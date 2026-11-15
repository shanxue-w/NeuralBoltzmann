"""Stream FFT Boltzmann labels directly into a resumable NPY dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from generate_boltzmann_dataset import (
    ALPHA_BANDS,
    COMPONENT_COUNTS,
    KERNEL_CONSTANT,
    POLYNOMIAL_MODULATION_PROBABILITY,
    SAMPLING_FAMILY,
    TARGET_MASS,
    build_solver,
    sample_distributions,
    synchronize_backend,
    to_numpy,
    truncation_condition_value,
)
from npy_direct_io import NpyDirectWriter


GIB = 1024**3


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": "boltzmann_direct_npy",
        "backend": args.backend,
        "n": args.n,
        "L": args.L,
        "R": args.R,
        "num_samples": args.num_samples,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "test_count": args.test_count,
        "radial_accuracy": args.n,
        "angular_accuracy": args.angular_accuracy,
        "angular_chunk": args.angular_chunk,
        "batch_chunk": args.batch_chunk,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "storage_dtype": args.storage_dtype,
        "sampling_family": SAMPLING_FAMILY,
        "use_rfft": not args.no_rfft,
        "collision_kernel_homogeneity": 0.0,
    }


def _split_plan(args: argparse.Namespace) -> list[str]:
    plan = ["train"] * args.train_count + ["val"] * args.val_count + ["test"] * args.test_count
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(plan)
    return plan


def _alpha_slot_plan(args: argparse.Namespace) -> list[int]:
    repeats = math.ceil(args.num_samples / len(ALPHA_BANDS))
    plan = list(range(len(ALPHA_BANDS))) * repeats
    rng = np.random.default_rng(args.seed + 130_363)
    rng.shuffle(plan)
    return plan[:args.num_samples]


def _required_array_bytes(args: argparse.Namespace) -> int:
    return 2 * args.num_samples * args.n**3 * np.dtype(args.storage_dtype).itemsize


def _check_free_space(args: argparse.Namespace, partial_dir: Path) -> None:
    free = shutil.disk_usage(args.output_dir.parent.resolve()).free
    required = _required_array_bytes(args)
    progress_path = partial_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress["completed_groups"])
        required = math.ceil(required * max(0.0, 1.0 - completed / args.num_samples))
    reserve = int(args.free_space_reserve_gib * GIB)
    if required + reserve > free:
        raise OSError(f"insufficient free space: need {(required + reserve) / GIB:.2f} GiB, have {free / GIB:.2f} GiB")


def _open_arrays(
    partial_dir: Path, args: argparse.Namespace, *, create: bool
) -> dict[str, dict[str, NpyDirectWriter]]:
    dtype = np.dtype(args.storage_dtype)
    arrays: dict[str, dict[str, NpyDirectWriter]] = {}
    for split, count in (("train", args.train_count), ("val", args.val_count), ("test", args.test_count)):
        split_dir = partial_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shape = (count, args.n, args.n, args.n)
        arrays[split] = {
            "f": NpyDirectWriter(split_dir / "f.npy", dtype=dtype, shape=shape, create=create),
            "q": NpyDirectWriter(split_dir / "q.npy", dtype=dtype, shape=shape, create=create),
        }
    return arrays


def _load_or_create(
    args: argparse.Namespace, partial_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, NpyDirectWriter]], np.random.Generator]:
    progress_path = partial_dir / "progress.json"
    if partial_dir.exists():
        if not args.resume:
            raise FileExistsError(f"incomplete dataset exists at {partial_dir}; pass --resume")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        arrays = _open_arrays(partial_dir, args, create=False)
        rng = np.random.default_rng(args.seed)
        if progress["rng_state"] is not None:
            rng.bit_generator.state = progress["rng_state"]
        return progress, arrays, rng

    if args.output_dir.exists():
        raise FileExistsError(f"completed output already exists: {args.output_dir}")
    partial_dir.mkdir(parents=True)
    progress = {
        "contract": _contract(args),
        "completed_groups": 0,
        "written": {"train": 0, "val": 0, "test": 0},
        "q_sum2": {"train": 0.0, "val": 0.0, "test": 0.0},
        "q_abs_max": {"train": 0.0, "val": 0.0, "test": 0.0},
        "rng_state": None,
        "generation_seconds": 0.0,
    }
    arrays = _open_arrays(partial_dir, args, create=True)
    rng = np.random.default_rng(args.seed)
    progress["rng_state"] = rng.bit_generator.state
    _atomic_json(progress_path, progress)
    return progress, arrays, rng


def _direct_collision_labels(solver, states: np.ndarray, *, batch_chunk: int) -> np.ndarray:
    state_device = solver.xp.asarray(states, dtype=solver.dtype)
    labels = solver.collision(state_device, batch_chunk=batch_chunk)
    synchronize_backend(solver.backend)
    return np.asarray(to_numpy(labels), dtype=np.float64)


def _flush(arrays: dict[str, dict[str, NpyDirectWriter]]) -> None:
    for split_arrays in arrays.values():
        split_arrays["f"].flush()
        split_arrays["q"].flush()


def _final_manifest(
    args: argparse.Namespace,
    solver,
    progress: dict[str, Any],
    split_plan: list[str],
    alpha_slot_plan: list[int],
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    point_count = args.n**3
    for name, count in (("train", args.train_count), ("val", args.val_count), ("test", args.test_count)):
        alpha_band_counts = [0] * len(ALPHA_BANDS)
        for split, alpha_slot in zip(split_plan, alpha_slot_plan, strict=True):
            if split == name:
                alpha_band_counts[alpha_slot] += 1
        splits[name] = {
            "count": count,
            "q_rms": float(math.sqrt(progress["q_sum2"][name] / (count * point_count)) if count else 0.0),
            "q_abs_max": float(progress["q_abs_max"][name]),
            "source": "direct_streaming_generation",
            "base_family_counts": {SAMPLING_FAMILY: split_plan.count(name)},
            "alpha_band_counts": alpha_band_counts,
        }
    return {
        "format": "boltzmann_fq_npy",
        "input_dir": None,
        "dtype": np.dtype(args.storage_dtype).name,
        "group_size": 1,
        "seed": args.split_seed,
        "shape": [args.n, args.n, args.n],
        "n": args.n,
        "L": args.L,
        "R": args.R,
        "dv": float(solver.dv),
        "dv3": float(solver.dv3),
        "kernel_constant": KERNEL_CONSTANT,
        "collision_kernel_homogeneity": 0.0,
        "omega": float(solver.omega),
        "radial_accuracy": int(solver.radial_accuracy),
        "angular_accuracy": int(solver.angular_accuracy),
        "source_dtype": "float64",
        "target_mass": TARGET_MASS,
        "sampling_family": SAMPLING_FAMILY,
        "component_count_sampling": {"kind": "uniform", "values": list(COMPONENT_COUNTS)},
        "polynomial_modulation": {
            "form": "epsilon + (1-epsilon) |v-u|^2 / (3 T)",
            "center": "discrete mean velocity of the Gaussian mixture",
            "applied_probability": POLYNOMIAL_MODULATION_PROBABILITY,
            "epsilon_sampling": {
                "zero_probability": 0.10,
                "near_zero_probability": 0.20,
                "near_zero_range": [0.0, 0.10],
                "regular_probability": 0.70,
                "regular_range": [0.10, 1.0],
            },
        },
        "independent_states": True,
        "base_family_counts": {SAMPLING_FAMILY: len(split_plan)},
        "collision_evaluations": args.num_samples,
        "label_generation": "direct FFT evaluation for every independent state",
        "alpha_sampling": {"kind": "stratified_uniform", "bands": [list(band) for band in ALPHA_BANDS]},
        "generation_seconds": float(progress["generation_seconds"]),
        "splits": splits,
    }


def generate(args: argparse.Namespace) -> Path:
    partial_dir = args.output_dir.with_name(args.output_dir.name + ".partial")
    _check_free_space(args, partial_dir)
    progress, arrays, rng = _load_or_create(args, partial_dir)
    plan = _split_plan(args)
    alpha_plan = _alpha_slot_plan(args)

    setup_start = time.perf_counter()
    solver = build_solver(
        n=args.n, L=args.L, R=args.R, angular_accuracy=args.angular_accuracy, backend=args.backend,
        angular_chunk=args.angular_chunk, use_rfft=not args.no_rfft)
    synchronize_backend(args.backend)
    print(f"quadrature radial={solver.radial_accuracy} angular={solver.angular_accuracy}; "
          f"setup={time.perf_counter() - setup_start:.3f}s", flush=True)

    completed = int(progress["completed_groups"])
    total_groups = len(plan)
    while completed < total_groups:
        batch_groups = min(args.groups_per_batch, total_groups - completed)
        started = time.perf_counter()
        v = to_numpy(solver.velocity_grid())
        alpha_slots = alpha_plan[completed : completed + batch_groups]
        f_host, _ = sample_distributions(
            v, count=batch_groups, rng=rng, L=solver.L, R=solver.R, alpha_slots=alpha_slots)
        q_host = _direct_collision_labels(solver, f_host, batch_chunk=args.batch_chunk)

        for local_group in range(batch_groups):
            split = plan[completed + local_group]
            start = int(progress["written"][split])
            stop = start + 1
            arrays[split]["f"][start:stop] = f_host[local_group:local_group + 1]
            arrays[split]["q"][start:stop] = q_host[local_group:local_group + 1]
            q64 = q_host[local_group:local_group + 1]
            progress["q_sum2"][split] += float(np.sum(q64 * q64))
            progress["q_abs_max"][split] = max(float(progress["q_abs_max"][split]), float(np.max(np.abs(q64))))
            progress["written"][split] = stop

        elapsed = time.perf_counter() - started
        completed += batch_groups
        progress["completed_groups"] = completed
        progress["rng_state"] = rng.bit_generator.state
        progress["generation_seconds"] += elapsed
        _flush(arrays)
        _atomic_json(partial_dir / "progress.json", progress)
        print(f"samples {completed}/{args.num_samples}; batch={elapsed:.3f}s", flush=True)

    _flush(arrays)
    for split_arrays in arrays.values():
        split_arrays["f"].close()
        split_arrays["q"].close()
    del arrays
    np.save(partial_dir / "v.npy", to_numpy(solver.velocity_grid()).astype(np.float64))
    manifest = _final_manifest(args, solver, progress, plan, alpha_plan)
    _atomic_json(partial_dir / "dataset.json", manifest)
    (partial_dir / "progress.json").replace(partial_dir / "generation.json")
    partial_dir.replace(args.output_dir)
    print(f"completed dataset: {args.output_dir}", flush=True)
    return args.output_dir / "dataset.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--R", type=float, default=9.0)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--train", dest="train_count", type=int, default=8_000)
    parser.add_argument("--val", dest="val_count", type=int, default=1_000)
    parser.add_argument("--test", dest="test_count", type=int, default=1_000)
    parser.add_argument("--angular-accuracy", type=int, default=23)
    parser.add_argument("--angular-chunk", type=int, default=16)
    parser.add_argument("--batch-chunk", type=int, default=16)
    parser.add_argument("--groups-per-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--storage-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--free-space-reserve-gib", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-rfft", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("boltzmann_dataset_n64_L10_R9_r64_a23_10000_mixed_gaussian_npy_float64"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    condition = truncation_condition_value(args.L, args.R)
    print(f"truncation condition satisfied={condition <= args.L}; required L>={condition:.6f}", flush=True)
    generate(args)


if __name__ == "__main__":
    main()
