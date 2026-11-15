"""Small runtime helpers shared by the one-off unsteady cases."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable

import torch
import numpy as np


def filename_float(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace("+", "").replace(".", "p")


def snapshot_targets(t_end: float, interval: float) -> list[float]:
    """Absolute output times, including the initial and exact terminal times."""
    if not math.isfinite(t_end) or t_end < 0:
        raise ValueError("T must be finite and nonnegative")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("snapshot-interval must be finite and positive")
    targets = [index * interval for index in range(math.floor(t_end / interval) + 1)]
    if math.isclose(targets[-1], t_end, rel_tol=0.0, abs_tol=32 * math.ulp(max(1.0, t_end))):
        targets[-1] = t_end
    else:
        targets.append(t_end)
    return targets


def clipped_time_step(time_value: float, nominal_dt: float, target: float) -> tuple[float, float]:
    """Shorten the last step to the target, allowing only clock roundoff.

    Preserve the nominal step when it already reaches the target to floating
    precision. This avoids tiny stochastic updates and keeps continuation at
    nominal-step boundaries bitwise reproducible.
    """
    remaining = target - time_value
    roundoff = 32 * math.ulp(max(abs(time_value), abs(target), nominal_dt))
    if abs(remaining - nominal_dt) <= roundoff:
        return nominal_dt, target
    if remaining < nominal_dt:
        return remaining, target
    return nominal_dt, time_value + nominal_dt


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def guard_output_directory(path: Path, *, overwrite: bool, resume: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not (overwrite or resume):
        raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite to replace it")


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    *,
    case: str,
    state: torch.Tensor,
    time_value: float,
    completed_steps: int,
    nominal_dt: float,
    signature: dict[str, Any],
    **extra: Any,
) -> None:
    payload = {
        "format_version": 1,
        "case": case,
        "state": state.detach().to(device="cpu").contiguous(),
        "time": float(time_value),
        "completed_steps": int(completed_steps),
        "nominal_dt": float(nominal_dt),
        "signature": dict(signature),
        **extra,
    }
    atomic_torch_save(path, payload)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    nominal_dt: float,
    signature: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload["signature"] != signature:
        raise ValueError("resume checkpoint parameters do not match this run")
    if not math.isclose(float(payload["nominal_dt"]), nominal_dt, rel_tol=1e-14):
        raise ValueError("checkpoint time step does not match this run")
    payload["state"] = payload["state"].to(device=device, dtype=torch.float64)
    return payload


def run_fixed_time(
    *,
    initial_state: Callable[[], torch.Tensor],
    advance: Callable[[torch.Tensor, float], torch.Tensor],
    t_end: float,
    nominal_dt: float,
    device: torch.device,
    case: str,
    progress: bool,
    checkpoint_path: Path | None,
    checkpoint_interval_seconds: float,
    resume_path: Path | None,
    signature: dict[str, Any],
    checkpoint_time_interval: float | None = None,
    snapshot_interval: float | None = None,
    snapshot_callback: Callable[[torch.Tensor, float, int], None] | None = None,
) -> tuple[torch.Tensor, int]:
    targets = [t_end] if snapshot_interval is None else snapshot_targets(t_end, snapshot_interval)
    if snapshot_interval is not None and snapshot_callback is None:
        raise ValueError("snapshot_callback is required with snapshot_interval")
    if resume_path is None:
        state, time_value, completed_steps = initial_state(), 0.0, 0
    else:
        payload = load_checkpoint(resume_path, device=device, nominal_dt=nominal_dt, signature=signature)
        state = payload["state"]
        time_value = float(payload["time"])
        if time_value > t_end:
            raise ValueError("checkpoint time exceeds requested T")
        completed_steps = int(payload["completed_steps"])
        if progress:
            print(f"resumed {resume_path} at t={time_value:.10g}, completed steps={completed_steps}", flush=True)

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, case=case, state=state, time_value=time_value,
                        completed_steps=completed_steps, nominal_dt=nominal_dt, signature=signature)
    if snapshot_callback is not None and time_value in targets:
        snapshot_callback(state, time_value, completed_steps)

    future_targets = [target for target in targets if target > time_value]
    previous_times = [time_value, *future_targets[:-1]]
    remaining_steps = sum(max(1, math.ceil((target - previous) / nominal_dt - 1e-12))
                          for previous, target in zip(previous_times, future_targets))
    final_steps = completed_steps + remaining_steps
    report_stride = max(1, remaining_steps // 10)
    last_checkpoint = time.monotonic()
    next_time_checkpoint = math.inf
    if checkpoint_time_interval is not None and checkpoint_time_interval > 0.0:
        next_time_checkpoint = (
            math.floor(time_value / checkpoint_time_interval + 1.0e-12) + 1
        ) * checkpoint_time_interval
    current_step = completed_steps
    interrupted = False

    def defer_interrupt(*_) -> None:
        nonlocal interrupted
        interrupted = True

    previous_sigint_handler = signal.signal(signal.SIGINT, defer_interrupt)
    try:
        target_index = 0
        while time_value < t_end:
            target = future_targets[target_index]
            step_dt, next_time = clipped_time_step(time_value, nominal_dt, target)
            state = advance(state, step_dt)
            time_value = next_time
            current_step += 1
            local_step = current_step - completed_steps
            snapshot_due = time_value == target
            report = progress and (local_step == 1 or local_step % report_stride == 0 or time_value == t_end)
            if report:
                minimum, maximum = torch.stack((torch.amin(state), torch.amax(state))).detach().cpu().tolist()
                print(f"step {current_step}/{final_steps}, t={time_value:.10g}, min(f)={minimum:.6e}", flush=True)
                if not math.isfinite(minimum) or not math.isfinite(maximum):
                    raise FloatingPointError(
                        f"state contains nonfinite values after step {current_step}/{final_steps}, "
                        f"t={time_value:.10g}, min(f)={minimum}, max(f)={maximum}")
            time_checkpoint_due = time_value + 1.0e-12 >= next_time_checkpoint
            wall_checkpoint_due = (
                checkpoint_path is not None
                and checkpoint_interval_seconds > 0.0
                and time.monotonic() - last_checkpoint >= checkpoint_interval_seconds
            )
            if checkpoint_path is not None and (time_checkpoint_due or wall_checkpoint_due or
                                                (snapshot_due and snapshot_callback is not None)):
                save_checkpoint(checkpoint_path, case=case, state=state, time_value=time_value,
                                completed_steps=current_step, nominal_dt=nominal_dt, signature=signature)
                last_checkpoint = time.monotonic()
                while time_value + 1.0e-12 >= next_time_checkpoint:
                    next_time_checkpoint += checkpoint_time_interval
            if snapshot_due:
                if snapshot_callback is not None:
                    snapshot_callback(state, time_value, current_step)
                target_index += 1
            if interrupted:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, case=case, state=state, time_value=time_value,
                            completed_steps=current_step, nominal_dt=nominal_dt, signature=signature)
            print(f"interrupted; checkpoint saved at t={time_value:.10g}: {checkpoint_path}", file=sys.stderr, flush=True)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError("final state contains nonfinite values")
    if checkpoint_path is not None and snapshot_callback is None:
        save_checkpoint(checkpoint_path, case=case, state=state, time_value=time_value,
                        completed_steps=current_step, nominal_dt=nominal_dt, signature=signature)
    return state, current_step
