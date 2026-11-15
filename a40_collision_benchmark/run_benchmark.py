#!/usr/bin/env python3
"""Portable GPU-resident Q(f,f) accuracy/throughput benchmark; see README.md."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
REPO_CHECKPOINT = REPO_ROOT / 'runs' / 'boltzmann_tail_simplified_gate_fno_w36_l3_neuralop_10000_b8_s1234_bkw' / 'best.pt'
DEFAULT_CHECKPOINT = REPO_CHECKPOINT if REPO_CHECKPOINT.is_file() else ROOT / 'checkpoints' / 'best.pt'
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.cache' / 'triton'))
os.environ.setdefault('CUPY_CACHE_DIR', str(ROOT / '.cache' / 'cupy'))
os.environ.setdefault('CUDA_CACHE_PATH', str(ROOT / '.cache' / 'cuda'))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.cache' / 'torchinductor'))
sys.path.insert(0, str(ROOT / 'vendor'))

import numpy as np
import torch
import cupy as cp
from boltzmann_fft_gpu import BoltzmannFFT
from boltzmann_simplified_gate_fno import load_simplified_gate_model
from homogeneous_initial import make_grid, initial_condition


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sync():
    # Same CUDA device; includes all Torch and CuPy streams.
    cp.cuda.runtime.deviceSynchronize()


def clean():
    gc.collect()
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()


@lru_cache(maxsize=None)
def canonical_initial(case, L):
    grid=make_grid(64,L)
    f,metadata=initial_condition(case,grid)
    return grid.v,f,metadata


def initial(case, n, L):
    if n not in (32,64):
        raise ValueError('Canonical homogeneous initial state supports Nv32/64')
    v,f,metadata=canonical_initial(case,L)
    stride=64//n
    return v[::stride].copy(),f[::stride,::stride,::stride].copy(),{
        **metadata,'normalization':'homogeneous Nv64 normalization only; exact node sampling at Nv32',
        'canonical_nv':64,'sampling_stride':stride}


def solver(args, n, dtype, angular, radial=None):
    return BoltzmannFFT(n=n, L=args.L, R=args.R,
        radial_accuracy=radial or args.radial, angular_accuracy=angular,
        omega=1.0, kernel_constant=1/(4*np.pi), backend='cupy',
        dtype=np.dtype(dtype).type, lebedev_dir=ROOT/'vendor'/'Lebedev',
        loss_mode='sinc', enforce_mass=True, use_rfft=True,
        angular_chunk=args.angular_chunk)


def metrics(q, ref, f, v):
    q, ref = q.astype(np.float64), ref.astype(np.float64)
    if not np.isfinite(q).all():
        raise FloatingPointError('Nonfinite collision output')
    dv3 = (v[1]-v[0])**3
    d = q-ref
    xyz = np.meshgrid(v, v, v, indexing='ij')
    psi = [np.ones_like(q), *xyz, sum(x*x for x in xyz)]
    return dict(relative_l1=float(np.abs(d).sum()/np.abs(ref).sum()),
        relative_l2=float(np.linalg.norm(d.ravel())/np.linalg.norm(ref.ravel())),
        absolute_l2=float(np.sqrt(dv3*np.square(d).sum())),
        absolute_linf=float(np.abs(d).max()),
        mass_rate=float(dv3*q.sum()),
        momentum_x_rate=float(dv3*(q*psi[1]).sum()),
        momentum_y_rate=float(dv3*(q*psi[2]).sum()),
        momentum_z_rate=float(dv3*(q*psi[3]).sum()),
        energy_rate=float(dv3*(q*psi[4]).sum()),
        entropy_rate=float(dv3*(q*np.log(np.maximum(f, 1e-30))).sum()))


def timed(call, args):
    for _ in range(args.warmup):
        call()
    sync()
    samples = []
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        out = call()
        sync()
        samples.append(time.perf_counter()-start)
        del out
    per = np.asarray(samples)*1000/args.batch_size
    return dict(batch_seconds=samples, mean_ms_per_state=float(per.mean()),
        median_ms_per_state=float(np.median(per)), p95_ms_per_state=float(np.quantile(per, .95)),
        std_ms_per_state=float(per.std(ddof=1)) if len(per)>1 else 0.0,
        min_ms_per_state=float(per.min()), max_ms_per_state=float(per.max()),
        states_per_second=float(1000/per.mean()))


def evaluate(args, f, n, dtype, method, angular=None):
    """Construct, validate, then time; no automatic OOM batch changes."""
    clean()
    sync()
    start = time.perf_counter()
    if method == 'neural':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model, config = load_simplified_gate_model(checkpoint, n=n)
        if float(config['L']) != args.L:
            raise ValueError('L differs from checkpoint domain')
        model.to(device='cuda', dtype=getattr(torch, dtype)).eval()
        del checkpoint
        state = torch.as_tensor(f, device='cuda', dtype=getattr(torch, dtype))[None].repeat(args.batch_size,1,1,1)
        chunk = args.batch_size
        call = lambda: model(state)
        copy = lambda x: x.detach().cpu().numpy()
        internal = {'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
                    'model_config': config}
    else:
        model = solver(args, n, dtype, angular)
        state = cp.repeat(cp.asarray(f, dtype=dtype)[None], args.batch_size, axis=0)
        chunk = args.batch_size
        call = lambda: model.collision(state, batch_chunk=chunk)
        copy = cp.asnumpy
        internal = dict(fft_dtype=str(model._values_to_hat_half(state[:1]).dtype),
                        loss_kernel_dtype=str(model.loss_kernel_half.dtype),
                        radial_nodes=len(model.radial_nodes), hemisphere_nodes=len(model.angular_weights))
    sync()
    setup_seconds = time.perf_counter()-start
    with torch.inference_mode():
        out = call()
        sync()
        if str(out.dtype).replace('torch.', '') != dtype:
            raise TypeError(f'Requested {dtype}, actual output {out.dtype}')
        batch_output = copy(out)
        q = batch_output[0]
        # Compare all repeated states, and compare against a true single-state call.
        spread = float(copy(abs(out-out[:1]).max()))
        single = model(state[:1]) if method == 'neural' else model.collision(state[:1], batch_chunk=1)
        single_q = copy(single[0])
        batch_difference = float(np.linalg.norm((q-single_q).ravel()) / max(np.linalg.norm(single_q.ravel()), 1e-30))
        if spread > (1e-5 if dtype == 'float32' else 1e-11) or batch_difference > (1e-4 if dtype == 'float32' else 1e-9):
            raise AssertionError(f'Batch consistency failed: {spread=}, {batch_difference=}')
        if not np.isfinite(q).all():
            raise FloatingPointError('Nonfinite output before timing')
        del out, single
        torch.cuda.reset_peak_memory_stats()
        timing = timed(call, args)
    info = dict(setup_seconds=setup_seconds, effective_chunk=chunk, output_dtype=str(q.dtype),
                batch_max_absolute_spread=spread, batch_vs_single_relative_l2=batch_difference,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cupy_pool_total_bytes=cp.get_default_memory_pool().total_bytes(), **internal)
    del call, model, state
    clean()
    return q, timing, info, batch_output


def reference(args, f, n, angular, radial):
    model = solver(args, n, 'float64', angular, radial)
    start = time.perf_counter()
    q = cp.asnumpy(model.collision(cp.asarray(f), batch_chunk=1))
    sync()
    seconds = time.perf_counter()-start
    del model
    clean()
    return q, seconds


def summary(out, rows):
    """Write data summaries only; plotting is a separate local postprocessing task."""
    with (out/'summary.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(out/'summary.json', rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nv',type=int,nargs='+',default=[32,64])
    p.add_argument('--angles',type=int,nargs='+',default=[7,9,11,13])
    p.add_argument('--dtypes',nargs='+',choices=['float32','float64'],default=['float32','float64'])
    p.add_argument('--cases',nargs='+',choices=['two_stream','three_stream'],default=['two_stream','three_stream'])
    p.add_argument('--L',type=float,default=10); p.add_argument('--R',type=float,default=9)
    p.add_argument('--radial',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--angular-chunk',type=int,default=4)
    p.add_argument('--warmup',type=int,default=5); p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT,
                   help='Weights path; default: sibling runs directory, then bundled checkpoints/best.pt')
    p.add_argument('--output',default=time.strftime('run_%Y%m%d_%H%M%S'))
    args=p.parse_args()
    args.reference_angle=23
    for key in ['radial','batch_size','angular_chunk','repeats']:
        if getattr(args,key)<1: p.error(f'{key} must be positive')
    if args.warmup<0 or args.L<=0 or args.R<=0 or any(n not in (32,64) for n in args.nv): p.error('Require Nv=32/64, positive domain, nonnegative warmup')
    if max(args.angles)>=args.reference_angle: p.error('Require tested angles < reference angle 23')
    out=(ROOT/'outputs'/args.output).resolve()
    if not out.is_relative_to(ROOT/'outputs'): p.error('Output must stay inside this folder/outputs')
    args.checkpoint=args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        p.error(f'Checkpoint not found: {args.checkpoint}. Pass --checkpoint with a path accessible on this node. Default candidates: {REPO_CHECKPOINT}, {ROOT / "checkpoints" / "best.pt"}')
    print(f'Checkpoint: {args.checkpoint}', flush=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.cuda.set_device(0); cp.cuda.Device(0).use()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(1234)
    out.mkdir(parents=True,exist_ok=False)
    log=(out/'run.log').open('a',buffering=1)
    def report(s): print(s,flush=True); print(s,file=log,flush=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    metadata=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        checkpoint_sha256=sha(args.checkpoint),checkpoint_model_config=checkpoint['model_config'],
        checkpoint_epoch=checkpoint.get('epoch'),python=sys.version,platform=platform.platform(),
        gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch_version=torch.__version__,cupy_version=cp.__version__,numpy_version=np.__version__,
        cuda_torch=torch.version.cuda,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        cuda_driver=cp.cuda.runtime.driverGetVersion(),tf32=False,
        sources={str(x.relative_to(ROOT)):sha(x) for x in [Path(__file__),*sorted((ROOT/'vendor').rglob('*.py')),*sorted((ROOT/'vendor'/'Lebedev').glob('*.txt'))]},
        timing='Synchronized wall time of full resident batch Q(f,f), divided by batch size; excludes setup, transfers, validation, serialization and reference. Both methods process the full batch in one call.',
        artifact_schema='Each method NPZ stores canonical f, actual dtype-cast f_input, q, full q_batch, sampled q_reference, grid v, and JSON metadata. These are single collision outputs, not time-evolved final distributions.',
        memory='Torch peak allocated and CuPy retained pool bytes are backend-specific, not combined device peak.',
        accuracy='Common Nv64 FP64 spectral a23 reference. Initial state is built once by the original homogeneous initial_condition on Nv64. Nv32 input and reference use exact coincident-node sampling [::2,::2,::2], without interpolation, low-pass filtering, or renormalization. Includes velocity discretization relative to Nv64; not an exact continuum solution.',
        reference_nv=64,reference_angle=23,
        status='running')
    del checkpoint
    write_json(out/'manifest.json',metadata)
    rows=[]
    try:
        references={}
        for case in args.cases:
            report(f'{case}: common Nv64 FP64 a23 reference')
            v64,f64,case_info=initial(case,64,args.L)
            ref64,seconds=reference(args,f64,64,23,args.radial)
            refpath=out/'reference_N64'/case; refpath.mkdir(parents=True)
            write_json(refpath/'reference.json',dict(n=64,angular=23,dtype='float64',
                case=case_info,seconds=seconds,radial_accuracy=args.radial))
            np.savez_compressed(refpath/'reference.npz',v=v64,f=f64,q=ref64)
            references[case]=(v64,f64,ref64,case_info)
        for n in args.nv:
            for case in args.cases:
                path=out/f'N{n}'/case; path.mkdir(parents=True)
                v64,f64,ref64,case_info=references[case]
                stride=64//n
                v,f,case_info=initial(case,n,args.L)
                selection=(slice(None,None,stride),)*3
                ref=ref64[selection].copy()
                expected_v=(np.arange(n)-n//2)*(2*args.L/n)
                np.testing.assert_allclose(v,expected_v,atol=1e-14,rtol=0)
                write_json(path/'reference.json',dict(case=case_info,reference_nv=64,
                    reference_angle=23,sampling_stride=stride,input_mass=float(f.sum()*(2*args.L/n)**3),
                    note='Canonical homogeneous Nv64 initial distribution and Q reference sampled at coincident nodes; no Nv32 renormalization.',
                    initial_vs_fine_sample_relative_l2=float(np.linalg.norm((f-f64[selection]).ravel())/np.linalg.norm(f64[selection].ravel()))))
                np.savez_compressed(path/'reference.npz',v=v,f=f,q=ref)
                for dtype in args.dtypes:
                    sub=path/dtype; sub.mkdir()
                    outputs={}; neural_time=None
                    for method,angular in [('neural',None),*[('spectral',a) for a in args.angles]]:
                        label=method if angular is None else f'spectral_a{angular}'
                        report(f'N={n} {case} {dtype} {label}: batch={args.batch_size}')
                        q,timing,info,q_batch=evaluate(args,f,n,dtype,method,angular)
                        outputs[label]=q
                        error=metrics(q,ref,f,v)
                        if method=='neural': neural_time=timing['mean_ms_per_state']
                        row=dict(n=n,case=case,dtype=dtype,method=label,angular=angular or 0,
                            reference_nv=64,reference_angle=23,
                            batch_size=args.batch_size,effective_chunk=info['effective_chunk'],
                            **error,**{k:v for k,v in timing.items() if k!='batch_seconds'},
                            speedup_over_neural=timing['mean_ms_per_state']/neural_time)
                        rows.append(row)
                        result=dict(**row,timing=timing,details=info,
                            L=args.L,R=args.R,radial_accuracy=args.radial,angular_chunk=args.angular_chunk,
                            checkpoint_sha256=metadata['checkpoint_sha256'],
                            neural_vs_this_spectral=metrics(outputs['neural'],q,f,v) if angular else None)
                        np.savez_compressed(sub/f'{label}.npz',v=v,f=f,
                            f_input=f.astype(dtype),q=q,q_batch=q_batch,q_reference=ref,
                            metadata_json=np.asarray(json.dumps(result,allow_nan=False)))
                        write_json(sub/f'{label}.json',result)
                        summary(out,rows)
                        report(f"  relL2={error['relative_l2']:.6g}; mean={timing['mean_ms_per_state']:.4f} ms/state")
        metadata['status']='complete'
    except BaseException:
        metadata['status']='failed'; metadata['error']=traceback.format_exc()
        report(metadata['error'])
        raise
    finally:
        write_json(out/'manifest.json',metadata); log.close()
    print(f'Results: {out}',flush=True)


if __name__=='__main__': main()
