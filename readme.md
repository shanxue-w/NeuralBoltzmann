# Learning the Boltzmann Collision Operator with Structural Constraints for Fast Kinetic Simulation

**Hao Wang and Zhenning Cai**  
Department of Mathematics, National University of Singapore

Implementation and numerical experiments for learning the nonlinear Boltzmann collision operator $f \mapsto \mathcal{Q}[f,f]$ and embedding it in deterministic kinetic solvers:

$$
\partial_t f + \mathbf{v}\cdot\nabla_{\mathbf{x}} f = \frac{1}{\mathrm{Kn}}\mathcal{Q}[f,f].
$$

The model pairs a 3D Fourier neural operator with a structured gain-loss parameterization, conservative moment projection, and entropy correction (for the normalized constant collision kernel $\mathcal{B} = 1/(4\pi)$). By construction, it enforces:

- **Conservation** of mass, momentum, and energy
- **Equilibrium preservation** and **entropy dissipation**
- **Quadratic density scaling**, with a gain–loss form supporting **positivity-preserving updates**

---

## Requirements

- **Core:** Python 3.11+, NumPy, PyTorch, Matplotlib
- **GPU & acceleration:** NVIDIA GPU (paper timings on A40), CuPy (for spectral evaluation), Triton (for fused inference kernels; required by the steady Couette solver)
- **DSMC references:** Numba

---

## Repository structure

### Model & spectral operators
| Path | Description |
| --- | --- |
| `boltzmann_simplified_gate_fno.py` | Structured learned collision operator and checkpoint loader |
| `boltzmann_fno_backbone.py`, `boltzmann_spectral.py` | 3D FNO layers and spectral readouts |
| `boltzmann_discrete_maxwellian.py` | Discrete moment-matched equilibrium solver |
| `boltzmann_triton.py` | Fused Triton GPU kernels |
| `boltzmann_fft_gpu.py`, `linear_collision_gpu.py` | Fourier spectral and linearized collision routines |
| `Lebedev/` | Angular quadrature tables |

### Data & training
| Path | Description |
| --- | --- |
| `generate_boltzmann_dataset.py` | Distribution sampling and spectral-label utilities |
| `generate_prepared_boltzmann_dataset.py`, `npy_direct_io.py` | Resumable train/val/test NPY dataset generation |
| `train_boltzmann_simplified_gate_fno.py` | Model training entry point |
| `reproduce_gate_test_metrics.py` | Test-set comparison with moment-matched BGK |
| `onsager_collision/` | Additional collision models and shared training/data utilities |

### Benchmarks & kinetic experiments
| Path | Description |
| --- | --- |
| `a40_collision_benchmark/` | Collision accuracy and GPU timing benchmarks |
| `experiments/bkw/` | Homogeneous BKW relaxation (analytic reference) |
| `experiments/homogeneous/` | Two-/three-stream relaxation, angular sweeps, and structural checks |
| `experiments/discontinuous/` | Generalization to discontinuous velocity distributions |
| `SteadyBoltzmann/` | 1D steady Couette flow and spectral refinement |
| `UnsteadyBoltzmann/sod_shock_1d/` | 1D normal shock and DSMC comparisons |
| `UnsteadyBoltzmann/lid_cavacity_2d/` | 2D lid-driven cavity and DSMC comparisons *(spelled `lid_cavacity_2d` in tree)* |

---

## Dataset generation and training

Samples are drawn from smooth Gaussian mixtures (with optional nonnegative polynomial modulation and interpolation toward a moment-matched equilibrium) and labeled by the Fourier spectral method on a $64^3$ velocity grid ($L=10$, $R=9$, Lebedev accuracy 23). The default 8,000 / 1,000 / 1,000 split takes about 42 GB in `float64`, writes `dataset.json`, `v.npy`, and per-split `f.npy`/`q.npy`, and supports `--resume`.

```bash
DATA_DIR="$PWD/boltzmann_dataset_n64_L10_R9_a23_npy"
RUN_DIR="$PWD/runs/gate_fno"

# 1. Generate dataset
python generate_prepared_boltzmann_dataset.py \
  --backend cupy --n 64 --L 10 --R 9 --angular-accuracy 23 \
  --num-samples 10000 --train 8000 --val 1000 --test 1000 \
  --output-dir "$DATA_DIR"

# 2. Train model
python train_boltzmann_simplified_gate_fno.py \
  --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
  --fno-width 36 --fno-layers 3 --fno-modes 8 \
  --batch-size 8 --epochs 100 --seed 1234 --device cuda

# 3. Compare test errors against moment-matched BGK
python reproduce_gate_test_metrics.py \
  --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" --device cuda

CHECKPOINT="$RUN_DIR/best.pt"
```

Training saves its configuration, history, `best.pt`, `latest.pt`, and test metrics under `RUN_DIR`.

---

## Numerical experiments

All kinetic tests below reuse the same trained `CHECKPOINT` without retraining (or point `CHECKPOINT` to an existing checkpoint path).

### Collision accuracy and GPU timing

Compares learned and spectral evaluations against a $64^3$ FP64 spectral reference (Lebedev accuracy 23) using the bundled `vendor/` implementation. Results are saved under `a40_collision_benchmark/outputs/` (output names must be unique).

```bash
python a40_collision_benchmark/run_benchmark.py \
  --checkpoint "$CHECKPOINT" --cases two_stream \
  --nv 32 64 --angles 7 9 11 --dtypes float32 float64 \
  --output two_stream_benchmark
```

### Homogeneous relaxation

Each runner saves its configuration, trajectories, errors, and plots under its own `outputs/` directory.

```bash
# Analytic BKW relaxation
python experiments/bkw/run_bkw.py \
  --checkpoint "$CHECKPOINT" --K 0.6 --device cuda

# Asymmetric three-stream relaxation (also supports --case two_stream)
python experiments/homogeneous/run_homogeneous.py \
  --checkpoint "$CHECKPOINT" --case three_stream --device cuda

# Discontinuous two-stream distribution (jump across v_y = 0)
python experiments/discontinuous/run_discontinuous.py \
  --checkpoint "$CHECKPOINT" --device cuda
```

**Structural diagnostics (three-stream):** Verifies the archived rollout (including the checkpoint hash), checks nonnegativity at all time levels and SSPRK2 stages, and exports three PDF figures showing physical moment drifts, $\min_j f_j(t)$ at the time levels, and discrete relative entropy $\mathscr{H}_h(f(t)\mid M[f_0])$.

```bash
python experiments/homogeneous/verify_three_stream_structure.py \
  --case-dir experiments/homogeneous/outputs/three_stream --device cuda

python experiments/homogeneous/plot_three_stream_structure.py \
  --output-dir figures/three_stream_structure
```

### Spatially inhomogeneous flows

**Steady Couette flow** (with BGK comparison and spectral refinement):

```bash
python -m SteadyBoltzmann.run_couette_kn_sweep_1d \
  --checkpoint "$CHECKPOINT" --nx 32 --nv 64
```

**1D normal shock** (conservative semi-Lagrangian CWENO3 transport with Strang splitting):

```bash
python UnsteadyBoltzmann/sod_shock_1d/run_strang.py \
  --checkpoint "$CHECKPOINT" --nx 100 --nv 32 \
  --kn 0.1 --mach 2 --T 50 --dt 0.035 --mesh sinh --stretch 5
```

**2D lid-driven cavity** (uniform spatial grid):

```bash
python UnsteadyBoltzmann/lid_cavacity_2d/run.py \
  --checkpoint "$CHECKPOINT" --nx 32 --ny 32 --nv 64 \
  --kn 1 --T 5 --dt 0.005 --stretch 0 \
  --velocity-half-width 6 --lid-speed 1 --wall-temperature 1 \
  --time-integrator euler
```

**DSMC references:** Independent DSMC and ensemble scripts (`run_dsmc*.py`) are provided in `UnsteadyBoltzmann/sod_shock_1d/` and `UnsteadyBoltzmann/lid_cavacity_2d/`; use `--help` to configure particle counts, time steps, realizations, and outputs.

---

## Citation

If you use this code in your research, please cite our [paper](https://arxiv.org/abs/2611.28864):

```bibtex
@misc{wang2026learningboltzmanncollisionoperator,
  title={Learning the Boltzmann Collision Operator with Structural Constraints for Fast Kinetic Simulation},
  author={Hao Wang and Zhenning Cai},
  year={2026},
  eprint={2611.28864},
  archivePrefix={arXiv},
  primaryClass={math.NA},
  url={https://arxiv.org/abs/2611.28864},
}
```
