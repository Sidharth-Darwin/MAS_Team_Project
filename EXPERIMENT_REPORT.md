# Executed verification and initial numerical experiment

Date: 2026-10-31. Environment: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10 CPU, Matplotlib 3.10.8. No CUDA device detected.

## Verification

Executed `python verify_math.py`: **32 passed, 1 skipped** (CUDA-only comparison). This includes negative tests for incompatible agent matrices, directed/disconnected graph input, unstable targets, invalid input coefficients; matching, pseudoinverse, exact dynamics, spectral projection, frozen inference, cost terms, terminal and threshold handling, and safe checkpoint loading.

A separate three-size, two-activation, three-effort-weight **full-suite execution smoke test** completed all 18 training configurations with only 3 optimization epochs per configuration; it demonstrates that the experiment orchestration executes, **not** that these configurations are converged or statistically validated. Its full logs are available separately, not used for the main numerical conclusions.

## Reproducible initial training setup

`python main.py --suite quick --run-name verified_final --epochs 100 --horizon 70 --batch 96 --eval-samples 16 --inference-sizes 100 1000`

- Reproducible seed: 16; training system uses physical seed 316 (3 agents).
- Target `[[1,.1],[-.2,.8]]`; eigenvalues `0.9 +/- 0.1j`; target spectral radius `~0.905539`.
- Connected undirected training graph: path on 3 agents; true `L_rw=I-D^-1 A`.
- Exact-matching maximum residual: `0` at reported precision.
- Scalar transfer gain `gamma0 = 6.90423107655`, peak `omega = 0.12349094616`.
- Train full-network gain `gamma_train = 1.75168738533`, `eta=.95`; cap `0.5423342`.
- Training-system baseline cost scale: `J_E0=96.194`, `J_U0=54.925`; ratio `~1.7514`; positive lambdas scale to 2% and 10% of that ratio.
- 16 held-out test trajectories per evaluation, same initial conditions for paired comparisons. Only the identity activation was optimized to 100 epochs in this initial run.

## Three-agent held-out comparisons (means over 16 trajectories)

| Training lambda | Learned W (approx) | ||W||2 | Mean J_E (baseline = 106.457) | Mean J_U (baseline = 52.793) | J_E reduction | J_U reduction | Pairwise J_E win rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `[-.070206, -.178565]` | 0.191871 | 99.6462 | 48.8553 | 6.398% | 7.459% | 100% |
| 0.0350272 | `[-.058096, -.177540]` | 0.186803 | 99.7146 | 48.5200 | 6.334% | 8.094% | 100% |
| 0.175136 | `[-.019228, -.173777]` | 0.174838 | 100.0167 | 47.5570 | 6.050% | 9.918% | 100% |

For the `lambda=0` controller, the training loop-gain product is `~0.33610` (<`0.95`), so its numerical sufficient small-gain condition holds on training system. Mean finite-horizon settling time (1% initial disagreement norm, observed through horizon 70) was `50.375` baseline versus `45.125` learned. First crossing was `50.375` baseline versus `44.6875` learned. These two metrics are distinct.

## Frozen-W larger-network inference: lambda=0

| N | Graph | gamma_N | gamma_N / gamma_train | loop gain g_N | Mean disagreement improvement | Mean physical energy reduction |
|---:|---|---:|---|---:|---:|---:|
| 100 | ring | 1.734876 | below | 0.332872 | 6.060% | 7.263% |
| 100 | wide ring | 1.087902 | below | 0.208737 | 4.218% | 8.214% |
| 1000 | ring | 1.755075 | above | 0.336748 | 5.917% | 6.382% |
| 1000 | wide ring | 1.108232 | below | 0.212638 | 4.102% | 7.006% |

The same frozen weight was used across all inference scenarios without retraining, projection, or rescaling. Both `gamma_N <= gamma_train` and `gamma_N > gamma_train` were observed without manipulating physical input matrices to manufacture an outcome. In particular, gamma_N exceeding gamma_train on the 1000-agent ring did **not** cause `g_N` to exceed 1. The learned controller improved average transient disagreement under both topology classes for this seed and test set. No universal size-independent certificate follows from this evidence.

## Limitations and negative findings

- The project provides a complete rational stationary-point calculation for the two-state scalar gain **in exact mathematical formulation**. Its floating-point implementation does not produce an interval-certified upper bound; a formal numerical certificate would require validated rounding/error enclosures. The independent dense frequency-grid diagnostics agree to resolution, but the grid itself is not a proof.
- Gains and trade-offs shown are empirical, sample-dependent and from a single initial 100-epoch experiment, not multi-seed statistical guarantees. Performance is not proven for all conditions; larger network generalization depends on physical parameters and graph topology.
- The wide-ring and ring graphs were chosen as structurally different *undirected* graphs; they share sampled physical agents at each N to limit confounding, but topology differences prevent claims based only on N.
- Full N_train=5,10 and tanh experiments have been **smoke-tested**, not trained to convergence in this delivered short-run report. Execute the configurable full suite for their meaningful research comparison.
- No CUDA hardware was available, so CUDA numerical equivalence remains unverified.
- Model omits actuator saturation, realistic robot hardware, adversarial disturbances, collisions, real-time networking and Pygame. The core interface is ready for those applications to import without altering the research controller.

## Files and exact data

See `results/verified_final/configuration.json`, `overview.csv`, `run_summary.json`, each training subfolder's `summary.json`, `training_history.csv`, per-trajectory paired CSV, trajectory NPZ, checkpoint NPZ, and plots (`PNG` and `SVG`). `inference_systems/` holds all exact physical matrices and sparse graph edge data for the tested large systems.
