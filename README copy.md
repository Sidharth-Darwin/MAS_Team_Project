# Distributed Neural Consensus Control — Executable Research Project

**Architecture:** one shared learnable `W` of shape `(2,1)` applied to the one-hop random-walk-Laplacian disagreement. The main controller is linear (`identity`), with `tanh` offered only as a configured comparison. There are no extra learned parameters, no hidden layers and no bias. Every rollout updates the original heterogeneous physical matrices.

## Setup and execution

From the `distributed_consensus` directory, with Python >=3.10:

```bash
python -m pip install -e '.[test]'
python verify_math.py
python main.py --suite quick --run-name my_first_run
```

The **quick suite** trains 3 agents with identity activation and 3 effort weights; it evaluates the frozen controller on 3, 100 and 1000 agents on a `ring` and an 8-neighbor-per-side `wide_ring` graph. The **full suite** adds training sizes 5 and 10, and tanh, for 3 x 2 x 3 = 18 training runs by default. Expect the full suite to take substantially longer.

```bash
python main.py --suite full --run-name exhaustive --epochs 300 --horizon 100 \
    --batch 256 --eval-samples 48 --device auto
python main.py --suite quick --lambda-values 0 0.03 0.15 --epochs 300
python main.py --help
```

The `--lambda-values` override is optional. Otherwise the program first computes local-only training-system costs and uses `lambda = [0, 0.02 J_E0/J_U0, 0.1 J_E0/J_U0]`. These are **scale-normalized exploratory** choices, not an assertion that one is optimal. All test comparisons within a train size use identical initial-state random seeds, fixed physical agents and graph, and matching starting weights. Train initial states are newly sampled each epoch from a reproducible generator; held-out validation uses a distinct generator and is fixed.

This package assumes `m=1`, `p=2` (as requested). The numerical gain factorization and the scalar maximizer implementation also rely on this structure. For default training examples, the first target row is `[1,0.1]`; each physical agent has an independently sampled second row and independent nonzero `b_i`, without any mass-spring-damper interpretation. Exact matching is checked and incompatible systems are rejected. **Controllability is never required.**

## The seven equations, exactly mapped to Python

| Eq. | Mathematics | Implementation |
|:---:|---|---|
| 1 | `x_i(k+1) = A_i x_i(k) + B_i u_i(k)` | `plants.py` and `dynamics.step_numpy`, `TorchWorld.step` |
| 2 | `K_i = pinv(B_i)(A_target - A_i)` | `plants.build_system` (uses `np.linalg.pinv`; compatibility checked) |
| 3 | `L_rw = I - D^{-1} A_g`, `E_mat = L_rw X_mat` | `graphs.Graph.laplacian`, `.error`, `TorchWorld.error` |
| 4 | `V_mat = phi(E_mat W)` | `controllers.correction_numpy`, `dynamics.TorchWorld.step` |
| 5 | `u_i = K_i x_i + v_i` | `dynamics.step_numpy` / `TorchWorld.step`, saved separate components |
| 6 | `E^+ = (I_N ⊗ A_target) E + (L_rw ⊗ I_2) B_block V` | `tests/test_math.py::test_16_centralized_error_agreement` verifies original physical updates equal this expression |
| 7 | `gamma_N = ||(zI_{2N} - A_e)^{-1} B_e||_H∞`; `||W||_2 <= 0.95 / gamma_train` | `plant_gain.full_gain`, `training.train`, `controllers.project_spectral_` |

**Objective:** `training.TorchWorld.costs` adds `sum_{k=0}^{T-1} ||E_mat(k)||_F²` and `lambda * sum_{k=0}^{T-1} ||U_mat(k)||_F²`, where `U_mat` is **total physical input**. The terminal disagreement is computed separately. `training.train` holds the environment fixed and updates only `W` with projected Adam. All matrices use double precision.

## Exact frequency maximization characterization (not a formal interval certificate)

For the two-state target `A = [[a11,a12],[a21,a22]]`, with `tau = trace(A)` and `delta = det(A)`, let `t = cos(omega)` and `b0 = [0,1]^T`. The squared SISO-input/vector-output frequency gain is

```text
F(t) = [a12^2 + 1 + a11^2 - 2*a11*t] /
       [4*delta*t^2 - 2*tau*(1+delta)*t + (1-delta)^2 + tau^2]
     for -1 <= t <= 1.
```

The denominator is positive on this interval for Schur-stable `A`. Hence `max F(t)` is obtained by checking `t=-1`, `t=1`, and **all real roots in [-1,1] of `F'(t)=0`**, a polynomial of degree at most 2. `plant_gain.scalar_gain` computes these candidates, not a finite frequency grid. This is a complete global maximization procedure in exact arithmetic; however, the implemented floating-point roots and spatial singular values **are numerical estimates, not certified outward-rounded interval bounds**. The separate `direct_grid_gain` is only a diagnostic verification on small graphs and deliberately not a certificate.

Because `B_block = diag(b_i) ⊗ [0,1]^T` and the same `A_target` is used for every agent, the transfer matrix factorizes exactly as `(L_rw diag(b_i)) ⊗ [(zI-A_target)^{-1}[0,1]^T]`. The full plant gain is therefore `gamma0 * ||L_rw diag(b_i)||_2`. The large spatial spectral norm is computed with sparse eigenvalue operations, avoiding any `2N x 2N` dense Kronecker matrix. `small_gain` reports `gamma_N*||W||_2`, separately from actual convergence, and uses the sufficient-condition interpretation. **For formal floating-point proof certificates, validated interval arithmetic and error-enclosed singular-value bounds are still needed.**

## Public interfaces for future independent formation visualization

The core does **not** use or import Pygame, and performs no rendering:

```python
import numpy as np
from helper_functions.reporting import load_checkpoint
from helper_functions.dynamics import step_numpy, rollout_numpy

system, graph, W, activation, lam = load_checkpoint(
    'results/my_first_run/trainN3_identity_lambda0/checkpoint.npz')
initial = np.zeros((system.n, 2))
initial[:,0] = [1., -1., .4]
step0 = step_numpy(initial, system, graph, W, activation=activation)
# step0['states'], ['next_states'], ['disagreement'], ['local'],
# ['correction'], ['total'] have distinct, documented shapes.
base = rollout_numpy(initial, system, graph, W=None, horizon=100)
learned = rollout_numpy(initial, system, graph, W=W, activation=activation, horizon=100)
```

Users can supply compatible physical matrices directly through `plants.build_system(A,B,A_target)` and their own connected symmetric adjacency through `graphs.Graph(...)`. They need not use random generation. For a future stationary formation, an external application can pass error coordinates `[p_i-r_i, v_i]` with physical `A_i=[[1,h],[0,d_i]]`, `B_i=[[0],[b_i]]`, target first row `[1,h]`, and fixed offsets. For moving formation, the external wrapper must add the prescribed reference feedforward and include it in reported actuator energy. Two planar axes can be independent copies of the one-dimensional project. The project itself models neither geometry/collision avoidance nor actuator limits.

**The simulation remains separate.** There is no `formation_pygame_demo` dependency or implementation.

## Generated artifacts

Each unique run is placed at `results/RUN_NAME/`, with:

- A `configuration.json`, `overview.csv`, and `run_summary.json`.
- `inference_systems/` with the exact large-network matrices, gains, and sparse graph edges for reproducibility.
- A folder for each `(N_train, activation, lambda)` with numeric-only `checkpoint.npz` (load using `np.load(..., allow_pickle=False)`), histories, per-sample paired CSVs, summary JSON, mean disagreement curve CSVs, and PNG/SVG figures.
- Complete representative trajectories for each comparison as compressed numeric-only `.npz`, including original states, local inputs, learned corrections, total physical input, and disagreement.

`J_E` and `J_U` report cumulative **squared** costs, not averaged per timestep; metrics are averaged over held-out trajectories when prefixed `mean_`. The initial threshold is `0.01*||E(0)||_2`. `first_crossing` is the first timestep at/below threshold; `settling_time` is the first timestep after which the error stays below it throughout the **observed** horizon. `None` means it did not happen by the terminal timestep. Terminal error at `T` is not added to the optimization objective.

## Limitations

- Exact matching is impossible for incompatible first rows and is intentionally rejected; no relaxed match or alternate controller is silently used.
- Asymptotic stability of the local-only controller already follows from Schur stability of the common target. Any benefit claimed is transient and empirical.
- A favorable small-gain bound is sufficient, not necessary; failure to satisfy it is not evidence of instability. Certification on training graph need not carry over to larger graphs.
- Sparse graph operations implement the neighbor computation, but the training itself runs in one process; this verifies the local information structure, **not a deployed multi-process distributed runtime**.
- No disturbance model, actuator saturation, collisions, moving-reference feedforward, or robust adversarial guarantees have been added to the primary experiments.
- Finite-horizon settling does not prove asymptotic settling; numerical floating-point gain maxima do not constitute formal interval certificates.
- Results depend on agent/graph samples, seeds, and optimizer convergence; limited seeds are not universal claims. Full-sweep user runs are recommended for broader research conclusions.
- CUDA behavior is only verified when CUDA hardware is actually present.

## Incremental topology, tuner and stress-test update (2026-10-31)

See [INCREMENTAL_CHANGES.md](INCREMENTAL_CHANGES.md) for a file-by-file change log, metric definitions, validation selection protocol, reproducible commands and verification outcomes; see [TOPOLOGY_FINDINGS.md](TOPOLOGY_FINDINGS.md) for executed star results.

The original `python main.py --suite quick|full` and old `checkpoint.npz` remain compatible. New results additionally save `final_checkpoint.npz` and `best_validation_checkpoint.npz` using the same numeric-only format. Optional `--early-stopping-patience`, `--val-every`, and `--min-delta` flags control validation-based early stopping.

```bash
python experiments/compare_topologies.py --checkpoint results/verified_final/trainN3_identity_lambda0/checkpoint.npz --sizes 3 5 10 100 1000
python experiments/tune_hyperparameters.py --out results/tuning --search random --max-candidates 12 --refine-top 2
python experiments/stress_test.py --checkpoint results/verified_final/trainN3_identity_lambda0/checkpoint.npz --sizes 100 1000
```

The checkpoint used in the example is included in the **complete** ZIP, but not necessarily present in a code-only archive. Replace it with any previously saved compatible checkpoint. All new experiments reuse existing physical equations, controller and training primitives; they do not depend on Pygame.
