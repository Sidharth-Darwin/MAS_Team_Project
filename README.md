# Attempt on Providing Convergence Gurantees for Leanred Consensus Controller

Research code for direct, one-hop graph-Laplacian feedback on heterogeneous discrete-time linear multi-agent systems. A shared `(2, 1)` parameter matrix `W` is learned with projected Adam under a **numerically evaluated** discrete-time small-gain constraint.

The core library, experiments, and tests are separate. This repository does not contain a robot renderer or a deployed multi-process communication system.

## Install and verify

Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) are recommended. From the repository root:

```bash
uv sync --extra test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra test python verify_math.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated system-wide pytest plugins (for example from ROS). The tests check the physical update, exact local matching, graph definitions, controller, gain factorization, training constraint, checkpoint loading, and evaluation behavior.

## Train a controller

```bash
uv run python main.py train \
  --suite quick --train-topology star \
  --epochs 100 --horizon 100 --lambda-values 0 \
  --run-name star_training
```

The training script saves `results/star_training/trainN3_identity_lambda0/checkpoint.npz` (best validation model) and `final_checkpoint.npz` (last epoch). Training does **not** automatically perform large-network inference. The `full` preset considers more training sizes and both supported activations. Run `uv run python main.py train --help` for configurable options, including a supplied training adjacency.

## Compare on new graphs without retraining

```bash
uv run python main.py topologies \
  --checkpoint results/star_training/trainN3_identity_lambda0/checkpoint.npz \
  --sizes 3 5 10 100 1000 \
  --topologies star chain ring wide_ring random \
  --out results/topology_study
```

Supply your own dense NumPy `.npy` adjacency or compatible sparse `.npz` graph instead:

```bash
uv run python main.py topologies \
  --checkpoint results/star_training/trainN3_identity_lambda0/checkpoint.npz \
  --adjacency my_graph.npy \
  --out results/custom_graph_study
```

With `--adjacency`, the comparison generates a compatible heterogeneous system by default; use `--system` with a saved physical-system `.npz` to test specific physical matrices. Graphs must be connected, undirected, symmetric, with positive degrees and no self loops. Frozen `W` is not rescaled or retrained at inference time.

## Stress testing and tuning

```bash
uv run python main.py stress \
  --checkpoint results/star_training/trainN3_identity_lambda0/checkpoint.npz \
  --sizes 100 1000 --out results/stress_study

uv run python main.py tune --help
```

The tuner selects candidates by a stated common validation score, not by comparing training objectives with different lambda values. Results are written under `results/`, which is Git-ignored by default.

## Reusable Python simulation interface

```python
import numpy as np
from distributed_consensus.graphs import Graph
from distributed_consensus.control.dynamics import step_numpy
from distributed_consensus.evaluation.reporting import load_checkpoint

system, _, W, activation, lam = load_checkpoint("path/to/checkpoint.npz")
adjacency = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
graph = Graph(adjacency)
states = np.zeros((system.n, 2))

result = step_numpy(states, system, graph, W, activation=activation)
states = result["next_states"]
# Also available: result["disagreement"], ["local"], ["correction"], ["total"]
```

An external simulation can supply new states and a new `Graph(adjacency)` at each timestep. The physical `System` must have the same number of agents as the graph, and the trained `W` shape and state/input dimensions must remain compatible. For a different system size, construct a new compatible system with `distributed_consensus.control.plants.build_system(A, B, target)`; don't reuse the training system matrices with a new size.

## Mathematical specification

For agent `i`, the implemented physical update is `x_i(k+1) = A_i x_i(k) + B_i u_i(k)`, with `B_i = [0, b_i]^T`. Compatible physical agents are exactly locally matched using `K_i = pinv(B_i)(A_target - A_i)`, with a compatibility residual check. For a connected undirected graph, `L_rw = I - D^{-1} A_graph`, `E_mat = L_rw X_mat`, `V_mat = phi(E_mat W)`, and `u_i = K_i x_i + v_i`. Activations are `identity` (primary) and `tanh`; `W` has no bias or hidden layers.

Training minimizes the average of `sum_{k=0}^{T-1} (||E_mat(k)||_F^2 + lambda ||U_mat(k)||_F^2)`, where `U_mat` is **total commanded physical control**. The weight is projected after each update to `||W||_2 <= eta / gamma_plant,train`, with default `eta=0.95`. For the specified `B_i`, the full-network gain factorizes as `gamma_plant,N = gamma0 ||L_rw diag(b_i)||_2`.

## Scope and limitations

- The plant-gain maximization and sparse spectral norms are computed numerically, **not** using rigorously outward-rounded certified upper bounds. Products above 1 mean the sufficient small-gain test is inconclusive, **not** proof of instability.
- The stable matched local target already yields asymptotic agreement toward zero in the undisturbed model. The learned correction is evaluated for transient disagreement and energy tradeoffs.
- Reported improvements are finite-horizon empirical measurements, dependent on physical matrices, graph, initialization, seed and horizon; they are not universal guarantees.
- Switching communication graphs and external disturbances can be simulated, but the fixed-graph small-gain result does not automatically certify arbitrary switching.
- No collision avoidance, actuator saturation, realistic robot dynamics or reference tracking is included in the core. A separate formation/robot simulation can be built on the NumPy one-step interface.

## Source layout

- `src/distributed_consensus/control/`: physical systems, controller, dynamics, plant gain
- `src/distributed_consensus/graphs.py`: graph generation, validation and Laplacian
- `src/distributed_consensus/learning/`: optimizer and validation
- `src/distributed_consensus/evaluation/`: metrics, checkpoints and reporting
- `experiments/`: training, topology comparison, stress testing, hyperparameter tuning
- `tests/`, `verify_math.py`: mathematical and integration verification
- `main.py`: command-line dispatcher
