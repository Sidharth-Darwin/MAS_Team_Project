# `distributed_consensus` package

This package contains the reusable control and evaluation code. The command-line experiments live at the repository root in `experiments/`, and `main.py` dispatches to them. The library itself does not render robots, implement network transport, or train while running inference.

## File map

| File or subfolder | Responsibility | Corresponding mathematical object |
| --- | --- | --- |
| `__init__.py` | Python package marker | None |
| `config.py` | `Config` dataclass for experiment defaults | Numerical parameters such as horizon `T`, margin `eta`, learning rate, batch size |
| `graphs.py` | `Graph`, `from_edges`, `make_graph`, `load_graph`; sparse graph validation and disagreement | `A_graph`, `D`, `P_rw = D^{-1} A_graph`, `L_rw = I - P_rw`, `E = L_rw X` |
| [`control/`](control/README.md) | Physical systems, controller, simulation and plant gain | `A_i`, `B_i`, `K_i`, `W`, physical updates, `gamma_plant` |
| [`learning/`](learning/README.md) | Projected-Adam training | `J_E + lambda J_U`, constrained `W` |
| [`evaluation/`](evaluation/README.md) | Paired evaluation, checkpoints and outputs | Finite-horizon costs, improvement percentages |

## Graph and controller interface

For a connected, undirected, weighted graph with no self-loops, the adjacency matrix is symmetric, nonnegative, and has zero diagonal. The degree of agent `i` is the sum of its edge weights. For stacked agent states `X` of shape `(N, 2)`:

$$
P_{rw}=D^{-1}A_{graph},\qquad L_{rw}=I_N-P_{rw},\qquad E=L_{rw}X.
$$

`Graph.error(X)` returns `E` in shape `(N, 2)` using sparse neighborhood multiplication. This is the **only** graph operation used before the learned correction `V = phi(EW)`; there is no additional graph filter.

```python
import numpy as np
from distributed_consensus.graphs import Graph

adjacency = np.array([[0., 1., 1.],
                      [1., 0., 0.],
                      [1., 0., 0.]])
graph = Graph(adjacency)
states = np.zeros((3, 2))
error = graph.error(states)
```

`make_graph()` generates `star`, `chain`, `ring`, `wide_ring`, `complete`, and `random` graphs. `load_graph()` accepts supported `.npy` or `.npz` graph files. `Graph` validates symmetry, positive finite edge weights, connectivity and node count; invalid graphs are rejected.

## Scope

All present physical-system and training APIs assume **two states and one input per agent**, with a shared `(2, 1)` controller weight. All graph sizes can vary, provided the graph and physical system have the same number of agents. See the control README for the physical model and limitations of the numerical small-gain calculation.
