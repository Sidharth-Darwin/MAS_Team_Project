# Initial topology experiment — 2026-10-31

**Important source limitation:** The available archived project includes only its previous three-agent-trained checkpoint `results/verified_final/trainN3_identity_lambda0/checkpoint.npz`, with learned `W=[-0.07020633,-0.17856523]^T` and `||W||_2=0.191870970552`. The user's newer **300-epoch**, N=5/N=10 trained checkpoints were **not in the available files**; their stated N=1,000 ring products of 1.0344 and 1.0459 cannot be independently recomputed here. The user's 0.9518 product for the three-agent-trained newer controller is between 0.95 and 1, not at-or-below 0.95.

The following values are from fresh executed inference using the available older W. `N`, topology, and `lambda` are not retrained. For each N, identical physical matrices and initial states were used across all graphs. `T=45`, `samples=2`, identity activation.

| N | Topology | `||L_rw||_2` | Full plant gamma | Product gamma*||W|| | Mean-centered Jpair improvement | Original JE improvement |
|---:|---|---:|---:|---:|---:|---:|
| 3 | star | 2.1213 | 1.8573 | 0.3564 | +4.35% | +4.60% |
| 3 | ring | 1.5000 | 1.3669 | 0.2623 | +4.49% | +4.49% |
| 5 | star | 2.5000 | 2.1492 | 0.4124 | +2.30% | +2.78% |
| 5 | ring | 1.8090 | 1.3290 | 0.2550 | +3.06% | +3.31% |
| 10 | star | 3.3333 | 1.9753 | 0.3790 | +5.28% | +8.02% |
| 10 | ring | 2.0000 | 1.6717 | 0.3208 | +4.09% | +6.35% |
| 100 | star | 10.0504 | 6.0201 | **1.1551** | +4.09% | +4.68% |
| 100 | ring | 2.0000 | 1.7349 | 0.3329 | +4.09% | +6.12% |
| 1000 | star | 31.6386 | 18.7619 | **3.5999** | +3.76% | +8.50% |
| 1000 | ring | 2.0000 | 1.7551 | 0.3367 | +3.57% | +5.66% |

**Interpretation:** The large star fails the *numerically evaluated sufficient small-gain test* despite empirical finite-horizon improvement for these selected initial conditions. This does **not** prove stability. The different percentages for `J_E` and `J_pair` show why raw Laplacian disagreement should not be the only topology comparison metric. When the star hub's physical `b_i` changes, the full gain can change strongly, even though `||L_rw||_2` is unchanged: in an additional N=100 stress run with the same agent set, different star hubs generated gamma values approximately 4.547, 5.133, and 9.598 (products approximately 0.872, 0.985, and 1.842) while `||W||_2` remained 0.19187. This directly exposes sensitivity to how heterogeneous inputs align with graph degree.

For a star with N vertices, the exact unweighted norm identity is `||L_rw||_2=N/sqrt(N-1)`. With heterogeneous `b_i`, the actual plant factor is `||L_rw diag(b)||_2`; it is *not* just the unweighted norm. In particular, hub column entries imply `||L_rw diag(b)||_2 >= |b_hub| sqrt(N-1)` (positive b assumed in these generators). This demonstrates how growth with N can occur even though all local plants are Schur stable after homogenization.

The gamma figures use stationary-point candidate evaluation with sparse floating-point spectral norm estimation, not formally verified interval upper bounds. Two trajectories over 45 steps is an **exploratory illustration**, not statistical evidence across physical/graph seeds or a formal robustness/stability guarantee. The original control-input `J_U` measures commanded physical energy, not energy injected by exogenous disturbances.

Reproducible output: `results/star_initial/topology_comparison.csv`, `.json`, `.png`, `.svg`; `results/stress_initial/stress_results.csv` and `results/stress_N1000_initial/stress_results.csv`.
