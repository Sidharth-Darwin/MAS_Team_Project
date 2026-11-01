# `evaluation/` — measured performance and saved results

## `metrics.py` — trajectory and paired metrics

`trajectory_metrics(history)` uses a trajectory of `T+1` states and disagreements and `T` actions. The primary finite-horizon costs (no terminal stage cost) are

$$
J_E=\sum_{k=0}^{T-1}\|E(k)\|_F^2,\qquad
J_U=\sum_{k=0}^{T-1}\|U_{total}(k)\|_F^2,
$$

with correction-only energy

$$
J_{corr}=\sum_{k=0}^{T-1}\|V(k)\|_F^2.
$$

It also computes the alternative **mean-centered** disagreement cost

$$
J_{pair}=\sum_{k=0}^{T-1}\frac1N\sum_{i=1}^{N}\|x_i(k)-\bar x(k)\|_2^2,
\qquad \bar x(k)=\frac1N\sum_{i=1}^{N}x_i(k).
$$

`J_pair` is a separate **evaluation metric**, not the training loss. The code also reports terminal disagreement, maximum disagreement and relative-threshold crossing/settling times. *Settling time* is the first time after which all remaining observed samples stay below the threshold, not merely the first crossing.

`evaluate_pair()` rolls out the local-only baseline (`W=None`) and the learned controller on **identical** initial states, graph, physical system and optional disturbances, then aggregates the results over samples. For any positive baseline cost `J`, the reported percentage improvement is

$$
100\,\frac{J_{baseline}-J_{learned}}{J_{baseline}}.
$$

When the baseline cost is zero, relative improvement is undefined and recorded as `None` (JSON `null` / blank CSV), not as an invented zero or infinity. `paired_win_rate` is the fraction of trajectories with lower learned `J_E`.

These are observed finite-horizon results, not universal guarantees or stability certificates.

## `reporting.py` — output, plots, checkpoints

`write_json`, `write_csv`, and the plotting helpers produce experiment artifacts. `save_training_plot`, `save_curve_plot`, and `save_tradeoff_plot` visualize training and evaluation results; plotting changes **no** controller weights.

`save_checkpoint()` writes a numeric-only compressed `.npz` containing physical `A`, `B`, `K`, target, graph edges, `W`, activation and lambda. `load_checkpoint()` reconstructs the physical system using `build_system()` (including matching checks), validates the saved `K`, and returns:

```python
system, graph, W, activation, lam = load_checkpoint("path/to/checkpoint.npz")
```

`save_system()` writes physical matrices and graph data without a learned controller. `save_curve_csv()` writes the paired mean-disagreement curves.

**Checkpoint convention used by experiments:** `checkpoint.npz` = best validation weights; `final_checkpoint.npz` = last-epoch weights. Generated checkpoints, graphs, CSVs and figures typically live under Git-ignored `results/`.
