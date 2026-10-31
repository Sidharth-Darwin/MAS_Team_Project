# Incremental update: graph topology, selection, tuning, robustness

This update was applied to the **existing source tree extracted from the original completed-project ZIP**. Unmentioned original source files remain byte-for-byte unchanged. The seven mathematical equations, heterogeneous physical matrices, exact pseudoinverse homogenization, shared direct controller `phi(E @ W)`, stage objective `J_E + lambda*J_U`, and norm projection are unmodified.

## Modified existing files

| File | Targeted changes |
| --- | --- |
| `helper_functions/graphs.py` | Add connected undirected `star`, `chain` alias for old `path`, and `complete`; allow an optional star `hub` index; retain old ring, wide-ring and random graph generators. |
| `helper_functions/plant_gain.py` | Extend returned small-gain dictionary with explicit status (`at_or_below_eta`, `between_eta_and_one`, `at_or_above_one`), the numerical gain method, and `rigorously_certified_upper_bound=False`; preserve legacy fields. |
| `helper_functions/dynamics.py` | Optional inference-only actuator-channel disturbance in physical NumPy stepping/rollouts. The original `total` is the **commanded** physical control; external disturbances affect state dynamics but do not change training costs. |
| `helper_functions/evaluation.py` | Add mean-centered `J_pair`, per-agent and initial-error-normalized costs, centered RMS values and finite-horizon centered settling metrics. Add optional supplied initial-state arrays and paired external disturbances. Old `J_E`, `J_U`, raw disagreement curves and settling metrics remain. |
| `helper_functions/training.py` | Keep the `W` return as **final** weights for API compatibility; add `best_W`, `best_epoch`, validation objective, optional early stopping, fixed externally supplied validation initial states, norm margin history. No changes to training objective/optimizer/projection. |
| `main.py` | Keep `checkpoint.npz` (final weights) for compatibility; also write `final_checkpoint.npz` and `best_validation_checkpoint.npz` with the old safe format. Add optional `--early-stopping-patience`, `--val-every`, `--min-delta`. Other suite behavior remains unchanged. |
| `README.md` | Append usage instructions and definitions for new experiments. |

## Added files

| File | Purpose |
| --- | --- |
| `experiments/compare_topologies.py` | Independently evaluate frozen checkpoint on star, chain, ring, wide-ring, complete and random undirected networks. Same `A_i`, `B_i`, `K_i` and initial states across topologies at fixed `N`. Produces CSV/JSON and PNG/SVG comparisons. |
| `experiments/tune_hyperparameters.py` | Reuses `train`, `evaluate_pair`, `save_checkpoint` and existing plant/graph functions; coarse grid or reproducible random search; optional local refinement; fixed validation dataset; numeric-only best/final checkpoints; completed-run resume; CSV/JSON/Pareto/score plots; limited finalist large-network inference. |
| `experiments/stress_test.py` | Reproducible frozen-controller inference with alternate heterogeneous physical inputs, star-hub assignment, topology, initial conditions, and external actuator-input pulses; no altered training equations. |
| `tests/test_incremental.py` | Regression checks for new graph families, star identity, costs, fair evaluation, disturbances, gain classification, training selection, safe checkpoints and tuner helpers. |
| `TOPOLOGY_FINDINGS.md` | Measured initial results and limitations. |

## How to run

From inside `distributed_consensus/`:

```bash
pip install -e '.[test]'
python verify_math.py

# Original main remains compatible:
python main.py --suite quick --run-name updated_quick
# Optional early stopping (patience counts validation checks):
python main.py --suite quick --run-name updated_early --val-every 5 --early-stopping-patience 8

# Frozen checkpoint, identical physical system and x0 across topologies:
python experiments/compare_topologies.py \
  --checkpoint results/verified_final/trainN3_identity_lambda0/checkpoint.npz \
  --sizes 3 5 10 100 1000 --samples 3 --horizon 60 --out results/topologies

# Full search; user-defined feasible grid, no retraining/rescaling at inference:
python experiments/tune_hyperparameters.py --out results/tuning \
  --search random --max-candidates 12 --refine-top 2 \
  --learning-rates 0.001 0.002 0.005 --batch-sizes 64 128 \
  --epochs 150 300 --horizons 70 100 --activations identity tanh \
  --lambda-values 0 0.035 0.175 --seeds 7 11 \
  --patience 10 --val-every 10 --finalists 2
# Re-run the exact command to skip completed candidates and completed finalist evaluations.

python experiments/stress_test.py \
  --checkpoint results/verified_final/trainN3_identity_lambda0/checkpoint.npz \
  --sizes 100 1000 --samples 3 --horizon 70 --out results/stress
```

The 1,000-node complete graph has ~1 million directed CSR entries; the topology script **skips it by default** to control memory/time. Add `--include-large-complete` to evaluate it explicitly. Other supported topologies run at N=1,000. The topology experiment never retrains, projects or scales its loaded `W`.

## Metric definitions and comparisons

* Original `J_E = sum_{k=0}^{T-1} ||L_rw X(k)||_F^2` and `J_U = sum_{k=0}^{T-1} ||U(k)||_F^2` are preserved.
* `J_pair = sum_{k=0}^{T-1} (1/N) sum_i ||x_i(k) - xbar(k)||_2^2` is graph-independent and comparable across topologies, provided physical models, initial conditions and time horizons are shared.
* `J_E_per_agent=J_E/N`, `J_U_per_agent=J_U/N`; `J_E_initial_normalized=J_E/max(||E(0)||_2^2, 1e-14)`. The last metric is still **graph-dependent** because `E=L_rw X`.
* Original settling threshold is `0.01 ||E(0)||_2`; additional `pair_settling_time` uses `0.01` times initial centered RMS for a topology-independent analog. Both are *finite-horizon* definitions, not asymptotic convergence proofs. `None` means no observed settling.
* Paired percentage improvement `(baseline - learned)/baseline * 100` is computed separately for `J_E`, `J_pair`, and physical energy. Star graphs must **not** be judged worse based on unnormalized `J_E` alone.
* Small-gain categories are based on **computed floating-point** `gamma_N ||W||_2`: `<=0.95`, `(0.95,1)`, and `>=1`. This gain uses stationary-frequency candidates and sparse spectral analysis but is **not** a formally outward-rounded interval bound. `>=1` makes the sufficient bound inconclusive, **not** unstable.

## Validation protocol and tuning fairness

* All candidates use the same fixed physical plant, graph and saved common validation initial states, with no test information used to select candidate weights, epochs or hyperparameters.
* `best_W` is selected using each candidate's validation `J_E + lambda*J_U` at that candidate's own training horizon; this is **within-candidate** model selection, not a cross-lambda score.
* To fairly rank different lambdas **and training horizons**, the tuner evaluates every candidate's best validation checkpoint at one common `--selection-horizon`, computes `(J_E/J_E,local) + selection_energy_weight*(J_U/J_U,local)` (default energy weight `0.1`), and reports the two-dimensional `J_E` vs `J_U` Pareto frontier. This cross-candidate score is for **selection only**, not a new training objective.
* Only top `--finalists` are evaluated on large networks, with initial states drawn independently from selection-validation states. Their outcomes are not fed back into selection.
* Resume skips completed candidate folders identified by deterministic configuration hash; a completed candidate has both `result.json` and `best_validation_checkpoint.npz`. A fixed selection protocol is enforced by `search_spec.json`.
* Existing `checkpoint.npz` files remain readable; new best/final files use identical numeric-only `.npz` checkpoint format.

## Verification executed in this environment

* Original code before modifications: **32 passed / 1 CUDA skipped**.
* After changes: **52 passed / 1 CUDA skipped**.
* `main.py` original quick suite executed with short smoke settings.
* Coarse/refined tuner and resume behavior executed (completed runs skipped on repeat).
* Initial topology study: star/chain/ring/wide-ring/random/complete at N=3,5,10,100; sparse families at N=1,000.
* Stress test executed with N=10 and 100 including varying star hub and external pulses (56 comparisons); also N=1,000 (28 comparisons).

The smoke runs only verify executability. They do not establish optimizer convergence for 300 epochs. CPU-only PyTorch was available; CUDA checks remain unexecuted. Certified interval H-infinity upper bounds and universal asymptotic claims are **not** established by these numerical tests.
