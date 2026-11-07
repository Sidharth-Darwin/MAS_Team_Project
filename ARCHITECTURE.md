# Repository Architecture and Technical Review

This document describes the Multi-Agent Systems (MAS) repository represented by the supplied `repo_contents.txt` snapshot. It records the mathematical model, software boundaries, execution paths, metric conventions, and issues found during the follow-up review.

**Snapshot caveat.** The uploaded text snapshot predates the two local edits reported by the maintainer (the `plant_gain.py` comment correction and the additional README clarification). Those edits are assessed below, but the latest local files were not re-uploaded for direct diff verification. Binary model checkpoints and Git history were also not present in the text snapshot.

## 1. Project boundaries

The repository has three related but distinct parts:

1. **Core consensus-control library** (`src/distributed_consensus/`): heterogeneous plants, graph operators, local and learned control, differentiable training dynamics, numerical small-gain analysis, evaluation, and optional resilient-message utilities.
2. **Research experiments** (`experiments/`, dispatched by `main.py`): training and paired evaluation studies that use the core library. These are offline experiments, not a real-time multi-process deployment.
3. **Scalar W-MSR demonstration** (`simulation/`): a separate GUI/headless test bench. It applies scalar W-MSR filtering to one scalar coordination message per agent while retaining the core two-state physical plant update.

The core controller's fixed-graph small-gain analysis and the simulation's W-MSR graph-robustness checks are different mathematical arguments. Neither automatically certifies the other system.

## 2. Core mathematical model

### 2.1 Heterogeneous physical agents and exact local matching

Each agent has two states and one scalar input:

\[
 x_i(k+1)=A_i x_i(k)+B_i u_i(k),\qquad
 A_i\in\mathbb R^{2\times2},\quad
 B_i=\begin{bmatrix}0\\b_i\end{bmatrix},\quad b_i\ne0.
\]

The common target matrix is

\[
 A_\star=\begin{bmatrix}1&0.1\\-0.2&0.8\end{bmatrix}
\]

in the default setup. It is Schur stable. `plants.py` computes

\[
 K_i=B_i^\dagger(A_\star-A_i),
\]

and verifies the compatibility condition needed for

\[
 A_i+B_iK_i=A_\star
\]

(up to the configured floating-point tolerance). This does **not** replace the physical matrices by the target matrix: the actual NumPy and Torch updates still use each agent's original `A_i` and `B_i`.

`generate_system()` creates heterogeneous systems that satisfy this matching constraint. `build_system()` is the validation/construction path for externally supplied matrices. The supported implementation is intentionally specialized to two states and one input per agent.

### 2.2 Communication graph and local disagreement

`graphs.py` represents a connected, undirected, weighted graph with symmetric adjacency matrix \(A_g\), degree matrix \(D\), and random-walk transition matrix

\[
 P_{rw}=D^{-1}A_g,\qquad L_{rw}=I-P_{rw}.
\]

For an agent-major state matrix \(X\in\mathbb R^{N\times2}\),

\[
 E=L_{rw}X,\qquad e_i=x_i-\frac{1}{d_i}\sum_{j\in\mathcal N_i}a_{ij}x_j.
\]

`Graph.error(X)` evaluates this operation with sparse multiplication. Each row is computed from an agent's own state and its one-hop neighbors; no all-pairs distance matrix is required for the controller update.

Supported generated topologies include chain, star, ring, wide ring, complete, and connected random graphs. `load_graph()` accepts the documented NumPy/SciPy adjacency formats and saved edge-list representation. Graph validation requires symmetry, positive finite edge weights, zero diagonal, positive degrees, and connectivity.

### 2.3 Controller and physical rollout

The learned correction is shared across all agents:

\[
 V=\phi(EW),\qquad W\in\mathbb R^{2\times1},\qquad
 \phi\in\{\mathrm{identity},\tanh\}.
\]

There is no bias, hidden layer, or agent-specific learned matrix in the core small-gain controller. The physical command is

\[
 u_i(k)=K_ix_i(k)+v_i(k),\qquad
 x_i(k+1)=A_ix_i(k)+B_i\big(u_i(k)+d_i(k)\big),
\]

where `d_i` is an optional actuator-channel disturbance for NumPy evaluation. `W=None` selects the local-only matched baseline. The returned `total`/`U_total` is the commanded input and excludes the external disturbance.

`control/dynamics.py` owns the NumPy stepping and rollout interface. For a horizon of \(T\) actions, the state and disagreement histories contain \(T+1\) samples, and action histories contain \(T\) samples. `TorchWorld` implements the corresponding fixed-graph batched update for differentiable training; it does not model actuator disturbances.

Exact local matching gives the disagreement dynamics

\[
 E(k+1)=(I_N\otimes A_\star)E(k)
 +(L_{rw}\otimes I_2)\operatorname{blkdiag}(B_1,\ldots,B_N)V(k).
\]

### 2.4 Numerical plant gain and small-gain constraint

For the special input structure \(B_i=[0,b_i]^T\), `plant_gain.py` factorizes the full-network induced gain as

\[
 \gamma_{\mathrm{plant},N}
 =\gamma_0\left\|L_{rw}\operatorname{diag}(b_1,\ldots,b_N)\right\|_2.
\]

- `scalar_gain(A_star)` computes the two-state transfer contribution \(\gamma_0\) by checking endpoints and real stationary candidates after substituting \(t=\cos\omega\).
- `spatial_norm(graph,b)` computes the spectral norm of \(L_{rw}\operatorname{diag}(b)\), using a dense norm for small graphs and a sparse largest-eigenvalue calculation for larger ones.
- `full_gain()` combines those factors.
- `direct_grid_gain()` is an independent finite-frequency diagnostic for small systems, not a proof.
- `small_gain()` reports \(\gamma_{\mathrm{plant},N}\|W\|_2\).

The numerical training projection is

\[
 \|W\|_2\le c,\qquad c=\frac{\eta}{\gamma_{\mathrm{plant,train}}},
\]

with default margin \(\eta=0.95\). The implementation uses floating-point numerical estimates, not outward-rounded certified upper bounds. A tested product above one means the sufficient small-gain condition is inconclusive; it does not, by itself, prove instability. A cap calculated on one training graph does not certify arbitrary graph sizes or topologies.

**Polynomial-order clarification:** for \(v(t)=N(t)/D(t)\), with a linear numerator and quadratic denominator, the numerator of the quotient-rule derivative is

\[
 P(t)=(-n_1d_2)t^2+(-2n_0d_2)t+(n_1d_0-n_0d_1).
\]

Therefore the coefficient array

```python
[-n1*d2, -2*n0*d2, n1*d0 - n0*d1]
```

is already ordered as \([c_2,c_1,c_0]\), i.e. **descending powers**, as required by `numpy.roots`. The earlier word “ascending” in the comment was incorrect; correcting that comment is sufficient. Do not reverse the array. The default-target numerical checks cited in the review are consistent with the independent frequency-grid test.

### 2.5 Training objective and update order

`learning/training.py` optimizes only the shared \(W\); the physical matrices, local matching gains, target, and training graph stay fixed. For \(T\) actions,

\[
 J_E=\sum_{k=0}^{T-1}\|E(k)\|_F^2,\qquad
 J_U=\sum_{k=0}^{T-1}\|U_{\mathrm{total}}(k)\|_F^2,
\]

\[
 \mathcal L(W)=\mathbb E[J_E+\lambda J_U].
\]

`J_U` is the energy of the total commanded physical input \(K_ix_i+v_i\), not just the learned correction. There is no terminal stage cost in this objective; terminal disagreement is returned separately by the Torch rollout.

The training sequence is:

1. Compute the training-system gain and set \(c=\eta/\gamma_{\mathrm{plant,train}}\).
2. Initialize \(W=0\) and create separate seeded training and fixed-validation streams.
3. Roll out the differentiable physical system and calculate \(J_E+\lambda J_U\).
4. Backpropagate and take an Adam step.
5. Project \(W\) immediately after every optimizer step.
6. Evaluate on the fixed validation batch at the configured cadence and retain the lowest-validation-objective `best_W`.

`min_delta` controls early-stopping patience; it does not change which checkpoint has the lowest validation objective. Training-history objective/norm fields are intentionally measured on different sides of the update: `train_*` and `pre_update_*` are pre-update, while projected norm and constraint-margin fields are post-update. Experiment inference should use the best-validation checkpoint unless a study explicitly says otherwise.

## 3. Metric conventions — keep the names distinct

There are two separate metric namespaces in the repository. The symbol `J_E` is overloaded in the standalone simulation; it does not mean the same formula in both contexts.

### 3.1 Core library and research experiments

The core `J_E` is the Laplacian disagreement cost used in training:

\[
 \boxed{J_E^{\mathrm{core}}=
 \sum_{k=0}^{T-1}\|L_{rw}X(k)\|_F^2.}
\]

The core `J_U` is total commanded-input energy, and `J_corr` is correction-only energy.

The core evaluator also reports `J_pair`, a **separate evaluation metric**, not the training loss:

\[
 J_{pair}=\sum_{k=0}^{T-1}\frac1N
 \sum_{i=1}^{N}\|x_i(k)-\bar x(k)\|_2^2,
 \qquad \bar x(k)=\frac1N\sum_{i=1}^{N}x_i(k).
\]

The name `J_pair` is mathematically defensible. At each time step, the identity

\[
 \frac1N\sum_i\|x_i-\bar x\|_2^2
 =\frac{1}{2N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}\|x_i-x_j\|_2^2
\]

shows that this is the normalized all-pairs squared-disagreement quantity. The implementation uses the equivalent centered-state computation, avoiding an \(O(N^2)\) pairwise-distance calculation. **Do not rename `J_pair` to `J_centered` merely for style.** Its existing result keys and CSV fields are already documented and tested; renaming them would be a schema/API change without fixing a mathematical error.

The evaluator also reports maximum/terminal disagreement, relative-threshold crossing and settling times. Settling time means the first sample after which every remaining observed sample stays below the selected threshold; it is not simply the first threshold crossing. Relative improvements with a zero baseline are undefined and should remain `None`/blank rather than being fabricated as zero or infinity.

### 3.2 Standalone adversarial W-MSR simulation

In `simulation/`, the GUI label `J_E` is used for a **different** diagnostic: cumulative mean-centred disagreement over normal agents only. If \(\mathcal N\) is the set of non-compromised agents and \(N_n=|\mathcal N|\), then

\[
 \bar x_{\mathcal N}(t)=\frac1{N_n}\sum_{i\in\mathcal N}x_i(t),
 \qquad
 e_{\mathrm{RMS}}(t)=
 \sqrt{\frac1{N_n}\sum_{i\in\mathcal N}
 \|x_i(t)-\bar x_{\mathcal N}(t)\|_2^2},
\]

\[
 \boxed{J_E^{\mathrm{sim}}(k)=
 \sum_{t=0}^{k-1}\sum_{i\in\mathcal N}
 \|x_i(t)-\bar x_{\mathcal N}(t)\|_2^2
 =N_n\sum_{t=0}^{k-1}e_{\mathrm{RMS}}(t)^2.}
\]

The simulation's cumulative `J_U` is likewise summed over normal agents' total commanded inputs. Compromised agents' states are excluded from these normal-agent metrics. This is why the simulation README clarification is important: in this test bench, `J_E` does **not** mean the full-graph Laplacian cost \(\sum\|L_{rw}X\|_F^2\) used by the core library. Keep the scope explicit in prose, plots, and exported CSV metadata. The two metrics should not be compared numerically as though they shared a definition.

The simulation's `e_RMS` is an instantaneous metric; its cumulative `J_E` and `J_U` are separate nondecreasing accumulated diagnostics. They are not added together to form the current error.

## 4. Research experiment architecture

`main.py` dispatches four report experiments:

| Command | Script | Purpose |
| --- | --- | --- |
| `activation` | `experiments/activation_comparison.py` | Identity versus tanh controller comparison |
| `size` | `experiments/network_size_generalization.py` | Train on a small network and evaluate frozen weights at larger sizes |
| `gain` | `experiments/plant_gain_transfer.py` | Transfer a frozen controller to systems with different plant gains |
| `topology` | `experiments/topology_generalization.py` | Train on one graph family and evaluate on other connected topologies |

`experiments/_common.py` holds common graph/system creation, paired rollouts, summary metrics, plots, and output writing. It should not become a second implementation of the physical model or controller equations.

`experiments/_paper_gnn_baseline.py` implements the separate paper-style comparison model. It uses the unnormalized Laplacian \(L_u=D-A_g\), whereas the core small-gain controller uses the random-walk Laplacian \(L_{rw}=I-D^{-1}A_g\). The paper baseline has hidden features and output gains; its weights are intentionally not projected by the small-gain constraint. In the locally matched comparisons, the learned correction is added to the same local homogenization input. The report experiments use the helper with settings appropriate to each comparison.

`experiments/small_gain_violation_diagnostic.py` is an additional diagnostic script, not one of the four `main.py` report commands. It explores frozen-checkpoint behavior outside the nominal numerical small-gain envelope; an above-one loop-gain product is not itself a stability verdict.

**Documentation discrepancy:** the supplied `experiments/README.md` says there are exactly three independent report experiments. That is stale: the dispatcher exposes four. It should be updated to include `topology_generalization.py` and explain that the small-gain-violation diagnostic and paper-GNN helper are supporting/diagnostic code rather than separate `main.py` commands.

## 5. Optional scalar W-MSR implementation

`src/distributed_consensus/robust/wmsr.py` is separate from the core learned controller. For each normal receiver, it:

1. Compares each scalar neighbour value with the receiver's own value.
2. Removes up to `F` smallest values strictly below its own.
3. Removes up to `F` largest values strictly above its own.
4. Retains equal values and every other value.
5. Averages its own value with the retained neighbour values.

For `F=1`, one low and one high neighbour may both be removed in the same update. That is the specified two-sided trimming rule, not permission for two malicious neighbours. The update is synchronous: all receivers use the same pre-update signal vector.

The module also contains exact small-graph robustness checks. Its documented graph conditions are:

- F-total malicious: \((F+1,F+1)\)-robustness (necessary and sufficient under the classical fixed-graph W-MSR result).
- F-local malicious: \((2F+1)\)-robustness is sufficient and \((F+1)\)-robustness is necessary.
- The Byzantine variants are assessed on the induced normal-node graph under the selected model.

These are scalar W-MSR graph-topology conditions. They do not prove stability of the full two-state heterogeneous plant, the paper GNN, or the original fixed-Laplacian small-gain controller after state-dependent message filtering. In the supplied GUI demonstration, the attack is a **malicious broadcast**: each compromised sender sends one common false scalar value to all neighbours. Byzantine equivocation (different lies to different recipients), physical actuator attacks, and communication jamming are not implemented.

The robustness predicates enumerate disjoint node-set pairs and have exponential complexity (on the order of \(3^N\)). The default exact-check limit is 14 nodes; above the selected limit the helper returns `None` rather than treating connectivity as a robustness certificate.

### 5.1 Additional robustness API hardening findings

The supplied snapshot exposes several input-validation gaps in the public helpers:

- `validate_adversary_set()` converts IDs with `int(i)`, so values such as `1.9` and `True` can be accepted as node 1 instead of rejected.
- The same truncation issue exists in `wmsr_local_with_details()` for neighbour IDs.
- `validate_adversary_set()` uses `startswith()` to choose total/local behavior and does not enforce the same exact threat-model vocabulary as `assess_wmsr_topology()`; unsupported labels can therefore bypass the intended count checks.
- `assess_wmsr_topology()` does not fully validate supplied adversary IDs for the malicious branches; invalid IDs can lead to misleading `normal_N` metadata even though the full graph is checked.
- `is_r_robust()` and `is_rs_robust()` default to `max_nodes=14` but the supplied implementation permits callers to raise that value. Since the search is exponential, this can make an apparently harmless override trigger excessive CPU or memory use. Validate that `max_nodes` is an integer and enforce a deliberate hard ceiling if the 14-node guard is intended to be non-overridable.

The simulation constructor currently performs stricter adversary-ID checks before calling the helper, so some of these defects are easiest to reproduce through direct library calls. They still matter because the helpers are public APIs.

### 5.2 Simulation metric and history cost

`normal_state_metrics()` also constructs the full pairwise-difference tensor for `normal_pairwise_diameter`. A repository-wide search of the supplied snapshot found no consumer of that returned field. This creates an \(O(N_n^2)\) temporary on repeated metric calls during each simulation step and CSV export. Either remove the unused field/calculation or calculate the diameter lazily only when a display/export actually requests it. Separately, the GUI model retains complete state, error, and action histories in memory; this is convenient for short demonstrations but grows with the run horizon. For substantially longer runs, use bounded plotting history and/or buffered file output rather than retaining every per-agent message record.

## 6. W-MSR simulation execution path

`simulation/` is a matched five-policy test bench:

1. Load the source small-gain checkpoint and the benchmark checkpoint.
2. Reconstruct the matched physical system and communication graph.
3. Create one common initial physical-state matrix for all five policies.
4. For each method and each synchronous step, compute the sender's scalar projection, inject the attacker's common extreme broadcast, optionally run scalar W-MSR at normal receivers, and construct each receiver's correction.
5. Apply the physical update through `advance_numpy()` from the core dynamics module.
6. Store state, message-audit, action, and metric histories; export CSV summaries and figures on request.

The five policies are local-only, paper GNN raw, paper GNN plus W-MSR, small-gain raw, and small-gain plus W-MSR. Compromised agents do not follow the filtering protocol and use their local matched control in this demonstration; only their communicated scalar signal is malicious. Normal-agent metrics omit compromised agents.

There is also a stale comment in `simulation/train_benchmarks.py` that describes the default comparison as Byzantine and refers to a 2-robust normal-node graph. The actual `MultiPolicySimulation` path validates `F-local-malicious` and checks the full graph for the \((2F+1)\)-robust sufficient condition. The comment should describe the implemented malicious-broadcast model, not a different Byzantine setup.

### 6.1 Confirmed benchmark-default incompatibility

`simulation.train_benchmarks.parse_args()` in the supplied snapshot defaults `--paper-hidden` to **2**. However, `simulation.model.load_benchmark()` explicitly accepts only

\[
 H\in\mathbb R^{1\times2},\qquad K_{paper}\in\mathbb R^{N\times1},
\]

and rejects other channel counts. A default benchmark training run can therefore save a checkpoint that the simulator refuses to load. The error message itself instructs users to retrain with `--paper-hidden 1`, but the trainer default is inconsistent with that requirement.

**Recommended fix:** change the simulation benchmark trainer default to `--paper-hidden 1` and add a test that calls `parse_args([])` and verifies the resulting checkpoint can be loaded. Alternatively, extend the message protocol and simulation equations deliberately to support multiple channels; merely relaxing the shape check would not suffice because the present W-MSR message is scalar.

## 7. Checkpoints, outputs, and data flow

- Core small-gain checkpoints are numeric-only `.npz` files containing physical `A`, `B`, `K`, target, graph edges/weights, `W`, activation, and `lambda`. `load_checkpoint()` reconstructs the system and graph, recomputes `K`, and validates the stored matching gain.
- Paper-GNN checkpoints store trained paper weights and the training plant/graph information.
- Simulation benchmark checkpoints combine physical matrices, graph, small-gain weights, paper-GNN weights, gain metadata, and JSON configuration metadata.
- Research outputs include validation histories, paired trajectory metrics, CSV/JSON summaries, plots, manifests, and `.npz` checkpoints, usually under `results/`.
- The simulation can export one row per method/agent/time-step with scalar message and filtering audit information, plus summary CSVs and PNG figures.

The uploaded text snapshot did not contain the binary default checkpoints expected under `simulation/assets/`. The runtime instructions that refer to those files are only reproducible when the user supplies/restores the assets or runs the documented checkpoint-training workflow with its required source checkpoint.

## 8. File and directory map

### Repository root

- `README.md` — installation, core model specification, CLI examples, and scope limitations.
- `pyproject.toml`, `uv.lock` — project metadata, runtime dependencies, optional test dependency, and locked environment.
- `main.py` — dispatcher for the four report experiment commands.
- `verify_math.py` — standalone mathematical verification entry point.
- `TEST_README.md` — supplementary test instructions/diagnostics; should be consolidated into the main README or a focused testing document if retained.
- `THREE_WAY_COMPARISON_README.md` — comparison-specific notes; historical/supporting documentation rather than core architecture.
- `requirements.txt` — the supplied snapshot contains a machine-specific editable path (`file:///home/sid/Downloads/distributed_consensus`) and a full environment freeze. This is not a portable project dependency file; `pyproject.toml` and `uv.lock` should remain authoritative.

### `src/distributed_consensus/`

- `graphs.py` — graph construction, validation, sparse random-walk Laplacian, graph loading.
- `control/plants.py` — compatible heterogeneous physical systems and exact local matching.
- `control/controllers.py` — shared correction activation, NumPy controller helpers, spectral projection.
- `control/dynamics.py` — authoritative NumPy physical step/rollout and differentiable `TorchWorld`.
- `control/plant_gain.py` — scalar transfer gain, spatial spectral norm, full gain factorization, small-gain status.
- `learning/training.py` — projected-Adam training, validation checkpoint selection, seeded streams.
- `evaluation/metrics.py` — core `J_E`, `J_U`, `J_corr`, `J_pair`, settling and paired metrics.
- `evaluation/reporting.py` — JSON/CSV writers, plots, numeric checkpoint save/load, system serialization.
- `robust/wmsr.py` — scalar W-MSR, receiver audit results, adversary validation, exact robustness predicates.

### `experiments/`

- `activation_comparison.py`, `network_size_generalization.py`, `plant_gain_transfer.py`, `topology_generalization.py` — four dispatched report experiments.
- `small_gain_violation_diagnostic.py` — separate frozen-checkpoint diagnostic.
- `_paper_gnn_baseline.py` — supporting train/evaluate/rollout implementation for the paper-style baseline.
- `_common.py` — shared experiment utilities and metrics; not a second core model implementation.
- `README.md` — currently has the stale three-versus-four experiment count noted above.

### `simulation/`

- `model.py` — benchmark loading, attack model, per-receiver W-MSR decisions, five-policy synchronous simulation, histories, CSV/figure export.
- `gui.py` — desktop interface and visual message/graph audit.
- `preview.py` — one headless demonstration run and artifacts.
- `evaluate.py` — paired multi-trial evaluation and aggregate outputs.
- `train_benchmarks.py` — trains and saves the paper-GNN/small-gain benchmark pair.
- `tests/test_simulation.py` — simulation-specific regression tests.
- `assets/` — expected trained checkpoints and benchmark metadata/history. Binary checkpoints were not present in the uploaded text snapshot.

### `tests/`

- `test_math.py` — plant matching, graph, dynamics and gain mathematics.
- `test_training_fixes.py` — projected training, validation and checkpoint behavior.
- `test_evaluation_fixes.py`, `test_incremental.py` — metrics, paired comparisons and reporting behavior.
- `test_dispatch_and_graph_files.py`, `test_cli_and_simulation.py` — command and file-interface behavior.
- `test_report_experiments.py`, `test_paper_baseline_integration.py` — experiment integration and paper baseline.
- `test_wmsr.py` — scalar W-MSR and graph-robustness behavior.
- `conftest.py` — shared fixtures.

## 9. Documentation corrections to retain

1. Keep the `plant_gain.py` derivative coefficient array in descending order; correct the comment only.
2. Keep `J_pair` as the core centered/pairwise-equivalent evaluation metric. Do not rename its output fields unless an intentional breaking schema migration is planned.
3. Retain the simulation README clarification that its `J_E` is a normal-agent mean-centred cumulative metric and differs from core `J_E`. The note should explicitly identify the normal-agent set.
4. Update `experiments/README.md` from three to four report experiments.
5. In `src/distributed_consensus/README.md`, remove the `config.py` file-map row unless that file is restored; it is absent from the supplied tree.
6. In the same package README, remove the claim that `robust/wmsr.py` provides a coordinate-wise vector adapter. The supplied simulation uses a scalar projection message and scalar W-MSR filter.
7. Keep the W-MSR limitation prominent: graph robustness for the scalar protocol is not a stability theorem for the filtered heterogeneous two-state plants.
8. Align `simulation.train_benchmarks`'s default hidden-channel count with the single-channel restriction in `load_benchmark()`.
9. Correct the stale Byzantine/2-robust comment in `simulation/train_benchmarks.py` to match the active F-local malicious-broadcast scenario.
10. Remove or make lazy the unused pairwise-diameter calculation in `normal_state_metrics()`; consider bounded/streamed histories for long simulation runs.

## 10. Verification notes and limits

The follow-up targeted checks on the supplied snapshot completed:

```text
python -m compileall -q src experiments simulation tests
# passed

PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/test_wmsr.py tests/test_math.py -q
# 44 passed, 1 skipped
```

These are targeted checks, not a claim that every test and every experiment has passed in one uninterrupted full-suite run. The CUDA-specific skip is expected on a machine without CUDA. The exact source snapshot and restored binary checkpoints should be re-tested together after the validation and benchmark-default fixes are applied.
