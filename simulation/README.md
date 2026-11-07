# Scalar W-MSR consensus-control test bench

## 1. Purpose and scope

This directory contains an **abstract consensus-control test bench** for inspecting scalar local Weighted-Mean-Subsequence-Reduced (W-MSR) filtering inside a two-state multi-agent feedback system.

The application compares five controller policies from identical initial physical states. Its main view combines a communication graph, a normal-agent disagreement plot, a controller scorecard, and a per-agent message audit.

This is **not** a robot, vehicle, formation-control, or hardware-network simulator. It does not model physical positions, actuator faults, communication delays, packet loss, or a distributed deployment. The agents and their dynamics are mathematical test systems.

The central experiment is a comparison of each learned controller with and without scalar W-MSR filtering under the same malicious-broadcast scenario:

- Paper-style controller, using raw neighbour messages.
- The same paper-style controller, with W-MSR filtering.
- Small-gain controller, using raw neighbour messages.
- The same small-gain controller, with W-MSR filtering.

A local-only controller is included as a reference; it does not use neighbour messages.

## 2. The five policies

Every policy uses the same heterogeneous physical system, communication graph, initial conditions, and discrete-time horizon. Each policy has its own copy of the evolving state matrix.

| Policy | Neighbour information | Role |
|---|---|---|
| **Local-only reference** | None | Applies only the local homogenizing feedback. It is a reference, not a communication-based consensus controller. |
| **Paper GNN (raw)** | All received scalar messages | Paper-style learned controller without message filtering. |
| **Paper GNN + W-MSR** | Retained scalar messages | The same paper-style controller with local scalar W-MSR filtering. |
| **Small-gain (raw)** | All received scalar messages | Trained small-gain controller without message filtering. |
| **Small-gain + W-MSR** | Retained scalar messages | The same small-gain controller with local scalar W-MSR filtering. |

The primary comparisons are **Paper GNN (raw) vs. Paper GNN + W-MSR** and **Small-gain (raw) vs. Small-gain + W-MSR**. The local-only curve provides context, but it should not be interpreted as a raw-versus-filtered communication comparison.

## 3. Physical plant and controller signals

Each agent has a two-dimensional state and a scalar physical input:

\[
 x_i(k+1)=A_i x_i(k)+B_i u_i(k),
 \qquad x_i(k)\in\mathbb R^2,\quad u_i(k)\in\mathbb R.
\]

The heterogeneous plants use the local homogenizing feedback supplied by the core library:

\[
 u_i^{\mathrm{local}}(k)=K_i x_i(k).
\]

A learned controller may add a scalar coordination correction:

\[
 u_i(k)=K_i x_i(k)+v_i(k).
\]

The simulation advances the physical state using the core library's NumPy dynamics implementation. W-MSR does **not** replace the two-state plant update. It filters one scalar communication signal before the relevant controller forms its correction.

### One scalar message per agent

For the small-gain controller, agent \(i\) broadcasts

\[
 s_i^{\mathrm{small}}(k)=W^T x_i(k),
 \qquad W\in\mathbb R^{2\times1}.
\]

For the paper-style controller, it broadcasts

\[
 s_i^{\mathrm{paper}}(k)=H x_i(k),
 \qquad H\in\mathbb R^{1\times2}.
\]

The bundled paper model must have one hidden channel, so each agent sends exactly one scalar. The retained messages are then used to form that controller's feedback correction. The paper-style correction also uses its learned per-agent output gain and its `tanh` nonlinearity.

Compromised agents in this test bench do not apply a malicious actuator input. They continue to use their local plant feedback and send a false **communication** value. Their physical states are simulated, but they are excluded from the normal-agent disagreement and control-energy metrics described below.

## 4. Exact scalar W-MSR update

The implementation in `src/distributed_consensus/robust/wmsr.py` applies the supplied scalar W-MSR rule separately at each normal receiver. Let \(s_i\) be receiver \(i\)'s own scalar signal and let \(s_j\) be a neighbour's received scalar signal.

For every normal receiver \(i\):

1. Compare each neighbour's scalar value with the receiver's own value.
2. Remove up to `F` of the smallest neighbour values that are **strictly below** the receiver's own value.
3. Remove up to `F` of the largest neighbour values that are **strictly above** the receiver's own value.
4. Retain values equal to the receiver's own value, together with all other neighbour values.
5. Average the receiver's own value together with the retained neighbour values.

The update is synchronous: every receiver's result is computed from the same pre-update signal vector. Ties equal to the receiver's own value are retained. If there are fewer than `F` values on one side, all values on that side are removed.

**For `F = 1`, up to two neighbour values may be ignored in one update:** the smallest value below the receiver and the largest value above it. Those values may come from two different senders. This does **not** mean that two malicious neighbours are allowed. An honest neighbour with an extreme value can also be removed by the rule.

The GUI's orange dashed arrow identifies a sender-to-receiver message that was ignored by W-MSR for that receiver. The underlying communication link remains in the graph; filtering does not delete edges.

## 5. Adversary model and the meaning of `F`

The implemented attack is **malicious broadcast**:

- A compromised sender transmits the same false scalar value to all of its neighbours.
- The false value is selected automatically just outside the current range of normal-agent scalar signals.
- The attack alternates between an upper extreme and a lower extreme across steps, exercising both W-MSR trimming directions.
- The attack changes the communicated scalar value; it does not directly alter the sender's physical state or actuator input.

The simulation uses the **F-local malicious** assumption: every normal agent has at most `F` malicious neighbours. This is a local bound, not a bound of `F` malicious agents over the entire graph. Multiple compromised agents can be present if their placement satisfies the local-neighbour constraint.

This demo does **not** implement Byzantine equivocation, where one compromised sender sends different values to different receivers. It also does not implement actuator attacks, communication jamming, or packet loss. The W-MSR filter does not know which agents are compromised; it only compares received scalar values with the receiver's own value.

### Graph robustness warning

Connectivity alone is not the sufficient graph condition reported by the application. For the F-local malicious scalar W-MSR case, the sufficient condition used by the topology checker is \((2F+1)\)-robustness. In particular, when `F = 1`, the checker evaluates 3-robustness.

The application can still run when the condition is not met, but the corresponding sufficient scalar W-MSR guarantee must not be claimed for that graph. Exact robustness checking is intended for small graphs (the helper's default limit is 14 nodes); above that limit the checker can report that the condition was not checked rather than treating ordinary connectivity as a substitute.

This graph test concerns the scalar W-MSR protocol. It is **not** a stability certificate for the complete two-state physical feedback system.

## 6. Reading the metrics

The controller scorecard contains three different quantities. They are not combined into a weighted score.

- **`e_RMS`** is the *current* mean-centred RMS disagreement of the normal agents. This is the quantity plotted over time.
- **`J_cent`** (`disagreement_cost` in the simulation model) is the *cumulative* mean-centred squared disagreement cost over normal agents up to the current step.
- **`J_U`** (`energy` in the simulation model) is the *cumulative* squared physical control-input energy over normal agents up to the current step.

Let \(\mathcal V_{\mathrm{normal}}\) be the set of non-compromised agents and let \(N_{\mathrm{norm}}=|\mathcal V_{\mathrm{normal}}|\). Define their mean state as

\[
 \bar{x}_{\mathrm{norm}}(k)
 =\frac{1}{N_{\mathrm{norm}}}
   \sum_{i\in\mathcal V_{\mathrm{normal}}}x_i(k).
\]

For the normal-agent states \(x_i(k)\) and scalar **total physical inputs** \(u_i(k)\), the simulation reports

\[
 e_{\mathrm{RMS}}(k)
 =\sqrt{\frac{1}{N_{\mathrm{norm}}}
   \sum_{i\in\mathcal V_{\mathrm{normal}}}
   \left\|x_i(k)-\bar{x}_{\mathrm{norm}}(k)\right\|_2^2},
\]

\[
 J_{\mathrm{cent}}(k)
 =\sum_{t=0}^{k-1}\sum_{i\in\mathcal V_{\mathrm{normal}}}
   \left\|x_i(t)-\bar{x}_{\mathrm{norm}}(t)\right\|_2^2
 =N_{\mathrm{norm}}\sum_{t=0}^{k-1}e_{\mathrm{RMS}}(t)^2,
\]

\[
 J_U(k)
 =\sum_{t=0}^{k-1}\sum_{i\in\mathcal V_{\mathrm{normal}}}u_i(t)^2.
\]

Here \(u_i(t)\) is the total commanded physical input, including both local feedback and the controller's coordination correction. The simulation accumulates each step's contribution using the state and input **before** that step's physical update. Consequently, at step \(k\), `e_RMS` describes the current state \(X(k)\), while the cumulative costs include the stages \(t=0,\ldots,k-1\). At step zero, both cumulative costs are zero.

### Distinction from the core library's `J_E` and `J_pair`

The core training/evaluation code in `src/distributed_consensus/` uses different, explicitly named metrics:

- \(J_E=\sum_{t=0}^{T-1}\|L_{rw}X(t)\|_F^2\) is the full-graph random-walk Laplacian disagreement cost used in the core training objective.
- \(J_{\mathrm{pair}}=\sum_{t=0}^{T-1}\frac{1}{N}\sum_{i=1}^{N}\|x_i(t)-\bar{x}(t)\|_2^2\) is the separate, per-agent mean-centred disagreement evaluation metric, with \(\bar{x}(t)=\frac1N\sum_{i=1}^N x_i(t)\). It is not the core training loss.

The adversarial simulator evaluates only the normal-agent set. If \(J_{\mathrm{pair,normal}}(k)\) denotes the same per-agent mean-centred cost accumulated over normal agents only, then

\[
 J_{\mathrm{cent}}(k)
 =N_{\mathrm{norm}}J_{\mathrm{pair,normal}}(k).
\]

Thus the simulator's `J_cent` is **not** the core library's `J_E`. The simulator's `J_cent` is also unnormalised by the number of normal agents, whereas `J_pair` is divided by the relevant agent count.

`J_cent` and `J_U` are accumulated diagnostic costs, not components that are added together to produce the current error. They can stay constant or increase, but cannot decrease. They appear to flatten when new per-step contributions become very small. Meanwhile, the current `e_RMS` can continue decreasing. The GUI displays `J_cent` and `J_U` separately; it does not compute a combined weighted total.

The controller table shows current `e_RMS` and cumulative `J_cent` and `J_U`. Only `e_RMS` is shown as a time-series chart.

## 7. GUI controls and interaction

### Main controls

- **Play / Pause:** start or stop automatic stepping.
- **Step:** advance the simulation by one discrete-time step.
- **Restart:** reset all five policies to the same initial conditions and return to step zero.
- **Save Full Run:** save trajectories, summary rows, configuration, and PNG figures for all five policies under `simulation/logs/run_YYYYMMDD_HHMMSS/`. It saves the data accumulated from step zero through the current step.
- **Help:** open the in-app explanation of the controller comparison, W-MSR rule, threat model, and metrics.

### Configuration

- **Graph:** choose the communication topology: `complete`, `wide_ring`, `ring`, `chain`, `star`, or `random`.
- **Malicious agent ID(s):** enter 1-based agent IDs separated by commas. Use `0` (or `off`) to disable the attacker. IDs must be unique and within the network size. At least two normal agents must remain. The selected malicious-agent placement must also satisfy the F-local neighbour bound for the chosen graph; invalid configurations are rejected.
- **`F`:** maximum number of malicious neighbours allowed at any normal agent under the F-local assumption. It is also the number of extreme values the scalar W-MSR rule may trim on each side of the receiver's own value. The GUI currently offers `F = 1` and `F = 2`.
- **Graph-condition status:** reports whether the sufficient W-MSR robustness condition was met by the selected graph. A warning does not prevent an illustrative run.

### Scorecard and graph

- **`e_RMS`:** current mean-centred RMS disagreement among normal agents.
- **`J_cent`:** cumulative mean-centred squared disagreement cost over normal agents (`disagreement_cost`).
- **`J_U`:** cumulative squared total physical control-input energy over normal agents (`energy`).
- **Orange dashed arrow:** the selected receiver ignored that sender's scalar value. The communication link remains present.

Select a controller row to highlight its curve and corresponding information. Select an agent node to open its message audit, including received values, retained neighbours, ignored neighbours, and the W-MSR mean. A compromised node is identified separately from normal nodes.

### Keyboard shortcuts

- `Space`: play or pause.
- `Right Arrow`: advance one step.
- `Escape`: pause.
- `Ctrl + R`: restart.
- `Ctrl + +` or `Ctrl + =`: zoom in.
- `Ctrl + -`: zoom out.
- `Ctrl + 0`: reset zoom to 100%.

The GUI run advances to step 250 and then pauses. The controller table displays the current values at the current step; the disagreement chart plots the `e_RMS` history.

## 8. Setup and benchmark checkpoints

The repository targets Python 3.10 or later. From the repository root, install the project and its test dependencies with:

```bash
uv sync --extra test
```

The GUI uses Tkinter. Ensure that Tkinter is installed for the Python interpreter used by `uv` (some Linux distributions package it separately).

The simulation loads a trained benchmark checkpoint by default from:

```text
simulation/assets/robustness_benchmark.npz
```

Benchmark training also requires a compatible physical-system checkpoint. The default source path in the code is:

```text
simulation/assets/topology_rep00_identity.npz
```

If either checkpoint is absent, supply a valid source checkpoint from the core experiment outputs. Its physical system must have the same number of agents requested by `--agents` (10 by default).

Create the benchmark with:

```bash
uv run python -m simulation.train_benchmarks \
  --source-checkpoint results/path/to/checkpoint.npz \
  --paper-hidden 1
```

The output defaults to `simulation/assets/robustness_benchmark.npz`. Use `--out PATH` to save it elsewhere, and then pass that path with `--benchmark PATH` when launching the GUI or headless commands.

**Important compatibility detail:** the current `load_benchmark()` implementation accepts only a one-hidden-channel paper model (`paper_H` shape `(1, 2)` and `paper_K` shape `(N, 1)`). In the supplied code snapshot, the training CLI's `--paper-hidden` default is `2`, so explicitly pass `--paper-hidden 1` as shown above. A checkpoint made with two hidden channels will be rejected by the simulator.

For a fast wiring/smoke test only, benchmark training supports:

```bash
uv run python -m simulation.train_benchmarks \
  --source-checkpoint results/path/to/checkpoint.npz \
  --paper-hidden 1 --quick
```

`--quick` uses only a few short training iterations. These weights are for testing the software path, not for research-quality comparisons or reported results.

The benchmark trainer trains the paper-style and small-gain controllers using matched training/validation streams and saves the selected model weights with benchmark metadata. The small-gain weight is projected using the core library's numerical plant-gain constraint; the paper-style model is not projected.

## 9. Run the GUI

From the repository root:

```bash
uv run python -m simulation
```

To use a non-default benchmark checkpoint, modify the GUI launch call in Python or invoke the `DemoApp` with the desired benchmark path. The headless commands below expose a `--benchmark` option directly.

## 10. Headless preview

Run one deterministic demonstration without opening Tkinter:

```bash
uv run python -m simulation.preview \
  --topology complete --adversary 10 --F 1 --steps 250 \
  --out results/wmsr_local_preview
```

Disable the attacker with `--adversary 0`. Multiple malicious IDs can be supplied as space-separated arguments or comma-separated values, for example `--adversary 3 7` or `--adversary 3,7`.

Useful options include:

| Option | Meaning |
|---|---|
| `--benchmark PATH` | Benchmark checkpoint to load. |
| `--topology NAME` | One of `complete`, `wide_ring`, `ring`, `chain`, `star`, `random`. |
| `--seed N` | Initial-condition and graph seed. |
| `--steps N` | Number of physical simulation steps. |
| `--adversary IDS` | 1-based malicious agent IDs; use `0` alone to disable. |
| `--F N` | F-local bound and W-MSR trimming count per side. |
| `--threshold VALUE` | Normal-agent disagreement threshold used for settling statistics. |
| `--out PATH` | Output directory. |

A preview writes `trajectories.csv`, `summary.csv`, `configuration.json`, and a `figures/` directory containing:

- `normal_disagreement_all_controllers.png`
- `normal_state_1_by_controller.png`
- `normal_state_2_by_controller.png`
- `normal_control_by_controller.png`
- `wmsr_discarded_values.png`

## 11. Multi-trial evaluation

Run matched trials across different initial-condition seeds with:

```bash
uv run python -m simulation.evaluate \
  --trials 20 --steps 250 --topology complete \
  --adversary 10 --F 1 \
  --out results/wmsr_local_evaluation
```

Each trial runs all five policies under the same trial configuration. The raw-versus-W-MSR comparisons are paired by trial. The evaluation writes:

- `trial_summary.csv`: one result per method and trial.
- `aggregate_summary.csv`: across-trial settling, final disagreement, cumulative disagreement-cost, control-energy, and filtering summaries.
- `paired_wmsr_effect.csv`: paired raw-versus-filtered results for the paper-style and small-gain controllers.
- `aggregate_curves.csv`: per-step mean, median, and quartile disagreement summaries.
- `aggregate_disagreement.png`: median disagreement curves with interquartile bands.
- `aggregate_settling.png`: settling success fractions and median settling steps.
- `representative_run/`: trajectory CSV and figures for the first trial.
- `configuration.json`: benchmark path, settings, topology-check report, and comparison definitions.

The strict settling time is the first sample after which every remaining observed `e_RMS` sample stays at or below the requested threshold. A separate eight-step hold indicator is also recorded in the single-run summary. A finite-horizon run that never satisfies the strict condition is marked unsettled; this does not establish that it could never settle at a longer horizon.

## 12. Main code map

| File | Responsibility |
|---|---|
| `simulation/model.py` | Loads and validates benchmark checkpoints; constructs the five matched policies; applies malicious broadcasts and scalar W-MSR; advances the physical system; computes normal-agent metrics; exports trajectories, summaries, and figures. |
| `simulation/gui.py` | Tkinter application, graph drawing, scorecard, animation controls, message audit, and run saving. |
| `simulation/train_benchmarks.py` | Trains and saves the paper-style and small-gain benchmark controllers. |
| `simulation/preview.py` | Runs one headless configuration and writes a complete run output. |
| `simulation/evaluate.py` | Runs matched multi-trial comparisons and aggregate summaries/plots. |
| `src/distributed_consensus/robust/wmsr.py` | Scalar W-MSR update, retained/ignored sender details, adversary validation, and graph-robustness checks. |
| `src/distributed_consensus/control/dynamics.py` | Physical two-state plant update used by the simulation. |

The simulation is a standalone application layer built on selected core-library components. It does not change the core training or evaluation implementation.

## 13. Interpretation and limitations

Keep the following distinctions in mind when interpreting a run:

1. **A graph-condition result is not a plant-stability proof.** The W-MSR robustness check concerns a scalar communication-filtering theorem. It does not prove stability or convergence of the entire heterogeneous, two-state closed-loop system with the controller-specific feedback corrections.
2. **The attack is deliberately limited.** Only a scalar message is falsified. Byzantine equivocation, actuator attacks, packet loss, delays, and jamming are outside the model.
3. **Filtering is scalar.** The filter is not applied independently to both physical state coordinates. Each agent sends one controller-specific scalar signal.
4. **The compromised agent is not an actuator attacker.** It continues local plant feedback; its false broadcast is the adversarial action. Compromised-agent states and inputs are excluded from the reported normal-agent `e_RMS`, `J_cent`, and `J_U`.
5. **The metrics are finite-horizon diagnostics.** A lower final `e_RMS`, `J_cent`, or `J_U` in one run is an empirical comparison for the specified benchmark, topology, seed, adversary placement, and horizon—not a universal guarantee.
6. **The accumulated costs are separate.** `J_cent` measures normal-agent disagreement accumulated over time; `J_U` measures normal-agent physical input energy. The GUI does not add them or apply a weighting coefficient.
7. **A failed or unchecked robustness condition must be reported honestly.** A run can still be useful for observing behaviour, but it must not be described as satisfying a sufficient W-MSR graph condition when the checker reports `False` or `None`.

For mathematical definitions used by the core controller-training experiments, see `src/distributed_consensus/README.md` and `src/distributed_consensus/evaluation/README.md`. This simulation README uses `J_cent` for its normal-agent, unnormalised mean-centred cumulative disagreement cost to keep it distinct from the core library's Laplacian cost `J_E` and per-agent mean-centred evaluation metric `J_pair`.
