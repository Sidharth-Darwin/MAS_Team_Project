# Scalar W-MSR consensus-control test bench

## What this is

This is an **abstract consensus-control test bench**, not a practical robot or vehicle simulation. Its main view is a consensus-error chart plus a communication graph and a per-agent message audit.

It compares five policies from identical initial physical states:

1. **Local-only reference** — ignores neighbour messages.
2. **Paper GNN (raw)** — trained paper-style controller using unfiltered messages.
3. **Paper GNN + W-MSR** — the same trained controller using the scalar W-MSR filter.
4. **Small-gain (raw)** — your trained controller using unfiltered messages.
5. **Small-gain + W-MSR** — the same trained controller using the scalar W-MSR filter.

The primary comparisons are raw versus W-MSR for each learned controller. Local-only is only a reference because it does not use communication.

## The exact W-MSR rule

`src/distributed_consensus/robust/wmsr.py` implements the scalar algorithm supplied for this project. Every normal receiver:

1. Compares the scalar value from each neighbour with its own value.
2. Removes up to `F` smallest neighbour values that are strictly below its own.
3. Removes up to `F` largest neighbour values that are strictly above its own.
4. Keeps equal values and all other neighbour values.
5. Averages its own value together with the retained neighbour values.

The implementation uses simultaneous updates from the same pre-update signal vector. `wmsr_local(X, N, F)` returns the updated scalar vector; `wmsr_local_with_details(...)` additionally returns retained/ignored sender IDs for the GUI and tests.

**For `F=1`, the supplied algorithm can ignore two different neighbours in one update:** the smallest value below the receiver's own value and the largest value above it. This is intentional and follows the supplied pseudocode. It does not mean two malicious neighbours are allowed. An honest neighbour with an extreme value can also be ignored.

## How it is used by the controllers

Each agent sends one scalar controller signal:

- Paper GNN: `H @ x_i` (the bundled paper model has one hidden channel).
- Small-gain controller: `W.T @ x_i`.

W-MSR filters these scalar neighbour messages. The retained values are then used to form the corresponding controller's original neighbour-feedback correction. The two-state physical plant is still propagated by `src/distributed_consensus/control/dynamics.py`; the scalar W-MSR average does not replace the physical state update.

This is a scalar message-filtering adaptation inside the feedback loop. The classical scalar W-MSR theorem does **not** automatically prove stability of the complete two-state physical feedback system, and the original fixed-Laplacian small-gain theorem does not automatically cover this state-dependent filtering operation.


## Reading the metrics

The controller table contains three different quantities:

- **e_RMS** is the *current* mean-centred RMS disagreement of the normal agents. This is the quantity plotted over time.
- **J_E** is the *cumulative* squared disagreement cost up to the current step.
- **J_U** is the *cumulative* squared physical control-input energy up to the current step.

For normal-agent states \(x_i(k)\), their mean \(\bar{x}(k)\), and scalar inputs \(u_i(k)\):

\[
e_{\mathrm{RMS}}(k)=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\|x_i(k)-\bar{x}(k)\|_2^2},
\]
\[
J_E(k)=\sum_{t=0}^{k-1}\sum_{i=1}^{N}\|x_i(t)-\bar{x}(t)\|_2^2,\qquad
J_U(k)=\sum_{t=0}^{k-1}\sum_{i=1}^{N}u_i(t)^2.
\]

So **the current error is not formed by adding J_E and J_U**. \(J_E\) and \(J_U\) are accumulated diagnostic costs. They can only stay the same or increase; they appear to flatten when the new per-step contributions become very small. Meanwhile, the current \(e_{\mathrm{RMS}}\) can keep decreasing. The GUI shows these two costs separately and does not combine them into a weighted total.

The controller table shows the current \(e_{\mathrm{RMS}}\) and the accumulated \(J_E,J_U\) values at the current step; only \(e_{\mathrm{RMS}}\) is drawn as a time-series chart.

## Adversary and `F`

The demo uses one simple attack: **malicious broadcast**. A compromised agent sends the same false scalar value to all its neighbours. The false value is automatically chosen just outside the current normal-signal range; there is no manual attack-strength control.

**Byzantine equivocation** is different: a compromised sender sends different false values to different neighbours. Equivocation is not included in this simplified demo. Physical actuator attacks and communication jamming are also not included.

The threat model is **F-local**: every normal agent has at most `F` malicious neighbours. This does not limit the total number of malicious agents if they are sufficiently separated.

For the standard scalar F-local malicious W-MSR result, `(2F+1)`-robustness is a sufficient fixed-graph condition. For `F=1`, the GUI checks 3-robustness. Connectivity by itself is insufficient. A failed condition does not stop an illustrative run, but no W-MSR guarantee should be claimed for that graph.

## GUI controls and numbers

- **Graph:** choose the communication topology.
- **Malicious agent IDs:** 1-based IDs; use `0` to disable the attacker.
- **F:** maximum malicious neighbours allowed at any normal agent under the F-local assumption.
- **e_RMS:** current mean-centred RMS disagreement among normal agents.
- **J_E:** cumulative squared disagreement cost.
- **J_U:** cumulative squared physical control-input energy.
- **Orange dashed arrow:** the selected receiver ignored that sender's scalar value. The communication link remains present.

Use **Ctrl + +** (or **Ctrl + =**) to zoom in, **Ctrl + -** to zoom out, and **Ctrl + 0** to reset the interface size.

Select a controller row to highlight it. Select a node to open a message audit showing received values, retained neighbours, ignored neighbours, and the W-MSR mean.

**SAVE FULL RUN** saves the complete run for all five controllers: trajectories, summary, configuration and PNG figures. There is no separate network-snapshot selector.

## Run

```bash
uv run python -m simulation
```

Headless preview:

```bash
uv run python -m simulation.preview \
  --topology complete --adversary 10 --F 1 --steps 250 \
  --out results/wmsr_local_preview
```

Multi-trial evaluation:

```bash
uv run python -m simulation.evaluate \
  --trials 20 --steps 250 --topology complete \
  --adversary 10 --F 1 \
  --out results/wmsr_local_evaluation
```

Use `--adversary 0` to disable the attacker. Try `chain`, `ring`, `wide_ring`, `star`, and `random` to see the graph-condition warning.
