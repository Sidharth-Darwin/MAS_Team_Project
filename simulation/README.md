# Multi-Agent Control Lab — interactive alignment demonstration

An **add-on**, not a rewrite of `MAS_Team_Project`. All code here is inside
`simulation/`. It **imports** your existing library and requires no changes
to `src/`, `experiments/`, tests, or trained models.

## Setup and launch (recommended)

Extract the downloaded `simulation_only.zip` into the **root** of your existing
`~/Downloads/distributed_consensus/` repository, so that it contains
`simulation/`, `src/`, and `pyproject.toml` as siblings.

```bash
cd ~/Downloads/distributed_consensus
uv run python -m simulation
```

Alternative if not using uv: activate the environment with the original
project installed (`pip install -e .`), then run `python -m simulation`.

**GUI dependency:** Python's built-in `tkinter`. On Ubuntu/Debian, if it is not
installed, run `sudo apt install python3-tk` using your system Python. NumPy,
SciPy, PyTorch, and Matplotlib are already dependencies of the original project.
There is **no** Pygame/PyQt/browser/graphics engine dependency.

To run the headless comparison without tkinter:

```bash
uv run python -m simulation.preview --steps 200
uv run python -m simulation.preview --steps 200 --pulse-at 85 --pulse-agent 3
uv run python -m unittest discover -s simulation/tests -v
```

To export a headless CSV, add `--csv simulation/logs/preview.csv`.

## Alternative independent directory

The `simulation/` directory may live in a different codebase as long as Python
can import `distributed_consensus` from the original project's `src` package.
For example, from the parent folder containing the standalone `simulation/`:

```bash
PYTHONPATH="$HOME/Downloads/distributed_consensus/src:$PWD" python -m simulation
```

Use a Python environment with dependencies from the original project.

## What you see

- **Left:** local-only controller, `u_i = K_i x_i`.
- **Right:** fixed, pretrained learned controller,
  `u_i = K_i x_i + phi(W^T e_i)`.
- Identical physical matrices, graph, time steps, initial states, and injected
  disturbances on both sides.
- Every rendered lifting column is an **independent one-degree-of-freedom
  linear actuator**. Its height is drawn from actual `q_i` (state 0); its
  velocity is state 1. The dashed linking line is a **virtual alignment guide**,
  **not** a mechanically rigid platform. No force coupling, collision avoidance,
  saturations, or stress model has been invented.
- Bottom left: actual undirected communication graph; selected agent and its
  neighbors; signed state, learned correction, and commanded total input.
- Bottom right: live RMS disagreement curves on a logarithmic scale.

The default controller is the unchanged, numeric-only Experiment 04 checkpoint
`rep00_identity.npz` copied byte-for-byte into `simulation/assets/`. It is loaded
by `distributed_consensus.evaluation.reporting.load_checkpoint()`.
Original project code calculates and validates the agent physical matrices and K.
No weights are trained in this demo.

## Controls

| Action | What it does |
|---|---|
| Play / Pause | Advance both simulations synchronously |
| STEP / right arrow | Advance exactly one discrete-time step |
| RESTART / Ctrl+R | Reset both to identical saved starting conditions |
| Click a column | Select the same agent index in both worlds |
| Drag a column vertically | Apply **the same additive position displacement** to both worlds at the same simulation step |
| PULSE BOTH | Apply the same additive actuator-input disturbance for three steps to the selected agent on both sides |
| Agents | 3, 5, 10, 20 (requires APPLY / RESET) |
| Graph | Original checkpoint graph, chain, star, ring, wide ring, complete, or random |
| Seed | Set deterministic initial states; APPLY / RESET |
| Steps/sec | Playback rate, **not** the physical system timestep |
| SAVE CSV | Export actual signed states, controls and disturbances to `simulation/logs/` |
| Esc / Space | Pause / toggle playback |

**Network-size changes:** at N=10, the original physical matrices are used.
For other N, the original library's `generate_system()` builds an appropriate
heterogeneous, exact-matchable system. The **same frozen W** is reused without
retraining. Generalization quality and small-gain margins can change; these
settings are exploratory, not guaranteed to improve performance.

**Graph changes:** physical matrices and W remain fixed for a given N.
`checkpoint` with a different N uses chain as a fallback, not the saved graph.

## Predefined success criterion

A system is *operationally ready* when all three are true:

```
max_i |q_i| <= 0.05
max_i |v_i| <= 0.05
max_i q_i - min_i q_i <= 0.07
```

and this is observed continuously for **8 samples**. The UI retroactively
reports the **first** sample of that confirmed 8-sample sequence. Readiness
is invalidated if the condition is later violated; intervention resets the
readiness clock. This is a finite-horizon operational metric, **not** a formal
infinite-horizon settling-time proof.

The disagreement threshold is `e_RMS <= 1e-3`, likewise displayed after 8
consecutive samples. The project's research *settling time* is stricter: it
requires staying within the bound through the entire observation horizon.
The GUI does **not** relabel the 8-step operational indicator as that stronger
research metric.

Cumulative `J_E` is sum of `||E(k)||_F^2` over acted-upon states and `J_U` is
sum of `||U(k)||_F^2` for the **commanded physical input**; neither includes
the extra disturbance input. The GUI displays both as a comparison, not as
necessarily better in all scenarios.

## Reproduced example (original Experiment 04 checkpoint; seed 3, chain)

For 200 physics steps, initial `q ~ Uniform(-2,2)` and
`v ~ Uniform(-0.6,0.6)` in fixed NumPy RNG order:

| Metric | Baseline | Learned |
|---|---:|---:|
| Operational readiness first sample | 43 | 38 |
| Retrospective research disagreement settling (`e_RMS <= 1e-3`) | 79 | 58 |
| Cumulative disagreement cost | 318.391791 | 295.758207 |
| Cumulative total commanded-input energy | 930.492353 | 859.118587 |

Adding a **matched** amplitude `+8` three-step actuator-input pulse to agent
3 at step 85 gives operational readiness after intervention at 115 (baseline)
versus 113 (learned). These are observed cases, not guarantees for other seeds,
topologies, disturbances, or network sizes.

## Fidelity / local-only information

Each local agent computes, using only a snapshot of its own state and the
messages from the neighbor IDs in its routing table:

```
e_i = x_i - (sum_j a_ij x_j) / sum_j a_ij
u_local = K_i x_i
v_i = phi(W.T e_i)  # zero for baseline
u_i = u_local + v_i
x_i_next = A_i x_i + B_i (u_i + external_disturbance_i)
```

Neighbor messages are all from timestep `k`; every decision is computed
before *any* next-state is committed. Global graph operations are used **only**
for visual metrics and for comparison tests, not for agent control.

The reference `step_numpy()` and `rollout_numpy()` functions from the original
package provide numerical ground truth in the test suite. The per-agent method
must match them to numerical precision. We do not edit either function.

**Interventions:** a mouse drag is an explicit externally applied instantaneous
position change; a pulse is an additive disturbance in the **actuator input
channel**, applied identically to both worlds. A pulse is not a physically
modeled arbitrary force on a rigid platform. CSV separates commanded input,
disturbance, and external position jumps. Manual interventions can produce
slow recovery or worse learned performance; the display always shows real data.

## File ownership

```
simulation/
  __init__.py
  __main__.py           # GUI entry point
  gui.py                # Tkinter drawing and interaction only
  model.py              # per-agent local-information synchronous simulator
  preview.py            # headless compare/export
  assets/
    topology_rep00_identity.npz  # UNCHANGED COPIED checkpoint
  tests/
    test_simulation.py   # agreement with original step_numpy()/rollout_numpy()
  README.md
```

`simulation/logs/` is created only if you export CSV. No research files are
written or modified. This is a visual presentation, not a fifth research run.
