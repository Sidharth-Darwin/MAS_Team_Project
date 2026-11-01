# Multi-agent consensus — experiment results

Four **independent** experiments; generated figures are **PNG only**. All simulations use the original heterogeneous `A_i`, `B_i` and physical control `u_i=K_i x_i + v_i`. The mathematical implementation under `src/distributed_consensus/` is untouched.

## Runs

Run from the project root. Run each independently; do not append data from a different experiment to its folder. To rerun, move or remove only that experiment's result directory.

```bash
uv run python experiments/activation_comparison.py --out results/01_activation_comparison --sizes 3 --repeats 5 --initials 32 --epochs 600 --train-horizon 100 --eval-horizon 200 --batch 128 --val-batch 128 --threshold 0.001 --eta 0.95
uv run python experiments/network_size_generalization.py --out results/02_network_size_generalization --sizes 3 50 100 --repeats 5 --initials 32 --epochs 600 --train-horizon 100 --eval-horizon 200 --batch 128 --val-batch 128 --threshold 0.001 --eta 0.95
uv run python experiments/plant_gain_transfer.py --out results/03_plant_gain_transfer --sizes 3 50 100 --repeats 5 --initials 32 --epochs 600 --train-horizon 100 --eval-horizon 200 --batch 128 --val-batch 128 --threshold 0.001 --eta 0.95 --gamma-train 0.75
uv run python experiments/topology_generalization.py --out results/04_topology_generalization --n 10 --train-topology chain --repeats 5 --initials 32 --epochs 600 --train-horizon 100 --eval-horizon 200 --batch 128 --val-batch 128 --threshold 0.001 --eta 0.95
```

For a short pipeline check, append `--quick` and use **different** output folders; smoke results are not report evidence.

## Experiment descriptions

**01 Activation:** Train independent shared weights with identity and tanh activation on the same three-agent system and graph, with identical initial training/validation sampling seeds. Test at N=3 against the local-only baseline. Compare dynamics and control costs.

**02 Network size:** Train an identity controller on N=3. Freeze W, test with N=3, 50, 100, connected undirected Erdős–Rényi graphs and independent heterogeneous test systems. Each network's full plant gain is calibrated to match the training network's target value; no test-time weight rescaling.

**03 Gain transfer:** Train one identity W on N=3 at `gamma_train=0.75`. Test at full-network gains `0.75*gamma_train` and `1.25*gamma_train` for N=3, 50, 100. Test cases share the **same base A, base B, ER graph and initial states** for each size/repetition; half of the agents (one vs two when N=3) receive a weaker `0.5` input-effectiveness multiplier, the others a stronger `1.5` multiplier, and all inputs are then scaled by a common factor to achieve the target full gain. Effective multipliers and group labels are saved in `traces/`. W is frozen. Record `gamma_test*||W||₂` for both cases; values >=1 make the sufficient small-gain condition inconclusive, not necessarily unstable.

**04 Topologies:** Train one identity controller on a ten-agent **chain** graph. Freeze W; test on N=10 **chain, star, ring, wide ring (2 hops), complete and connected ER**. Reuse **identical physical A, B, K and initial states** across all six graphs. Draw each actual graph separately. Only graph topology changes; full plant gains may differ.

## What to inspect

Each experiment's `figures/` contains:

- `*_error_convergence.png`: network-level Laplacian disagreement RMS vs timestep, threshold marked.
- `*_centered_disagreement.png`: mean-centered disagreement vs timestep.
- `*_agent_error.png`: **individual agents' disagreement norms** vs timestep for a representative trajectory.
- `*_state_x1.png` and `*_state_x2.png`: **individual signed state-component trajectories** vs timestep, baseline and learned side by side.
- `*_control_total.png`: **individual signed physical actuator commands** vs timestep, baseline and learned side by side.
- `*_correction_input.png`: learned correction input `v_i(k)` vs timestep (baseline is zero).
- `*_cumulative_energy.png`: cumulative **physical input energy per agent**; not an RMS curve.
- Settling comparison PNGs. Experiment 03 also directly overlays below/above plant-gain cases and plots measured plant/loop gains. Experiment 04 saves **each topology drawing** and compares gain and settling by topology.

For N=3 and N=10 all agent curves are individually labeled. For N=50 and N=100 every agent is shown with transparent lines and the mean highlighted to avoid unreadable legends. Per-agent figures use the **first held-out trajectory** of each realization; pooled curves aggregate held-out trajectories and must not be confused with individual-agent motion.

## Saved files

- `summary.csv`: means and conditional settling summaries per experiment realization.
- `trajectory_metrics.csv`: per-initial-condition metrics and settled flags.
- `curves.csv`: sampled mean/median/quartile curves.
- `checkpoints/`: numeric learned weights and training-history PNG/CSV files.
- `traces/*.npz`: compressed numeric histories for one representative initial condition (agent states, disagreement, local control, learned correction and total input) plus physical test matrices and graph adjacency. Experiment 03 additionally saves stronger/weaker groups and final effective multipliers.
- `configuration.json`: CLI settings, Python/NumPy versions and Git hash if available.

No SVG files, large reproducibility snapshots, or separate combined report runner.

## Convergence definition and caveats

For N agents and states `X(k)` with two features, `E(k)=L_rw X(k)` and disagreement RMS = `||E(k)||_F/sqrt(N)`. A trajectory settles at the **first timestep after the last violation** of the absolute `0.001` threshold (including the terminal state). Never encode an unsettled trajectory as settling at the final timestep. The settled fraction and median *among settled trajectories* must be reported together.

Physical control plots show `u_i=K_i x_i+v_i`; the correction is shown separately. Control energy sums `u_i(k)^2` over action timesteps, normalized by N for cumulative plots. Absolute state convergence to zero is not the same thing as consensus; compare disagreement independently. The `eta=0.95` projection and plant gains are numerical floating-point sufficient-condition checks, not formal interval certificates.
