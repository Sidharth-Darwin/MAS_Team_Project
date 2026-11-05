# Rerunning all four consensus studies with the trained paper GNN as baseline

## What changes

Your four original experiment questions remain **unchanged**:

1. Activation: independently train projected identity and projected tanh controllers against a trained paper GNN (plus local-only reference), N=3.
2. Size transfer: train on N=3 and deploy *frozen* models to N=3, 50, 100.
3. Plant-gain transfer: train on N=3 and deploy *frozen* models to below/above train gain at N=3, 50, 100.
4. Topology transfer: train on chain N=10, deploy *frozen* models to six connected graph types, keeping A_i,B_i and initial conditions fixed.

All saved figures include paper-GNN traces alongside learned small-gain and local-only traces:
- Error RMS versus discrete timestep and absolute 0.001 threshold
- Individual-agent disagreements
- Individual-agent physical state components x1 and x2
- Actual signed physical actuator command u_i
- Learned correction input v_i
- Mean-centered disagreement
- Cumulative physical input energy
- Settling fraction and conditional settling time; numerical CSVs and representative raw trajectory NPZs

### Paper-style neural baseline

The Wang--Lin (FASTA 2025) equation (4) at one layer is:

    z_i = tanh(H sum_{j in neighbors(i)}(x_i - x_j))
    v_i = K_gnn,i z_i

where H is trainable and K_gnn,i is trainable. The graph operator inside the paper network is **L_un = D - A**, not the L_rw = I - D^-1 A used for your controller and the common disagreement metric.

**Neither H nor K_gnn,i is projected, clipped, or otherwise norm-constrained.** The GNN is genuinely trained, not given handpicked weights.

The local homogenization u_local,i = K_local,i x_i is applied to both methods for these experiments; the paper GNN supplies the extra v_i. This is a *paper-architecture baseline adapted to our homogenized plant*, not a literal reproduction of the original paper's unmodified plant experiment. The reference local-only controller has v_i=0.

With N=3 fixed, Experiment 01 learns distinct K_gnn,i. Experiment 04 (N=10) also learns distinct K_gnn,i. In Experiments 02 and 03, **K_gnn,i is tied/shared across agents**, because otherwise a three-agent-trained output layer has no defined weights for agent 4 through agent 100. This is an explicit restricted scalable case of the paper architecture. Do not present it as the paper's general per-agent output layer transferred unchanged to a new N.

### Fairness and limitations

- Both train on identical A_i, B_i, graph, target A_target, and local K_local, with the **same seeded train and validation initial conditions**. They use Adam, same learning rate, same epochs, same horizon, and the existing objective `J_E + lambda J_U`, where J_U is the total *physical* command energy.
- Initialization of model weights differs because the architectures differ. Training dynamics and model capacity differ; the paper GNN has more trainable parameters.
- Default `--lam 0` retains your previous four-experiment study definition. It is **not a direct reproduction of the paper's Q=I,R=I training objective**. To study energy-regularized optimization, set the same positive `--lam` on both models.
- The small-gain W is constrained against the **training** plant gain in these four original experiments. This is distinct from the additional paper-transfer demonstration that used a **predeclared maximum deployment gain**. Accordingly, if the test graph/system has a larger gamma than training, the training-time guarantee may no longer apply. Check actual test gamma x norm(W); never assume it remains <= 0.95.
- Same test x0 and test physical matrices for all methods in each condition; no deployment retraining/rescaling.
- A paper GNN's bounded tanh is not necessarily sufficient to guarantee consensus. Nonzero equilibria and oscillation can occur. Observed outcomes should be reported, not pre-selected.
- The original user-requested tests with A=0.9 I_2 and 8x B_i scaling are a **separate specialized comparison**, and their trained weights/checkpoints should not be transplanted into these four different plant families.

## Install

Extract `four_experiments_trained_paper_baseline_patch.zip` from Downloads **over your existing project**. The archive contains only changed experiment files, not `src/`.

    cd ~/Downloads/distributed_consensus
    unzip -o ~/Downloads/four_experiments_trained_paper_baseline_patch.zip -d .

Check:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra test python -m pytest -q tests/test_paper_baseline_integration.py tests/test_report_experiments.py

Run a quick smoke test in separate paths:

    uv run python experiments/activation_comparison.py --quick --out results/threeway_smoke_01
    uv run python experiments/network_size_generalization.py --quick --out results/threeway_smoke_02
    uv run python experiments/plant_gain_transfer.py --quick --out results/threeway_smoke_03
    uv run python experiments/topology_generalization.py --quick --out results/threeway_smoke_04

Then run **all full experiments**, archiving only the new four folders:

    bash run_threeway_reports.sh

The script fails rather than overwrites existing paths. It uses five repeats, 32 held-out initial states, 600 epochs, T_train=100, T_test=200, batch=128 and val_batch=128, eta=0.95, gamma_train=0.75, threshold=0.001. All generated images are PNG only.

The full study is CPU-heavy. **Smoke tests and source-level checks do not mean the full 600-epoch experiments were executed.** The script must finish to obtain the final results.
