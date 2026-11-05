# Out-of-envelope small-gain violation: frozen-checkpoint diagnostic

This is an **additional test**, NOT retraining. It uses the exact three trained controller checkpoints from the earlier Wang–Lin-style paper comparison. Same physical N=3 chain setup, with common `A_i=0.9 I_2`. Both GNN and our W are frozen; no projection, fine-tuning, or hand-selected destabilizing weights occur during testing.

## What changed?

The original heterogeneous single-input matrices are:

```
B1 = [0.20, 0.10]^T
B2 = [0.10, 0.10]^T
B3 = [0.24, 0.12]^T
```

All **three** input matrices are multiplied by the same scalar `s`, using the predefined sweep:

`s = 1, 8, 12, 16, 32, 64, 100, 128, 160, 200, 220, 240, 256`.

The graph, `A_i`, saved `W`, saved paper `H,K_i`, and each seed's held-out initial conditions are held fixed across scales. Thus a change in `s` is the physical intervention.

## Plant gain method

The original `src/distributed_consensus/control/plant_gain.py` evaluates the complete discrete-time **H-infinity disagreement plant gain** using a factorization specifically valid for `B_i=[0,b_i]^T` and a general two-state Schur target, using an exact (up to floating-point) frequency-peak candidate search.

Here the paper-adapted matrices have **two nonzero components** but one input per agent, so that original factorization **cannot be applied directly**. Since this test has `A_i=A_target=alpha*I_2` with `alpha=0.9`, the full transfer function simplifies to:

`G(z) = (z-alpha)^(-1) (L_rw kron I_2) B_blk`.

Hence the full plant gain, without frequency gridding, is:

`gamma = || (L_rw kron I_2) B_blk ||_2 / (1-alpha)`.

It is the **same mathematical H-infinity gain**, computed via a special-case formula. An independent 129-frequency full-resolvent check matched the formula to numerical precision; for the original project's special `[0,b_i]` setting, the original `full_gain` function agreed to numerical precision.

The paper GNN itself uses the *unnormalized* Laplacian, while the small-gain controller uses `L_rw`.

Because every `B_i` is multiplied by `s`, `gamma(s)=s*gamma(1)`; `gamma(1)≈3.8703638`.

## Main answer: does loss of sufficient condition cause failure?

No—not necessarily. All values below come from all **three saved model checkpoints**, each evaluated on **eight paper-range held-out initial conditions** for **250 timesteps** (24 cases total per scale):

| B multiplier | Measured gamma*||W|| | Ours zero-equilibrium spectral radius (median) | Ours settled | Paper GNN settled |
|---:|---:|---:|---:|---:|
| 1× | 0.0896 | 0.9000 | 24/24 | 24/24 |
| 8× | 0.7167 | 0.9000 | 24/24 | 0/24 |
| 12× | **1.0751** | **0.9000** | **24/24** | 0/24 |
| 64× | **5.7339** | **0.9000** | **24/24** | 0/24 |
| 128× | **11.4678** | **0.9000** | **24/24** | 0/24 |
| 200× | **17.9184** | **0.9000** | **24/24** | 0/24 |
| 220× | **19.7103** | **1.0448** | **0/24** | 0/24 |
| 240× | **21.5021** | **1.2216** | **0/24** | 0/24 |

The small-gain product crosses 1 at about **11.16×**, but linear closed-loop instability begins only around **215×** (varies slightly by trained seed). There is therefore a large gap between the sufficient bound and the actual stability limit.

**Important:** `rho<1` is an *exact finite-dimensional linear closed-loop stability criterion* for our identity activation; unlike the sufficient small-gain inequality, it is necessary and sufficient for asymptotic stability. We compute it using the actual frozen W and each changed physical B.

The paper GNN uses tanh; `rho>1` refers to local zero-equilibrium Jacobian instability. Its bounded output and stable `A=0.9I` prevent unbounded state growth, even when it produces persistent disagreement.

These extremely large input multipliers are diagnostic stress cases, not a claim about realistic actuator uncertainty or a rigorous percentage robustness margin for all different plant perturbations.

## Package contents

- `experiments/small_gain_violation_diagnostic.py`: runnable no-training test script.
- `pretrained_models/models_rep00.npz` through `models_rep02.npz`: the **actual saved trained** weights.
- `results/sg_violation_diagnostic/`: saved summary and per-trajectory CSVs, PNGs, configuration JSON.

## Run in existing repository

Extract the code add-on at the project root, then:

```bash
uv run python experiments/small_gain_violation_diagnostic.py \
    --models pretrained_models \
    --scales 1 8 12 16 32 64 100 128 160 200 220 240 256 \
    --horizon 250 --initials 8 \
    --out results/sg_violation_diagnostic
```

The `--out` directory must be absent or empty. If extracting the precomputed results ZIP first, choose a **different** output directory to rerun.

All tests are CPU-ready. The script uses only NumPy and Matplotlib and is independent of the old training experiment implementation. It does not update `src/`.
