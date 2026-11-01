# `learning/` — training the shared controller

## `training.py`

This is the only optimizer module. It trains the shared `W` while keeping `A_i`, `B_i`, `K_i`, the target and the **training graph** fixed.

For a horizon of `T` physical actions, the objective averaged over sampled initial conditions is

$$
\mathcal L(W)=\mathbb E\left[J_E+\lambda J_U\right],
$$

where

$$
J_E=\sum_{k=0}^{T-1}\|E(k)\|_F^2,\qquad
J_U=\sum_{k=0}^{T-1}\|U_{total}(k)\|_F^2.
$$

Here `E = L_rw X`, and `U_total` is the total **commanded** physical input (`K_i x_i + v_i`). The objective has no added loss terms, terminal cost or penalty that is hidden from this formula. `lambda >= 0` controls the energy/disagreement tradeoff.

`train()` performs the following:

1. Computes the training plant gain and the numerical cap `cap = eta / gamma_plant,train`.
2. Initializes a shared `(2,1)` `W` at zero, with no bias.
3. Draws training batches of initial states independently of a fixed validation batch.
4. Rolls out the physical `TorchWorld`, backpropagates `J_E + lambda J_U`, and updates **only W** with Adam.
5. Applies `project_spectral_(W,cap)` **after every optimizer step**.
6. Evaluates validation loss at scheduled epochs, retaining the lowest-validation-loss `best_W`. `min_delta` affects early-stopping patience only.

`baseline_scales()` estimates local-only (`W=None`) costs for selecting or reporting lambda scales. `set_reproducible()` seeds NumPy and PyTorch; this does not guarantee bit-for-bit deterministic execution on all CUDA systems.

### Returned results and history

`train()` returns `W` (last epoch), `best_W` (best validation), their metadata, cap and training history. Experiment scripts save `best_W` as `checkpoint.npz` and the last `W` as `final_checkpoint.npz`.

`train_J_E`, `train_J_U`, and `train_objective` are measured **before** the epoch's optimizer update. `pre_update_W_spectral_norm` uses those same weights. `W_spectral_norm` and `constraint_margin` are measured **after** the update and projection. Validation loss is computed after the update.

The trained checkpoint applies a fixed `W` during inference; the inference code does not retrain or automatically rescale it for a new graph. The numerical small-gain cap was computed for the **training** plant and graph.
