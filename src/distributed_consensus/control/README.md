# `control/` — physical model, controller and plant gain

The files here implement the agreed **two-state, single-input, heterogeneous** physical agents, exact local matching, direct Laplacian correction and the numerical small-gain test. No physical `B_i` is replaced by a common matrix.

## `plants.py` — physical agents and local matching

`System` stores `A`, `B`, `K` and `target`. Shapes are `A: (N,2,2)`, `B: (N,2,1)`, `K: (N,1,2)` and `target: (2,2)`.

For each agent:

$$
x_i(k+1)=A_i x_i(k)+B_i u_i(k),\qquad B_i=\begin{bmatrix}0\\b_i\end{bmatrix},\quad b_i\ne0.
$$

`build_system(A,B,target)` computes the **true Moore–Penrose pseudoinverse** and local gain:

$$
K_i=B_i^\dagger(A_{target}-A_i).
$$

It checks exact-matching compatibility (within numerical tolerance), so the accepted systems satisfy

$$
A_i+B_iK_i=A_{target}.
$$

`check_target()` requires a Schur-stable target (`rho(A_target) < 1`). `generate_system()` samples heterogeneous, **compatible** systems; it is not a generator of arbitrary incompatible agents. `matching_residual()` reports numerical matching error.

## `controllers.py` — one shared correction

`correction_numpy(E,W,activation)` computes

$$
V=\phi(EW),\qquad E\in\mathbb{R}^{N\times2},\quad W\in\mathbb{R}^{2\times1},\quad V\in\mathbb{R}^{N\times1}.
$$

`phi` is either `identity` (primary baseline) or `tanh`. There is **one shared W**, no bias, no hidden layer, and no extra graph filtering. `project_spectral_()` projects `W` onto `||W||_2 <= cap` after each optimizer step; `weight_norm()` reports the spectral norm.

## `dynamics.py` — real physical updates

`step_numpy()` implements one time step; `rollout_numpy()` repeats it. Each agent receives

$$
u_i(k)=K_ix_i(k)+v_i(k),\quad v_i(k)=\phi\big((L_{rw}X(k))_iW\big).
$$

The **actual** next state is computed using the original `A_i` and `B_i`, not by substituting the target matrix directly. `W=None` runs the local-only baseline. Optional NumPy disturbances enter at the actuator channel, while returned `total` means **commanded** control, excluding the disturbance.

`TorchWorld` implements differentiable batched versions of the same fixed-graph update for training. It does not implement actuator disturbances. `TorchWorld.costs()` accumulates costs for actions `k=0,...,T-1` and returns the terminal disagreement separately; there is no terminal stage-cost addition.

With exact matching, the corresponding full-network disagreement dynamics are

$$
E(k+1)=(I_N\otimes A_{target})E(k)+(L_{rw}\otimes I_2)\operatorname{blkdiag}(B_1,\ldots,B_N)V(k).
$$

## `plant_gain.py` — numerical small-gain analysis

`scalar_gain(target)` calculates `gamma0` for the unit-input target transfer. `spatial_norm(graph,b)` calculates the spectral norm of `L_rw diag(b_1,...,b_N)`. `full_gain()` uses the factorization

$$
\gamma_{plant,N}=\gamma_0\left\|L_{rw}\operatorname{diag}(b_1,\ldots,b_N)\right\|_2.
$$

`small_gain()` evaluates the product `gamma_plant,N * ||W||_2`; the training cap is `||W||_2 <= eta / gamma_plant,train`. `direct_grid_gain()` independently checks small systems on a finite frequency grid; it is a diagnostic, not a proof.

**Limit:** the gain and sparse spectral computations are floating-point estimates, **not formally certified upper bounds**. Product below 1 supports the numerical sufficient-condition check; product at or above 1 means the sufficient condition is inconclusive, **not** that the network is unstable. A cap computed for the training graph does not automatically certify all other graph sizes or topologies.
