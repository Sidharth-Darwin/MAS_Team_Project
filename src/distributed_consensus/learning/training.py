"""Fixed-plant, differentiable projected-Adam optimization of shared W only."""

import numpy as np
import torch
from distributed_consensus.control.controllers import project_spectral_
from distributed_consensus.control.dynamics import TorchWorld
from distributed_consensus.control.plant_gain import full_gain


def set_reproducible(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def baseline_scales(system, graph, horizon=70, samples=128, seed=291, device="cpu"):
    """Estimate train-system relative cost scales before choosing positive lambdas."""
    world = TorchWorld.from_system(system, graph, device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch = torch.randn((samples, system.n, 2), dtype=torch.float64, generator=gen).to(
        device
    )
    with torch.no_grad():
        je, ju, _ = world.costs(batch, None, horizon)
    return float(je.item()), float(ju.item())


def train(
    system,
    graph,
    lam=0.0,
    activation="identity",
    epochs=200,
    horizon=80,
    batch=128,
    val_batch=128,
    lr=0.002,
    eta=0.95,
    seed=7,
    device="cpu",
    val_every=10,
    early_stopping_patience=None,
    min_delta=0.0,
    validation_initial_states=None,
):
    """Learn W:(2,1) only, with J_E+lambda*J_U, projected after EACH step.

    All matrices, graph and K_i fixed across iterations. Train and validation RNG
    streams are independent and seed-separated; repeated runs with same seed/lambda
    use identical initial weights and training samples.

    History convention: train_J_E, train_J_U and train_objective are computed
    BEFORE the optimizer step, on the sampled training batch. The
    pre_update_W_spectral_norm and pre_update_constraint_margin correspond to
    those same weights. W_spectral_norm and constraint_margin
    are POST-update, POST-projection diagnostics.
    Validation losses are evaluated after the update on fixed validation data.

    min_delta affects early-stopping patience only. best_W is always selected
    by the lowest finite validation objective among evaluated epochs.
    """
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("lambda must be >=0 and finite")
    if activation not in ("identity", "tanh"):
        raise ValueError("activation")
    if (
        epochs < 1
        or horizon < 1
        or batch < 1
        or val_batch < 1
        or val_every < 1
        or lr <= 0
    ):
        raise ValueError("invalid training settings")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("patience is measured in validation checks and must be >=1")
    if min_delta < 0 or not np.isfinite(min_delta):
        raise ValueError("min_delta must be >=0")
    gamma = full_gain(system, graph).gamma
    cap = eta / gamma
    set_reproducible(seed)
    world = TorchWorld.from_system(system, graph, device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10001)
    valgen = torch.Generator(device="cpu").manual_seed(seed + 20003)
    if validation_initial_states is None:
        valX = torch.randn(
            (val_batch, system.n, 2), dtype=torch.float64, generator=valgen
        ).to(device)
    else:
        valX = torch.as_tensor(
            np.asarray(validation_initial_states, np.float64).copy(),
            dtype=torch.float64,
            device=device,
        )
        if valX.ndim != 3 or valX.shape[1:] != (system.n, 2) or valX.shape[0] < 1:
            raise ValueError("validation_initial_states shape must be (M,N,2)")
    if not bool(torch.isfinite(valX).all().item()):
        raise ValueError("validation initial states must be finite")
    W = torch.nn.Parameter(torch.zeros((2, 1), dtype=torch.float64, device=device))
    opt = torch.optim.Adam([W], lr=lr)
    history = []
    best_W = None
    best_epoch = None
    best_validation = float("inf")
    stale_checks = 0
    patience_reference = float("inf")
    for epoch in range(1, epochs + 1):
        initial = torch.randn(
            (batch, system.n, 2), dtype=torch.float64, generator=generator
        ).to(device)
        opt.zero_grad(set_to_none=True)
        pre_norm = float(torch.linalg.matrix_norm(W.detach(), ord=2).item())
        je, ju, _ = world.costs(initial, W, horizon, activation)
        loss = je + lam * ju
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite objective during training")
        loss.backward()
        opt.step()
        norm = project_spectral_(W, cap)
        if norm > cap + 1e-10:
            raise AssertionError("spectral projection failed")
        record = {
            "epoch": epoch,
            "train_J_E": float(je.item()),
            "train_J_U": float(ju.item()),
            "train_objective": float(loss.item()),
            "pre_update_W_spectral_norm": pre_norm,
            "pre_update_constraint_margin": float(cap - pre_norm),
            "W_spectral_norm": norm,
            "constraint_margin": float(cap - norm),
            "cap": cap,
            "val_J_E": None,
            "val_J_U": None,
            "val_objective": None,
        }
        if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
            with torch.no_grad():
                vje, vju, _ = world.costs(valX, W, horizon, activation)
            val_objective = float((vje + lam * vju).item())
            if not (
                bool(torch.isfinite(vje).item())
                and bool(torch.isfinite(vju).item())
                and np.isfinite(val_objective)
            ):
                raise ArithmeticError(f"nonfinite validation costs at epoch {epoch}")
            record.update(
                val_J_E=float(vje.item()),
                val_J_U=float(vju.item()),
                val_objective=val_objective,
            )

            # Checkpoint selection: always retain the lowest validation loss.
            if val_objective < best_validation:
                best_validation = val_objective
                best_epoch = epoch
                best_W = W.detach().cpu().numpy().copy()

            if val_objective < patience_reference - min_delta:
                patience_reference = val_objective
                stale_checks = 0
            else:
                stale_checks += 1
        history.append(record)
        if (
            early_stopping_patience is not None
            and stale_checks >= early_stopping_patience
        ):
            break
    return {
        "W": W.detach().cpu().numpy().copy(),
        "best_W": best_W,
        "best_epoch": best_epoch,
        "best_validation_objective": best_validation,
        "stopped_early": len(history) < epochs,
        "history": history,
        "cap": cap,
        "gamma_train": gamma,
        "lambda": float(lam),
        "activation": activation,
        "seed": seed,
        "train_seed": seed + 10001,
        "validation_seed": seed + 20003,
    }
