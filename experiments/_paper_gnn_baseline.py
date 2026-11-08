"""Trained one-layer Wang--Lin-style GNN baseline for the four comparison studies.

Original paper: u = K tanh((L_unweighted tensor H) X), K=blockdiag(K_i).
For a locally matched heterogeneous plant, this module learns the *correction*
v_i and applies u_i=K_local,i x_i + v_i for BOTH compared controllers.

The N-transfer experiment uses tied K_i to permit frozen deployment to unseen N;
fixed-N experiments use individual trainable K_i. Neither paper weight is
projected, clipped, or stabilized during training or testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    metrics_from_history,
    settling_time_absolute,
    aggregate_method,
    summarize_time_series,
    write_csv_any,
)


class PaperGNN(torch.nn.Module):
    def __init__(self, n: int, hidden: int = 4, shared_output: bool = False):
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden channels must be positive")
        self.shared_output = shared_output
        self.H = torch.nn.Parameter(torch.empty((hidden, 2), dtype=torch.float64))
        self.K = torch.nn.Parameter(
            torch.empty((1 if shared_output else n, hidden), dtype=torch.float64)
        )
        torch.nn.init.normal_(self.H, mean=0.0, std=0.035)
        torch.nn.init.normal_(self.K, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor, L_un: torch.Tensor) -> torch.Tensor:
        # X^(1) = tanh((L_un tensor H) X^(0)), v_i = K_i X_i^(1)
        hidden = torch.tanh((L_un @ x) @ self.H.T)
        if self.shared_output:
            return (hidden @ self.K[0]).unsqueeze(-1)
        if hidden.shape[1] != self.K.shape[0]:
            raise ValueError(
                "per-agent paper output K_i cannot be transferred to a different N; use shared K"
            )
        return torch.einsum("bnh,nh->bn", hidden, self.K).unsqueeze(-1)


def unnormalized_laplacian(graph) -> np.ndarray:
    # This is the paper's D-A, NOT our random-walk Laplacian I-D^-1 A.
    return (np.diag(graph.degrees) - graph.adjacency.toarray()).astype(np.float64)


def train_paper_gnn(system, graph, args, seed: int, shared_output: bool):
    """Train on the SAME train/validation initial-state streams as train_model().

    Uses the existing physical dynamics, total actuator-energy objective and
    fixed B/A/K_local; Adam parameters, epochs and validation checkpoints match.
    Paper H,K_i unconstrained: absolutely no projection, gain cap or clipping.
    """
    # Small dense graph models are much faster with one CPU thread.
    if args.device == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(seed + 794)
    model = PaperGNN(system.n, getattr(args, "paper_hidden", 4), shared_output).to(
        args.device
    )
    A = torch.as_tensor(np.array(system.A), dtype=torch.float64, device=args.device)
    B = torch.as_tensor(np.array(system.B), dtype=torch.float64, device=args.device)
    K_local = torch.as_tensor(
        np.array(system.K), dtype=torch.float64, device=args.device
    )
    L_rw = torch.as_tensor(
        graph.laplacian.toarray(), dtype=torch.float64, device=args.device
    )
    L_un = torch.as_tensor(
        unnormalized_laplacian(graph), dtype=torch.float64, device=args.device
    )

    def objective(X0):
        x = X0
        je = torch.zeros((), device=args.device, dtype=torch.float64)
        ju = torch.zeros_like(je)
        for _ in range(args.train_horizon):
            e = L_rw @ x
            v = model(x, L_un)
            local = torch.einsum("nij,bnj->bni", K_local, x)
            u = local + v
            je = je + e.square().sum((1, 2)).mean()
            ju = ju + u.square().sum((1, 2)).mean()
            x = torch.einsum("nij,bnj->bni", A, x) + torch.einsum("nij,bnj->bni", B, u)
        return je + args.lam * ju, je, ju

    generator = torch.Generator(device="cpu").manual_seed(seed + 10001)
    val_gen = torch.Generator(device="cpu").manual_seed(seed + 20003)
    valid = torch.randn(
        (args.val_batch, system.n, 2), generator=val_gen, dtype=torch.float64
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best = None
    best_epoch = 0
    best_val = float("inf")
    stale_checks = 0
    patience_ref = float("inf")
    for epoch in range(1, args.epochs + 1):
        initial = torch.randn(
            (args.batch, system.n, 2), generator=generator, dtype=torch.float64
        ).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss, je, ju = objective(initial)
        if not bool(torch.isfinite(loss).item()):
            raise ArithmeticError(f"paper GNN nonfinite training loss, epoch={epoch}")
        loss.backward()
        optimizer.step()  # intentionally no projection or clipping
        val_loss = None
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                vl, _, _ = objective(valid)
            val_loss = float(vl.item())
            if not np.isfinite(val_loss):
                raise ArithmeticError(
                    f"paper GNN nonfinite validation loss, epoch={epoch}"
                )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if val_loss < patience_ref - args.min_delta:
                patience_ref = val_loss
                stale_checks = 0
            else:
                stale_checks += 1
        history.append(
            {
                "epoch": epoch,
                "train_J_E": float(je.detach()),
                "train_J_U": float(ju.detach()),
                "train_objective": float(loss.detach()),
                "val_objective": val_loss,
            }
        )
        if (
            args.early_stopping_patience is not None
            and stale_checks >= args.early_stopping_patience
        ):
            break
    model.load_state_dict(best)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_objective": best_val,
        "history": history,
        "paper_shared_output": int(shared_output),
    }


def save_paper_checkpoint(path, model, system, graph, train_info):
    """Numeric-only checkpoint including trained H,K and the training plant/graph."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paper_H=model.H.detach().cpu().numpy(),
        paper_K=model.K.detach().cpu().numpy(),
        shared_output=np.array(int(model.shared_output)),
        A=system.A,
        B=system.B,
        K_local=system.K,
        A_target=system.target,
        adjacency=graph.adjacency.toarray(),
        best_epoch=np.array(train_info["best_epoch"]),
    )


def rollout_paper(x0, system, graph, model, horizon):
    """Actual physical total u; frozen paper weights, no projection or clipping."""
    x = np.asarray(x0, dtype=np.float64).copy()
    L_un = unnormalized_laplacian(graph)
    H = model.H.detach().cpu().numpy()
    K = model.K.detach().cpu().numpy()
    xx = [x.copy()]
    ee = [graph.error(x)]
    local_list = []
    correction_list = []
    total_list = []
    for k in range(horizon):
        hidden = np.tanh((L_un @ x) @ H.T)
        v = (
            (hidden @ K[0]) if model.shared_output else np.einsum("nh,nh->n", hidden, K)
        )[:, None]
        local = np.einsum("nij,nj->ni", system.K, x)
        total = local + v
        xnext = np.einsum("nij,nj->ni", system.A, x) + np.einsum(
            "nij,nj->ni", system.B, total
        )
        if not np.isfinite(xnext).all():
            raise FloatingPointError(f"nonfinite paper-GNN physical rollout at k={k}")
        local_list.append(local)
        correction_list.append(v)
        total_list.append(total)
        x = xnext
        xx.append(x.copy())
        ee.append(graph.error(x))
    return {
        "states": np.stack(xx),
        "disagreement": np.stack(ee),
        "local": np.stack(local_list),
        "correction": np.stack(correction_list),
        "total": np.stack(total_list),
    }


def evaluate_paper(
    model,
    system,
    graph,
    initials,
    horizon,
    threshold,
    meta,
    summary_rows,
    trajectory_rows,
    curve_rows,
):
    """Append paper outcomes with same schema as other methods; return plotting data."""
    from distributed_consensus.evaluation.metrics import trajectory_metrics

    histories = []
    raw = None
    rows = []
    for sample, x0 in enumerate(initials):
        h = rollout_paper(x0, system, graph, model, horizon)
        if sample == 0:
            raw = h
        m = metrics_from_history(h)
        settle = settling_time_absolute(m["error_rms"], threshold)
        tm = trajectory_metrics(h)
        rows.append(
            {
                "sample": sample,
                "paper_gnn_settling_time": settle,
                "paper_gnn_settled": int(settle is not None),
                "paper_gnn_J_E": tm["J_E"],
                "paper_gnn_J_U": tm["J_U"],
                "paper_gnn_J_pair": tm["J_pair"],
                "paper_gnn_terminal_E": tm["terminal_E"],
                "paper_gnn_final_error_rms": float(m["error_rms"][-1]),
            }
        )
        histories.append(m)
    for r in rows:
        # Prefix the keys of the dedicated paper method to match aggregate_method.
        trajectory_rows.append({**meta, "method": "paper_gnn", **r})
    stats = aggregate_method(rows, "paper_gnn")
    summary_rows.append(
        {
            **meta,
            "method": "paper_gnn",
            **stats,
            "paper_hidden_channels": int(model.H.shape[0]),
            "paper_shared_output": int(model.shared_output),
        }
    )
    for row in summarize_time_series(
        histories, ("error_rms", "centered_rms", "state_rms", "control_rms")
    ):
        curve_rows.append({**meta, "method": "paper_gnn", **row})
    return histories, raw, stats


def save_training_history(path, history):
    write_csv_any(path, history)
