"""Out-of-envelope transfer test with frozen trained comparison checkpoints.

Same Wang--Lin-adapted A=0.9 I_2, N=3 chain, heterogeneous one-input B_i
as in compare_paper_gnn.py. Scale ALL B_i by s, without any retraining,
projection, clipping or test-time rescaling of W/H/K.

Run from existing project root after copying this file and the model files:
  uv run python experiments/small_gain_violation_diagnostic.py \
      --models pretrained_models --out results/sg_violation_diagnostic

Note: this is an empirical diagnostic under gross plant changes, not an
additional stability guarantee. H-infinity computed exactly via special-case
factorization for A=alpha I; floating-point spectral norms.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

A = 0.9
B0 = np.array([[0.2, 0.1], [0.1, 0.1], [0.24, 0.12]], dtype=np.float64)
L_UN = np.array(
    [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]], dtype=np.float64
)
L_RW = np.array(
    [[1.0, -1.0, 0.0], [-0.5, 1.0, -0.5], [0.0, -1.0, 1.0]], dtype=np.float64
)
SCALES = (1, 8, 12, 16, 32, 64, 100, 128, 160, 200, 220, 240, 256)
SAVED_PLOTS = (8, 12, 64, 128, 240)


def gain_for(B):
    n = len(B)
    blk = np.zeros((2 * n, n))
    for i in range(n):
        blk[2 * i : 2 * i + 2, i] = B[i]
    gain = np.linalg.norm(np.kron(L_RW, np.eye(2)) @ blk, 2) / (1.0 - A)
    return float(gain)


def linear_closed_matrix(B, W):
    n = len(B)
    M = A * np.eye(2 * n)
    w = W.reshape(-1)
    for i in range(n):
        for j in range(n):
            M[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += np.outer(B[i], w) * L_RW[i, j]
    return M


def paper_jacobian(B, H, K):
    n = len(B)
    M = A * np.eye(2 * n)
    F = K @ H
    for i in range(n):
        for j in range(n):
            M[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += np.outer(B[i], F[i]) * L_UN[i, j]
    return M


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def make_initials(seed, count, profile):
    gen = np.random.default_rng(
        seed + 500000 + 70001 * 3 + (1000000 if profile == "near_zero" else 0)
    )
    if profile == "paper_range":
        return gen.uniform(-10.0, 10.0, size=(count, 3, 2)) + np.array([50.0, 0.0])
    if profile == "near_zero":
        return gen.normal(size=(count, 3, 2)) * 0.01
    raise ValueError(profile)


def simulate(B, x0, W, H, K, horizon):
    """Same x0 for paper, our frozen W, and open-loop baseline."""
    batch = len(x0)
    histories = {}
    for method in ("paper_gnn", "small_gain", "local_only"):
        x = x0.copy()
        states = [x.copy()]
        controls = []
        for _ in range(horizon):
            if method == "paper_gnn":
                z = np.tanh((L_UN @ x) @ H.T)
                u = np.einsum("bnh,nh->bn", z, K)
            elif method == "small_gain":
                u = ((L_RW @ x) @ W).squeeze(-1)
            else:
                u = np.zeros((batch, 3))
            controls.append(u.copy())
            x = A * x + B[None, :, :] * u[:, :, None]
            states.append(x.copy())
        states = np.stack(states)
        controls = np.stack(controls)
        disagreements = L_RW @ states
        rms = np.linalg.norm(disagreements, axis=(2, 3)) / np.sqrt(3)
        histories[method] = {
            "states": states,
            "controls": controls,
            "disagreement_rms": rms,
            "per_agent_disagreement": np.linalg.norm(disagreements, axis=3),
        }
    return histories


def settling_steps(rms, threshold):
    out = []
    for curve in rms.T:
        bad = np.flatnonzero(curve > threshold)
        if len(bad) and bad[-1] == len(curve) - 1:
            out.append(None)
        else:
            out.append(int(bad[-1] + 1) if len(bad) else 0)
    return out


# The same colors and line patterns are used across every saved figure.
METHOD_COLORS = {
    "paper_gnn": "#CB662A",
    "small_gain": "#176EA3",
    "local_only": "#677486",
}
METHOD_LABELS = {
    "paper_gnn": "Trained paper GNN (unconstrained)",
    "small_gain": "Our frozen small-gain controller",
    "local_only": "Local-only / no learned correction",
}
AGENT_STYLES = ("-", "--", ":")
PLOT_DPI = 175


def _finish_axis(ax, ylabel, *, log=False, symlog=None, zero=False):
    ax.set_xlabel("Discrete-time step $k$")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if symlog is not None:
        ax.set_yscale("symlog", linthresh=symlog)
    if zero:
        ax.axhline(0, color="#333333", lw=0.8, alpha=0.45, zorder=0)
    ax.grid(True, which="major", alpha=0.20)
    if log or symlog is not None:
        ax.grid(True, which="minor", alpha=0.07)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def _common_legend(fig, *, include_agents=True):
    methods = [
        Line2D([0], [0], color=METHOD_COLORS[m], lw=2.5, label=METHOD_LABELS[m])
        for m in ("paper_gnn", "small_gain", "local_only")
    ]
    if include_agents:
        agents = [
            Line2D(
                [0],
                [0],
                color="#303030",
                ls=AGENT_STYLES[i],
                lw=1.8,
                label=f"Agent {i + 1} (line style)",
            )
            for i in range(3)
        ]
        handles = methods + agents
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    else:
        fig.legend(
            handles=methods,
            loc="lower center",
            ncol=3,
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )


def graph_plot(path, histories, scale, seed, gamma, loop, rsg, rp, threshold, sample=0):
    """Six fully labelled panels using the same held-out trajectory for all methods.

    Method = color; agent identity = line style. All nine agent/method curves
    are drawn on agent-level panels. The two error panels use actual norms;
    the 1e-14 floor exists only so zero/underflow can be displayed on log axes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=False)
    fig.subplots_adjust(
        left=0.073, right=0.985, bottom=0.115, top=0.83, wspace=0.27, hspace=0.39
    )

    # Force a consistent draw order across runs/checkpoints.
    for method in ("local_only", "small_gain", "paper_gnn"):
        h = histories[method]
        c = METHOD_COLORS[method]
        x = h["states"][:, sample, :, :]  # T+1, N, 2
        u = h["controls"][:, sample, :]  # T, N
        rms = h["disagreement_rms"][:, sample]  # T+1
        errors = h["per_agent_disagreement"][:, sample, :]  # T+1, N
        t_state = np.arange(len(x))
        t_action = np.arange(len(u))
        label = METHOD_LABELS[method]

        axes[0, 0].plot(t_state, np.maximum(rms, 1e-14), color=c, lw=2.1, label=label)
        axes[1, 1].plot(
            np.arange(len(u) + 1),
            np.r_[0, np.cumsum(np.sum(u * u, axis=1) / 3)],
            color=c,
            lw=2.1,
            label=label,
        )

        for i in range(3):
            style = AGENT_STYLES[i]
            kw = dict(color=c, linestyle=style, lw=1.25, alpha=0.87)
            axes[0, 1].plot(t_state, x[:, i, 0], **kw)
            axes[0, 2].plot(t_state, x[:, i, 1], **kw)
            axes[1, 0].plot(t_action, u[:, i], **kw)
            axes[1, 2].plot(t_state, np.maximum(errors[:, i], 1e-14), **kw)

    axes[0, 0].axhline(threshold, color="#171717", ls="--", lw=1.45)
    axes[1, 2].axhline(threshold, color="#171717", ls="--", lw=1.45)
    _finish_axis(axes[0, 0], "Disagreement RMS per agent", log=True)
    _finish_axis(axes[0, 1], "$x_{i,1}(k)$ (state component 1)", symlog=1.0, zero=True)
    _finish_axis(axes[0, 2], "$x_{i,2}(k)$ (state component 2)", symlog=1.0, zero=True)
    _finish_axis(axes[1, 0], "Total signed input $u_i(k)$", symlog=0.1, zero=True)
    _finish_axis(
        axes[1, 1],
        r"Cumulative energy per agent, $\frac{1}{3}\sum_{t=0}^{k-1}\sum_i u_i(t)^2$",
        symlog=0.1,
    )
    _finish_axis(axes[1, 2], r"Agent disagreement $\|e_i(k)\|_2$", log=True)

    axes[0, 0].set_title("(a) Network disagreement convergence", fontweight="semibold")
    axes[0, 1].set_title(
        "(b) Individual agent states: component 1", fontweight="semibold"
    )
    axes[0, 2].set_title(
        "(c) Individual agent states: component 2", fontweight="semibold"
    )
    axes[1, 0].set_title(
        "(d) Actual actuator inputs (not corrections only)", fontweight="semibold"
    )
    axes[1, 1].set_title("(e) Total physical control effort", fontweight="semibold")
    axes[1, 2].set_title("(f) Individual agent disagreement", fontweight="semibold")

    fig.suptitle(
        f"Frozen-controller transfer | Chain graph, $N=3$ | All $B_i$ multiplied by {scale:g}×",
        x=0.5,
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.942,
        f"Checkpoint seed {seed}  |  held-out trajectory #{sample + 1}  |  "
        rf"plant gain $\gamma={gamma:.3f}$  |  "
        rf"small-gain product $\gamma\|W\|_2={loop:.3f}$",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.910,
        rf"Our linear closed-loop $\rho={rsg:.4f}$  |  "
        rf"Paper GNN Jacobian at zero $\rho={rp:.4f}$  |  "
        rf"Error threshold $\varepsilon={threshold:g}$ (black dashed line)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.871,
        "Method = color; agent = line style. States and inputs use symmetric-log axes; "
        "disagreement uses log axes.",
        ha="center",
        fontsize=9.5,
        color="#4f5561",
    )
    _common_legend(fig, include_agents=True)
    fig.savefig(path, dpi=PLOT_DPI, facecolor="white")
    plt.close(fig)


def agent_detail_plot(path, histories, scale, seed, threshold, sample=0):
    """Additional 3×3 view: each controller gets its own row to avoid overplotting.

    Columns: x1(k), x2(k), physical input u(k). Each individual agent appears
    as a separately labelled trace, with consistent line styles.
    """
    methods = ("paper_gnn", "small_gain", "local_only")
    fig, axs = plt.subplots(3, 3, figsize=(17, 12), sharex="col")
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.88, bottom=0.105, wspace=0.20, hspace=0.30
    )
    titles = ("State component 1", "State component 2", "Physical control input")
    for col in range(3):
        axs[0, col].set_title(titles[col], fontsize=12.5, fontweight="semibold", pad=12)
    for row, method in enumerate(methods):
        h = histories[method]
        x = h["states"][:, sample]
        u = h["controls"][:, sample]
        c = METHOD_COLORS[method]
        for col in range(3):
            ax = axs[row, col]
            arr = x[:, :, col] if col < 2 else u
            t = np.arange(len(arr))
            for i in range(3):
                ax.plot(
                    t,
                    arr[:, i],
                    color=c,
                    ls=AGENT_STYLES[i],
                    lw=1.7,
                    alpha=0.94,
                    label=f"Agent {i + 1}",
                )
            _finish_axis(
                ax,
                (
                    "$x_{i,1}(k)$"
                    if col == 0
                    else "$x_{i,2}(k)$"
                    if col == 1
                    else "Total $u_i(k)$"
                ),
                symlog=(1 if col < 2 else 0.1),
                zero=True,
            )
            if col == 0:
                ax.text(
                    -0.17,
                    0.5,
                    METHOD_LABELS[method],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11.2,
                    color=c,
                    fontweight="semibold",
                )
            if col == 2:
                ax.legend(loc="best", fontsize=8.5, frameon=True)
    fig.suptitle(
        f"Individual agent states and physical inputs | $B_i$ scale {scale:g}× | seed {seed}",
        y=0.977,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "Every panel shows agents 1–3 separately; columns share physical quantities "
        "and rows separate controllers. Symmetric-log vertical axes.",
        ha="center",
        fontsize=10,
        color="#505866",
    )
    fig.savefig(path, dpi=PLOT_DPI, facecolor="white")
    plt.close(fig)


def _median_range_by_scale(table, method, key, scales):
    med, lo, hi = [], [], []
    for scale in scales:
        vals = np.array(
            [
                float(r[key])
                for r in table
                if float(r["scale"]) == float(scale) and r["method"] == method
            ],
            dtype=float,
        )
        if len(vals) == 0:
            raise ValueError(f"missing metric {key} at scale {scale} for {method}")
        med.append(float(np.median(vals)))
        lo.append(float(vals.min()))
        hi.append(float(vals.max()))
    return np.array(med), np.array(lo), np.array(hi)


def plot_stability(path, table):
    """Zero-equilibrium Jacobian spectral radius, median + min/max over seeds."""
    scales = np.array(sorted(set(float(r["scale"]) for r in table)))
    fig, ax = plt.subplots(figsize=(11.6, 6.3))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.21)
    for method in ("paper_gnn", "small_gain"):
        median, lo, hi = _median_range_by_scale(
            table, method, "rho_closed_loop", scales
        )
        c = METHOD_COLORS[method]
        ax.fill_between(scales, lo, hi, color=c, alpha=0.13)
        ax.plot(
            scales,
            median,
            marker="o",
            markersize=4.8,
            lw=2.2,
            color=c,
            label=f"{METHOD_LABELS[method]}: seed median",
        )
    ax.axhline(
        1,
        ls="--",
        color="#171717",
        linewidth=1.3,
        label=r"Local stability boundary $\rho(J)=1$",
    )
    ax.axhline(
        A,
        ls=":",
        color=METHOD_COLORS["local_only"],
        linewidth=1.5,
        label=rf"Local-only reference $\rho={A:.1f}$",
    )
    ax.set_xlabel("Common multiplier $s$ applied to all three $B_i$")
    ax.set_ylabel(r"Zero-equilibrium Jacobian spectral radius $\rho(J)$")
    ax.set_title(
        "Local stability diagnostic across actuator scaling", fontweight="semibold"
    )
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    fig.suptitle(
        "Frozen trained controllers | Three-agent chain graph",
        fontsize=14.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.907,
        "Lines show median over independently trained seeds; transparent bands show "
        "min–max across seeds.",
        ha="center",
        fontsize=10,
        color="#4f5561",
    )
    fig.text(
        0.5,
        0.08,
        r"For the nonlinear paper GNN, $\rho(J)>1$ means the zero equilibrium is locally unstable; "
        "it does not by itself establish unbounded trajectories.",
        ha="center",
        fontsize=9.4,
        color="#4f5561",
    )
    fig.savefig(path, dpi=PLOT_DPI, facecolor="white")
    plt.close(fig)


def plot_loop(path, table):
    """Plant H-infinity gain times frozen ||W||_2, median + range over seeds."""
    scales = np.array(sorted(set(float(r["scale"]) for r in table)))
    med, lo, hi = _median_range_by_scale(
        table, "small_gain", "small_gain_product", scales
    )
    fig, ax = plt.subplots(figsize=(11.6, 6.3))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.19)
    ax.fill_between(
        scales,
        lo,
        hi,
        color=METHOD_COLORS["small_gain"],
        alpha=0.14,
        label="Across-seed min–max",
    )
    ax.plot(
        scales,
        med,
        color=METHOD_COLORS["small_gain"],
        lw=2.4,
        marker="o",
        markersize=5.3,
        label=r"Median full-plant $\gamma(s)\|W\|_2$",
    )
    ax.axhline(
        1,
        color="#171717",
        ls="--",
        linewidth=1.4,
        label="Sufficient small-gain boundary = 1",
    )
    ax.axhline(
        0.95,
        color="#707781",
        ls=":",
        linewidth=1.4,
        label=r"Training design margin $\eta=0.95$",
    )
    ax.set(
        xlabel="Common multiplier $s$ applied to all three $B_i$",
        ylabel=r"Full disagreement-plant gain $\gamma(s)\|W\|_2$",
        title="Where the small-gain sufficient condition stops applying",
    )
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    fig.suptitle(
        "Frozen constrained controller | Three-agent chain graph",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.085,
        "Above 1: the sufficient condition provides no conclusion; "
        "instability is not implied. Controller weights were never changed.",
        ha="center",
        fontsize=9.5,
        color="#4f5561",
    )
    fig.savefig(path, dpi=PLOT_DPI, facecolor="white")
    plt.close(fig)


def plot_settlement(path, table, threshold):
    """Settling rate versus scaling, with trajectories and profile fixed in table."""
    scales = np.array(sorted(set(float(r["scale"]) for r in table)))
    fig, ax = plt.subplots(figsize=(11.6, 6.3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.19)
    for method in ("paper_gnn", "small_gain", "local_only"):
        vals = [r for r in table if r["method"] == method]
        rate = [
            np.mean([r["settled"] for r in vals if float(r["scale"]) == s])
            for s in scales
        ]
        ax.plot(
            scales,
            rate,
            "-o",
            markersize=5.2,
            lw=2.1,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set(
        xlabel="Common multiplier $s$ applied to all three $B_i$",
        ylabel="Fraction of held-out trajectories settled",
        ylim=(-0.04, 1.06),
        title=rf"Settling success rate (persistent RMS threshold $\varepsilon={threshold:g}$)",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="center right")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    fig.suptitle(
        "Frozen-controller agreement after transfer | Paper-range initial states",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.085,
        "A trajectory counts as settled only if disagreement remains below the threshold "
        "for every remaining observed timestep.",
        ha="center",
        fontsize=9.4,
        color="#4f5561",
    )
    fig.savefig(path, dpi=PLOT_DPI, facecolor="white")
    plt.close(fig)


def write_rows(path, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def run(args):
    if any(s <= 0 for s in args.scales):
        raise ValueError("scales must be positive")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty out: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir()
    models = sorted(Path(args.models).glob("models_rep*.npz"))
    if not models:
        raise FileNotFoundError("no models_repXX.npz checkpoints found")
    allrows = []
    for rep, path in enumerate(models):
        with np.load(path, allow_pickle=False) as z:
            W = z["small_W"].copy()
            H = z["paper_H"].copy()
            K = z["paper_K"].copy()
            b_train = z["B_train"].copy()
            assert np.allclose(b_train, B0)
            assert np.isclose(z["A_target"], A * np.eye(2)).all()
        if K.shape[0] != 3 or W.shape != (2, 1):
            raise ValueError(f"expected N3 per-agent checkpoint: {path}")
        wnorm = float(np.linalg.norm(W, ord=2))
        seed = args.seed + 10007 * rep
        for s in args.scales:
            B = B0 * s
            gamma = gain_for(B)
            loop = gamma * wnorm
            rsg = rho(linear_closed_matrix(B, W))
            rp = rho(paper_jacobian(B, H, K))
            representative = None
            for profile in args.profiles:
                initial = make_initials(seed, args.initials, profile)
                histories = simulate(B, initial, W, H, K, args.horizon)
                if profile == "paper_range":
                    representative = histories
                for method, h in histories.items():
                    settles = settling_steps(h["disagreement_rms"], args.threshold)
                    for i in range(args.initials):
                        e = h["disagreement_rms"][:, i]
                        u = h["controls"][:, i]
                        allrows.append(
                            {
                                "rep": rep,
                                "seed": seed,
                                "profile": profile,
                                "sample": i,
                                "N": 3,
                                "topology": "chain",
                                "scale": s,
                                "method": method,
                                "gamma_plant": gamma,
                                "W_norm": wnorm if method == "small_gain" else None,
                                "small_gain_product": loop
                                if method == "small_gain"
                                else None,
                                "small_gain_below_one": int(loop < 1)
                                if method == "small_gain"
                                else None,
                                "rho_closed_loop": rsg
                                if method == "small_gain"
                                else (rp if method == "paper_gnn" else A),
                                "settled": int(settles[i] is not None),
                                "settling_step": settles[i],
                                "final_disagreement_rms": float(e[-1]),
                                "late_disagreement_rms": float(
                                    np.mean(e[-min(50, len(e)) :])
                                ),
                                "max_disagreement_rms": float(e.max()),
                                "J_E_per_agent": float(np.sum(e[:-1] ** 2)),
                                "J_U_per_agent": float(np.sum(u**2) / 3),
                                "final_state_max": float(
                                    np.max(abs(h["states"][-1, i]))
                                ),
                            }
                        )
            if s in args.plot_scales:
                if representative is None:
                    raise ValueError(
                        "paper_range profile is required for detailed plots"
                    )
                graph_plot(
                    out / "figures" / f"chain_scale{s:g}_rep{rep:02d}.png",
                    representative,
                    s,
                    seed,
                    gamma,
                    loop,
                    rsg,
                    rp,
                    args.threshold,
                )
                agent_detail_plot(
                    out / "figures" / f"chain_scale{s:g}_rep{rep:02d}_agent_traces.png",
                    representative,
                    s,
                    seed,
                    args.threshold,
                )
            print(
                f"rep={rep}, scale={s:g}× gamma={gamma:.6g} loop={loop:.6g} rho_small_gain={rsg:.6g} rho_paper={rp:.6g}",
                flush=True,
            )
    write_rows(out / "trajectory_metrics.csv", allrows)
    agg = []
    for s in args.scales:
        for profile in args.profiles:
            for method in ("small_gain", "paper_gnn", "local_only"):
                rows = [
                    r
                    for r in allrows
                    if r["scale"] == s
                    and r["profile"] == profile
                    and r["method"] == method
                ]
                settled = [r["settling_step"] for r in rows if r["settled"]]
                agg.append(
                    {
                        "scale": s,
                        "profile": profile,
                        "method": method,
                        "samples": len(rows),
                        "settled_count": len(settled),
                        "settled_fraction": len(settled) / len(rows),
                        "median_settling_if_settled": float(np.median(settled))
                        if settled
                        else None,
                        "median_final_disagreement_rms": float(
                            np.median([r["final_disagreement_rms"] for r in rows])
                        ),
                        "median_late_disagreement_rms": float(
                            np.median([r["late_disagreement_rms"] for r in rows])
                        ),
                        "median_rho": float(
                            np.median([r["rho_closed_loop"] for r in rows])
                        ),
                        "median_loop_product": float(
                            np.median([r["small_gain_product"] for r in rows])
                        )
                        if method == "small_gain"
                        else None,
                    }
                )
    write_rows(out / "summary.csv", agg)
    plot_stability(
        out / "figures" / "spectral_radius_vs_scale.png",
        [r for r in allrows if r["profile"] == "paper_range" and r["sample"] == 0],
    )
    plot_loop(
        out / "figures" / "small_gain_product_vs_scale.png",
        [r for r in allrows if r["profile"] == "paper_range" and r["sample"] == 0],
    )
    plot_settlement(
        out / "figures" / "settling_fraction_vs_scale.png",
        [r for r in allrows if r["profile"] == "paper_range"],
        args.threshold,
    )
    (out / "configuration.json").write_text(
        json.dumps(
            {
                **vars(args),
                "weights_frozen": True,
                "all_Bi_scaled": True,
                "gamma_method": "exact A=0.9I temporal 10 × spectral norm of full E input matrix",
                "original_source_gain_method": "same Hinf induced norm but stationary-candidates factorization for B=[0,b]",
            },
            indent=2,
        )
    )
    return {
        "out": str(out),
        "checkpoints": len(models),
        "scales": len(args.scales),
        "trajectory_rows": len(allrows),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("--models", default="pretrained_models")
    p.add_argument("--out", default="results/sg_violation_diagnostic")
    p.add_argument("--scales", nargs="+", type=float, default=list(SCALES))
    p.add_argument("--plot-scales", nargs="+", type=float, default=list(SAVED_PLOTS))
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--initials", type=int, default=8)
    p.add_argument(
        "--profiles",
        nargs="+",
        choices=["paper_range", "near_zero"],
        default=["paper_range", "near_zero"],
    )
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=16)
    return p.parse_args(argv)


if __name__ == "__main__":
    print("Completed:", run(parse_args()))
