"""Finite-horizon trajectory metrics and paired baseline/learned evaluation."""

import numpy as np

from distributed_consensus.control.dynamics import rollout_numpy


def _crossing_and_settling(norms, fraction):
    """First crossing and first index after which the threshold always holds."""
    inside = norms <= fraction * norms[0]
    crossing = np.flatnonzero(inside)
    violations = np.flatnonzero(~inside)
    first = int(crossing[0]) if crossing.size else None
    settle = (
        None
        if violations.size and violations[-1] == len(norms) - 1
        else int(violations[-1] + 1)
        if violations.size
        else 0
    )
    return first, settle


def _percentage_reduction(baseline, learned):
    """Return None when baseline is zero (percentage reduction is undefined)."""
    if baseline == 0:
        return None
    value = 100.0 * (baseline - learned) / baseline
    if not np.isfinite(value):
        raise ArithmeticError("nonfinite percentage reduction")
    return float(value)


def trajectory_metrics(history, epsilon_fraction=0.01):
    """Metrics for T actions and T+1 states/disagreement samples.

    No terminal stage cost is added. Relative quantities are None when the
    reference disagreement is exactly zero; no arbitrary epsilon floor is used.
    """
    if not np.isfinite(epsilon_fraction) or epsilon_fraction < 0:
        raise ValueError("epsilon_fraction must be finite and nonnegative")

    e = np.asarray(history["disagreement"], dtype=np.float64)
    u = np.asarray(history["total"], dtype=np.float64)
    corr = np.asarray(history["correction"], dtype=np.float64)

    if (
        e.ndim != 3
        or e.shape[0] < 2
        or e.shape[1] < 1
        or u.shape != (len(e) - 1, e.shape[1], 1)
        or corr.shape != u.shape
    ):
        raise ValueError("expected disagreement (T+1,N,p) and actions (T,N,1)")
    if not (np.isfinite(e).all() and np.isfinite(u).all() and np.isfinite(corr).all()):
        raise ValueError("trajectory contains nonfinite values")

    t = len(u)
    n = e.shape[1]
    norms = np.linalg.norm(e.reshape(t + 1, -1), axis=1)
    if not np.isfinite(norms).all():
        raise ArithmeticError("nonfinite disagreement norm")
    initial = float(norms[0])
    cross, settle = _crossing_and_settling(norms, epsilon_fraction)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        je = float(np.square(e[:-1]).sum())
        ju = float(np.square(u).sum())
        jcorr = float(np.square(corr).sum())
        normalized = float(np.square(e[:-1] / initial).sum()) if initial != 0 else None

    states = history.get("states")
    jpair = pairmax = pairterminal = pairsettle = None
    if states is not None:
        x = np.asarray(states, dtype=np.float64)
        if x.shape != e.shape or not np.isfinite(x).all():
            raise ValueError("states must be finite and have shape (T+1,N,p)")
        with np.errstate(over="raise", invalid="raise"):
            centered = x - x.mean(axis=1, keepdims=True)
            paircurve = np.square(centered).sum(axis=(1, 2)) / n
            pairnorm = np.sqrt(paircurve)
            jpair = float(paircurve[:-1].sum())
        if not np.isfinite(pairnorm).all():
            raise ArithmeticError("nonfinite mean-centered disagreement")
        pairmax = float(pairnorm.max())
        pairterminal = float(pairnorm[-1])
        _, pairsettle = _crossing_and_settling(pairnorm, epsilon_fraction)

    return {
        "J_pair": jpair,
        "J_E_per_agent": je / n,
        "J_U_per_agent": ju / n,
        "J_E_initial_normalized": normalized,
        "max_centered_rms": pairmax,
        "terminal_centered_rms": pairterminal,
        "pair_settling_time": pairsettle,
        "J_E": je,
        "J_U": ju,
        "J_corr": jcorr,
        "first_crossing": cross,
        "settling_time": settle,
        "settled": settle is not None,
        "max_E": float(norms.max()),
        "terminal_E": float(norms[-1]),
        "initial_E": initial,
        "threshold": epsilon_fraction * initial,
        "norm_curve": norms,
    }


def evaluate_pair(
    system,
    graph,
    W,
    horizon=100,
    samples=24,
    seed=413,
    activation="identity",
    initial_states=None,
    disturbances=None,
):
    """Compare local-only and frozen W using identical initial states/disturbances."""
    if samples < 1 or horizon < 1:
        raise ValueError("samples and horizon must be positive")
    if graph.n != system.n:
        raise ValueError("graph/system size mismatch")
    if W is not None:
        weights = np.asarray(W, dtype=np.float64)
        if weights.shape != (2, 1) or not np.isfinite(weights).all():
            raise ValueError("W must be finite with shape (2,1)")

    if initial_states is not None:
        initial_states = np.asarray(initial_states, dtype=np.float64)
        if (
            initial_states.shape != (samples, system.n, 2)
            or not np.isfinite(initial_states).all()
        ):
            raise ValueError("initial_states must be finite with shape (samples,N,2)")
    if disturbances is not None:
        disturbances = np.asarray(disturbances, dtype=np.float64)
        if (
            disturbances.shape != (samples, horizon, system.n, 1)
            or not np.isfinite(disturbances).all()
        ):
            raise ValueError("disturbances must be finite with shape (samples,T,N,1)")

    rng = np.random.default_rng(seed)
    records = []
    curves_baseline, curves_learned = [], []
    representative = {}

    for sample in range(samples):
        x0 = (
            rng.standard_normal((system.n, 2))
            if initial_states is None
            else initial_states[sample]
        )
        d = None if disturbances is None else disturbances[sample]
        baseline = rollout_numpy(
            x0, system, graph, None, horizon=horizon, disturbances=d
        )
        learned = rollout_numpy(
            x0, system, graph, W, activation, horizon, disturbances=d
        )
        bm, lm = trajectory_metrics(baseline), trajectory_metrics(learned)
        if sample == 0:
            representative = {"baseline": baseline, "learned": learned}
        row = {"sample": sample}
        for label, metric in (("baseline", bm), ("learned", lm)):
            row.update(
                {
                    f"{label}_{key}": val
                    for key, val in metric.items()
                    if key != "norm_curve"
                }
            )
        for key in ("J_E", "J_U", "J_pair", "terminal_E"):
            row[f"{key}_improvement_pct"] = _percentage_reduction(bm[key], lm[key])
        row["paired_win_J_E"] = lm["J_E"] < bm["J_E"]
        records.append(row)
        curves_baseline.append(bm["norm_curve"])
        curves_learned.append(lm["norm_curve"])

    def describe(label):
        def mean_of(key):
            values = [row[f"{label}_{key}"] for row in records]
            defined = [value for value in values if value is not None]
            return float(np.mean(defined)) if defined else None

        crossing = [r[f"{label}_first_crossing"] for r in records]
        settling = [r[f"{label}_settling_time"] for r in records]
        pair_settling = [r[f"{label}_pair_settling_time"] for r in records]
        return {
            "mean_J_pair": mean_of("J_pair"),
            "mean_J_E_per_agent": mean_of("J_E_per_agent"),
            "mean_J_U_per_agent": mean_of("J_U_per_agent"),
            "mean_J_E_initial_normalized": mean_of("J_E_initial_normalized"),
            "mean_max_centered_rms": mean_of("max_centered_rms"),
            "pair_settled_fraction": float(
                np.mean([v is not None for v in pair_settling])
            ),
            "mean_pair_settling_if_settled": mean_of("pair_settling_time"),
            "mean_J_E": mean_of("J_E"),
            "mean_J_U": mean_of("J_U"),
            "mean_J_corr": mean_of("J_corr"),
            "mean_max_E": mean_of("max_E"),
            "mean_terminal_E": mean_of("terminal_E"),
            "crossed_fraction": float(np.mean([v is not None for v in crossing])),
            "settled_fraction": float(np.mean([v is not None for v in settling])),
            "mean_first_crossing_if_crossed": mean_of("first_crossing"),
            "mean_settling_if_settled": mean_of("settling_time"),
        }

    base_stats, learned_stats = describe("baseline"), describe("learned")
    return {
        "baseline": base_stats,
        "learned": learned_stats,
        "J_pair_improvement_pct": _percentage_reduction(
            base_stats["mean_J_pair"], learned_stats["mean_J_pair"]
        ),
        "J_E_improvement_pct": _percentage_reduction(
            base_stats["mean_J_E"], learned_stats["mean_J_E"]
        ),
        "J_U_reduction_pct": _percentage_reduction(
            base_stats["mean_J_U"], learned_stats["mean_J_U"]
        ),
        "paired_win_rate": float(np.mean([row["paired_win_J_E"] for row in records])),
        "records": records,
        "curves": {
            "baseline": np.mean(curves_baseline, axis=0),
            "learned": np.mean(curves_learned, axis=0),
        },
        "representative": representative,
    }
