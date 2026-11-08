"""Paired multi-trial evaluation for scalar local W-MSR controller comparisons.

All five policies use matched initial physical states and the same F-local
malicious-broadcast scenario. The paper GNN and small-gain controllers are
compared both without filtering and with the exact scalar local W-MSR update.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from distributed_consensus.evaluation.reporting import write_csv, write_json
from .model import (
    DEFAULT_BENCHMARK_CHECKPOINT,
    METHODS,
    METHOD_LABELS,
    MultiPolicySimulation,
    load_benchmark,
)


def _parse_ids(values, n: int) -> tuple[int, ...]:
    tokens = [
        part.strip() for value in values for part in value.split(",") if part.strip()
    ]
    try:
        numbers = [int(v) for v in tokens]
    except ValueError as exc:
        raise ValueError(
            "--adversary expects 1-based integer IDs; use 0 to disable"
        ) from exc
    if numbers == [0]:
        return ()
    if (
        not numbers
        or len(set(numbers)) != len(numbers)
        or any(v < 1 or v > n for v in numbers)
    ):
        raise ValueError(
            f"adversary IDs must be unique values in [1,{n}], or use 0 alone"
        )
    return tuple(v - 1 for v in numbers)


def _aggregate_trials(rows: list[dict]) -> list[dict]:
    aggregate = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        settled = [
            row["strict_settling_step"]
            for row in selected
            if row["strict_settling_step"] is not None
        ]
        final = np.asarray(
            [row["final_normal_disagreement_rms"] for row in selected], dtype=float
        )
        je = np.asarray(
            [row["cumulative_disagreement_cost"] for row in selected], dtype=float
        )
        ju = np.asarray(
            [row["cumulative_physical_control_energy"] for row in selected], dtype=float
        )
        discarded = [row["scalar_values_discarded_total"] for row in selected]
        aggregate.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "trials": len(selected),
                "settled_trials": len(settled),
                "settled_fraction": len(settled) / len(selected) if selected else 0.0,
                "median_settling_steps_if_settled": float(np.median(settled))
                if settled
                else None,
                "median_final_disagreement_rms": float(np.median(final))
                if len(final)
                else None,
                "q25_final_disagreement_rms": float(np.percentile(final, 25))
                if len(final)
                else None,
                "q75_final_disagreement_rms": float(np.percentile(final, 75))
                if len(final)
                else None,
                "median_disagreement_cost": float(np.median(je)) if len(je) else None,
                "median_physical_control_energy": float(np.median(ju))
                if len(ju)
                else None,
                "median_scalar_values_discarded": float(np.median(discarded))
                if discarded
                else 0.0,
            }
        )
    return aggregate


def _paired_effect(rows: list[dict]) -> list[dict]:
    pairs = (
        ("paper_gnn_raw", "paper_gnn_wmsr", "Paper GNN"),
        ("small_gain_raw", "small_gain_wmsr", "Small-gain controller"),
    )
    output = []
    for raw_method, filtered_method, label in pairs:
        raw_by_trial = {
            row["trial"]: row for row in rows if row["method"] == raw_method
        }
        filtered_by_trial = {
            row["trial"]: row for row in rows if row["method"] == filtered_method
        }
        for trial in sorted(set(raw_by_trial) & set(filtered_by_trial)):
            raw = raw_by_trial[trial]
            filt = filtered_by_trial[trial]
            denominator = max(float(raw["final_normal_disagreement_rms"]), 1e-15)
            output.append(
                {
                    "trial": trial,
                    "controller": label,
                    "raw_final_error": raw["final_normal_disagreement_rms"],
                    "wmsr_final_error": filt["final_normal_disagreement_rms"],
                    "wmsr_to_raw_error_ratio": filt["final_normal_disagreement_rms"]
                    / denominator,
                    "raw_settled": raw["settled"],
                    "wmsr_settled": filt["settled"],
                    "raw_settling_step": raw["strict_settling_step"],
                    "wmsr_settling_step": filt["strict_settling_step"],
                    "raw_control_energy": raw["cumulative_physical_control_energy"],
                    "wmsr_control_energy": filt["cumulative_physical_control_energy"],
                }
            )
    return output


def _save_figures(
    out: Path,
    curves: dict[str, list[np.ndarray]],
    aggregate: list[dict],
    threshold: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "local_only": "#8d98a7",
        "paper_gnn_raw": "#e28a43",
        "paper_gnn_wmsr": "#e2bd55",
        "small_gain_raw": "#5c91d8",
        "small_gain_wmsr": "#28b69a",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.2), layout="constrained")
    for method in METHODS:
        values = np.stack(curves[method])
        q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
        ax.plot(
            np.maximum(median, 1e-12),
            label=METHOD_LABELS[method],
            color=colors[method],
            linewidth=2,
        )
        ax.fill_between(
            np.arange(values.shape[1]),
            np.maximum(q25, 1e-12),
            np.maximum(q75, 1e-12),
            color=colors[method],
            alpha=0.12,
        )
    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"threshold {threshold:g}",
    )
    ax.set_yscale("log")
    ax.set(
        xlabel="Discrete-time step k",
        ylabel="Normal-agent disagreement RMS",
        title="Median disagreement with interquartile range",
    )
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out / "aggregate_disagreement.png", dpi=160)
    plt.close(fig)

    labels = [row["method_label"] for row in aggregate]
    x = np.arange(len(aggregate))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    axes[0].bar(
        x,
        [row["settled_fraction"] for row in aggregate],
        color=[colors[row["method"]] for row in aggregate],
    )
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction settled")
    axes[0].set_title("Settling success rate")
    axes[0].grid(axis="y", alpha=0.2)
    medians = [row["median_settling_steps_if_settled"] for row in aggregate]
    axes[1].bar(
        x,
        [np.nan if value is None else value for value in medians],
        color=[colors[row["method"]] for row in aggregate],
    )
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Discrete-time steps")
    axes[1].set_title("Median settling time (successful trials)")
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(out / "aggregate_settling.png", dpi=160)
    plt.close(fig)


def run_evaluation(args) -> Path:
    if args.trials < 1 or args.steps < 1 or args.F < 0:
        raise ValueError("trials/steps must be positive and F must be nonnegative")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark(args.benchmark)
    adversaries = _parse_ids(args.adversary, benchmark.system.n)
    first = MultiPolicySimulation(
        args.benchmark,
        topology=args.topology,
        seed=args.seed,
        adversary_ids=adversaries,
        F=args.F,
        threshold=args.threshold,
    )
    first_report = first.topology_report

    trial_rows = []
    curves = {method: [] for method in METHODS}
    for trial in range(args.trials):
        seed = args.seed + trial
        sim = MultiPolicySimulation(
            args.benchmark,
            topology=args.topology,
            seed=seed,
            adversary_ids=adversaries,
            F=args.F,
            threshold=args.threshold,
        )
        for _ in range(args.steps):
            sim.step()
        trial_rows.extend(
            {"trial": trial, "seed": seed, **row} for row in sim.summary_rows()
        )
        for method in METHODS:
            curves[method].append(
                np.asarray(sim.error_histories[method], dtype=float).copy()
            )
        if trial == 0:
            sim.export_csv(out / "representative_run" / "trajectories.csv")
            sim.save_figures(out / "representative_run" / "figures")
        print(
            f"trial {trial + 1:02d}/{args.trials}: "
            + " | ".join(
                f"{method}={sim.error_histories[method][-1]:.3g}" for method in METHODS
            ),
            flush=True,
        )

    aggregate = _aggregate_trials(trial_rows)
    paired = _paired_effect(trial_rows)
    write_csv(out / "trial_summary.csv", trial_rows)
    write_csv(out / "aggregate_summary.csv", aggregate)
    write_csv(out / "paired_wmsr_effect.csv", paired)
    curve_rows = []
    for method in METHODS:
        values = np.stack(curves[method])
        q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
        for step in range(values.shape[1]):
            curve_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "step": step,
                    "mean_error": float(values[:, step].mean()),
                    "q25_error": float(q25[step]),
                    "median_error": float(median[step]),
                    "q75_error": float(q75[step]),
                }
            )
    write_csv(out / "aggregate_curves.csv", curve_rows)
    _save_figures(out, curves, aggregate, args.threshold)
    write_json(
        out / "configuration.json",
        {
            "benchmark": str(Path(args.benchmark).resolve()),
            "trials": args.trials,
            "seed_start": args.seed,
            "steps": args.steps,
            "topology": args.topology,
            "adversary_ids_1based": [i + 1 for i in adversaries],
            "F": args.F,
            "fault_model": "F-local malicious broadcast",
            "threshold": args.threshold,
            "methods": list(METHODS),
            "wmsr_topology_report": first_report,
            "primary_comparisons": [
                ["paper_gnn_raw", "paper_gnn_wmsr"],
                ["small_gain_raw", "small_gain_wmsr"],
            ],
            "local_only_role": "secondary reference; it does not consume neighbour messages",
            "wmsr_note": "Exact scalar local W-MSR on one controller-specific scalar message per agent.",
        },
    )
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_CHECKPOINT))
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument(
        "--topology",
        choices=("complete", "wide_ring", "ring", "chain", "star", "random"),
        default="complete",
    )
    p.add_argument(
        "--adversary",
        nargs="+",
        default=["10"],
        help="1-based malicious IDs; use 0 to disable",
    )
    p.add_argument(
        "--F", type=int, default=1, help="F-local adversary bound and W-MSR trim count"
    )
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--out", default="results/wmsr_local_evaluation")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = run_evaluation(args)
    print(f"\nEvaluation complete: {out.resolve()}")
    for row in _aggregate_trials_from_csv(out / "trial_summary.csv"):
        settling = row["median_settling_steps_if_settled"]
        text = "unsettled" if settling is None else f"{settling:.1f}"
        print(
            f"{row['method_label']:<28} settled={row['settled_fraction']:.0%} "
            f"median-settle={text:<10} final e={row['median_final_disagreement_rms']:.3g}"
        )
    return 0


def _aggregate_trials_from_csv(path):
    import csv

    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return _aggregate_trials(
        [
            {
                "method": row["method"],
                "strict_settling_step": None
                if row["strict_settling_step"] == ""
                else int(float(row["strict_settling_step"])),
                "final_normal_disagreement_rms": float(
                    row["final_normal_disagreement_rms"]
                ),
                "cumulative_disagreement_cost": float(
                    row["cumulative_disagreement_cost"]
                ),
                "cumulative_physical_control_energy": float(
                    row["cumulative_physical_control_energy"]
                ),
                "scalar_values_discarded_total": float(
                    row["scalar_values_discarded_total"]
                ),
            }
            for row in rows
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
