"""Run one headless scalar local W-MSR / multi-controller demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path

from distributed_consensus.evaluation.reporting import write_csv, write_json
from .model import DEFAULT_BENCHMARK_CHECKPOINT, METHODS, METHOD_LABELS, MultiPolicySimulation


def _parse_adversaries(values, n):
    tokens = [part.strip() for value in values for part in value.split(',') if part.strip()]
    try:
        numbers = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError('--adversary accepts positive 1-based IDs; use 0 alone to disable') from exc
    if numbers == [0]:
        return ()
    if not numbers or any(i < 1 or i > n for i in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError(f'adversary IDs must be unique integers in [1,{n}], or use 0 alone')
    return tuple(i - 1 for i in numbers)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--benchmark', default=str(DEFAULT_BENCHMARK_CHECKPOINT))
    p.add_argument('--topology', choices=('complete', 'wide_ring', 'ring', 'chain', 'star', 'random'), default='complete')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--adversary', nargs='+', default=['10'], help='1-based malicious agent IDs; use 0 to disable')
    p.add_argument('--F', type=int, default=1, help='F-local bound and W-MSR high/low trimming count')
    p.add_argument('--threshold', type=float, default=1e-3)
    p.add_argument('--out', default='results/wmsr_local_preview')
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error('--steps must be positive')
    if args.F < 0:
        p.error('--F must be nonnegative')
    try:
        from .model import load_benchmark
        benchmark = load_benchmark(args.benchmark)
        adversaries = _parse_adversaries(args.adversary, benchmark.system.n)
        sim = MultiPolicySimulation(
            args.benchmark, topology=args.topology, seed=args.seed,
            adversary_ids=adversaries, F=args.F, threshold=args.threshold,
        )
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))

    print(f'N={sim.n}, graph={args.topology}, seed={args.seed}, steps={args.steps}')
    print(f'adversaries={[i + 1 for i in sim.adversary_ids]}, F={sim.F} (F-local malicious model)')
    report = sim.topology_report
    status = 'MET' if report['condition_met'] is True else (
        'NOT MET' if report['condition_met'] is False else 'NOT CHECKED')
    print(f"W-MSR sufficient graph condition: {report['required_robustness']} = {status}.")
    if report['condition_met'] is not True:
        print('WARNING: this topology does not meet the sufficient F-local W-MSR condition.')

    for step in range(args.steps):
        sim.step()
        if (step + 1) % 50 == 0 or step + 1 == args.steps:
            errors = ' | '.join(f'{METHOD_LABELS[m]}={sim.error_histories[m][-1]:.3g}' for m in METHODS)
            print(f'k={step + 1:4d}: {errors}', flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim.export_csv(out / 'trajectories.csv')
    sim.save_figures(out / 'figures')
    rows = sim.summary_rows()
    write_csv(out / 'summary.csv', rows)
    write_json(out / 'configuration.json', {
        'benchmark': str(Path(args.benchmark).resolve()),
        'topology': args.topology, 'seed': args.seed, 'steps': args.steps,
        'adversary_ids_1based': [i + 1 for i in sim.adversary_ids],
        'F': args.F, 'fault_model': 'F-local malicious broadcast',
        'attack': 'same extreme scalar signal broadcast to every neighbour; sign alternates by step',
        'threshold': args.threshold,
        'methods': list(METHODS),
        'wmsr_topology_report': sim.topology_report,
        'wmsr_note': 'Exact scalar local W-MSR on one scalar coordination signal per agent.',
    })
    print('\nFINAL SUMMARY')
    for row in rows:
        settling = row['strict_settling_step']
        settling_text = 'unsettled' if settling is None else f'{settling} steps'
        print(f"{row['method_label']:<27} settle={settling_text:<14} "
              f"final error={row['final_normal_disagreement_rms']:.4g} "
              f"J_U={row['cumulative_physical_control_energy']:.4g} "
              f"discarded={row['scalar_values_discarded_total']}")
    print(f'RESULTS: {out.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
