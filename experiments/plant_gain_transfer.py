"""Experiment 03: compare inference plant gain below/above training gain.

Run from the project root:
  uv run python experiments/plant_gain_transfer.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import (
    ensure_empty_dir, er_connected_graph, system_at_gain, system_at_mixed_gain, generate_system, manifest, paired_rollouts,
    aggregate_method, summarize_time_series, save_training_artifacts,
    plot_metric_with_band, plot_representative_agents, plot_cumulative_energy,
    plot_settling_bars, save_checkpoint, write_csv_any, train_model, save_trace_npz, plot_settling_grouped, plot_settling_comparison, plot_numeric_conditions,
)

EXPERIMENT = '03_plant_gain_transfer'
SCENARIOS = {'below_training_gain': 0.75, 'above_training_gain': 1.25}


def run(args):
    out = ensure_empty_dir(Path(args.out))
    (out / 'checkpoints').mkdir()
    (out / 'figures').mkdir()
    manifest(out, args, EXPERIMENT, 'uv run python experiments/plant_gain_transfer.py ...')

    summary_rows = []
    trajectory_rows = []
    curve_rows = []

    for rep in range(args.repeats):
        train_graph = er_connected_graph(3, args.er_prob, seed=args.seed + 1000 * rep + 11)
        train_system = system_at_gain(3, train_graph, args.gamma_train, seed=args.seed + 1000 * rep + 37)
        result, W = train_model(train_system, train_graph, args, 'identity', seed=args.seed + 1000 * rep + 101)
        save_checkpoint(out / 'checkpoints' / f'rep{rep:02d}_identity.npz', train_system, train_graph, W, activation='identity', lam=args.lam)
        run_folder = out / 'checkpoints' / f'rep{rep:02d}_identity'
        run_folder.mkdir(exist_ok=True)
        save_training_artifacts(run_folder, result['history'])

        for n in args.sizes:
            test_graph = er_connected_graph(n, args.er_prob, seed=args.seed + 1000 * rep + 211 + n)
            initial_states = np.random.default_rng(args.seed + 1000 * rep + 411 + n).standard_normal((args.initials, n, 2))
            # Both gain scenarios share exactly the same A_i, original B_i, graph and X0.
            base_system = generate_system(n, seed=args.seed + 1000 * rep + 311 + n)
            group = np.zeros(n, dtype=np.int64)
            group[np.random.default_rng(args.seed + 1000 * rep + 711 + n).permutation(n)[:(n // 2 if rep % 2 == 0 else (n+1)//2)]] = 1
            scenario_histories = {}
            for scenario, ratio in SCENARIOS.items():
                gamma_test = args.gamma_train * ratio
                test_system, effective_multipliers = system_at_mixed_gain(base_system, test_graph, gamma_test, group)
                records, representative, baseline_histories, learned_histories = paired_rollouts(test_system, test_graph, args.eval_horizon, initial_states, learned_W=W, activation='identity', threshold=args.threshold)
                scenario_histories[scenario] = learned_histories
                base_stats = aggregate_method(records, 'baseline')
                learned_stats = aggregate_method(records, 'learned')
                wnorm = float(np.linalg.norm(W, 2))
                loop_gain = gamma_test * wnorm
                status = 'at_or_below_eta' if loop_gain <= args.eta else ('between_eta_and_one' if loop_gain < 1.0 else 'at_or_above_one')
                summary_rows.extend([
                    {'repeat': rep, 'N': n, 'scenario': scenario, 'method': 'baseline', 'gamma_train': args.gamma_train,
                     'gamma_test': gamma_test, 'gamma_test_over_train': ratio, 'weak_agents': int(np.sum(group == 0)), 'strong_agents': int(np.sum(group == 1)), 'W_norm': 0.0, 'loop_gain': 0.0,
                     'small_gain_status': 'baseline', **base_stats},
                    {'repeat': rep, 'N': n, 'scenario': scenario, 'method': 'learned', 'gamma_train': args.gamma_train,
                     'gamma_test': gamma_test, 'gamma_test_over_train': ratio, 'weak_agents': int(np.sum(group == 0)), 'strong_agents': int(np.sum(group == 1)), 'W_norm': wnorm, 'loop_gain': loop_gain,
                     'small_gain_status': status,
                     'J_E_improvement_pct': 100 * (base_stats['mean_J_E'] - learned_stats['mean_J_E']) / base_stats['mean_J_E'],
                     'J_U_reduction_pct': 100 * (base_stats['mean_J_U'] - learned_stats['mean_J_U']) / base_stats['mean_J_U'] if base_stats['mean_J_U'] > 0 else None,
                     'paired_win_rate': float(np.mean([r['paired_win_J_E'] for r in records])),
                     **learned_stats},
                ])
                trajectory_rows.extend([{'repeat': rep, 'N': n, 'scenario': scenario, 'method': 'baseline', **r} for r in records])
                trajectory_rows.extend([{'repeat': rep, 'N': n, 'scenario': scenario, 'method': 'learned', **r} for r in records])
                for label, histories in (('baseline', baseline_histories), ('learned', learned_histories)):
                    for row in summarize_time_series(histories, ('error_rms', 'centered_rms', 'state_rms', 'control_rms')):
                        curve_rows.append({'repeat': rep, 'N': n, 'scenario': scenario, 'method': label, **row})
                reps = {'baseline': baseline_histories[0], 'learned': learned_histories[0]}
                prefix = out / 'figures' / f'N{n}_{scenario}_rep{rep:02d}'
                plot_metric_with_band(prefix.with_name(prefix.name + '_error_convergence.png'), {'baseline': baseline_histories, 'learned': learned_histories}, 'error_rms', threshold=args.threshold,
                                      ylabel='RMS disagreement per agent', title=f'Experiment 03 error convergence, N={n}, {scenario}, rep={rep}')
                plot_representative_agents(prefix.with_name(prefix.name + '_agent_error.png'), reps, 'error_agent', 'Agent disagreement norm', f'Experiment 03 representative disagreement, N={n}, {scenario}, rep={rep}')
                plot_representative_agents(prefix.with_name(prefix.name + '_state_x1.png'), reps, 'state_component_1', 'State component x₁', f'Experiment 03 representative x₁, N={n}, {scenario}, rep={rep}')
                plot_representative_agents(prefix.with_name(prefix.name + '_state_x2.png'), reps, 'state_component_2', 'State component x₂', f'Experiment 03 representative x₂, N={n}, {scenario}, rep={rep}')
                plot_representative_agents(prefix.with_name(prefix.name + '_control_total.png'), reps, 'control_total', 'Physical control input', f'Experiment 03 representative physical control, N={n}, {scenario}, rep={rep}')
                plot_representative_agents(prefix.with_name(prefix.name + '_correction_input.png'), reps, 'control_correction', 'Learned correction input', f'Experiment 03 learned correction, N={n}, {scenario}, rep={rep}')
                plot_metric_with_band(prefix.with_name(prefix.name + '_centered_disagreement.png'), {'baseline': baseline_histories, 'learned': learned_histories}, 'centered_rms', ylabel='Mean-centered disagreement RMS', title=f'Experiment 03 mean-centered disagreement, N={n}, {scenario}, rep={rep}')
                plot_cumulative_energy(prefix.with_name(prefix.name + '_cumulative_energy.png'), reps, f'Experiment 03 cumulative control energy, N={n}, {scenario}, rep={rep}')
                save_trace_npz(out / 'traces' / f'N{n}_{scenario}_rep{rep:02d}.npz', representative, test_system, test_graph, initial_states[0])
                np.savez_compressed(out / 'traces' / f'N{n}_{scenario}_rep{rep:02d}_gain_groups.npz', group=group, initial_B=base_system.B, effective_multiplier=effective_multipliers, gamma_train=args.gamma_train, gamma_test=gamma_test)
                print(f'TEST gain-transfer rep={rep} N={n} {scenario}: gamma_train={args.gamma_train:.5f} gamma_test={gamma_test:.5f} ratio={ratio:.3f} loop_gain={loop_gain:.5f} status={status}', flush=True)

            for metric in ('error_rms', 'centered_rms', 'state_rms', 'control_rms'):
                plot_metric_with_band(out / 'figures' / f'N{n}_rep{rep:02d}_below_vs_above_{metric}.png', scenario_histories, metric,
                                      threshold=args.threshold if metric == 'error_rms' else None,
                                      ylabel=metric, title=f'Experiment 03 frozen learned controller: N={n}, {metric}')

            # comparison plot for both scenarios at this N, rep
            compare_rows = [r for r in summary_rows if r['repeat'] == rep and r['N'] == n and r['method'] == 'learned']
            plot_settling_bars(out / 'figures' / f'N{n}_rep{rep:02d}_settling_compare.png', compare_rows, 'scenario', f'Experiment 03 settling comparison, N={n}, rep={rep}')

    for n in args.sizes:
        plot_settling_comparison(out / 'figures' / f'N{n}_settling_by_gain.png', [r for r in trajectory_rows if r['N'] == n], 'scenario', f'Experiment 03: N={n} gain transfer')
        plot_numeric_conditions(out / 'figures' / f'N{n}_gamma_and_loop_gain.png', [r for r in summary_rows if r['N'] == n], 'scenario', f'N={n}: measured plant and loop gain', args.eta)
    write_csv_any(out / 'summary.csv', summary_rows)
    write_csv_any(out / 'trajectory_metrics.csv', trajectory_rows)
    write_csv_any(out / 'curves.csv', curve_rows)
    return {'out': str(out), 'trained_models': args.repeats, 'trajectory_rows': len(trajectory_rows), 'summary_rows': len(summary_rows)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', default='results/03_plant_gain_transfer')
    p.add_argument('--sizes', nargs='+', type=int, default=[3, 50, 100])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--initials', type=int, default=32)
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--train-horizon', type=int, default=100)
    p.add_argument('--eval-horizon', type=int, default=200)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--val-batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--eta', type=float, default=0.95)
    p.add_argument('--gamma-train', type=float, default=0.75)
    p.add_argument('--threshold', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=16)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--er-prob', type=float, default=None)
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--early-stopping-patience', type=int, default=None)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--quick', action='store_true', help='tiny smoke run; not report-quality data')
    a = p.parse_args(argv)
    if a.device == 'auto':
        import torch
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.quick:
        a.sizes = [3, 50]
        a.repeats = 1
        a.initials = 2
        a.epochs = 4
        a.train_horizon = 10
        a.eval_horizon = 15
        a.batch = 8
        a.val_batch = 8
    return a


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    print('Completed:', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
