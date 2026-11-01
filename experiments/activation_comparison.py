"""Experiment 01: compare separately trained identity and tanh controllers.

Run from the project root:
  uv run python experiments/activation_comparison.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import (
    ensure_empty_dir, er_connected_graph, default_er_probability, system_at_gain,
    manifest, paired_rollouts, aggregate_method, summarize_time_series,
    save_training_artifacts, plot_metric_with_band, plot_representative_agents,
    plot_cumulative_energy, plot_settling_bars, save_checkpoint, write_csv_any, save_trace_npz, plot_settling_grouped, plot_settling_comparison, plot_numeric_conditions,
    write_json, train_model,
)


EXPERIMENT = '01_activation_comparison'


def run(args):
    out = ensure_empty_dir(Path(args.out))
    (out / 'checkpoints').mkdir()
    (out / 'figures').mkdir()
    manifest(out, args, EXPERIMENT, 'uv run python experiments/activation_comparison.py ...')

    summary_rows = []
    trajectory_rows = []
    curve_rows = []

    for rep in range(args.repeats):
        train_graph = er_connected_graph(3, args.er_prob, seed=args.seed + 1000 * rep + 11)
        train_system = system_at_gain(3, train_graph, args.gamma_train, seed=args.seed + 1000 * rep + 37)
        trained = {}
        for activation in ('identity', 'tanh'):
            result, W = train_model(train_system, train_graph, args, activation, seed=args.seed + 1000 * rep + 101)
            ckpt = out / 'checkpoints' / f'rep{rep:02d}_{activation}.npz'
            save_checkpoint(ckpt, train_system, train_graph, W, activation=activation, lam=args.lam)
            run_folder = out / 'checkpoints' / f'rep{rep:02d}_{activation}'
            run_folder.mkdir(exist_ok=True)
            save_training_artifacts(run_folder, result['history'])
            trained[activation] = {'W': W, 'result': result}

        for n in args.sizes:
            test_graph = er_connected_graph(n, args.er_prob, seed=args.seed + 1000 * rep + 211 + n)
            test_system = system_at_gain(n, test_graph, args.gamma_train, seed=args.seed + 1000 * rep + 311 + n)
            initial_states = np.random.default_rng(args.seed + 1000 * rep + 411 + n).standard_normal((args.initials, n, 2))

            series_map = {}
            representatives = {}
            raw_representatives = {}
            baseline_records = None

            # Baseline via identity call (same baseline for both activations).
            for method in ('identity', 'tanh'):
                records, representative, baseline_histories, learned_histories = paired_rollouts(
                    test_system, test_graph, args.eval_horizon, initial_states,
                    learned_W=trained[method]['W'], activation=method, threshold=args.threshold,
                )
                if baseline_records is None:
                    baseline_records = records
                    base_stats = aggregate_method(records, 'baseline')
                    summary_rows.append({
                        'repeat': rep, 'N': n, 'method': 'baseline',
                        'gamma_train': args.gamma_train, 'gamma_test': args.gamma_train,
                        'W_norm': 0.0, 'loop_gain': 0.0,
                        **base_stats,
                    })
                    for row in records:
                        trajectory_rows.append({'repeat': rep, 'N': n, 'method': 'baseline', **row})
                    series_map['baseline'] = baseline_histories
                    representatives['baseline'] = baseline_histories[0]
                    raw_representatives['baseline'] = representative['baseline']
                    for row in summarize_time_series(baseline_histories, ('error_rms', 'centered_rms', 'state_rms', 'control_rms')):
                        curve_rows.append({'repeat': rep, 'N': n, 'method': 'baseline', **row})
                stats = aggregate_method(records, 'learned')
                sg = trained[method]['result']['gamma_train'] * float(np.linalg.norm(trained[method]['W'], 2))
                summary_rows.append({
                    'repeat': rep, 'N': n, 'method': method,
                    'gamma_train': args.gamma_train, 'gamma_test': args.gamma_train,
                    'W_norm': float(np.linalg.norm(trained[method]['W'], 2)),
                    'loop_gain': sg,
                    'J_E_improvement_pct': 100 * (base_stats['mean_J_E'] - stats['mean_J_E']) / base_stats['mean_J_E'],
                    'J_U_reduction_pct': 100 * (base_stats['mean_J_U'] - stats['mean_J_U']) / base_stats['mean_J_U'] if base_stats['mean_J_U'] > 0 else None,
                    'paired_win_rate': float(np.mean([r['paired_win_J_E'] for r in records])),
                    **stats,
                })
                for row in records:
                    trajectory_rows.append({'repeat': rep, 'N': n, 'method': method, **row})
                series_map[method] = learned_histories
                representatives[method] = learned_histories[0]
                raw_representatives[method] = representative['learned']
                for row in summarize_time_series(learned_histories, ('error_rms', 'centered_rms', 'state_rms', 'control_rms')):
                    curve_rows.append({'repeat': rep, 'N': n, 'method': method, **row})

            prefix = out / 'figures' / f'N{n}_rep{rep:02d}'
            plot_metric_with_band(prefix.with_name(prefix.name + '_error_convergence.png'), series_map, 'error_rms', threshold=args.threshold,
                                  ylabel='RMS disagreement per agent', title=f'Experiment 01 error convergence, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_agent_error.png'), representatives, 'error_agent',
                                       'Agent disagreement norm', f'Experiment 01 representative agent disagreement, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_state_x1.png'), representatives, 'state_component_1',
                                       'State component x₁', f'Experiment 01 representative x₁ trajectories, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_state_x2.png'), representatives, 'state_component_2',
                                       'State component x₂', f'Experiment 01 representative x₂ trajectories, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_control_total.png'), representatives, 'control_total',
                                       'Physical control input', f'Experiment 01 representative physical control, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_correction_input.png'), representatives, 'control_correction', 'Learned correction input', f'Experiment 01 learned correction, N={n}, rep={rep}')
            plot_metric_with_band(prefix.with_name(prefix.name + '_centered_disagreement.png'), series_map, 'centered_rms', ylabel='Mean-centered disagreement RMS', title=f'Experiment 01 mean-centered disagreement, N={n}, rep={rep}')
            save_trace_npz(out / 'traces' / f'N{n}_rep{rep:02d}.npz', raw_representatives, test_system, test_graph, initial_states[0])
            plot_cumulative_energy(prefix.with_name(prefix.name + '_cumulative_energy.png'), representatives,
                                   f'Experiment 01 cumulative control energy, N={n}, rep={rep}')

    # aggregate settling bar by method using per-repeat summaries
    latest = [r for r in summary_rows if r['N'] == args.sizes[-1]]
    plot_settling_comparison(out / 'figures' / 'settling_summary.png', trajectory_rows, 'method', 'Experiment 01: identity vs tanh settling')

    write_csv_any(out / 'summary.csv', summary_rows)
    write_csv_any(out / 'trajectory_metrics.csv', trajectory_rows)
    write_csv_any(out / 'curves.csv', curve_rows)
    write_json(out / 'README.json', {'description': 'Experiment 01 activation comparison', 'png_only': True})
    return {'out': str(out), 'trained_models': args.repeats * 2, 'trajectory_rows': len(trajectory_rows), 'summary_rows': len(summary_rows)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', default='results/01_activation_comparison')
    p.add_argument('--sizes', nargs='+', type=int, default=[3])
    p.add_argument('--repeats', type=int, default=5, help='independent training/test graph and system realizations')
    p.add_argument('--initials', type=int, default=32, help='paired initial-state trajectories per test graph')
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--train-horizon', type=int, default=100)
    p.add_argument('--eval-horizon', type=int, default=200)
    p.add_argument('--batch', type=int, default=128, help='fresh initial-state trajectories sampled per training epoch')
    p.add_argument('--val-batch', type=int, default=128, help='fixed held-out validation initial states')
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--lam', type=float, default=0.0, help='existing control-energy objective coefficient')
    p.add_argument('--eta', type=float, default=0.95)
    p.add_argument('--gamma-train', type=float, default=0.75, help='target full-network plant gain of the 3-agent training system')
    p.add_argument('--threshold', type=float, default=1e-3, help='absolute RMS disagreement settling threshold')
    p.add_argument('--seed', type=int, default=16)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--er-prob', type=float, default=None, help='ER edge probability; default is selected from N')
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--early-stopping-patience', type=int, default=None)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--quick', action='store_true', help='tiny smoke run; not report-quality data')
    a = p.parse_args(argv)
    if a.device == 'auto':
        import torch
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.quick:
        a.sizes = [3]
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
