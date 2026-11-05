"""Experiment 02: train on three agents and test on 3, 50 and 100 agents.

Run from the project root:
  uv run python experiments/network_size_generalization.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import (
    ensure_empty_dir, er_connected_graph, system_at_gain, manifest, paired_rollouts,
    aggregate_method, summarize_time_series, save_training_artifacts,
    plot_metric_with_band, plot_representative_agents, plot_cumulative_energy,
    plot_settling_bars, save_checkpoint, write_csv_any, save_trace_npz, plot_settling_grouped, plot_settling_comparison, plot_numeric_conditions, train_model,
)
from experiments._paper_gnn_baseline import (train_paper_gnn, save_paper_checkpoint,
                                              evaluate_paper, save_training_history)


PAPER_SHARED_OUTPUT = True

EXPERIMENT = '02_network_size_generalization'


def run(args):
    out = ensure_empty_dir(Path(args.out))
    (out / 'checkpoints').mkdir()
    (out / 'figures').mkdir()
    manifest(out, args, EXPERIMENT, 'uv run python experiments/network_size_generalization.py ...')

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

        # Matched training batches/validation seeds, same physical plant and objective.
        # No projection or clipping on paper H, K_i. For N-transfer tie K_i across nodes.
        paper, paper_info = train_paper_gnn(
            train_system, train_graph, args, seed=args.seed + 1000 * rep + 101,
            shared_output=PAPER_SHARED_OUTPUT)
        save_paper_checkpoint(out / 'checkpoints' / f'rep{rep:02d}_paper_gnn.npz',
                              paper, train_system, train_graph, paper_info)
        save_training_history(out / 'checkpoints' / f'rep{rep:02d}_paper_training.csv',
                              paper_info['history'])
        print(f'TRAIN paper_gnn rep={rep} best_epoch={paper_info["best_epoch"]} '
              f'best_validation={paper_info["best_validation_objective"]:.5g}', flush=True)

        for n in args.sizes:
            test_graph = er_connected_graph(n, args.er_prob, seed=args.seed + 1000 * rep + 211 + n)
            test_system = system_at_gain(n, test_graph, args.gamma_train, seed=args.seed + 1000 * rep + 311 + n)
            initial_states = np.random.default_rng(args.seed + 1000 * rep + 411 + n).standard_normal((args.initials, n, 2))
            records, representative, baseline_histories, learned_histories = paired_rollouts(test_system, test_graph, args.eval_horizon, initial_states, learned_W=W, activation='identity', threshold=args.threshold)
            base_stats = aggregate_method(records, 'baseline')
            learned_stats = aggregate_method(records, 'learned')
            sg = result['gamma_train'] * float(np.linalg.norm(W, 2))
            summary_rows.extend([
                {'repeat': rep, 'N': n, 'method': 'baseline', 'gamma_train': args.gamma_train, 'gamma_test': args.gamma_train,
                 'W_norm': 0.0, 'loop_gain': 0.0, **base_stats},
                {'repeat': rep, 'N': n, 'method': 'learned', 'gamma_train': args.gamma_train, 'gamma_test': args.gamma_train,
                 'W_norm': float(np.linalg.norm(W, 2)), 'loop_gain': sg,
                 'J_E_improvement_pct': 100 * (base_stats['mean_J_E'] - learned_stats['mean_J_E']) / base_stats['mean_J_E'],
                 'J_U_reduction_pct': 100 * (base_stats['mean_J_U'] - learned_stats['mean_J_U']) / base_stats['mean_J_U'] if base_stats['mean_J_U'] > 0 else None,
                 'paired_win_rate': float(np.mean([r['paired_win_J_E'] for r in records])),
                 **learned_stats},
            ])
            trajectory_rows.extend([{'repeat': rep, 'N': n, 'method': 'baseline', **r} for r in records])
            trajectory_rows.extend([{'repeat': rep, 'N': n, 'method': 'learned', **r} for r in records])
            for label, histories in (('baseline', baseline_histories), ('learned', learned_histories)):
                for row in summarize_time_series(histories, ('error_rms', 'centered_rms', 'state_rms', 'control_rms')):
                    curve_rows.append({'repeat': rep, 'N': n, 'method': label, **row})
            paper_histories, paper_rep, paper_stats = evaluate_paper(
                paper, test_system, test_graph, initial_states, args.eval_horizon,
                args.threshold, {'repeat':rep,'N':n,'gamma_train':args.gamma_train,
                                 'gamma_test':args.gamma_train},
                summary_rows, trajectory_rows, curve_rows)
            summary_rows[-2]['J_E_improvement_vs_paper_pct'] = (
                100*(paper_stats['mean_J_E'] - learned_stats['mean_J_E'])/paper_stats['mean_J_E']
                if paper_stats['mean_J_E'] > 0 else None)
            representative['paper_gnn'] = paper_rep

            reps = {'baseline': baseline_histories[0], 'learned': learned_histories[0], 'paper_gnn': paper_histories[0]}
            prefix = out / 'figures' / f'N{n}_rep{rep:02d}'
            plot_metric_with_band(prefix.with_name(prefix.name + '_error_convergence.png'), {'baseline': baseline_histories, 'learned': learned_histories, 'paper_gnn': paper_histories}, 'error_rms', threshold=args.threshold,
                                  ylabel='RMS disagreement per agent', title=f'Experiment 02 error convergence, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_agent_error.png'), reps, 'error_agent', 'Agent disagreement norm', f'Experiment 02 representative agent disagreement, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_state_x1.png'), reps, 'state_component_1', 'State component x₁', f'Experiment 02 representative x₁ trajectories, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_state_x2.png'), reps, 'state_component_2', 'State component x₂', f'Experiment 02 representative x₂ trajectories, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_control_total.png'), reps, 'control_total', 'Physical control input', f'Experiment 02 representative physical control, N={n}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_correction_input.png'), reps, 'control_correction', 'Learned correction input', f'Experiment 02 learned correction, N={n}, rep={rep}')
            plot_metric_with_band(prefix.with_name(prefix.name + '_centered_disagreement.png'), {'baseline': baseline_histories, 'learned': learned_histories, 'paper_gnn': paper_histories}, 'centered_rms', ylabel='Mean-centered disagreement RMS', title=f'Experiment 02 mean-centered disagreement, N={n}, rep={rep}')
            save_trace_npz(out / 'traces' / f'N{n}_rep{rep:02d}.npz', representative, test_system, test_graph, initial_states[0])
            plot_cumulative_energy(prefix.with_name(prefix.name + '_cumulative_energy.png'), reps, f'Experiment 02 cumulative control energy, N={n}, rep={rep}')
            print(f'TEST size-generalization rep={rep} N={n} gamma_test={args.gamma_train:.5f}', flush=True)

    learned = [r for r in summary_rows if r['method'] == 'learned']
    plot_settling_comparison(out / 'figures' / 'settling_vs_size.png', trajectory_rows, 'N', 'Experiment 02: train N=3, test N=3/50/100')
    write_csv_any(out / 'summary.csv', summary_rows)
    write_csv_any(out / 'trajectory_metrics.csv', trajectory_rows)
    write_csv_any(out / 'curves.csv', curve_rows)
    return {'out': str(out), 'trained_models': args.repeats * 2, 'trajectory_rows': len(trajectory_rows), 'summary_rows': len(summary_rows)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', default='results/02_network_size_generalization')
    p.add_argument('--sizes', nargs='+', type=int, default=[3, 50, 100])
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
    p.add_argument('--gamma-train', type=float, default=0.75)
    p.add_argument('--threshold', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=16)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--er-prob', type=float, default=None)
    p.add_argument('--paper-hidden', type=int, default=4, help='one-layer paper GNN hidden features')
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
