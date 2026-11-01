"""Experiment 04: train on the default topology and test on common N=10 topologies.

Run from the project root:
  uv run python experiments/topology_generalization.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import (
    TOPOLOGIES, ensure_empty_dir, er_connected_graph, system_at_gain, manifest, full_gain, make_graph,
    paired_rollouts, aggregate_method, summarize_time_series, save_training_artifacts,
    plot_metric_with_band, plot_representative_agents, plot_cumulative_energy,
    plot_settling_bars, draw_graph_figure, save_checkpoint, write_csv_any, save_trace_npz, plot_settling_grouped, plot_settling_comparison, plot_numeric_conditions, train_model,
)

EXPERIMENT = '04_topology_generalization'


def run(args):
    out = ensure_empty_dir(Path(args.out))
    (out / 'checkpoints').mkdir()
    figroot = out / 'figures'
    figroot.mkdir()
    manifest(out, args, EXPERIMENT, 'uv run python experiments/topology_generalization.py ...')

    summary_rows = []
    trajectory_rows = []
    curve_rows = []

    for rep in range(args.repeats):
        train_graph = er_connected_graph(args.n, args.er_prob, seed=args.seed + 1000 * rep + 11) if args.train_topology == 'random' else make_graph(args.n, args.train_topology)
        train_system = system_at_gain(args.n, train_graph, args.gamma_train, seed=args.seed + 1000 * rep + 37)
        result, W = train_model(train_system, train_graph, args, 'identity', seed=args.seed + 1000 * rep + 101)
        save_checkpoint(out / 'checkpoints' / f'rep{rep:02d}_identity.npz', train_system, train_graph, W, activation='identity', lam=args.lam)
        run_folder = out / 'checkpoints' / f'rep{rep:02d}_identity'
        run_folder.mkdir(exist_ok=True)
        save_training_artifacts(run_folder, result['history'])

        # same physical system and same initial states across all tested topologies
        base_system = train_system
        initial_states = np.random.default_rng(args.seed + 1000 * rep + 411).standard_normal((args.initials, args.n, 2))

        for topology in TOPOLOGIES:
            topo_fig = figroot / topology
            topo_fig.mkdir(exist_ok=True)
            graph = er_connected_graph(args.n, args.er_prob, seed=args.seed + 1000 * rep + 701) if topology == 'random' else make_graph(args.n, topology, neighborhood=2 if topology == 'wide_ring' else 8)
            draw_graph_figure(topo_fig / f'rep{rep:02d}_graph.png', graph, 'erdos_renyi' if topology == 'random' else topology)
            records, representative, baseline_histories, learned_histories = paired_rollouts(base_system, graph, args.eval_horizon, initial_states, learned_W=W, activation='identity', threshold=args.threshold)
            base_stats = aggregate_method(records, 'baseline')
            learned_stats = aggregate_method(records, 'learned')
            gamma = full_gain(base_system, graph).gamma
            sg = gamma * float(np.linalg.norm(W, 2))
            status = 'at_or_below_eta' if sg <= args.eta else ('between_eta_and_one' if sg < 1.0 else 'at_or_above_one')
            summary_rows.extend([
                {'repeat': rep, 'N': args.n, 'topology': topology, 'method': 'baseline', 'gamma_train': args.gamma_train, 'gamma_test': gamma,
                 'W_norm': 0.0, 'loop_gain': 0.0, 'small_gain_status': 'baseline', **base_stats},
                {'repeat': rep, 'N': args.n, 'topology': topology, 'method': 'learned', 'gamma_train': args.gamma_train, 'gamma_test': gamma,
                 'W_norm': float(np.linalg.norm(W, 2)), 'loop_gain': sg, 'small_gain_status': status,
                 'J_E_improvement_pct': 100 * (base_stats['mean_J_E'] - learned_stats['mean_J_E']) / base_stats['mean_J_E'],
                 'J_U_reduction_pct': 100 * (base_stats['mean_J_U'] - learned_stats['mean_J_U']) / base_stats['mean_J_U'] if base_stats['mean_J_U'] > 0 else None,
                 'paired_win_rate': float(np.mean([r['paired_win_J_E'] for r in records])),
                 **learned_stats},
            ])
            trajectory_rows.extend([{'repeat': rep, 'N': args.n, 'topology': topology, 'method': 'baseline', **r} for r in records])
            trajectory_rows.extend([{'repeat': rep, 'N': args.n, 'topology': topology, 'method': 'learned', **r} for r in records])
            for label, histories in (('baseline', baseline_histories), ('learned', learned_histories)):
                for row in summarize_time_series(histories, ('error_rms', 'centered_rms', 'state_rms', 'control_rms')):
                    curve_rows.append({'repeat': rep, 'N': args.n, 'topology': topology, 'method': label, **row})
            reps = {'baseline': baseline_histories[0], 'learned': learned_histories[0]}
            prefix = topo_fig / f'rep{rep:02d}'
            plot_metric_with_band(prefix.with_name(prefix.name + '_error_convergence.png'), {'baseline': baseline_histories, 'learned': learned_histories}, 'error_rms', threshold=args.threshold,
                                  ylabel='RMS disagreement per agent', title=f'Experiment 04 error convergence, topology={topology}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_agent_error.png'), reps, 'error_agent', 'Agent disagreement norm', f'Experiment 04 representative disagreement, {topology}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_state_x1.png'), reps, 'state_component_1', 'State component x₁', f'Experiment 04 representative x₁, {topology}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_state_x2.png'), reps, 'state_component_2', 'State component x₂', f'Experiment 04 representative x₂, {topology}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_control_total.png'), reps, 'control_total', 'Physical control input', f'Experiment 04 representative physical control, {topology}, rep={rep}')
            plot_representative_agents(prefix.with_name(prefix.name + '_correction_input.png'), reps, 'control_correction', 'Learned correction input', f'Experiment 04 learned correction, {topology}, rep={rep}')
            plot_metric_with_band(prefix.with_name(prefix.name + '_centered_disagreement.png'), {'baseline': baseline_histories, 'learned': learned_histories}, 'centered_rms', ylabel='Mean-centered disagreement RMS', title=f'Experiment 04 mean-centered disagreement, {topology}, rep={rep}')
            save_trace_npz(out / 'traces' / f'{topology}_rep{rep:02d}.npz', representative, base_system, graph, initial_states[0])
            plot_cumulative_energy(prefix.with_name(prefix.name + '_cumulative_energy.png'), reps, f'Experiment 04 cumulative control energy, {topology}, rep={rep}')
            print(f'TEST topology rep={rep} topology={topology} gamma_test={gamma:.5f}', flush=True)

    learned = [r for r in summary_rows if r['method'] == 'learned']
    plot_settling_comparison(out / 'figures' / 'settling_vs_topology.png', trajectory_rows, 'topology', 'Experiment 04: topology settling')
    plot_numeric_conditions(out / 'figures' / 'plant_and_loop_gain_vs_topology.png', summary_rows, 'topology', 'Experiment 04: plant and loop gain by topology', args.eta)
    write_csv_any(out / 'summary.csv', summary_rows)
    write_csv_any(out / 'trajectory_metrics.csv', trajectory_rows)
    write_csv_any(out / 'curves.csv', curve_rows)
    return {'out': str(out), 'trained_models': args.repeats, 'trajectory_rows': len(trajectory_rows), 'summary_rows': len(summary_rows)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', default='results/04_topology_generalization')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--train-topology', choices=('chain', 'star', 'ring', 'wide_ring', 'complete', 'random'), default='chain')
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
        a.repeats = 1
        a.initials = 2
        a.epochs = 4
        a.train_horizon = 10
        a.eval_horizon = 15
        a.batch = 8
        a.val_batch = 8
        a.n = 10
    return a


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    print('Completed:', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
