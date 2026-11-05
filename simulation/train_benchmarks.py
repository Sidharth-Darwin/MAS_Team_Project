"""Train matched paper-GNN and small-gain controllers for the adversarial demo.

Both learned methods see the same physical plant, graph, training batches,
validation batches, objective, optimizer settings, and checkpoint cadence.
The paper GNN follows the unnormalized-Laplacian architecture and is NOT
projected. The small-gain W is projected using the source library's H-infinity
plant gain implementation.

Run from the repository root:
  uv run python -m simulation.train_benchmarks --help
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from distributed_consensus.control.plant_gain import full_gain
from distributed_consensus.evaluation.reporting import load_checkpoint, write_csv, write_json
from distributed_consensus.graphs import make_graph
from distributed_consensus.learning.training import train
from experiments._paper_gnn_baseline import train_paper_gnn

from .model import DEFAULT_SOURCE_CHECKPOINT, DEFAULT_BENCHMARK_CHECKPOINT


def _save_benchmark(path: Path, system, graph, W: np.ndarray, paper_model,
                    train_info: dict, paper_info: dict, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gamma = full_gain(system, graph).gamma
    H = paper_model.H.detach().cpu().numpy().astype(np.float64)
    K_paper = paper_model.K.detach().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(W, 2))
    np.savez_compressed(
        path,
        A=np.asarray(system.A), B=np.asarray(system.B),
        K_local=np.asarray(system.K), target=np.asarray(system.target),
        adjacency=graph.adjacency.toarray(), W=np.asarray(W, dtype=np.float64),
        paper_H=H, paper_K=K_paper,
        paper_shared_output=np.array(int(paper_model.shared_output), dtype=np.int64),
        gamma_train=np.array(gamma), eta=np.array(config['eta_small_gain']),
        small_gain_W_norm=np.array(norm),
        small_gain_loop_gain=np.array(gamma * norm),
        paper_best_epoch=np.array(paper_info['best_epoch'], dtype=np.int64),
        small_gain_best_epoch=np.array(train_info['best_epoch'], dtype=np.int64),
        train_seed=np.array(config['seed'], dtype=np.int64),
        config_json=np.array(__import__('json').dumps(config, sort_keys=True)),
    )


def train_benchmarks(args) -> Path:
    source_system, _source_graph, _source_W, _activation, _lam = load_checkpoint(args.source_checkpoint)
    if source_system.n != args.agents:
        raise ValueError(
            f'the supplied physical checkpoint has N={source_system.n}; '
            f'this paper GNN uses per-agent output gains, so --agents must be {source_system.n}'
        )
    # A dense, redundant graph is deliberate: the default F=1 Byzantine
    # comparison needs the normal-node induced graph to be 2-robust.
    graph = make_graph(args.agents, args.topology, seed=args.seed, neighborhood=args.neighborhood)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.set_num_threads(max(1, args.threads))

    train_args = SimpleNamespace(
        lam=args.lam, epochs=args.epochs, horizon=args.train_horizon,
        train_horizon=args.train_horizon, batch=args.batch,
        val_batch=args.val_batch, lr=args.lr, eta=args.eta, seed=args.seed,
        device=device, val_every=args.val_every,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta, paper_hidden=args.paper_hidden,
    )

    print(f'Plant: N={source_system.n}, A/B from {args.source_checkpoint}', flush=True)
    print(f'Graph: {args.topology}; edges={graph.adjacency.nnz // 2}; '
          f'full-plant gain={full_gain(source_system, graph).gamma:.6g}', flush=True)
    print('Training small-gain W with projected Adam...', flush=True)
    small_result = train(
        source_system, graph, lam=args.lam, activation='identity',
        epochs=args.epochs, horizon=args.train_horizon, batch=args.batch,
        val_batch=args.val_batch, lr=args.lr, eta=args.eta, seed=args.seed,
        device=device, val_every=args.val_every,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
    )
    print(f"Small-gain W: best epoch={small_result['best_epoch']}, "
          f"||W||2={np.linalg.norm(small_result['best_W'], 2):.6g}, "
          f"gamma*||W||={small_result['gamma_train'] * np.linalg.norm(small_result['best_W'], 2):.6g}",
          flush=True)

    print('Training paper GNN on the same training/validation streams...', flush=True)
    paper_model, paper_info = train_paper_gnn(
        source_system, graph, train_args, seed=args.seed, shared_output=False)
    print(f"Paper GNN: best epoch={paper_info['best_epoch']}, "
          f"H shape={tuple(paper_model.H.shape)}, K shape={tuple(paper_model.K.shape)}",
          flush=True)

    config = {
        'agents': args.agents, 'topology': args.topology,
        'seed': args.seed, 'epochs': args.epochs,
        'train_horizon': args.train_horizon, 'batch': args.batch,
        'val_batch': args.val_batch, 'lr': args.lr, 'lambda': args.lam,
        'eta_small_gain': args.eta, 'paper_hidden': args.paper_hidden,
        'device': device, 'training_streams_matched': True,
        'paper_projection': False,
        'small_gain_projection': '||W||_2 <= eta / gamma_train',
        'objective': 'sum_k ||L_rw X(k)||_F^2 + lambda * ||U_total(k)||_F^2',
        'source_checkpoint': str(Path(args.source_checkpoint).resolve()),
        'paper_best_validation_objective': paper_info['best_validation_objective'],
        'small_gain_best_validation_objective': small_result['best_validation_objective'],
    }
    out = Path(args.out)
    _save_benchmark(out, source_system, graph, small_result['best_W'],
                    paper_model, small_result, paper_info, config)
    out_dir = out.parent
    write_csv(out_dir / 'small_gain_training_history.csv', small_result['history'])
    write_csv(out_dir / 'paper_gnn_training_history.csv', paper_info['history'])
    write_json(out_dir / 'benchmark_training_config.json', config)
    print(f'BENCHMARKS SAVED: {out.resolve()}', flush=True)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source-checkpoint', default=str(DEFAULT_SOURCE_CHECKPOINT))
    p.add_argument('--out', default=str(DEFAULT_BENCHMARK_CHECKPOINT))
    p.add_argument('--agents', type=int, default=10)
    p.add_argument('--topology', choices=('complete', 'wide_ring', 'ring', 'chain', 'star', 'random'), default='complete')
    p.add_argument('--neighborhood', type=int, default=3)
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--train-horizon', type=int, default=100)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--val-batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--eta', type=float, default=0.95)
    p.add_argument('--paper-hidden', type=int, default=2)
    p.add_argument('--seed', type=int, default=16)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--early-stopping-patience', type=int, default=None)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--quick', action='store_true', help='tiny smoke training; not report-quality weights')
    args = p.parse_args(argv)
    if args.quick:
        args.epochs = 3
        args.train_horizon = 8
        args.batch = 4
        args.val_batch = 4
    if args.agents < 3:
        p.error('--agents must be at least 3')
    if args.epochs < 1 or args.train_horizon < 1 or args.batch < 1 or args.val_batch < 1:
        p.error('training dimensions must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    train_benchmarks(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
