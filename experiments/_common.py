from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

# Make direct script execution work from the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from distributed_consensus.control.dynamics import rollout_numpy
from distributed_consensus.control.plant_gain import full_gain, small_gain
from distributed_consensus.control.plants import build_system, generate_system
from distributed_consensus.evaluation.metrics import trajectory_metrics
from distributed_consensus.evaluation.reporting import save_checkpoint, write_json, write_csv
from distributed_consensus.graphs import Graph, make_graph
from distributed_consensus.learning.training import train


TOPOLOGIES = ('chain', 'star', 'ring', 'wide_ring', 'complete', 'random')


def ensure_empty_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path




def write_csv_any(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (row.get(k, '') if row.get(k, None) is not None else '') for k in fields})

def default_er_probability(n: int) -> float:
    return float(min(0.85, max(0.12, 2.5 * np.log(n) / n)))


def er_connected_graph(n: int, p: float | None, seed: int, max_tries: int = 200) -> Graph:
    if n < 2:
        raise ValueError('N must be >= 2')
    if p is None:
        p = default_er_probability(n)
    if not (0.0 < p <= 1.0):
        raise ValueError('ER probability must lie in (0,1]')
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        r, c = np.triu_indices(n, k=1)
        keep = rng.random(len(r)) < p
        rows, cols = r[keep], c[keep]
        data = np.ones(2 * len(rows), dtype=np.float64)
        adj = sparse.coo_matrix((data, (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
        try:
            return Graph(adj)
        except ValueError:
            continue
    raise RuntimeError(f'ER G({n},{p}) failed to yield a connected graph in {max_tries} tries')


def system_at_gain(n: int, graph: Graph, gamma_target: float, seed: int):
    if not np.isfinite(gamma_target) or gamma_target <= 0:
        raise ValueError('target gain must be finite and positive')
    base = generate_system(n, seed=seed)
    gamma_before = full_gain(base, graph).gamma
    scaled = build_system(base.A, base.B * (gamma_target / gamma_before), base.target)
    actual = full_gain(scaled, graph).gamma
    if not np.isclose(actual, gamma_target, rtol=1e-8, atol=1e-10):
        raise AssertionError(f'gain construction mismatch: {actual} vs {gamma_target}')
    return scaled


def system_at_mixed_gain(base, graph: Graph, gamma_target: float, group: np.ndarray):
    """Hold A/base B fixed, apply 0.5/1.5 group ratio, then calibrate full gain."""
    group = np.asarray(group, dtype=np.int64)
    if group.shape != (base.n,) or not np.isin(group, [0,1]).all():
        raise ValueError('invalid stronger/weaker group assignment')
    base_multipliers = np.where(group == 0, 0.5, 1.5)
    intermediate = build_system(base.A, base.B * base_multipliers[:,None,None], base.target)
    common_scale = gamma_target / full_gain(intermediate, graph).gamma
    effective = common_scale * base_multipliers
    system = build_system(base.A, base.B * effective[:,None,None], base.target)
    measured = full_gain(system, graph).gamma
    if not np.isclose(measured, gamma_target, rtol=1e-8, atol=1e-10):
        raise AssertionError('mixed full gain calibration failed')
    return system, effective


def metrics_from_history(history: dict) -> dict[str, np.ndarray]:
    x = np.asarray(history['states'], dtype=np.float64)
    e = np.asarray(history['disagreement'], dtype=np.float64)
    u = np.asarray(history['total'], dtype=np.float64)
    v = np.asarray(history['correction'], dtype=np.float64)
    n = x.shape[1]
    centered = x - x.mean(axis=1, keepdims=True)
    error_agent = np.linalg.norm(e, axis=2)  # (T+1,N)
    return {
        'error_rms': np.sqrt(np.sum(e * e, axis=(1, 2)) / n),
        'centered_rms': np.sqrt(np.sum(centered * centered, axis=(1, 2)) / n),
        'state_rms': np.sqrt(np.sum(x * x, axis=(1, 2)) / n),
        'control_rms': np.sqrt(np.sum(u * u, axis=(1, 2)) / n),
        'correction_rms': np.sqrt(np.sum(v * v, axis=(1, 2)) / n),
        'error_agent': error_agent,
        'state_component_1': x[:, :, 0],
        'state_component_2': x[:, :, 1],
        'control_total': u[:, :, 0],
        'control_correction': v[:, :, 0],
        'cumulative_control_energy': np.concatenate([[0.0], np.cumsum(np.sum(u[:, :, 0] ** 2, axis=1) / n)]),
    }


def settling_time_absolute(norm_curve: np.ndarray, threshold: float) -> int | None:
    values = np.asarray(norm_curve, np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('expected a finite 1-D error curve')
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError('threshold must be nonnegative and finite')
    bad = np.flatnonzero(values > threshold)
    return None if (len(bad) and bad[-1] == len(values) - 1) else (int(bad[-1] + 1) if len(bad) else 0)


def sample_initials(seed: int, samples: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((samples, n, 2))


def paired_rollouts(system, graph, horizon: int, initial_states: np.ndarray, learned_W=None, activation='identity', threshold=1e-3):
    records = []
    representative = None
    baseline_histories = []
    learned_histories = []
    for idx, x0 in enumerate(initial_states):
        base = rollout_numpy(x0, system, graph, None, horizon=horizon)
        learned = rollout_numpy(x0, system, graph, learned_W, activation=activation, horizon=horizon)
        bm = trajectory_metrics(base)
        lm = trajectory_metrics(learned)
        b_curve = metrics_from_history(base)['error_rms']
        l_curve = metrics_from_history(learned)['error_rms']
        b_settle = settling_time_absolute(b_curve, threshold)
        l_settle = settling_time_absolute(l_curve, threshold)
        if representative is None:
            representative = {'baseline': base, 'learned': learned}
        row = {
            'sample': idx,
            'baseline_J_E': bm['J_E'],
            'baseline_J_U': bm['J_U'],
            'baseline_J_pair': bm['J_pair'],
            'baseline_terminal_E': bm['terminal_E'],
            'baseline_settling_time': b_settle,
            'learned_J_E': lm['J_E'],
            'learned_J_U': lm['J_U'],
            'learned_J_pair': lm['J_pair'],
            'learned_terminal_E': lm['terminal_E'],
            'learned_settling_time': l_settle,
            'paired_win_J_E': int(lm['J_E'] < bm['J_E']),
            'baseline_settled': int(b_settle is not None),
            'learned_settled': int(l_settle is not None),
            'baseline_final_error_rms': float(b_curve[-1]),
            'learned_final_error_rms': float(l_curve[-1]),
        }
        records.append(row)
        baseline_histories.append(metrics_from_history(base))
        learned_histories.append(metrics_from_history(learned))
    return records, representative, baseline_histories, learned_histories


def aggregate_method(records: list[dict], prefix: str, threshold_name: str = 'settling_time') -> dict:
    settled = [r[f'{prefix}_{threshold_name}'] for r in records if r[f'{prefix}_{threshold_name}'] is not None]
    return {
        'samples': len(records),
        'mean_J_E': float(np.mean([r[f'{prefix}_J_E'] for r in records])),
        'mean_J_U': float(np.mean([r[f'{prefix}_J_U'] for r in records])),
        'mean_J_pair': float(np.mean([r[f'{prefix}_J_pair'] for r in records])),
        'mean_terminal_E': float(np.mean([r[f'{prefix}_terminal_E'] for r in records])),
        'settled_fraction': float(np.mean([r[f'{prefix}_settled'] for r in records])),
        'median_settling_if_settled': (float(np.median(settled)) if settled else None),
        'q25_settling_if_settled': (float(np.percentile(settled, 25)) if settled else None),
        'q75_settling_if_settled': (float(np.percentile(settled, 75)) if settled else None),
    }


def summarize_time_series(series: list[dict], keys: Iterable[str]) -> list[dict]:
    rows = []
    for metric in keys:
        values = np.stack([s[metric] for s in series])
        q25, med, q75 = np.percentile(values, [25, 50, 75], axis=0)
        mean = values.mean(axis=0)
        for t in range(values.shape[1]):
            rows.append({
                'metric': metric,
                't': t,
                'mean': float(mean[t]),
                'q25': float(q25[t]),
                'median': float(med[t]),
                'q75': float(q75[t]),
            })
    return rows


def save_training_artifacts(folder: Path, history: list[dict]) -> None:
    rows = history
    write_csv(folder / 'training_history.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    epochs = [r['epoch'] for r in rows]
    for ax, label in zip(axes, ('J_E', 'J_U')):
        ax.plot(epochs, [r[f'train_{label}'] for r in rows], label='train')
        vv = [(r['epoch'], r[f'val_{label}']) for r in rows if r[f'val_{label}'] is not None]
        if vv:
            ax.plot([v[0] for v in vv], [v[1] for v in vv], label='validation')
        ax.set(xlabel='Epoch', ylabel=label, title=f'Training {label}')
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(folder / 'training_metrics.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), layout='constrained')
    ax.plot(epochs, [r['W_spectral_norm'] for r in rows], label='post-projection ||W||₂')
    ax.plot(epochs, [r['cap'] for r in rows], '--', label='cap')
    ax.set(xlabel='Epoch', ylabel='Spectral norm', title='Constraint tracking')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(folder / 'training_weight_norm.png', dpi=150)
    plt.close(fig)


def plot_metric_with_band(path: Path, series_map: dict[str, list[dict]], metric: str, threshold: float | None = None,
                          ylabel: str = '', title: str = '') -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), layout='constrained')
    for label, series in series_map.items():
        values = np.stack([s[metric] for s in series])
        q25, med, q75 = np.percentile(values, [25, 50, 75], axis=0)
        if 'error' in metric or 'centered' in metric:
            med = np.maximum(med, 1e-15)
            q25 = np.maximum(q25, 1e-15)
            q75 = np.maximum(q75, 1e-15)
        t = np.arange(values.shape[1])
        ax.plot(t, med, label=label, linewidth=1.8)
        ax.fill_between(t, q25, q75, alpha=0.15)
    if threshold is not None:
        ax.axhline(threshold, color='black', linestyle='--', linewidth=1, label=f'threshold {threshold:g}')
    if 'error' in metric or 'centered' in metric:
        ax.set_yscale('log')
    ax.set(xlabel='Timestep', ylabel=ylabel or metric, title=title or metric)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_panel_agents(ax, values: np.ndarray, label: str, ylabel: str):
    # values shape (T,N)
    n = values.shape[1]
    t = np.arange(values.shape[0])
    if n <= 12:
        for i in range(n):
            ax.plot(t, values[:, i], linewidth=1.4, label=f'agent {i}')
        ax.legend(fontsize=7, ncol=2 if n > 6 else 1)
    else:
        for i in range(n):
            ax.plot(t, values[:, i], linewidth=0.8, alpha=0.28)
        ax.plot(t, values.mean(axis=1), linewidth=2.2, color='black', label='agent mean')
        ax.legend(fontsize=8)
    ax.set(xlabel='Timestep', ylabel=ylabel, title=label)
    ax.grid(alpha=.22)


def plot_representative_agents(path: Path, data_map: dict[str, dict], metric: str, ylabel: str, title: str) -> None:
    labels = list(data_map.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.6 * len(labels), 4.4), squeeze=False, layout='constrained')
    for ax, label in zip(axes[0], labels):
        _plot_panel_agents(ax, np.asarray(data_map[label][metric]), label, ylabel)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cumulative_energy(path: Path, data_map: dict[str, dict], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), layout='constrained')
    for label, values in data_map.items():
        ax.plot(values['cumulative_control_energy'], label=label, linewidth=1.8)
    ax.set(xlabel='Timestep', ylabel='Cumulative control energy per agent', title=title)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_settling_bars(path: Path, rows: list[dict], group_key: str, title: str) -> None:
    labels = [str(r[group_key]) for r in rows]
    values = [r['median_settling_if_settled'] if r['median_settling_if_settled'] is not None else np.nan for r in rows]
    settled = [r['settled_fraction'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    x = np.arange(len(rows))
    axes[0].bar(x, settled)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Settled fraction')
    axes[0].set_title('Fraction reaching and staying below threshold')
    axes[0].grid(axis='y', alpha=.22)

    axes[1].bar(x, values)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel('Median settling timestep')
    axes[1].set_title('Median settling time among settled trajectories')
    axes[1].grid(axis='y', alpha=.22)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def topology_positions(graph: Graph, topology: str | None = None):
    n = graph.n
    if topology == 'chain':
        xs = np.linspace(0, 1, n)
        ys = np.zeros(n)
        return np.column_stack([xs, ys])
    if topology == 'star':
        coords = np.zeros((n, 2))
        coords[0] = np.array([0.0, 0.0])
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)[1:]
        coords[1:, 0] = np.cos(ang)
        coords[1:, 1] = np.sin(ang)
        return coords
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def draw_graph_figure(path: Path, graph: Graph, topology: str) -> None:
    coords = topology_positions(graph, topology)
    rows, cols = graph.edges()
    fig, ax = plt.subplots(figsize=(5.2, 5.2), layout='constrained')
    used = set()
    for i, j in zip(rows, cols):
        if (j, i) in used:
            continue
        used.add((i, j))
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color='0.6', linewidth=1.2)
    ax.scatter(coords[:, 0], coords[:, 1], s=110, color='#1f77b4')
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha='center', va='center', color='white', fontsize=9, weight='bold')
    ax.set_title(f'{topology} topology')
    ax.axis('off')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_trace_npz(path: Path, history_by_method: dict[str, dict], system, graph, initial):
    """One representative physical trajectory per method, plus exact test conditions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'A': np.asarray(system.A), 'B': np.asarray(system.B),
        'K': np.asarray(system.K), 'target': np.asarray(system.target),
        'adjacency': graph.adjacency.toarray(), 'initial': np.asarray(initial),
    }
    for method, history in history_by_method.items():
        for field in ('states', 'disagreement', 'local', 'correction', 'total'):
            data[f'{method}_{field}'] = np.asarray(history[field])
    np.savez_compressed(path, **data)


def plot_settling_grouped(path: Path, rows: list[dict], group_key: str, title: str):
    """Pooled trajectory outcomes, not averages of per-repeat medians."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    labels = list(sorted(groups, key=str))
    frac, median = [], []
    for label in labels:
        items = groups[label]
        times = [r['baseline_settling_time'] if r['method'] == 'baseline'
                 else r['learned_settling_time'] for r in items]
        settled = [float(v) for v in times if v is not None]
        frac.append(len(settled) / len(times))
        median.append(float(np.median(settled)) if settled else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.2 * len(labels)), 4.5), layout='constrained')
    for ax, vals, title0, ylabel in ((axes[0], frac, 'Settled fraction', 'Fraction'),
                                      (axes[1], median, 'Conditional median settling step', 'Timestep')):
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=22, ha='right')
        ax.set(title=title0, ylabel=ylabel)
        ax.grid(axis='y', alpha=.25)
    axes[0].set_ylim(0, 1.05)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def manifest(out: Path, args, experiment_name: str, command: str) -> None:
    data = {
        'experiment': experiment_name,
        'command': command,
        'arguments': vars(args),
        'git_revision': git_revision(),
        'python': sys.version,
        'numpy': np.__version__,
    }
    write_json(out / 'configuration.json', data)


def train_model(system, graph, args, activation: str, seed: int):
    result = train(
        system,
        graph,
        lam=args.lam,
        activation=activation,
        epochs=args.epochs,
        horizon=args.train_horizon,
        batch=args.batch,
        val_batch=args.val_batch,
        lr=args.lr,
        eta=args.eta,
        seed=seed,
        device=args.device,
        val_every=args.val_every,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
    )
    W = result['best_W']
    if W is None:
        raise RuntimeError('training did not produce best-validation weights')
    return result, W


def plot_settling_comparison(path: Path, rows: list[dict], x_key: str, title: str) -> None:
    """Settling fraction and conditional median from the raw paired trajectory outcomes."""
    if not rows:
        return
    labels = sorted(set(str(r[x_key]) for r in rows), key=lambda x: (int(x) if x.isdigit() else x))
    methods = [m for m in ('baseline', 'identity', 'tanh', 'learned') if any(r['method'] == m for r in rows)]
    if not methods:
        return
    x = np.arange(len(labels), dtype=float)
    width = 0.78 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.5*len(labels)+4), 4.8), layout='constrained')
    for j, method in enumerate(methods):
        settled_fraction = []
        median_time = []
        for label in labels:
            values = [r['baseline_settling_time' if method == 'baseline' else 'learned_settling_time']
                      for r in rows if str(r[x_key]) == label and r['method'] == method]
            good = [float(v) for v in values if v is not None]
            settled_fraction.append(float(len(good)/len(values)) if values else np.nan)
            median_time.append(float(np.median(good)) if good else np.nan)
        offset = (j - (len(methods)-1)/2) * width
        axes[0].bar(x+offset, settled_fraction, width=width*.9, label=method)
        axes[1].bar(x+offset, median_time, width=width*.9, label=method)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel(x_key)
        ax.grid(axis='y', alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set(ylabel='Fraction', title='Settled trajectories', ylim=(0, 1.05))
    axes[1].set(ylabel='Timesteps (among settled)', title='Median sustained settling step')
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_numeric_conditions(path: Path, rows: list[dict], xkey: str, title: str, eta: float):
    """Compare full plant gain with actual frozen-weight loop gain (not just gamma)."""
    rows = [r for r in rows if r.get('method') != 'baseline']
    groups = defaultdict(list)
    for r in rows:
        groups[str(r[xkey])].append(r)
    labels = sorted(groups)
    if not labels:
        return
    x = np.arange(len(labels))
    gammas = [np.mean([float(r['gamma_test']) for r in groups[g]]) for g in labels]
    loops = [np.mean([float(r['loop_gain']) for r in groups[g]]) for g in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.3), 4.7), layout='constrained')
    ax.bar(x-0.2, gammas, 0.38, label='gamma_test')
    ax.bar(x+0.2, loops, 0.38, label='gamma_test × ||W||₂')
    ax.axhline(1, color='black', linestyle=':', label='loop-gain boundary 1')
    ax.axhline(eta, color='gray', linestyle='--', label=f'eta={eta:g}')
    ax.set(xticks=x, xticklabels=labels, ylabel='Gain magnitude', title=title)
    ax.tick_params(axis='x', rotation=22)
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
