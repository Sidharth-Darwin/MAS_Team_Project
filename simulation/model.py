"""Matched multi-controller simulation with scalar local W-MSR filtering.

Five policies are run from identical initial physical states:
  * local-only reference;
  * trained paper GNN, raw scalar messages;
  * trained paper GNN, scalar W-MSR messages;
  * trained small-gain controller, raw scalar messages;
  * trained small-gain controller, scalar W-MSR messages.

Each agent broadcasts ONE scalar coordination signal. For the paper GNN it
is H x_i (the benchmark uses one hidden channel); for the small-gain method it
is W^T x_i. The exact local scalar W-MSR algorithm from
src/distributed_consensus/robust/wmsr.py filters these messages once per normal
receiver. The original two-state physical plant is propagated only through
src/distributed_consensus/control/dynamics.py.

The adversary uses a single malicious-broadcast model: it sends the same
extreme false scalar signal to all of its neighbours. This is an F-local
threat model, not Byzantine equivocation and not an actuator attack.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np

from distributed_consensus.control.plants import build_system
from distributed_consensus.control.dynamics import advance_numpy
from distributed_consensus.evaluation.reporting import load_checkpoint
from distributed_consensus.graphs import Graph, make_graph
from distributed_consensus.robust.wmsr import (
    assess_wmsr_topology,
    validate_adversary_set,
    wmsr_local_with_details,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_CHECKPOINT = ROOT / 'assets' / 'topology_rep00_identity.npz'
DEFAULT_BENCHMARK_CHECKPOINT = ROOT / 'assets' / 'robustness_benchmark.npz'
TOPOLOGIES = ('complete', 'wide_ring', 'ring', 'chain', 'star', 'random')

METHODS = (
    'local_only',
    'paper_gnn_raw',
    'paper_gnn_wmsr',
    'small_gain_raw',
    'small_gain_wmsr',
)
METHOD_LABELS = {
    'local_only': 'Local-only reference',
    'paper_gnn_raw': 'Paper GNN (no filter)',
    'paper_gnn_wmsr': 'Paper GNN + W-MSR',
    'small_gain_raw': 'Small-gain (no filter)',
    'small_gain_wmsr': 'Small-gain + W-MSR',
}
METHOD_COLORS = {
    'local_only': '#8d98a7',
    'paper_gnn_raw': '#e28a43',
    'paper_gnn_wmsr': '#e2bd55',
    'small_gain_raw': '#5c91d8',
    'small_gain_wmsr': '#28b69a',
}

DISAGREEMENT_TOL = 1e-3
HOLD_STEPS = 8


@dataclass(frozen=True)
class BenchmarkModels:
    system: object
    training_graph: Graph
    W: np.ndarray
    paper_H: np.ndarray
    paper_K: np.ndarray
    gamma_train: float
    eta: float
    metadata: dict


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_CHECKPOINT) -> BenchmarkModels:
    """Load trained, matched two-state controllers with a one-scalar GNN layer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing trained benchmark checkpoint: {path}\n'
            'Create it with: uv run python -m simulation.train_benchmarks --paper-hidden 1'
        )
    with np.load(path, allow_pickle=False) as z:
        required = {
            'A', 'B', 'K_local', 'target', 'adjacency', 'W', 'paper_H', 'paper_K',
            'gamma_train', 'eta', 'config_json',
        }
        missing = required - set(z.files)
        if missing:
            raise ValueError(f'benchmark checkpoint is missing keys: {sorted(missing)}')
        system = build_system(z['A'], z['B'], z['target'])
        if not np.allclose(system.K, z['K_local'], atol=1e-10, rtol=1e-10):
            raise ValueError('saved local homogenization gains do not match A/B/target')
        graph = Graph(z['adjacency'])
        W = np.asarray(z['W'], dtype=np.float64).copy()
        H = np.asarray(z['paper_H'], dtype=np.float64).copy()
        K_paper = np.asarray(z['paper_K'], dtype=np.float64).copy()
        gamma_train = float(z['gamma_train'])
        eta = float(z['eta'])
        saved_w_norm = float(z['small_gain_W_norm']) if 'small_gain_W_norm' in z else float(np.linalg.norm(W, 2))
        metadata = json.loads(str(z['config_json'].item()))
    n = system.n
    if graph.n != n or W.shape != (2, 1):
        raise ValueError('incompatible benchmark system/graph/W shapes')
    if H.shape != (1, 2) or K_paper.shape != (n, 1):
        raise ValueError(
            'scalar local W-MSR requires a one-hidden-channel paper GNN; '
            'retrain the benchmark with --paper-hidden 1'
        )
    if not all(np.isfinite(a).all() for a in (W, H, K_paper)):
        raise ValueError('benchmark weights contain non-finite values')
    if not np.isfinite(gamma_train) or gamma_train <= 0 or not (0 < eta < 1):
        raise ValueError('invalid saved small-gain parameters')
    if not np.isclose(np.linalg.norm(W, 2), saved_w_norm, rtol=1e-8, atol=1e-10):
        raise ValueError('saved small-gain weight norm metadata is inconsistent')
    return BenchmarkModels(system, graph, W, H, K_paper, gamma_train, eta, metadata)


def initial_conditions(n: int, seed: int = 3) -> np.ndarray:
    """Same initial physical states for every policy; columns are two plant states."""
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(-2.0, 2.0, n), rng.uniform(-0.6, 0.6, n)))


def normal_state_metrics(states: np.ndarray, normal_ids: tuple[int, ...]) -> dict:
    """Disagreement and state magnitude over normal agents only."""
    x = np.asarray(states, dtype=np.float64)
    ids = np.asarray(normal_ids, dtype=int)
    if x.ndim != 2 or x.shape[1] != 2 or len(ids) < 2:
        raise ValueError('states must be (N,2) and at least two normal IDs are required')
    benign = x[ids]
    centered = benign - benign.mean(axis=0, keepdims=True)
    error_rms = float(np.linalg.norm(centered) / np.sqrt(len(benign)))
    pairwise = benign[:, None, :] - benign[None, :, :]
    return {
        'error_rms': error_rms,
        'state_rms': float(np.linalg.norm(benign) / np.sqrt(len(benign))),
        'normal_state_1_spread': float(np.ptp(benign[:, 0])),
        'normal_state_2_spread': float(np.ptp(benign[:, 1])),
        'normal_pairwise_diameter': float(np.max(np.linalg.norm(pairwise, axis=2))),
        'normal_agent_count': int(len(benign)),
    }


def strict_settling_time(error_curve, threshold: float = DISAGREEMENT_TOL):
    """First sample after which every remaining observed error stays <= threshold."""
    values = np.asarray(error_curve, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('error curve must be nonempty, finite and one-dimensional')
    violating = np.flatnonzero(values > threshold)
    if len(violating) == 0:
        return 0
    if violating[-1] == len(values) - 1:
        return None
    return int(violating[-1] + 1)


@dataclass
class ActionRecord:
    scalar_signal: float
    local: float
    correction: float
    total: float
    neighbor_messages: tuple[tuple[int, float], ...]
    retained_ids: tuple[int, ...]
    removed_high_ids: tuple[int, ...]
    removed_low_ids: tuple[int, ...]

    @property
    def removed_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.removed_high_ids) | set(self.removed_low_ids)))


class MultiPolicySimulation:
    """Synchronous comparison under one F-local malicious broadcast attack."""

    def __init__(self, benchmark_path=DEFAULT_BENCHMARK_CHECKPOINT, *,
                 topology='complete', seed=3, adversary_ids=(9,), F=1,
                 threshold=DISAGREEMENT_TOL):
        self.benchmark_path = Path(benchmark_path)
        self.models = load_benchmark(self.benchmark_path)
        self.system = self.models.system
        self.n = self.system.n
        self.seed = int(seed)
        if topology not in TOPOLOGIES:
            raise ValueError(f'topology must be one of {TOPOLOGIES}')
        self.topology = topology
        self.graph = (self.models.training_graph if topology == 'complete'
                      and self.models.metadata.get('topology') == 'complete'
                      else make_graph(self.n, topology, seed=seed + 600, neighborhood=3))
        if isinstance(F, bool) or int(F) != F or F < 0:
            raise ValueError('F must be a nonnegative integer')
        self.F = int(F)
        raw_ids = tuple(adversary_ids)
        if any(isinstance(i, bool) or not np.isfinite(i) or int(i) != i for i in raw_ids):
            raise ValueError('adversary IDs must be finite integer indices')
        self.adversary_ids = validate_adversary_set(
            self.graph, raw_ids, self.F, 'F-local-malicious')
        self.normal_ids = tuple(i for i in range(self.n) if i not in set(self.adversary_ids))
        if len(self.normal_ids) < 2:
            raise ValueError('at least two normal agents are required')
        self.topology_report = assess_wmsr_topology(
            self.graph, self.F, 'F-local-malicious', adversary_ids=self.adversary_ids)
        self.neighbor_map = self._build_neighbor_map()
        self.routes = self.neighbor_map
        self.initial = initial_conditions(self.n, self.seed)
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('threshold must be finite and positive')
        self.reset()

    def _build_neighbor_map(self) -> dict[int, tuple[int, ...]]:
        adj = self.graph.adjacency.tocsr()
        return {
            i: tuple(int(j) for j in adj.indices[adj.indptr[i]:adj.indptr[i + 1]])
            for i in range(self.n)
        }

    def _projection(self, method: str) -> np.ndarray:
        if method.startswith('paper_gnn'):
            return self.models.paper_H[0]
        return self.models.W[:, 0]

    def _messages(self, states: np.ndarray, method: str):
        """Create one scalar feature per sender and inject an extreme broadcast lie."""
        projection = self._projection(method)
        true_signals = states @ projection
        delivered = true_signals.copy()
        attack_value = None
        if self.adversary_ids:
            normal_values = true_signals[np.asarray(self.normal_ids, dtype=int)]
            lower = float(np.min(normal_values))
            upper = float(np.max(normal_values))
            margin = max(1.0, upper - lower)
            # Broadcast just outside the current normal-signal range. The
            # sign alternates so the attack exercises both trimming branches.
            attack_value = upper + margin if self.k % 2 == 0 else lower - margin
            delivered[np.asarray(self.adversary_ids, dtype=int)] = attack_value
        return true_signals, delivered, attack_value

    def _decision(self, method: str, agent_id: int, own_state: np.ndarray,
                  own_signal: float, delivered: np.ndarray,
                  wmsr_details=None) -> ActionRecord:
        local = float(self.system.K[agent_id, 0] @ own_state)
        neighbors = self.neighbor_map[agent_id]
        inbox = tuple((j, float(delivered[j])) for j in neighbors)

        # Compromised senders do not cooperate with the protocol; they only
        # send the fixed extreme scalar broadcast and follow local plant feedback.
        if agent_id in self.adversary_ids or method == 'local_only':
            return ActionRecord(own_signal, local, 0.0, local, inbox, (), (), ())

        use_wmsr = method.endswith('_wmsr')
        if use_wmsr:
            retained = wmsr_details.retained_ids[agent_id]
            removed_high = wmsr_details.removed_high_ids[agent_id]
            removed_low = wmsr_details.removed_low_ids[agent_id]
            retained_count = len(retained)
            updated_signal = float(wmsr_details.values[agent_id])
            # The W-MSR update is the mean of [own value] + retained neighbours.
            # Recover the retained-neighbour average for the controller's
            # original Laplacian/neighbor-average operation.
            if retained_count:
                retained_average = (
                    (retained_count + 1) * updated_signal - own_signal
                ) / retained_count
                average_error = own_signal - retained_average
                retained_laplacian = (retained_count + 1) * (own_signal - updated_signal)
            else:
                average_error = 0.0
                retained_laplacian = 0.0
        else:
            retained = neighbors
            removed_high = ()
            removed_low = ()
            retained_count = len(neighbors)
            neighbor_mean = float(np.mean([delivered[j] for j in neighbors])) if neighbors else own_signal
            average_error = own_signal - neighbor_mean
            retained_laplacian = float(sum(own_signal - delivered[j] for j in neighbors))

        if method.startswith('small_gain'):
            # W^T e_i is exactly the scalar disagreement in the W projection.
            correction = float(average_error)
        elif method.startswith('paper_gnn'):
            # For one hidden channel, H sum_j(x_i-x_j) = sum_j(Hx_i-Hx_j).
            correction = float(self.models.paper_K[agent_id, 0] * np.tanh(retained_laplacian))
        else:
            raise ValueError(f'unknown controller mode: {method}')

        return ActionRecord(
            own_signal, local, correction, local + correction, inbox,
            tuple(retained), tuple(removed_high), tuple(removed_low),
        )

    def reset(self):
        self.states = {name: self.initial.copy() for name in METHODS}
        self.k = 0
        self.state_histories = {name: [self.initial.copy()] for name in METHODS}
        self.error_histories = {
            name: [normal_state_metrics(self.initial, self.normal_ids)['error_rms']]
            for name in METHODS
        }
        self.action_histories = {name: [] for name in METHODS}
        self.energy = {name: 0.0 for name in METHODS}
        self.disagreement_cost = {name: 0.0 for name in METHODS}
        self.filtered_count = {name: 0 for name in METHODS}
        self.filtered_history = {name: [0] for name in METHODS}
        self.last_attack_value = {name: None for name in METHODS}
        self.hold_count = {name: 0 for name in METHODS}
        self.hold_step = {name: None for name in METHODS}

    def step(self):
        outputs = {}
        next_states_by_method = {}
        for method in METHODS:
            snapshot = self.states[method].copy()
            true_signals, delivered, attack_value = self._messages(snapshot, method)
            wmsr_details = None
            if method.endswith('_wmsr'):
                wmsr_details = wmsr_local_with_details(delivered, self.neighbor_map, self.F)
            decisions = [
                self._decision(method, i, snapshot[i], float(true_signals[i]),
                               delivered, wmsr_details)
                for i in range(self.n)
            ]
            local = np.asarray([[d.local] for d in decisions], dtype=np.float64)
            correction = np.asarray([[d.correction] for d in decisions], dtype=np.float64)
            total = local + correction
            # One authoritative physical update from src. The compromised
            # agents keep their normal local action; only the communicated
            # scalar signal is malicious in this demonstration.
            next_states = advance_numpy(snapshot, self.system, total)
            next_states_by_method[method] = next_states

            removed_ids = set()
            receiver_sender_pairs = set()
            for receiver, action in enumerate(decisions):
                if receiver in self.adversary_ids:
                    continue
                for sender in action.removed_ids:
                    removed_ids.add(sender)
                    receiver_sender_pairs.add((receiver, sender))
            outputs[method] = {
                'local': local,
                'correction': correction,
                'total': total,
                'actions': decisions,
                'delivered_signals': delivered.copy(),
                'attack_value': attack_value,
                'filtered_values': sum(len(d.removed_ids) for i, d in enumerate(decisions)
                                       if i not in self.adversary_ids),
                'filtered_sender_ids': tuple(sorted(removed_ids)),
                'filtered_receiver_sender_pairs': tuple(sorted(receiver_sender_pairs)),
            }

        self.states = next_states_by_method
        self.k += 1
        for method in METHODS:
            out = outputs[method]
            before = normal_state_metrics(self.state_histories[method][-1], self.normal_ids)
            after = normal_state_metrics(self.states[method], self.normal_ids)
            self.disagreement_cost[method] += before['error_rms'] ** 2 * len(self.normal_ids)
            normal_total = out['total'][np.asarray(self.normal_ids), 0]
            self.energy[method] += float(np.sum(normal_total ** 2))
            self.filtered_count[method] += int(out['filtered_values'])
            self.last_attack_value[method] = out['attack_value']
            self.hold_count[method] = self.hold_count[method] + 1 if after['error_rms'] <= self.threshold else 0
            if self.hold_count[method] >= HOLD_STEPS and self.hold_step[method] is None:
                self.hold_step[method] = self.k - HOLD_STEPS + 1
            self.state_histories[method].append(self.states[method].copy())
            self.error_histories[method].append(after['error_rms'])
            self.action_histories[method].append(out)
            self.filtered_history[method].append(int(out['filtered_values']))
        return outputs

    def latest(self, method):
        if method not in METHODS:
            raise KeyError(method)
        states = self.states[method]
        return {
            **normal_state_metrics(states, self.normal_ids),
            'step': self.k,
            'settled_8step': self.hold_step[method],
            'strict_settling_time': strict_settling_time(self.error_histories[method], self.threshold),
            'energy': self.energy[method],
            'disagreement_cost': self.disagreement_cost[method],
            'filtered_values_this_step': int(self.filtered_history[method][-1]) if self.k else 0,
            'filtered_values_total': int(self.filtered_count[method]),
            'filtered_sender_ids_this_step': (
                self.action_histories[method][-1]['filtered_sender_ids'] if self.k else ()
            ),
            'attack_value': self.last_attack_value[method],
            'states': states.copy(),
            'adversary_ids': self.adversary_ids,
            'normal_ids': self.normal_ids,
            'method': method,
            'topology_report': self.topology_report,
        }

    def export_csv(self, output_path: str | Path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = (
            'step', 'method', 'agent', 'is_adversary', 'state_1', 'state_2',
            'scalar_signal', 'local_input', 'learned_correction', 'total_commanded_input',
            'received_scalar_messages', 'retained_neighbor_ids',
            'removed_high_neighbor_ids', 'removed_low_neighbor_ids',
            'normal_disagreement_rms', 'normal_state_rms',
            'cumulative_control_energy', 'cumulative_disagreement_cost',
            'attack_model', 'broadcast_value', 'F', 'filtered_values_this_step',
        )
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for method in METHODS:
                for k, output in enumerate(self.action_histories[method]):
                    states = self.state_histories[method][k]
                    metrics = normal_state_metrics(states, self.normal_ids)
                    for i, action in enumerate(output['actions']):
                        inbox = [{'sender': sender + 1, 'value': value}
                                 for sender, value in action.neighbor_messages]
                        writer.writerow({
                            'step': k, 'method': method, 'agent': i + 1,
                            'is_adversary': int(i in self.adversary_ids),
                            'state_1': states[i, 0], 'state_2': states[i, 1],
                            'scalar_signal': action.scalar_signal,
                            'local_input': action.local,
                            'learned_correction': action.correction,
                            'total_commanded_input': action.total,
                            'received_scalar_messages': json.dumps(inbox, separators=(',', ':')),
                            'retained_neighbor_ids': ';'.join(str(v + 1) for v in action.retained_ids),
                            'removed_high_neighbor_ids': ';'.join(str(v + 1) for v in action.removed_high_ids),
                            'removed_low_neighbor_ids': ';'.join(str(v + 1) for v in action.removed_low_ids),
                            'normal_disagreement_rms': metrics['error_rms'],
                            'normal_state_rms': metrics['state_rms'],
                            'cumulative_control_energy': self.energy[method],
                            'cumulative_disagreement_cost': self.disagreement_cost[method],
                            'attack_model': 'F-local malicious broadcast',
                            'broadcast_value': output['attack_value'] if output['attack_value'] is not None else '',
                            'F': self.F,
                            'filtered_values_this_step': output['filtered_values'],
                        })
        return path

    def save_figures(self, output_dir: str | Path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        t = np.arange(self.k + 1)

        fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
        for method in METHODS:
            y = np.maximum(np.asarray(self.error_histories[method]), 1e-12)
            ax.plot(t, y, label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=2)
        ax.axhline(self.threshold, color='black', linestyle='--', linewidth=1.2,
                   label=f'settling threshold {self.threshold:g}')
        ax.set_yscale('log')
        ax.set(xlabel='Discrete-time step k', ylabel='Normal-agent disagreement RMS',
               title='Matched controllers under a malicious scalar broadcast')
        ax.grid(True, which='both', alpha=.22)
        ax.legend(fontsize=8)
        fig.savefig(root / 'normal_disagreement_all_controllers.png', dpi=170)
        plt.close(fig)

        for component, label in ((0, 'state_1'), (1, 'state_2')):
            fig, axes = plt.subplots(len(METHODS), 1, figsize=(10, 2.0 * len(METHODS)),
                                     sharex=True, layout='constrained')
            for ax, method in zip(axes, METHODS):
                x = np.stack(self.state_histories[method])[:, self.normal_ids, component]
                for j, agent_id in enumerate(self.normal_ids):
                    ax.plot(t, x[:, j], linewidth=.9, alpha=.75, label=f'agent {agent_id + 1}')
                ax.set_ylabel(label.replace('_', ' '))
                ax.set_title(METHOD_LABELS[method], loc='left', fontsize=9)
                ax.grid(alpha=.2)
            axes[-1].set_xlabel('Discrete-time step k')
            fig.savefig(root / f'normal_{label}_by_controller.png', dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(len(METHODS), 1, figsize=(10, 2.0 * len(METHODS)),
                                 sharex=True, layout='constrained')
        for ax, method in zip(axes, METHODS):
            u = np.stack([o['total'][:, 0] for o in self.action_histories[method]])
            for agent_id in self.normal_ids:
                ax.plot(np.arange(self.k), u[:, agent_id], linewidth=.9, alpha=.75,
                        label=f'agent {agent_id + 1}')
            ax.set_ylabel('input')
            ax.set_title(METHOD_LABELS[method], loc='left', fontsize=9)
            ax.grid(alpha=.2)
        axes[-1].set_xlabel('Discrete-time step k')
        fig.savefig(root / 'normal_control_by_controller.png', dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
        for method in ('paper_gnn_wmsr', 'small_gain_wmsr'):
            ax.plot(np.arange(self.k + 1), self.filtered_history[method],
                    label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=1.8)
        ax.set(xlabel='Discrete-time step k', ylabel='Scalar neighbour values discarded',
               title=f'W-MSR local filtering activity (F={self.F})')
        ax.grid(alpha=.22)
        ax.legend()
        fig.savefig(root / 'wmsr_discarded_values.png', dpi=160)
        plt.close(fig)
        return root

    def summary_rows(self):
        rows = []
        for method in METHODS:
            result = self.latest(method)
            rows.append({
                'method': method,
                'method_label': METHOD_LABELS[method],
                'settled': int(result['strict_settling_time'] is not None),
                'strict_settling_step': result['strict_settling_time'],
                '8step_hold_first': result['settled_8step'],
                'final_normal_disagreement_rms': result['error_rms'],
                'final_normal_state_rms': result['state_rms'],
                'cumulative_disagreement_cost': result['disagreement_cost'],
                'cumulative_physical_control_energy': result['energy'],
                'scalar_values_discarded_total': result['filtered_values_total'],
            })
        return rows
