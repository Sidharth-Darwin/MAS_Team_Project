"""Synchronous, neighbor-only control simulator for the existing MAS library.

State is (position deviation, velocity); each drawn column is an independent
1-DOF actuator. No rigid-platform forces, position clamps, or fabricated motion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

from distributed_consensus.control.plants import generate_system
from distributed_consensus.evaluation.reporting import load_checkpoint
from distributed_consensus.graphs import make_graph

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / 'assets' / 'topology_rep00_identity.npz'
TOPOLOGIES = ('chain', 'star', 'ring', 'wide_ring', 'complete', 'random')

# Operational conditions specified independently of observed performance.
POSITION_TOL = 0.05
VELOCITY_TOL = 0.05
SPREAD_TOL = 0.07
HOLD_STEPS = 8
DISAGREEMENT_TOL = 1e-3
DT = 0.1  # Model's discrete position integration coefficient, not wall-clock duration.


@dataclass
class ControlDecision:
    disagreement: np.ndarray  # (2,)
    local: float
    correction: float
    total: float


class LocalAgent:
    """One physical agent; controller receives ONLY self-state and its inbox."""
    def __init__(self, A, B, K, W, activation, learned):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64).reshape(2)
        self.K = np.asarray(K, dtype=np.float64).reshape(2)
        self.W = np.asarray(W, dtype=np.float64).reshape(2)
        self.activation = activation
        self.learned = learned

    def decide(self, own_state, inbox):
        """inbox entries are (neighbor_state, edge_weight), not global X."""
        own = np.asarray(own_state, dtype=np.float64)
        weighted_sum = np.zeros(2, dtype=np.float64)
        degree = 0.0
        for neighbor_state, weight in inbox:
            weighted_sum += weight * neighbor_state
            degree += weight
        if degree <= 0:
            raise ValueError('Agent has no neighbors')
        error = own - weighted_sum / degree
        local = float(self.K @ own)
        correction = 0.0
        if self.learned:
            value = float(self.W @ error)
            correction = value if self.activation == 'identity' else float(np.tanh(value))
        return ControlDecision(error, local, correction, local + correction)

    def move(self, own_state, decision, disturbance):
        """x+ = A_i x + B_i (u_i + disturbance_i). No clipping/saturation."""
        return self.A @ own_state + self.B * (decision.total + disturbance)


class LocalWorld:
    """Synchronous snapshot -> neighbor-only communication -> simultaneous commit."""
    def __init__(self, system, graph, W, activation, learned, initial):
        if system.n != graph.n:
            raise ValueError('system and graph sizes differ')
        if activation not in ('identity', 'tanh'):
            raise ValueError('unknown activation')
        self.system, self.graph, self.W, self.activation = system, graph, W, activation
        self.learned = learned
        self.agents = [LocalAgent(system.A[i], system.B[i], system.K[i], W, activation, learned)
                       for i in range(system.n)]
        adj = graph.adjacency.tocsr()
        # Static routing table: a row lists ONLY the neighbor IDs and weights.
        self.routes = [tuple((int(adj.indices[j]), float(adj.data[j]))
                             for j in range(adj.indptr[i], adj.indptr[i + 1]))
                       for i in range(system.n)]
        self.states = np.array(initial, dtype=np.float64, copy=True)
        if self.states.shape != (system.n, 2) or not np.isfinite(self.states).all():
            raise ValueError('initial shape must be (N,2) and finite')

    def step(self, disturbance):
        d = np.asarray(disturbance, dtype=np.float64)
        if d.shape != (self.system.n, 1) or not np.isfinite(d).all():
            raise ValueError('disturbance must have shape (N,1) and be finite')
        # A read-only communication snapshot. We never pass it to agents.
        snapshot = self.states.copy()
        # The inbox contains *only* neighbor state broadcasts (copied).
        messages = [tuple((snapshot[j].copy(), weight) for j, weight in route)
                    for route in self.routes]
        decisions = [agent.decide(snapshot[i].copy(), messages[i])
                     for i, agent in enumerate(self.agents)]
        next_states = np.stack([agent.move(snapshot[i], decisions[i], float(d[i, 0]))
                                for i, agent in enumerate(self.agents)])
        self.states = next_states  # simultaneous barrier: commit after all decisions
        return {
            'disagreement': np.stack([decision.disagreement for decision in decisions]),
            'local': np.array([[decision.local] for decision in decisions]),
            'correction': np.array([[decision.correction] for decision in decisions]),
            'total': np.array([[decision.total] for decision in decisions]),
            'next_states': next_states.copy(),
        }


def get_system(checkpoint=DEFAULT_CHECKPOINT, n=10, topology='chain', seed=3):
    """N=10 uses exact saved system; other N uses existing generator, never retrains W."""
    saved_system, saved_graph, W, activation, _ = load_checkpoint(checkpoint)
    if n == saved_system.n:
        system = saved_system
    else:
        system = generate_system(n, target=saved_system.target, seed=seed + 400)
    if topology == 'checkpoint' and n == saved_system.n:
        graph = saved_graph
    else:
        graph = make_graph(n, topology='chain' if topology == 'checkpoint' else topology,
                           seed=seed + 600)
    return system, graph, W, activation


def initial_conditions(n, seed=3):
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(-2., 2., n), rng.uniform(-.6, .6, n)))


def state_metrics(states, graph):
    x = np.asarray(states, dtype=np.float64)
    e = graph.error(x)  # Global metrics are reporting only, NOT agent control.
    n = len(x)
    return {
        'error_rms': float(np.linalg.norm(e) / np.sqrt(n)),
        'state_rms': float(np.linalg.norm(x) / np.sqrt(n)),
        'position_max': float(np.max(np.abs(x[:, 0]))),
        'velocity_max': float(np.max(np.abs(x[:, 1]))),
        'height_spread': float(np.ptp(x[:, 0])),
        'ready_now': bool(np.max(np.abs(x[:, 0])) <= POSITION_TOL and
                          np.max(np.abs(x[:, 1])) <= VELOCITY_TOL and
                          np.ptp(x[:, 0]) <= SPREAD_TOL),
    }


class DualSimulation:
    """Both worlds advance from the same initial states with the same perturbations."""
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, n=10, topology='checkpoint', seed=3):
        self.checkpoint = Path(checkpoint)
        self.n, self.topology, self.seed = int(n), topology, int(seed)
        self.system, self.graph, self.W, self.activation = get_system(
            self.checkpoint, self.n, self.topology, self.seed)
        self.initial = initial_conditions(self.n, self.seed)
        self.reset()

    def reset(self):
        self.worlds = {
            'baseline': LocalWorld(self.system, self.graph, self.W, self.activation, False, self.initial),
            'learned': LocalWorld(self.system, self.graph, self.W, self.activation, True, self.initial),
        }
        self.k = 0
        self.events = []
        self.pulses = []  # (agent_index, amplitude, remaining_steps)
        self.segment_start = 0
        self.segments = 0
        self.energy = {name: 0. for name in self.worlds}
        self.je = {name: 0. for name in self.worlds}
        self.hold = {name: 0 for name in self.worlds}
        self.ready_step = {name: None for name in self.worlds}
        self.error_hold = {name: 0 for name in self.worlds}
        self.error_step = {name: None for name in self.worlds}
        self.histories = {name: [self.worlds[name].states.copy()] for name in self.worlds}
        self.error_histories = {name: [state_metrics(self.worlds[name].states, self.graph)['error_rms']]
                                for name in self.worlds}
        self.actions = {name: [] for name in self.worlds}
        self.disturbance_history = []

    def _new_intervention(self, label):
        self.segment_start = self.k
        self.segments += 1
        self.events.append((self.k, label))
        for name in self.worlds:
            self.ready_step[name] = None
            self.hold[name] = 0
            self.error_step[name] = None
            self.error_hold[name] = 0

    def drag(self, agent_index, position_delta):
        if not 0 <= agent_index < self.n or not np.isfinite(position_delta):
            raise ValueError('invalid drag')
        if abs(position_delta) < 1e-12:
            return
        for name, world in self.worlds.items():
            world.states[agent_index, 0] += position_delta
            self.histories[name][-1] = world.states.copy()
            self.error_histories[name][-1] = state_metrics(world.states, self.graph)['error_rms']
        self._new_intervention(f'drag agent {agent_index + 1}: {position_delta:+.2f}')

    def pulse(self, agent_index, amplitude=8., duration=3):
        if not 0 <= agent_index < self.n or not np.isfinite(amplitude) or duration < 1:
            raise ValueError('invalid pulse')
        self.pulses.append((int(agent_index), float(amplitude), int(duration)))
        self._new_intervention(f'pulse agent {agent_index + 1}: {amplitude:+.1f} x {duration}')

    def step(self):
        d = np.zeros((self.n, 1), dtype=np.float64)
        updated = []
        for i, amplitude, left in self.pulses:
            d[i, 0] += amplitude
            if left > 1:
                updated.append((i, amplitude, left - 1))
        self.pulses = updated
        self.disturbance_history.append(d.copy())
        for name, world in self.worlds.items():
            current_e = self.graph.error(world.states)
            output = world.step(d)
            self.je[name] += float(np.sum(current_e ** 2))
            self.energy[name] += float(np.sum(output['total'] ** 2))
            self.actions[name].append(output)
            self.histories[name].append(world.states.copy())
            self.error_histories[name].append(state_metrics(world.states, self.graph)['error_rms'])
        self.k += 1
        for name, world in self.worlds.items():
            metrics = state_metrics(world.states, self.graph)
            self.hold[name] = (self.hold[name] + 1) if metrics['ready_now'] else 0
            self.error_hold[name] = (self.error_hold[name] + 1) if (
                metrics['error_rms'] <= DISAGREEMENT_TOL) else 0
            if not metrics['ready_now']:
                self.ready_step[name] = None
            if metrics['error_rms'] > DISAGREEMENT_TOL:
                self.error_step[name] = None
            if self.hold[name] >= HOLD_STEPS and self.ready_step[name] is None:
                self.ready_step[name] = self.k - HOLD_STEPS + 1
            if self.error_hold[name] >= HOLD_STEPS and self.error_step[name] is None:
                self.error_step[name] = self.k - HOLD_STEPS + 1
        return d.copy()

    def latest(self, name):
        world = self.worlds[name]
        metrics = state_metrics(world.states, self.graph)
        return dict(metrics, step=self.k, ready_step=self.ready_step[name],
                    error_step=self.error_step[name], energy=self.energy[name],
                    disagreement_cost=self.je[name], states=world.states.copy(),
                    last_action=self.actions[name][-1] if self.actions[name] else None)

    def export_csv(self, output_path):
        """Signed per-agent state/feedback/action and matched injected disturbances."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ('k', 'time', 'segment', 'mode', 'agent', 'position', 'velocity',
                'error_position', 'error_velocity', 'local_input', 'correction_input',
                'physical_input', 'disturbance_input', 'next_position', 'next_velocity',
                'position_jump_after_step', 'events_at_step')
        event_steps = {k for k, _ in self.events}
        event_messages = {}
        for event_k, description in self.events:
            event_messages.setdefault(event_k, []).append(description)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for k, disturbance in enumerate(self.disturbance_history):
                for mode in ('baseline', 'learned'):
                    x = self.histories[mode][k]
                    nxt = self.histories[mode][k + 1]
                    output = self.actions[mode][k]
                    for i in range(self.n):
                        writer.writerow(dict(k=k, time=k * DT,
                          segment=sum(1 for event_k in event_steps if event_k <= k),
                          mode=mode, agent=i + 1,
                          position=x[i, 0], velocity=x[i, 1],
                          error_position=output['disagreement'][i, 0],
                          error_velocity=output['disagreement'][i, 1],
                          local_input=output['local'][i, 0],
                          correction_input=output['correction'][i, 0],
                          physical_input=output['total'][i, 0],
                          disturbance_input=disturbance[i, 0],
                          next_position=output['next_states'][i, 0],
                          next_velocity=output['next_states'][i, 1],
                          position_jump_after_step=nxt[i, 0] - output['next_states'][i, 0],
                          events_at_step='; '.join(event_messages.get(k, []))))
        return path
