"""Scalar Weighted-Mean-Subsequence-Reduced (W-MSR) update.

This module implements the local scalar algorithm used by the simulation:
for each normal node, sort its scalar neighbour values, discard up to F
values strictly above its own value and up to F strictly below, then average
its own value with the retained neighbours.

The physical simulation now sends one scalar coordination signal per agent,
so W-MSR is applied once per receiver (not separately to multiple plant-state
coordinates).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WMSRLocalResult:
    """Full result and audit trail for one simultaneous scalar W-MSR update."""
    values: np.ndarray
    retained_ids: tuple[tuple[int, ...], ...]
    removed_high_ids: tuple[tuple[int, ...], ...]
    removed_low_ids: tuple[tuple[int, ...], ...]

    @property
    def removed_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(sorted(set(high) | set(low)))
            for high, low in zip(self.removed_high_ids, self.removed_low_ids)
        )


def _validate_inputs(X, N: Mapping[int, Iterable[int]], F: int):
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError('X must be a finite one-dimensional vector with at least two nodes')
    if isinstance(F, (bool, np.bool_)) or int(F) != F or F < 0:
        raise ValueError('F must be a nonnegative integer')
    if not isinstance(N, Mapping):
        raise TypeError('N must be a dictionary mapping each node to its neighbour IDs')
    n = values.size
    neighbors = []
    for node in range(n):
        if node not in N:
            raise ValueError(f'N is missing neighbour list for node {node}')
        ids = tuple(int(j) for j in N[node])
        if len(ids) != len(set(ids)):
            raise ValueError(f'node {node} has duplicate neighbour IDs')
        if any(j < 0 or j >= n or j == node for j in ids):
            raise ValueError(f'node {node} has an invalid neighbour ID')
        neighbors.append(ids)
    return values, tuple(neighbors), int(F)


def wmsr_local_with_details(X, N: Mapping[int, Iterable[int]], F: int = 1) -> WMSRLocalResult:
    """Apply the user's local scalar W-MSR algorithm simultaneously.

    For every node i:
      1. Separate neighbour values below, equal to, and above X[i].
      2. Remove the F smallest values below X[i].
      3. Remove the F largest values above X[i].
      4. Average X[i] together with every retained neighbour value.

    Ties equal to the receiver's own value are retained. If fewer than F
    values exist on one side, all values on that side are removed. No weighted
    averaging or vector-coordinate filtering is used.

    Node indices are zero-based. ``N`` maps each node index to a sequence of
    neighbour indices. The returned values are a simultaneous update: every
    result is calculated from the original X, not from partially updated nodes.
    """
    values, neighbors, F = _validate_inputs(X, N, F)
    updated = values.copy()
    retained_all, removed_high_all, removed_low_all = [], [], []

    for node, ids in enumerate(neighbors):
        # Stable sorting makes equal-valued neighbour ordering deterministic.
        ordered = sorted(ids, key=lambda j: (values[j], j))
        lower = [j for j in ordered if values[j] < values[node]]
        middle = [j for j in ordered if values[j] == values[node]]
        upper = [j for j in ordered if values[j] > values[node]]

        # Match the supplied pseudocode literally:
        # lower = lower[F:] removes the F smallest below-own values;
        # upper = upper[:-F] removes the F largest above-own values.
        removed_low = lower[:F]
        removed_high = upper[-F:] if F > 0 else []
        retained_lower = lower[F:]
        retained_upper = upper[:-F] if F > 0 else upper
        retained = retained_lower + middle + retained_upper

        # Exactly: mean(concat(([X[node]], filtered_neighbors_values))).
        if retained:
            updated[node] = float(np.mean(np.concatenate(([values[node]], values[retained]))))
        else:
            updated[node] = values[node]

        retained_all.append(tuple(retained))
        removed_high_all.append(tuple(sorted(removed_high)))
        removed_low_all.append(tuple(sorted(removed_low)))

    return WMSRLocalResult(
        values=updated,
        retained_ids=tuple(retained_all),
        removed_high_ids=tuple(removed_high_all),
        removed_low_ids=tuple(removed_low_all),
    )


def wmsr_local(X, N: Mapping[int, Iterable[int]], F: int = 1) -> np.ndarray:
    """Return only X_new from the local scalar W-MSR algorithm.

    This matches the familiar ``wmsr_local(X, N, F=1)`` interface. Use
    ``wmsr_local_with_details`` only when an audit of removed neighbour IDs is
    required by a visualizer or test.
    """
    return wmsr_local_with_details(X, N, F).values


def _adjacency_neighbor_masks(graph) -> tuple[int, ...]:
    """Return Python-int neighbor masks from the project's Graph or adjacency."""
    adjacency = getattr(graph, 'adjacency', graph)
    a = adjacency.tocsr() if hasattr(adjacency, 'tocsr') else np.asarray(adjacency)
    n = int(a.shape[0])
    if len(a.shape) != 2 or a.shape[1] != n or n < 2:
        raise ValueError('graph adjacency must be square with N>=2')
    masks = []
    for i in range(n):
        if hasattr(a, 'indices') and hasattr(a, 'indptr'):
            neighbors = a.indices[a.indptr[i]:a.indptr[i + 1]]
        else:
            neighbors = np.flatnonzero(np.asarray(a[i]) != 0)
        mask = 0
        for j in neighbors:
            if int(j) != i:
                mask |= 1 << int(j)
        masks.append(mask)
    return tuple(masks)


def _reachable_counts(neighbor_masks: tuple[int, ...], r: int) -> tuple[list[int], list[int]]:
    n = len(neighbor_masks)
    full = (1 << n) - 1
    sizes = [0] * (1 << n)
    counts = [0] * (1 << n)
    for mask in range(1, 1 << n):
        sizes[mask] = mask.bit_count()
        outside = full ^ mask
        remaining = mask
        reachable = 0
        while remaining:
            bit = remaining & -remaining
            i = bit.bit_length() - 1
            if (neighbor_masks[i] & outside).bit_count() >= r:
                reachable += 1
            remaining ^= bit
        counts[mask] = reachable
    return sizes, counts


def is_r_robust(graph, r: int, *, max_nodes: int = 14) -> bool | None:
    """Exactly test r-robustness for small graphs; return None when too large.

    A graph is r-robust if, for every pair of nonempty disjoint node sets,
    at least one set contains a node with at least r neighbors outside it.
    The exact check costs O(3**N), so it is deliberately limited to small N.
    """
    if isinstance(r, bool) or int(r) != r or r < 1:
        raise ValueError('r must be a positive integer')
    r = int(r)
    masks = _adjacency_neighbor_masks(graph)
    n = len(masks)
    if n > max_nodes:
        return None
    sizes, counts = _reachable_counts(masks, r)
    full = (1 << n) - 1
    for s1 in range(1, full):
        remaining = full ^ s1
        s2 = remaining
        while s2:
            if s1 < s2 and counts[s1] == 0 and counts[s2] == 0:
                return False
            s2 = (s2 - 1) & remaining
    return True


def is_rs_robust(graph, r: int, s: int, *, max_nodes: int = 14) -> bool | None:
    """Exactly test (r,s)-robustness for small graphs; None above max_nodes."""
    if isinstance(r, bool) or int(r) != r or r < 1:
        raise ValueError('r must be a positive integer')
    if isinstance(s, bool) or int(s) != s or s < 1:
        raise ValueError('s must be a positive integer')
    r, s = int(r), int(s)
    masks = _adjacency_neighbor_masks(graph)
    n = len(masks)
    if s > n:
        raise ValueError('s cannot exceed the number of nodes')
    if n > max_nodes:
        return None
    sizes, counts = _reachable_counts(masks, r)
    full = (1 << n) - 1
    for s1 in range(1, full):
        remaining = full ^ s1
        s2 = remaining
        while s2:
            if s1 < s2:
                if not (
                    counts[s1] == sizes[s1]
                    or counts[s2] == sizes[s2]
                    or counts[s1] + counts[s2] >= s
                ):
                    return False
            s2 = (s2 - 1) & remaining
    return True


def assess_wmsr_topology(graph, F: int, threat_model: str = 'F-local-malicious', *,
                         adversary_ids: Iterable[int] = (), max_nodes: int = 14) -> dict:
    """Report textbook W-MSR topology condition; this is not a MAS plant proof.

    The graph argument is the communication graph. For Byzantine variants,
    this helper checks the induced graph of normal nodes, as required by the
    cited fixed-topology result. For malicious variants it checks the full graph.

    * F-total-malicious: (F+1,F+1)-robust (necessary and sufficient).
    * F-local-malicious: (2F+1)-robust is sufficient; (F+1)-robust is necessary.
    * F-total/local Byzantine: the normal-node induced graph must be (F+1)-robust.

    Exact robustness is checked only up to ``max_nodes``. For larger graphs,
    ``condition_met`` is None; this helper does not mistake connectivity for
    the stronger robustness property.
    """
    if isinstance(F, bool) or int(F) != F or F < 0:
        raise ValueError('F must be a nonnegative integer')
    F = int(F)
    supported = {
        'F-total-malicious': ('rs', F + 1, F + 1,
                              'necessary and sufficient: (F+1,F+1)-robust'),
        'F-local-malicious': ('r', 2 * F + 1, None,
                              'sufficient: (2F+1)-robust; necessary: (F+1)-robust'),
        'F-total-byzantine': ('normal-r', F + 1, None,
                              'normal-node induced graph must be (F+1)-robust'),
        'F-local-byzantine': ('normal-r', F + 1, None,
                              'normal-node induced graph must be (F+1)-robust'),
    }
    if threat_model not in supported:
        raise ValueError(f'threat_model must be one of {tuple(supported)}')
    kind, r, s, condition = supported[threat_model]
    full_n = int(graph.n if hasattr(graph, 'n') else len(_adjacency_neighbor_masks(graph)))
    checked_graph = graph
    checked_ids = tuple(sorted(set(int(i) for i in adversary_ids)))
    if kind == 'normal-r':
        if any(i < 0 or i >= full_n for i in checked_ids):
            raise ValueError('adversary IDs must be valid graph nodes')
        normal_ids = [i for i in range(full_n) if i not in set(checked_ids)]
        if len(normal_ids) < 2:
            return {
                'F': F, 'threat_model': threat_model, 'N': full_n,
                'normal_N': len(normal_ids), 'required_robustness': f'{r}-robust',
                'condition': condition, 'condition_met': False, 'exact_check': True,
                'max_nodes_for_exact_check': int(max_nodes),
                'interpretation': 'Fewer than two normal agents; no normal-network W-MSR consensus claim.',
            }
        adjacency = getattr(graph, 'adjacency', graph)
        checked_graph = adjacency.tocsr()[normal_ids][:, normal_ids]
    degrees = np.asarray(checked_graph.sum(axis=1)).ravel() if not hasattr(checked_graph, 'degrees') else np.asarray(checked_graph.degrees)
    checked_n = int(checked_graph.n if hasattr(checked_graph, 'n') else checked_graph.shape[0])
    exact = (is_rs_robust(checked_graph, r, s, max_nodes=max_nodes)
             if kind == 'rs' else is_r_robust(checked_graph, r, max_nodes=max_nodes))
    return {
        'F': F,
        'threat_model': threat_model,
        'N': full_n,
        'normal_N': full_n - len(checked_ids),
        'checked_network': 'normal-node induced graph' if kind == 'normal-r' else 'full communication graph',
        'minimum_degree': int(degrees.min()),
        'required_robustness': f'({r},{s})-robust' if kind == 'rs' else f'{r}-robust',
        'condition': condition,
        'condition_met': exact,
        'exact_check': exact is not None,
        'max_nodes_for_exact_check': int(max_nodes),
        'interpretation': (
            'Textbook scalar W-MSR topology condition met.' if exact is True else
            'Topology condition not met; no W-MSR theorem claim for this graph.' if exact is False else
            'Exact graph-robustness check skipped for this N; guarantee not certified by this helper.'
        ),
    }


def validate_adversary_set(graph, adversary_ids: Iterable[int], F: int, threat_model: str = 'F-local-malicious') -> tuple[int, ...]:
    """Validate the declared F-total or F-local adversary-count assumption."""
    if isinstance(F, bool) or int(F) != F or F < 0:
        raise ValueError('F must be a nonnegative integer')
    F = int(F)
    ids = tuple(sorted(set(int(i) for i in adversary_ids)))
    n = int(graph.n)
    if any(i < 0 or i >= n for i in ids):
        raise ValueError('adversary IDs must be valid zero-based graph nodes')
    if len(ids) >= n:
        raise ValueError('at least one normal agent is required')
    if threat_model.startswith('F-total') and len(ids) > F:
        raise ValueError(f'{threat_model} permits at most F={F} adversaries total; got {len(ids)}')
    if threat_model.startswith('F-local'):
        adj = graph.adjacency.tocsr()
        adv = set(ids)
        for i in range(n):
            if i in adv:
                continue
            neighbors = adj.indices[adj.indptr[i]:adj.indptr[i + 1]]
            count = sum(int(j) in adv for j in neighbors)
            if count > F:
                raise ValueError(
                    f'F-local violation: normal agent {i} has {count} adversarial neighbors, F={F}'
                )
    return ids
