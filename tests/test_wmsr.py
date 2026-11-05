"""Tests for the exact scalar local W-MSR algorithm supplied by the user."""
from copy import deepcopy

import numpy as np
import pytest

from distributed_consensus.graphs import make_graph
from distributed_consensus.robust.wmsr import (
    wmsr_local,
    wmsr_local_with_details,
    is_r_robust,
    is_rs_robust,
    assess_wmsr_topology,
    validate_adversary_set,
)


def reference_wmsr_local(X, N, F=1):
    """Literal readable reference matching the supplied pseudocode."""
    X_new = deepcopy(np.asarray(X, dtype=float))
    for node in range(len(X)):
        neighbors = N[node]
        neighbors_values = np.sort(np.asarray(X)[list(neighbors)])
        node_value = X[node]
        lower = neighbors_values[neighbors_values < node_value]
        middle = neighbors_values[neighbors_values == node_value]
        upper = neighbors_values[neighbors_values > node_value]
        lower = lower[F:]
        upper = upper[:-F] if F > 0 else upper
        filtered = np.concatenate((lower, middle, upper))
        X_new[node] = np.mean(np.concatenate(([node_value], filtered)))
    return X_new


def test_wmsr_matches_user_algorithm_exactly_for_all_nodes():
    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    N = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 3, 4],
        3: [1, 2, 4],
        4: [2, 3],
    }
    actual = wmsr_local(X, N, F=1)
    expected = reference_wmsr_local(X, N, F=1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_receiver_discards_smallest_low_and_largest_high_and_keeps_middle():
    # Node 2 has own value 2 and receives [0, 1, 3, 4].
    # F=1 removes 0 and 4, retaining 1 and 3; mean([2,1,3]) = 2.
    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    N = {0: [2], 1: [2], 2: [0, 1, 3, 4], 3: [2], 4: [2]}
    result = wmsr_local_with_details(X, N, F=1)
    assert result.removed_low_ids[2] == (0,)
    assert result.removed_high_ids[2] == (4,)
    assert result.retained_ids[2] == (1, 3)
    assert result.values[2] == pytest.approx(2.0)


def test_equal_neighbour_values_are_never_removed():
    X = np.array([2.0, 2.0, 2.0, 9.0, -4.0])
    N = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
    result = wmsr_local_with_details(X, N, F=1)
    assert 1 in result.retained_ids[0]
    assert 2 in result.retained_ids[0]
    assert result.removed_high_ids[0] == (3,)
    assert result.removed_low_ids[0] == (4,)


def test_f_zero_is_ordinary_mean_with_self_and_all_neighbours():
    X = np.array([0.0, 2.0, 4.0])
    N = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    result = wmsr_local(X, N, F=0)
    np.testing.assert_allclose(result, np.array([2.0, 2.0, 2.0]))


def test_f_larger_than_available_values_removes_all_strict_extremes():
    X = np.array([0.0, 1.0, 2.0])
    N = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    details = wmsr_local_with_details(X, N, F=5)
    # For node 1, both neighbours are strictly lower and higher respectively;
    # both are removed, leaving the node's own value unchanged.
    assert details.values[1] == pytest.approx(X[1])
    assert details.removed_low_ids[1] == (0,)
    assert details.removed_high_ids[1] == (2,)


def test_updates_are_simultaneous_and_do_not_mutate_input():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    original = X.copy()
    N = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    expected = reference_wmsr_local(X, N, F=1)
    actual = wmsr_local(X, N, F=1)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(X, original)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match='finite one-dimensional'):
        wmsr_local(np.array([[0.0, 1.0]]), {0: [1], 1: [0]}, F=1)
    with pytest.raises(ValueError, match='nonnegative integer'):
        wmsr_local(np.array([0.0, 1.0]), {0: [1], 1: [0]}, F=-1)
    with pytest.raises(ValueError, match='missing neighbour list'):
        wmsr_local(np.array([0.0, 1.0]), {0: [1]}, F=1)
    with pytest.raises(ValueError, match='duplicate neighbour'):
        wmsr_local(np.array([0.0, 1.0]), {0: [1, 1], 1: [0]}, F=1)


def test_complete_graph_meets_f_local_malicious_sufficient_condition():
    graph = make_graph(10, 'complete')
    assert is_r_robust(graph, 3) is True
    report = assess_wmsr_topology(graph, F=1, threat_model='F-local-malicious')
    assert report['required_robustness'] == '3-robust'
    assert report['condition_met'] is True


def test_chain_is_connected_but_does_not_meet_f_local_wmsr_robustness():
    graph = make_graph(10, 'chain')
    report = assess_wmsr_topology(graph, F=1, threat_model='F-local-malicious')
    assert report['condition_met'] is False
    assert 'not met' in report['interpretation'].lower()


def test_f_local_adversary_distribution_is_validated():
    graph = make_graph(10, 'complete')
    assert validate_adversary_set(graph, (0,), F=1, threat_model='F-local-malicious') == (0,)
    # In a complete graph, two malicious nodes are neighbours of every normal
    # node, violating the F=1 local distribution.
    with pytest.raises(ValueError, match='F-local violation'):
        validate_adversary_set(graph, (0, 1), F=1, threat_model='F-local-malicious')


def test_f_local_model_can_have_more_than_f_total_malicious_nodes():
    # On a graph where compromised nodes are sufficiently separated, multiple
    # malicious nodes may exist globally while every normal node sees <= F.
    graph = make_graph(10, 'ring')
    ids = (0, 4)
    assert validate_adversary_set(graph, ids, F=1, threat_model='F-local-malicious') == ids


def test_exact_robustness_checker_does_not_claim_large_graph_result():
    graph = make_graph(20, 'complete')
    assert is_r_robust(graph, 3) is None
    report = assess_wmsr_topology(graph, 1, 'F-local-malicious')
    assert report['condition_met'] is None
    assert report['exact_check'] is False
