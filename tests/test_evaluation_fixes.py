"""Regression tests for exact-threshold evaluation and invalid inputs."""

import numpy as np
import pytest

from distributed_consensus.control.plants import generate_system
from distributed_consensus.graphs import make_graph
from distributed_consensus.evaluation.metrics import evaluate_pair, trajectory_metrics


def test_all_zero_initial_states_have_undefined_relative_improvement():
    system = generate_system(3, seed=16)
    graph = make_graph(3, "star")
    result = evaluate_pair(
        system,
        graph,
        np.array([[0.2], [-0.1]]),
        horizon=4,
        samples=1,
        initial_states=np.zeros((1, 3, 2)),
    )
    assert result["baseline"]["mean_J_E"] == 0.0
    assert result["learned"]["mean_J_E"] == 0.0
    assert result["J_E_improvement_pct"] is None
    assert result["J_U_reduction_pct"] is None
    assert result["J_pair_improvement_pct"] is None
    assert result["baseline"]["mean_J_E_initial_normalized"] is None
    assert result["baseline"]["mean_settling_if_settled"] == 0.0


def test_near_zero_initial_error_does_not_bypass_relative_threshold():
    e = np.array([[[1e-13]], [[5e-10]], [[5e-10]]])
    history = {
        "disagreement": e,
        "total": np.zeros((2, 1, 1)),
        "correction": np.zeros((2, 1, 1)),
    }
    metric = trajectory_metrics(history)
    assert metric["threshold"] == pytest.approx(1e-15)
    assert metric["first_crossing"] is None
    assert metric["settling_time"] is None
    assert metric["J_E_initial_normalized"] == pytest.approx(
        (1e-13 / 1e-13) ** 2 + (5e-10 / 1e-13) ** 2
    )


def test_zero_initial_error_then_nonzero_is_not_settled():
    history = {
        "disagreement": np.array([[[0.0]], [[1.0]], [[1.0]]]),
        "total": np.zeros((2, 1, 1)),
        "correction": np.zeros((2, 1, 1)),
    }
    metric = trajectory_metrics(history)
    assert metric["first_crossing"] == 0
    assert metric["settling_time"] is None
    assert metric["J_E_initial_normalized"] is None


def test_reject_bad_sample_counts_and_nonfinite_initial_states():
    system = generate_system(3, seed=16)
    graph = make_graph(3, "ring")
    W = np.zeros((2, 1))
    with pytest.raises(ValueError, match="samples"):
        evaluate_pair(system, graph, W, samples=0)
    with pytest.raises(ValueError, match="horizon"):
        evaluate_pair(system, graph, W, horizon=0)
    initial = np.zeros((1, 3, 2))
    initial[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="initial_states"):
        evaluate_pair(system, graph, W, samples=1, initial_states=initial)
    with pytest.raises(ValueError, match="W"):
        evaluate_pair(system, graph, np.array([[float("inf")], [0.0]]), samples=1)


def test_invalid_trajectory_values_or_nonfinite_accumulation_raise():
    e = np.array([[[1.0]], [[1.0]]])
    history = {
        "disagreement": e,
        "total": np.zeros((1, 1, 1)),
        "correction": np.zeros((1, 1, 1)),
    }
    history["disagreement"][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        trajectory_metrics(history)
    history["disagreement"][1, 0, 0] = 1.0
    history["total"][0, 0, 0] = 1e250
    with pytest.raises(ArithmeticError):
        trajectory_metrics(history)


def test_normal_nonzero_baseline_yields_zero_when_controller_is_zero():
    system = generate_system(3, seed=3)
    graph = make_graph(3, "chain")
    result = evaluate_pair(system, graph, np.zeros((2, 1)), horizon=5, samples=4)
    assert result["J_E_improvement_pct"] == pytest.approx(0.0)
    assert result["J_U_reduction_pct"] == pytest.approx(0.0)
    assert result["J_pair_improvement_pct"] == pytest.approx(0.0)


def test_nonfinite_disturbances_rejected():
    system = generate_system(3, seed=16)
    graph = make_graph(3, "chain")
    d = np.zeros((1, 3, 3, 1))
    d[0, 1, 0, 0] = np.inf
    with pytest.raises(ValueError, match="disturbances"):
        evaluate_pair(
            system, graph, np.zeros((2, 1)), horizon=3, samples=1, disturbances=d
        )
