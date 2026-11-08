"""Regression tests for validation selection, invalid data, and training logs."""

import numpy as np
import pytest

from distributed_consensus.control.plants import generate_system
from distributed_consensus.graphs import make_graph
from distributed_consensus.learning.training import train


def _case():
    return generate_system(3, seed=16), make_graph(3, "star")


def test_best_validation_selection_is_independent_of_min_delta():
    system, graph = _case()
    result = train(
        system,
        graph,
        lam=0.2,
        epochs=12,
        horizon=15,
        batch=16,
        val_batch=16,
        seed=7,
        val_every=1,
        min_delta=2.0,
        device="cpu",
    )
    rows = [row for row in result["history"] if row["val_objective"] is not None]
    best = min(rows, key=lambda r: r["val_objective"])
    assert result["best_epoch"] == best["epoch"]
    assert result["best_validation_objective"] == pytest.approx(best["val_objective"])
    # The test is meaningful: the selected model is NOT the first validation.
    assert result["best_epoch"] > 1


def test_reject_nonfinite_validation_initial_states():
    system, graph = _case()
    for invalid in (np.nan, np.inf, -np.inf):
        states = np.full((4, system.n, 2), invalid)
        with pytest.raises(ValueError, match="finite"):
            train(
                system,
                graph,
                epochs=2,
                horizon=5,
                batch=4,
                validation_initial_states=states,
            )


def test_reject_overflowed_validation_objective():
    system, graph = _case()
    # Inputs are finite, but quadratic costs overflow during validation.
    states = np.full((4, system.n, 2), 1e200)
    with pytest.raises(ArithmeticError, match="nonfinite validation"):
        train(
            system,
            graph,
            epochs=2,
            horizon=5,
            batch=4,
            val_every=1,
            validation_initial_states=states,
        )


def test_history_distinguishes_pre_and_post_update_weight_norms():
    system, graph = _case()
    result = train(
        system, graph, epochs=3, horizon=5, batch=8, val_batch=8, val_every=1, seed=3
    )
    first = result["history"][0]
    assert first["pre_update_W_spectral_norm"] == 0.0
    assert first["W_spectral_norm"] > 0
    assert first["pre_update_constraint_margin"] == pytest.approx(result["cap"])
    assert first["constraint_margin"] == pytest.approx(
        result["cap"] - first["W_spectral_norm"]
    )
    for prev, curr in zip(result["history"], result["history"][1:]):
        assert curr["pre_update_W_spectral_norm"] == pytest.approx(
            prev["W_spectral_norm"], abs=1e-12
        )
