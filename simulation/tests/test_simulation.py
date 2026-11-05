"""Regression tests for the scalar local W-MSR multi-controller simulator."""
from __future__ import annotations

import csv
import numpy as np
import pytest

from distributed_consensus.control.dynamics import advance_numpy
from simulation.model import (
    DEFAULT_BENCHMARK_CHECKPOINT,
    METHODS,
    MultiPolicySimulation,
    load_benchmark,
)
from distributed_consensus.robust.wmsr import wmsr_local_with_details


def make_sim(seed=3, adversary_ids=(9,), topology='complete', F=1):
    return MultiPolicySimulation(
        DEFAULT_BENCHMARK_CHECKPOINT,
        topology=topology,
        seed=seed,
        adversary_ids=adversary_ids,
        F=F,
    )


def test_bundled_benchmark_has_one_scalar_paper_message_per_agent():
    models = load_benchmark(DEFAULT_BENCHMARK_CHECKPOINT)
    assert models.system.n == 10
    assert models.training_graph.n == 10
    assert models.W.shape == (2, 1)
    assert models.paper_H.shape == (1, 2)
    assert models.paper_K.shape == (10, 1)
    assert models.metadata['training_streams_matched'] is True


def test_all_five_policies_start_from_identical_states():
    sim = make_sim()
    assert set(sim.states) == set(METHODS)
    for method in METHODS:
        np.testing.assert_array_equal(sim.states[method], sim.initial)


def test_complete_graph_meets_f_local_3_robust_sufficient_condition():
    sim = make_sim()
    assert sim.topology_report['required_robustness'] == '3-robust'
    assert sim.topology_report['condition_met'] is True
    assert len(sim.normal_ids) == 9


def test_wmsr_messages_are_scalar_and_match_user_algorithm():
    sim = make_sim()
    output = sim.step()
    for method in ('paper_gnn_wmsr', 'small_gain_wmsr'):
        delivered = output[method]['delivered_signals']
        expected = wmsr_local_with_details(delivered, sim.neighbor_map, sim.F)
        for receiver in sim.normal_ids:
            action = sim.action_histories[method][-1]['actions'][receiver]
            assert len(action.neighbor_messages) == len(sim.neighbor_map[receiver])
            assert action.retained_ids == expected.retained_ids[receiver]
            assert action.removed_high_ids == expected.removed_high_ids[receiver]
            assert action.removed_low_ids == expected.removed_low_ids[receiver]
            assert len(action.removed_high_ids) <= sim.F
            assert len(action.removed_low_ids) <= sim.F


def test_one_malicious_broadcast_is_received_and_trimmed_by_each_normal_receiver():
    sim = make_sim()
    outputs = sim.step()
    malicious_id = 9
    for method in ('paper_gnn_raw', 'small_gain_raw'):
        for receiver in sim.normal_ids:
            action = outputs[method]['actions'][receiver]
            assert malicious_id in [sender for sender, _ in action.neighbor_messages]
            assert malicious_id not in action.removed_ids
    for method in ('paper_gnn_wmsr', 'small_gain_wmsr'):
        for receiver in sim.normal_ids:
            action = outputs[method]['actions'][receiver]
            assert malicious_id in action.removed_ids
    assert outputs['paper_gnn_wmsr']['filtered_values'] > 0
    assert outputs['small_gain_wmsr']['filtered_values'] > 0


def test_f_one_means_up_to_one_high_and_one_low_scalar_value_per_receiver():
    sim = make_sim()
    outputs = sim.step()
    for method in ('paper_gnn_wmsr', 'small_gain_wmsr'):
        for receiver in sim.normal_ids:
            action = outputs[method]['actions'][receiver]
            assert len(action.removed_high_ids) <= 1
            assert len(action.removed_low_ids) <= 1
            assert len(action.removed_ids) <= 2


def test_local_only_reference_is_invariant_to_communication_attack():
    attacked = make_sim(seed=17, adversary_ids=(9,))
    clean = make_sim(seed=17, adversary_ids=())
    for _ in range(60):
        attacked.step()
        clean.step()
    np.testing.assert_allclose(
        attacked.states['local_only'], clean.states['local_only'], atol=1e-13
    )


def test_unfiltered_and_filtered_controllers_follow_different_message_paths():
    sim = make_sim(seed=3)
    outputs = sim.step()
    for method in ('paper_gnn_raw', 'small_gain_raw'):
        for receiver in sim.normal_ids:
            action = outputs[method]['actions'][receiver]
            assert action.removed_ids == ()
    for method in ('paper_gnn_wmsr', 'small_gain_wmsr'):
        assert any(
            outputs[method]['actions'][receiver].removed_ids
            for receiver in sim.normal_ids
        )


def test_no_adversary_runs_all_five_controllers():
    sim = make_sim(seed=5, adversary_ids=())
    assert sim.normal_ids == tuple(range(10))
    sim.step()
    assert all(np.isfinite(sim.states[m]).all() for m in METHODS)


def test_chain_topology_is_warned_as_not_meeting_f_local_sufficient_condition():
    sim = make_sim(seed=3, topology='chain')
    assert sim.topology_report['condition_met'] is False
    assert sim.topology_report['required_robustness'] == '3-robust'


def test_local_f_model_rejects_too_many_compromised_neighbours():
    with pytest.raises(ValueError, match='F-local violation'):
        make_sim(adversary_ids=(0, 1), topology='complete', F=1)


def test_authoritative_plant_step_matches_physical_update():
    system = load_benchmark(DEFAULT_BENCHMARK_CHECKPOINT).system
    rng = np.random.default_rng(11)
    states = rng.normal(size=(system.n, 2))
    inputs = rng.normal(size=(system.n, 1))
    expected = (
        np.einsum('nij,nj->ni', system.A, states)
        + np.einsum('nij,nj->ni', system.B, inputs)
    )
    np.testing.assert_allclose(advance_numpy(states, system, inputs), expected, atol=1e-13)


def test_small_gain_raw_scalar_signal_is_projection_of_neighbor_average_error():
    sim = make_sim(seed=9, adversary_ids=())
    out = sim.step()
    states = sim.initial
    W = sim.models.W[:, 0]
    for i in range(sim.n):
        neighbors = sim.neighbor_map[i]
        expected = float(W @ (states[i] - states[list(neighbors)].mean(axis=0)))
        actual = out['small_gain_raw']['actions'][i].correction
        assert actual == pytest.approx(expected)


def test_wmsr_small_gain_uses_retained_neighbor_average():
    sim = make_sim(seed=9, adversary_ids=())
    out = sim.step()
    signals = out['small_gain_wmsr']['delivered_signals']
    for i in sim.normal_ids:
        action = out['small_gain_wmsr']['actions'][i]
        retained = action.retained_ids
        expected = 0.0 if not retained else float(signals[i] - signals[list(retained)].mean())
        assert action.correction == pytest.approx(expected)


def test_export_contains_one_scalar_and_no_old_q_v_columns(tmp_path):
    sim = make_sim()
    for _ in range(5):
        sim.step()
    path = sim.export_csv(tmp_path / 'trajectories.csv')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(METHODS) * sim.n * sim.k
    assert 'scalar_signal' in rows[0]
    assert 'removed_high_neighbor_ids' in rows[0]
    assert 'removed_low_neighbor_ids' in rows[0]
    assert not any('q_' in key or 'v_' in key for key in rows[0])
    sim.save_figures(tmp_path / 'figures')
    assert (tmp_path / 'figures' / 'normal_disagreement_all_controllers.png').exists()
    assert (tmp_path / 'figures' / 'wmsr_discarded_values.png').exists()


def test_multi_trial_evaluator_writes_paired_results(tmp_path):
    from simulation.evaluate import parse_args, run_evaluation
    args = parse_args([
        '--trials', '2', '--steps', '12', '--topology', 'complete',
        '--adversary', '10', '--F', '1', '--out', str(tmp_path / 'evaluation'),
    ])
    out = run_evaluation(args)
    assert (out / 'trial_summary.csv').exists()
    assert (out / 'aggregate_summary.csv').exists()
    assert (out / 'paired_wmsr_effect.csv').exists()
    with (out / 'trial_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * len(METHODS)


def test_preview_cli_exposes_only_the_simple_broadcast_attack():
    from simulation.preview import main
    with pytest.raises(SystemExit) as error:
        main(['--help'])
    assert error.value.code == 0
