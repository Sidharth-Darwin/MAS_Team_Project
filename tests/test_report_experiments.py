"""Smoke tests for the four independent report experiments."""
from pathlib import Path


def test_activation_experiment_smoke(tmp_path):
    from experiments.activation_comparison import parse_args, run
    out = tmp_path / 'activation'
    result = run(parse_args(['--quick', '--out', str(out)]))
    assert result['trained_models'] == 3
    assert (out / 'summary.csv').exists()
    import csv
    with (out/'summary.csv').open() as f:
        methods = {r['method'] for r in csv.DictReader(f)}
    assert 'paper_gnn' in methods
    assert any((out / 'figures').glob('*_state_x1.png'))
    assert any((out / 'figures').glob('*_control_total.png'))


def test_size_experiment_smoke(tmp_path):
    from experiments.network_size_generalization import parse_args, run
    out = tmp_path / 'size'
    result = run(parse_args(['--quick', '--out', str(out)]))
    assert result['trained_models'] == 2
    assert (out / 'summary.csv').exists()
    import csv
    with (out/'summary.csv').open() as f:
        methods = {r['method'] for r in csv.DictReader(f)}
    assert 'paper_gnn' in methods
    assert (out / 'figures' / 'settling_vs_size.png').exists()


def test_gain_experiment_smoke(tmp_path):
    from experiments.plant_gain_transfer import parse_args, run
    out = tmp_path / 'gain'
    result = run(parse_args(['--quick', '--out', str(out)]))
    assert result['trained_models'] == 2
    assert (out / 'summary.csv').exists()
    import csv
    with (out/'summary.csv').open() as f:
        methods = {r['method'] for r in csv.DictReader(f)}
    assert 'paper_gnn' in methods
    assert any((out / 'figures').glob('*above_training_gain*_state_x2.png'))


def test_topology_experiment_smoke(tmp_path):
    from experiments.topology_generalization import parse_args, run
    out = tmp_path / 'topology'
    result = run(parse_args(['--quick', '--out', str(out)]))
    assert result['trained_models'] == 2
    assert (out / 'summary.csv').exists()
    import csv
    with (out/'summary.csv').open() as f:
        methods = {r['method'] for r in csv.DictReader(f)}
    assert 'paper_gnn' in methods
    assert (out / 'figures' / 'ring').exists()
    assert any((out / 'figures' / 'ring').glob('*_graph.png'))


def test_empty_output_dir_required(tmp_path):
    from experiments.activation_comparison import parse_args, run
    out = tmp_path / 'activation'
    out.mkdir()
    (out / 'old.txt').write_text('x')
    import pytest
    with pytest.raises(FileExistsError):
        run(parse_args(['--quick', '--out', str(out)]))


def test_absolute_threshold_is_used_in_paired_rollouts():
    import numpy as np
    from experiments._common import er_connected_graph, system_at_gain, paired_rollouts
    g = er_connected_graph(3, 0.85, 12)
    s = system_at_gain(3, g, .75, 5)
    x0 = np.random.default_rng(14).normal(size=(2,3,2))
    rows, _, _, _ = paired_rollouts(s, g, 12, x0, np.zeros((2,1)), threshold=1e-12)
    assert all(r['baseline_settled'] == 0 and r['learned_settled'] == 0 for r in rows)


def test_mixed_gain_uses_common_A_and_base_B():
    import numpy as np
    from experiments._common import er_connected_graph, system_at_mixed_gain
    from distributed_consensus.control.plants import generate_system
    from distributed_consensus.control.plant_gain import full_gain
    graph=er_connected_graph(10,.7,77)
    base=generate_system(10,seed=9)
    group=np.array([0]*5+[1]*5)
    below,ml=system_at_mixed_gain(base,graph,.5625,group)
    above,mh=system_at_mixed_gain(base,graph,.9375,group)
    np.testing.assert_array_equal(below.A, above.A)
    np.testing.assert_array_equal(below.A, base.A)
    np.testing.assert_allclose(below.B,base.B*ml[:,None,None])
    np.testing.assert_allclose(above.B,base.B*mh[:,None,None])
    assert np.isclose(full_gain(below,graph).gamma,.5625)
    assert np.isclose(full_gain(above,graph).gamma,.9375)
    assert np.all(ml[group==1] > ml[group==0])
