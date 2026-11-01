"""Regression checks for main.py model selection and external graph stepping."""
import json
import numpy as np
import pytest
from distributed_consensus.control.plants import generate_system
from distributed_consensus.evaluation.reporting import load_checkpoint


def test_external_simulator_accepts_direct_and_changing_adjacency():
    """The public single-step interface needs no saved graph and keeps physical states."""
    from distributed_consensus.graphs import Graph
    from distributed_consensus.control.dynamics import step_numpy

    s = generate_system(3, seed=17)
    x0 = np.array([[1., -1.], [.5, .4], [-.25, .8]])
    w = np.array([[.02], [-.05]])
    path_adj = np.array([[0,1,0], [1,0,1], [0,1,0]], dtype=float)
    star_adj = np.array([[0,1,1], [1,0,0], [1,0,0]], dtype=float)
    g1 = Graph(path_adj)
    g2 = Graph(star_adj)

    first = step_numpy(x0, s, g1, W=w)
    assert np.allclose(first['disagreement'], g1.laplacian @ x0)
    second = step_numpy(first['next_states'], s, g2, W=w)
    assert np.allclose(second['disagreement'], g2.laplacian @ first['next_states'])
    assert np.allclose(second['next_states'],
                       np.einsum('nij,nj->ni', s.A, first['next_states'])
                       + np.einsum('nij,nj->ni', s.B, second['total']))
    assert np.array_equal(w, np.array([[.02], [-.05]]))


def test_main_evaluates_best_and_preserves_final_checkpoint(tmp_path, monkeypatch):
    """Regression: the default evaluation must not quietly use last-epoch weights."""
    import sys
    from experiments import train_controller as runner
    original_train = runner.train

    def distinguish_best_from_final(*args, **kwargs):
        result = original_train(*args, **kwargs)
        result['best_W'] = np.zeros_like(result['W'])
        return result

    monkeypatch.setattr(runner, 'train', distinguish_best_from_final)
    monkeypatch.setattr(sys, 'argv', [
        'train_controller.py', '--suite', 'quick', '--run-name', 'smoke',
        '--results-root', str(tmp_path), '--epochs', '3', '--horizon', '4',
        '--batch', '4', '--eval-samples', '2', '--lambda-values', '0',
        '--train-topology', 'star', '--device', 'cpu'
    ])
    root = runner.execute(runner.parse_args())
    folder = root / 'trainN3_identity_lambda0'
    _, graph, default_w, _, _ = load_checkpoint(folder / 'checkpoint.npz')
    _, _, final_w, _, _ = load_checkpoint(folder / 'final_checkpoint.npz')
    assert not (folder / 'best_validation_checkpoint.npz').exists()
    assert np.array_equal(default_w, np.zeros((2,1)))
    assert not np.array_equal(default_w, final_w)
    assert graph.adjacency[0,2] == 1  # star with hub 0, not path graph
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['evaluated_checkpoint'] == 'checkpoint.npz'
    assert summary['training_topology'] == 'star'
    assert summary['local_eval']['J_E_improvement_pct'] == pytest.approx(0., abs=1e-12)
    meta = json.loads((root / 'configuration.json').read_text())
    assert meta['cli_arguments']['train_topology'] == 'star'


def test_train_cli_rejects_zero_instead_of_using_default(tmp_path):
    from experiments.train_controller import execute, parse_args
    args = parse_args(['--epochs', '0', '--run-name', 'bad',
                       '--results-root', str(tmp_path), '--device', 'cpu'])
    with pytest.raises(ValueError, match='must be positive'):
        execute(args)
    assert not (tmp_path / 'bad').exists()