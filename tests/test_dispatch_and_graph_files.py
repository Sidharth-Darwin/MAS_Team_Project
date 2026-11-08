"""Focused regression checks for graph loading and thin dispatcher."""

import numpy as np
import pytest
from scipy import sparse
from distributed_consensus.graphs import Graph, load_graph, make_graph
from distributed_consensus.control.plants import generate_system
from distributed_consensus.evaluation.reporting import save_system


def test_load_dense_numpy_adjacency(tmp_path):
    adj = make_graph(5, "star").adjacency.toarray()
    path = tmp_path / "graph.npy"
    np.save(path, adj)
    assert np.array_equal(load_graph(path).adjacency.toarray(), adj)


def test_load_sparse_scipy_adjacency(tmp_path):
    g = make_graph(1000, "ring")
    path = tmp_path / "graph.npz"
    sparse.save_npz(path, g.adjacency)
    loaded = load_graph(path)
    assert loaded.n == 1000
    assert loaded.adjacency.nnz == 2000
    assert (loaded.adjacency != g.adjacency).nnz == 0


def test_load_dense_npz_and_existing_saved_system_graph(tmp_path):
    g = make_graph(5, "ring")
    path = tmp_path / "adj.npz"
    np.savez_compressed(path, adjacency=g.adjacency.toarray())
    assert (load_graph(path).adjacency != g.adjacency).nnz == 0
    system = generate_system(5, seed=22)
    saved = tmp_path / "system.npz"
    save_system(saved, system, g)
    assert (load_graph(saved).adjacency != g.adjacency).nnz == 0


def test_load_invalid_graph_rejected(tmp_path):
    p = tmp_path / "graph.npy"
    np.save(p, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="undirected"):
        load_graph(p)
    with pytest.raises(ValueError, match="must be .npy or .npz"):
        load_graph(tmp_path / "adj.json")


def test_main_dispatches_to_activation_experiment(monkeypatch):
    import main
    from experiments import activation_comparison

    observed = []

    def fake_main(argv):
        observed.append(argv)

    monkeypatch.setattr(activation_comparison, "main", fake_main)
    main.main(["activation", "--quick"])
    assert observed == [["--quick"]]


def test_main_requires_an_explicit_command():
    import main
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main.main(["--help"]) == 0
    assert "activation|size|gain|topology" in buf.getvalue()
    with pytest.raises(SystemExit, match="Unknown command"):
        main.main(["--suite", "quick"])


def test_reject_even_tiny_asymmetry():
    with pytest.raises(ValueError, match="undirected"):
        Graph(np.array([[0.0, 1.0], [1.0 + 5e-13, 0.0]]))
