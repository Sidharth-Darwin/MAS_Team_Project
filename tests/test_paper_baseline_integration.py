"""Checks of the EXACT one-layer paper-formula and matched physical rollout."""

import numpy as np
import torch

from experiments._paper_gnn_baseline import (
    PaperGNN,
    rollout_paper,
    unnormalized_laplacian,
)
from distributed_consensus.graphs import make_graph
from distributed_consensus.control.plants import generate_system


def test_paper_one_layer_unnormalized_laplacian_and_per_agent_K():
    g = make_graph(3, "chain")
    model = PaperGNN(n=3, hidden=2, shared_output=False)
    H = np.array([[0.2, -0.5], [0.4, 0.3]])
    K = np.array([[0.5, 0.2], [-0.3, 0.8], [0.4, -0.2]])
    with torch.no_grad():
        model.H.copy_(torch.from_numpy(H))
        model.K.copy_(torch.from_numpy(K))
    x = np.array([[[1.0, 0.0], [0.0, 1.0], [0.5, -0.4]]])
    L = unnormalized_laplacian(g)
    predicted = (
        model(torch.from_numpy(x), torch.from_numpy(L)).detach().numpy()[0, :, 0]
    )
    expected = np.sum(np.tanh((L @ x[0]) @ H.T) * K, axis=1)
    np.testing.assert_allclose(predicted, expected, atol=1e-13)
    # Degree-2 agent uses 2*x2-x1-x3, NOT x2-(x1+x3)/2.
    assert L[1, 1] == 2.0


def test_paper_control_is_physically_applied_after_exact_local_matching():
    s = generate_system(3, seed=8)
    g = make_graph(3, "chain")
    x0 = np.array([[1, 0.3], [-0.3, 0.6], [0.7, -1.2]])
    model = PaperGNN(n=3, hidden=2, shared_output=False)
    h = rollout_paper(x0, s, g, model, horizon=3)
    assert h["states"].shape == (4, 3, 2)
    assert h["total"].shape == (3, 3, 1)
    np.testing.assert_allclose(h["total"], h["local"] + h["correction"])
    np.testing.assert_allclose(
        h["states"][1],
        np.einsum("nij,nj->ni", s.A, x0) + np.einsum("nij,nj->ni", s.B, h["total"][0]),
        atol=1e-13,
    )
    np.testing.assert_allclose(h["disagreement"][1], g.error(h["states"][1]))


def test_paper_tied_K_transfers_from_three_to_ten_agents():
    model = PaperGNN(n=3, hidden=3, shared_output=True)
    for n in (3, 10):
        g = make_graph(n, "chain")
        x = torch.ones((2, n, 2), dtype=torch.float64)
        v = model(x, torch.from_numpy(unnormalized_laplacian(g)))
        assert v.shape == (2, n, 1)
        np.testing.assert_allclose(v.detach().numpy(), 0.0, atol=1e-12)
