"""Regression checks for public simulation stepping."""
import numpy as np
from distributed_consensus.control.plants import generate_system


def test_external_simulator_accepts_direct_and_changing_adjacency():
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
