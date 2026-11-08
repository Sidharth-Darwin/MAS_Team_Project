"""Reproducible mathematical checks and negative tests (no universal guarantees claimed)."""

import numpy as np
import pytest
import torch
from scipy import sparse
from distributed_consensus.graphs import make_graph, from_edges, Graph
from distributed_consensus.control.plants import (
    generate_system,
    build_system,
    DEFAULT_TARGET,
    check_target,
    matching_residual,
)
from distributed_consensus.control.controllers import (
    correction_numpy,
    project_spectral_,
    weight_norm,
)
from distributed_consensus.control.dynamics import step_numpy, rollout_numpy, TorchWorld
from distributed_consensus.control.plant_gain import (
    scalar_gain,
    full_gain,
    direct_grid_gain,
    small_gain,
)
from distributed_consensus.learning.training import train, baseline_scales
from distributed_consensus.evaluation.metrics import trajectory_metrics, evaluate_pair
from distributed_consensus.evaluation.reporting import save_checkpoint, load_checkpoint


@pytest.fixture
def case():
    s = generate_system(3, seed=16)
    g = make_graph(3, "chain")
    return s, g


def test_01_physical_heterogeneity(case):
    s, _ = case
    assert len(np.unique(s.B[:, 1, 0])) == 3
    assert len(np.unique(s.A[:, 1, 1])) == 3
    assert np.array_equal(s.B[:, 0, 0], np.zeros(3))
    assert np.all(s.B[:, 1, 0] != 0)


def test_02_pseudoinverse_is_actual(case):
    s, _ = case
    pinvs = np.linalg.pinv(s.B)
    assert np.allclose(s.K, pinvs @ (s.target[None, :, :] - s.A))
    assert np.allclose(s.B @ pinvs @ s.B, s.B)


def test_03_exact_matching(case):
    assert matching_residual(case[0]) < 1e-12


def test_04_reject_incompatible(case):
    s, _ = case
    a = s.A.copy()
    a[0, 0, 0] += 0.0001
    with pytest.raises(ValueError, match="incompatible"):
        build_system(a, s.B, s.target)


def test_05_no_controllability_assumption():
    target = np.diag([0.7, 0.6])
    a = np.array([[[0.7, 0.0], [0.2, 0.3]], [[0.7, 0.0], [-0.1, 0.8]]])
    b = np.array([[[0.0], [0.1]], [[0.0], [0.15]]])
    s = build_system(a, b, target)
    assert matching_residual(s) < 1e-12
    assert np.linalg.matrix_rank(np.column_stack((b[0], a[0] @ b[0]))) == 1


def test_06_target_stable(case):
    rho = check_target(case[0].target)
    assert rho < 1 and abs(rho - np.sqrt(0.82)) < 1e-12


def test_07_reject_unstable_target(case):
    with pytest.raises(ValueError, match="not Schur stable"):
        check_target(np.eye(2))


def test_08_reject_invalid_b(case):
    s, _ = case
    b = s.B.copy()
    b[0, 1, 0] = 0
    with pytest.raises(ValueError):
        build_system(s.A, b, s.target)
    b = s.B.copy()
    b[0, 0, 0] = 0.1
    with pytest.raises(ValueError):
        build_system(s.A, b, s.target)


def test_09_graph_symmetry_and_connectivity(case):
    _, g = case
    assert (g.adjacency - g.adjacency.T).nnz == 0
    assert g.n == 3 and np.all(g.degrees > 0)


def test_10_reject_directed_or_disconnected_graph():
    with pytest.raises(ValueError, match="undirected"):
        from_edges(3, [0, 1], [1, 2])
    with pytest.raises(ValueError, match="connected"):
        from_edges(3, [0, 1], [1, 0])


def test_11_lrw_not_augmented(case):
    _, g = case
    expected = np.array([[1, -1, 0], [-0.5, 1, -0.5], [0, -1, 1]])
    assert np.allclose(g.laplacian.toarray(), expected)
    assert np.allclose(g.p_rw @ np.ones(3), np.ones(3))


def test_12_disagreement_is_local_sparse(case):
    _, g = case
    x = np.arange(6, dtype=float).reshape((3, 2))
    assert np.allclose(g.error(x), g.laplacian @ x)
    assert np.allclose(g.error(np.ones((3, 2))), 0.0)


def test_13_exact_direct_controller_no_bias(case):
    e = np.array([[1.0, 2.0], [-2, 1], [0, 0]])
    W = np.array([[0.2], [-0.3]])
    assert np.allclose(correction_numpy(e, W), e @ W)
    assert np.allclose(correction_numpy(e, W, "tanh"), np.tanh(e @ W))
    assert correction_numpy(e, W)[2, 0] == 0
    assert correction_numpy(e, W, "tanh")[2, 0] == 0


def test_14_activation_global_one_lipschitz_samples():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(10000), rng.standard_normal(10000)
    assert np.max(np.abs(np.tanh(a) - np.tanh(b)) - np.abs(a - b)) < 1e-12
    assert np.allclose(a - b, a - b)


def test_15_physical_update(case):
    s, g = case
    x = np.array([[1, 0.2], [-1, 0.3], [0.4, -0.5]])
    out = step_numpy(x, s, g, np.array([[0.1], [-0.2]]))
    assert np.allclose(out["total"], out["local"] + out["correction"])
    assert np.allclose(
        out["next_states"],
        np.einsum("nij,nj->ni", s.A, x) + np.einsum("nij,nj->ni", s.B, out["total"]),
    )


def test_16_centralized_error_agreement(case):
    s, g = case
    L = g.laplacian.toarray()
    x = np.array([[1, 0.2], [-1, 0.3], [0.4, -0.5]])
    W = np.array([[0.25], [-0.35]])
    out = step_numpy(x, s, g, W)
    E = g.error(x).ravel()
    Bblk = sparse.block_diag(list(s.B)).toarray()
    A_e = np.kron(np.eye(s.n), s.target)
    B_e = np.kron(L, np.eye(2)) @ Bblk
    predicted = A_e @ E + B_e @ out["correction"].ravel()
    assert np.allclose(predicted, g.error(out["next_states"]).ravel(), atol=1e-12)


def test_17_gamma0_global_candidates():
    gamma, omega = scalar_gain(DEFAULT_TARGET)
    assert abs(gamma - 6.90423107655106) < 1e-9
    assert abs(omega - 0.123490946162) < 1e-9


def test_18_full_gain_direct_grid(case):
    s, g = case
    exact = full_gain(s, g)
    grid = direct_grid_gain(s, g, points=4097)
    assert grid <= exact.gamma + 1e-9
    assert abs(exact.gamma - grid) / exact.gamma < 1e-5


def test_19_factorized_spatial_norm(case):
    s, g = case
    gamma = full_gain(s, g)
    expected = np.linalg.norm(g.laplacian.toarray() @ np.diag(s.B[:, 1, 0]), 2)
    assert np.isclose(gamma.spatial_norm, expected)
    assert np.isclose(gamma.gamma, expected * gamma.gamma0)


def test_20_large_sparse_gain():
    n = 100
    g = make_graph(n, "ring")
    s = generate_system(n, seed=123)
    result = full_gain(s, g)
    assert np.isfinite(result.gamma) and result.gamma > 0
    assert g.laplacian.nnz < 5 * n


def test_21_projection_spectral():
    W = torch.nn.Parameter(torch.tensor([[3.0], [-4.0]], dtype=torch.float64))
    norm = project_spectral_(W, 0.4)
    assert abs(norm - 0.4) < 1e-12
    assert np.isclose(weight_norm(W.detach().numpy()), 0.4)
    W.data.zero_()
    assert project_spectral_(W, 0.4) == 0


def test_22_projection_after_every_training_epoch(case):
    s, g = case
    r = train(s, g, lam=0, epochs=4, horizon=6, batch=8, val_batch=8, lr=1.0, seed=1)
    assert all(h["W_spectral_norm"] <= r["cap"] + 1e-10 for h in r["history"])
    assert r["W"].shape == (2, 1)


def test_23_inference_frozen_weight(case):
    s, g = case
    W = np.array([[0.1], [-0.15]])
    before = W.copy()
    for _ in range(3):
        rollout_numpy(np.ones((s.n, 2)), s, g, W, horizon=5)
    assert np.array_equal(W, before)


def test_24_cost_action_indices_and_terminal(case):
    s, g = case
    x = np.arange(6, dtype=float).reshape(3, 2) / 4
    hist = rollout_numpy(x, s, g, np.zeros((2, 1)), horizon=2)
    metrics = trajectory_metrics(hist)
    assert np.isclose(metrics["J_E"], np.square(hist["disagreement"][:2]).sum())
    assert np.isclose(metrics["J_U"], np.square(hist["total"]).sum())
    assert metrics["terminal_E"] == pytest.approx(
        np.linalg.norm(hist["disagreement"][2])
    )


def test_25_objective_lambda_zero_and_positive(case):
    s, g = case
    world = TorchWorld.from_system(s, g)
    x = np.random.default_rng(17).standard_normal((s.n, 2))
    W = np.array([[0.1], [-0.2]])
    je, ju, terminal = world.costs(torch.from_numpy(x[None]), torch.from_numpy(W), 3)
    hist = rollout_numpy(x, s, g, W, horizon=3)
    assert je.item() == pytest.approx(np.square(hist["disagreement"][:3]).sum())
    assert ju.item() == pytest.approx(np.square(hist["total"]).sum())
    assert terminal.item() == pytest.approx(np.square(hist["disagreement"][-1]).sum())
    assert (je + 0.123 * ju).item() == pytest.approx(
        np.square(hist["disagreement"][:3]).sum()
        + 0.123 * np.square(hist["total"]).sum()
    )
    assert (je + 0 * ju).item() == pytest.approx(je.item())


def test_26_torch_numpy_step_match(case):
    s, g = case
    x = np.random.default_rng(91).standard_normal((s.n, 2))
    W = np.array([[0.1], [-0.2]])
    world = TorchWorld.from_system(s, g)
    for act in ("identity", "tanh"):
        ts = world.step(torch.from_numpy(x[None].copy()), torch.from_numpy(W), act)
        out = step_numpy(x, s, g, W, act)
        for a, b in zip(
            ts,
            (
                out["next_states"],
                out["disagreement"],
                out["local"],
                out["correction"],
                out["total"],
            ),
        ):
            assert np.allclose(a.detach().numpy()[0], b)


def test_27_training_repro_and_independent_validation(case):
    s, g = case
    a = train(s, g, lam=0.01, epochs=3, horizon=5, batch=7, val_batch=8, seed=3)
    b = train(s, g, lam=0.01, epochs=3, horizon=5, batch=7, val_batch=8, seed=3)
    assert np.array_equal(a["W"], b["W"])
    assert a["history"] == b["history"]
    assert a["train_seed"] != a["validation_seed"]
    assert a["history"][0]["val_J_E"] is not None


def test_28_baseline_and_learned_identical_initial_conditions(case):
    s, g = case
    W = np.zeros((2, 1))
    r = evaluate_pair(s, g, W, horizon=5, samples=5)
    assert r["J_E_improvement_pct"] == pytest.approx(0, abs=1e-10)
    assert r["J_U_reduction_pct"] == pytest.approx(0, abs=1e-10)
    assert r["paired_win_rate"] == 0


def test_29_settling_not_just_first_crossing():
    e = np.array([[[2.0]], [[0.05]], [[0.9]], [[0.001]]])
    h = {
        "disagreement": e,
        "total": np.zeros((3, 1, 1)),
        "correction": np.zeros((3, 1, 1)),
    }
    m = trajectory_metrics(h, epsilon_fraction=0.1)
    assert m["first_crossing"] == 1
    assert m["settling_time"] == 3
    e = np.array([[[2.0]], [[2.0]], [[2.0]], [[2.0]]])
    h["disagreement"] = e
    m = trajectory_metrics(h)
    assert m["first_crossing"] is None and m["settling_time"] is None
    h["disagreement"] = np.zeros_like(e)
    m = trajectory_metrics(h)
    assert m["first_crossing"] == 0 and m["settling_time"] == 0


def test_30_consensus_invariance(case):
    s, g = case
    x = np.tile([0.8, -0.4], (s.n, 1))
    for act in ("identity", "tanh"):
        out = step_numpy(x, s, g, np.array([[0.4], [-0.2]]), act)
        assert np.allclose(out["correction"], 0.0, atol=1e-14)
        assert np.allclose(out["next_states"], out["next_states"][0])


def test_31_bigger_network_gain_not_equivalent_to_loss_of_certificate(case):
    s, g = case
    small = full_gain(s, g).gamma
    big_s = generate_system(100, seed=86)
    big_g = make_graph(100, "ring")
    big = full_gain(big_s, big_g).gamma
    W = np.array([[0], [0.1]])
    info = small_gain(big_s, big_g, W)
    assert info["loop_gain"] == pytest.approx(big * 0.1)
    # This test does NOT assume the relationship between gamma_train and gamma_big.
    assert info["below_one"] == (info["loop_gain"] < 1)


def test_32_checkpoint_roundtrip(tmp_path, case):
    s, g = case
    W = np.array([[0.22], [-0.33]])
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, s, g, W, "tanh", 0.03)
    ss, gg, ww, act, lam = load_checkpoint(path)
    assert np.array_equal(ss.A, s.A) and np.array_equal(ss.B, s.B)
    assert np.allclose(gg.laplacian.toarray(), g.laplacian.toarray())
    assert np.array_equal(ww, W) and act == "tanh" and lam == 0.03


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not installed in runtime"
)
def test_33_cpu_cuda_consistency(case):
    s, g = case
    X = np.random.default_rng(10).standard_normal((8, 3, 2))
    W = np.array([[0.1], [-0.2]])
    cpu = TorchWorld.from_system(s, g, "cpu")
    gpu = TorchWorld.from_system(s, g, "cuda")
    a = cpu.step(torch.tensor(X), torch.tensor(W))
    b = gpu.step(torch.tensor(X, device="cuda"), torch.tensor(W, device="cuda"))
    for x, y in zip(a, b):
        assert np.allclose(x.cpu().numpy(), y.cpu().numpy(), atol=1e-11)
