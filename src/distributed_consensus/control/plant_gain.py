"""Discrete-time H-infinity gain via complete two-state stationary-point candidates.

For b0=[0,1]^T, the squared vector transfer norm is a ratio of a linear
and quadratic polynomial in t=cos(omega). Evaluating all interior stationary
points and both endpoints finds the global maximum in exact arithmetic.
Reported values are floating-point estimates, NOT outward-rounded interval proofs.
"""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from distributed_consensus.control.plants import check_target


@dataclass(frozen=True)
class Gain:
    gamma0: float
    omega_peak: float
    spatial_norm: float
    gamma: float
    method: str = "stationary candidates (floating-point) + sparse spectral norm"


def scalar_gain(target):
    """Return (global candidate-maximum gamma0, peak omega), for 2x2 real Schur target."""
    a = np.asarray(target, np.float64)
    check_target(a)
    if a.shape != (2, 2):
        raise ValueError("only 2-state target supported")
    tau = float(np.trace(a))
    delta = float(np.linalg.det(a))
    n0 = float(a[0, 1] ** 2 + 1 + a[0, 0] ** 2)
    n1 = float(-2 * a[0, 0])
    d0 = (1 - delta) ** 2 + tau**2
    d1 = -2 * tau * (1 + delta)
    d2 = 4 * delta
    # d/dt [N(t)/D(t)] numerator, in descending powers of t [c2, c1, c0]:
    polynomial = np.array([-n1 * d2, -2 * n0 * d2, n1 * d0 - n0 * d1])
    coef = polynomial.copy()
    while len(coef) > 1 and abs(coef[0]) < 1e-14:
        coef = coef[1:]
    roots = [] if len(coef) == 1 else np.roots(coef)
    candidates = [-1.0, 1.0]
    for r in roots:
        if abs(np.imag(r)) < 1e-9 and -1 - 1e-12 <= np.real(r) <= 1 + 1e-12:
            candidates.append(float(np.clip(np.real(r), -1, 1)))
    values = []
    for t in candidates:
        denom = d0 + d1 * t + d2 * t * t
        if denom <= 0:
            raise ArithmeticError(
                "nonpositive denominator; check target and conditioning"
            )
        v = (n0 + n1 * t) / denom
        if v < 0 and v < -1e-9:
            raise ArithmeticError("negative gain squared")
        values.append(max(0.0, v))
    idx = int(np.argmax(values))
    return float(np.sqrt(values[idx])), float(np.arccos(candidates[idx]))


def spatial_norm(graph, b):
    """Spectral 2-norm of sparse L_rw @ diag(b), no dense Kronecker assembly."""
    b = np.asarray(b, np.float64).reshape(-1)
    if b.size != graph.n or np.any(b == 0):
        raise ValueError("invalid input vector b")
    mat = graph.laplacian @ sparse.diags(b)
    if graph.n <= 16:
        return float(np.linalg.norm(mat.toarray(), 2))
    gram = (mat.T @ mat).asfptype()
    maximum = float(
        eigsh(
            gram, k=1, which="LA", tol=1e-11, return_eigenvectors=False, maxiter=10000
        )[0]
    )
    return float(np.sqrt(max(0, maximum)))


def full_gain(system, graph):
    """Actual full N-agent gain factorization for B_i=[0,b_i]^T."""
    if system.n != graph.n:
        raise ValueError("graph/system sizes differ")
    g0, omega = scalar_gain(system.target)
    sn = spatial_norm(graph, system.B[:, 1, 0])
    return Gain(g0, omega, sn, g0 * sn)


def direct_grid_gain(system, graph, points=2049):
    """Diagnostic ONLY: peak from full dense resolvent across finite frequency grid.

    Intended for small N; independent of the factorized formula.
    """
    n = system.n
    if n > 10:
        raise ValueError("dense direct check only for N<=10")
    a_e = np.kron(np.eye(n), system.target)
    blocks = sparse.block_diag([system.B[i] for i in range(n)]).toarray()
    be = np.kron(graph.laplacian.toarray(), np.eye(2)) @ blocks
    eye = np.eye(2 * n)
    maximum = 0.0
    for omega in np.linspace(0.0, np.pi, points):
        g = np.linalg.solve(np.exp(1j * omega) * eye - a_e, be)
        maximum = max(maximum, float(np.linalg.svd(g, compute_uv=False)[0]))
    return maximum


def small_gain(system, graph, W, eta=0.95):
    """Return measured gamma, W spectral norm, loop product, numeric sufficient-status."""
    gain = full_gain(system, graph)
    gamma = gain.gamma
    norm = float(np.linalg.norm(np.asarray(W), ord=2))
    product = gamma * norm
    status = (
        "at_or_below_eta"
        if product <= eta
        else "between_eta_and_one"
        if product < 1.0
        else "at_or_above_one"
    )
    return {
        "gamma": gamma,
        "W_spectral_norm": norm,
        "loop_gain": product,
        "below_one": bool(product < 1.0),
        "within_eta": bool(product <= eta),
        "status": status,
        "eta": eta,
        "gain_method": gain.method,
        "rigorously_certified_upper_bound": False,
    }
