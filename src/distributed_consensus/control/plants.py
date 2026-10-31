from dataclasses import dataclass
import numpy as np

DEFAULT_TARGET = np.array([[1., .1],[-.2, .8]], dtype=np.float64)


def check_target(target):
    """Check A_target (p,p) is real and Schur stable; return its spectral radius."""
    a = np.asarray(target, np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError('target must be finite square matrix')
    rho = float(np.max(np.abs(np.linalg.eigvals(a))))
    if rho >= 1: 
        raise ValueError(f'target not Schur stable: rho={rho}')
    return rho


@dataclass(frozen=True)
class System:
    """A:(N,p,p), B:(N,p,1), K:(N,1,p), target:(p,p), exact matching required."""
    A: np.ndarray
    B: np.ndarray
    K: np.ndarray
    target: np.ndarray

    @property
    def n(self): 
        return self.A.shape[0]
    @property
    def p(self): 
        return self.A.shape[1]


def build_system(A, B, target=DEFAULT_TARGET, tol=1e-10):
    """Compute K_i=pinv(B_i)(target-A_i), rejecting all incompatible agents.

    No assumption of controllability. The project uses p=2,m=1 and B_i=[0,b_i].
    """
    a = np.array(A, dtype=np.float64, copy=True)
    b = np.array(B, dtype=np.float64, copy=True)
    target = np.array(target, dtype=np.float64, copy=True)
    check_target(target)
    if a.ndim != 3 or a.shape[1:] != (2,2) or b.shape != (a.shape[0],2,1) or target.shape != (2,2):
        raise ValueError('expected A:(N,2,2), B:(N,2,1), target:(2,2)')
    if a.shape[0] < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('at least two agents and finite entries required')
    if np.max(np.abs(b[:,0,0])) > tol or np.any(np.abs(b[:,1,0]) <= tol):
        raise ValueError('B_i must equal [0,b_i]^T with b_i nonzero')
    pinvs = np.linalg.pinv(b)
    diffs = target[None,:,:] - a
    compat = diffs - b @ (pinvs @ diffs)
    if np.max(np.abs(compat)) > tol:
        raise ValueError(f'incompatible matrices: matching compatibility residual {np.max(np.abs(compat)):.3e}')
    K = pinvs @ diffs
    matching = a + b @ K
    if np.max(np.abs(matching - target)) > tol:
        raise ValueError('numeric matching residual exceeds tolerance')
    if np.unique(np.round(b[:,1,0], decimals=12)).size < 2 or np.unique(np.round(a[:,1,:], decimals=12),axis=0).shape[0] < 2:
        raise ValueError('agents must have heterogeneous B coefficients and second rows')
    for arr in (a,b,K,target): 
        arr.setflags(write=False)
    return System(a,b,K,target)


def generate_system(n, target=DEFAULT_TARGET, seed=5, b_range=(.065,.14)):
    """Compatible heterogeneous A_i: fixed first row, independently sampled second rows and b_i."""
    if n < 2: 
        raise ValueError('N>=2')
    rng = np.random.default_rng(seed)
    target = np.asarray(target,np.float64)
    a = np.broadcast_to(target, (n,2,2)).copy()
    a[:,1,0] = rng.uniform(-.42,-.06,n)
    a[:,1,1] = rng.uniform(.55,.95,n)
    b = np.zeros((n,2,1),dtype=np.float64)
    b[:,1,0] = rng.uniform(*b_range,size=n)
    return build_system(a,b,target)


def matching_residual(system):
    return float(np.max(np.abs(system.A + system.B @ system.K - system.target)))
