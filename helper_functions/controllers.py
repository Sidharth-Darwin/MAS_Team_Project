"""Exactly one shared W:(p,1), no bias or additional message-passing layers."""
import numpy as np
import torch


def activate(z, activation='identity'):
    if activation == 'identity': return z
    if activation == 'tanh':
        return torch.tanh(z) if isinstance(z,torch.Tensor) else np.tanh(z)
    raise ValueError('activation must be identity or tanh')


def correction_numpy(error, W, activation='identity'):
    """error:(N,p); W:(p,1); correction:(N,1)."""
    e, w = np.asarray(error), np.asarray(W)
    if e.ndim != 2 or w.shape != (e.shape[1],1): raise ValueError('error/W shapes')
    return activate(e @ w,activation)


def project_spectral_(W, cap):
    """Projected-gradient step onto ||W||_2<=cap, in-place; returns post-projection norm."""
    if cap <= 0 or not np.isfinite(cap): raise ValueError('cap must be finite positive')
    with torch.no_grad():
        norm = torch.linalg.matrix_norm(W, ord=2)
        if norm.item() > cap: W.mul_(cap / norm)
        return float(torch.linalg.matrix_norm(W,ord=2).item())


def weight_norm(W):
    """Singular-value spectral norm, not Frobenius (equal for rank-one column W)."""
    return float(np.linalg.norm(np.asarray(W),ord=2))
