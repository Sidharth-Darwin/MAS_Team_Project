"""Undirected simple graphs with random-walk Laplacian L = I - D^{-1} A."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Graph:
    """adjacency: N x N undirected CSR; graph operations act on agent-major arrays."""
    adjacency: sparse.csr_matrix

    def __post_init__(self):
        a = sparse.csr_matrix(self.adjacency, dtype=np.float64)
        a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
        if a.shape[0] < 2 or a.shape[0] != a.shape[1]:
            raise ValueError('Adjacency must be square with N>=2')
        if a.nnz and (np.any(a.data <= 0) or not np.all(np.isfinite(a.data))):
            raise ValueError('Edges must be positive finite weights')
        if np.any(a.diagonal() != 0):
            raise ValueError('Self-loops are not supported')
        if (a - a.T).nnz and np.max(np.abs((a-a.T).data)) > 1e-12:
            raise ValueError('Graph must be undirected (symmetric)')
        deg = np.asarray(a.sum(axis=1)).ravel()
        if np.any(deg <= 0) or connected_components(a, directed=False)[0] != 1:
            raise ValueError('Graph must be connected with positive node degrees')
        object.__setattr__(self, 'adjacency', a)

    @property
    def n(self):
        return self.adjacency.shape[0]

    @property
    def degrees(self):
        return np.asarray(self.adjacency.sum(axis=1)).ravel()

    @property
    def p_rw(self):
        return sparse.diags(1. / self.degrees) @ self.adjacency

    @property
    def laplacian(self):
        return sparse.eye(self.n, format='csr') - self.p_rw

    def error(self, states):
        """For X (N,p), return E=L_rw X, using CSR sparse multiplication."""
        x = np.asarray(states, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != self.n:
            raise ValueError('states must have shape (N,p)')
        return x - self.p_rw @ x

    def edges(self):
        rows, cols = self.adjacency.nonzero()
        return rows.astype(np.int64), cols.astype(np.int64)


def from_edges(n, rows, cols, weights=None):
    """Build graph from ALL directed CSR entries (both orientations required)."""
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    data = np.ones(len(rows)) if weights is None else np.asarray(weights, np.float64)
    return Graph(sparse.coo_matrix((data, (rows, cols)), shape=(n,n)).tocsr())


def make_graph(n, topology='path', seed=0, neighborhood=8):
    """Deterministic connected undirected path, ring, wide_ring, or random tree-plus-edges."""
    if n < 2: raise ValueError('N>=2 required')
    pairs = set()
    def add(i,j):
        if i != j: pairs.add((min(i,j), max(i,j)))
    if topology == 'path':
        for i in range(n-1): add(i,i+1)
    elif topology in ('ring','wide_ring'):
        k = 1 if topology == 'ring' else min(int(neighborhood), n-1)
        for i in range(n):
            for j in range(1,k+1): add(i,(i+j)%n)
    elif topology == 'random':
        rng = np.random.default_rng(seed)
        for i in range(1,n): add(i, int(rng.integers(0,i)))
        for i in range(n):
            for j in rng.integers(0,n,size=2): add(i,int(j))
    else:
        raise ValueError(f'unknown graph topology: {topology}')
    rows, cols = [], []
    for i,j in sorted(pairs):
        rows.extend((i,j)); cols.extend((j,i))
    return from_edges(n,rows,cols)
