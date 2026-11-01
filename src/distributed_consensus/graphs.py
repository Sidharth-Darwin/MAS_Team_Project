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
        if (a != a.T).nnz:
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


def make_graph(n, topology='chain', seed=0, neighborhood=8, hub=0):
    """Connected undirected star, chain, ring, wide_ring, complete, random graphs."""
    if n < 2: raise ValueError('N>=2 required')
    pairs = set()
    def add(i,j):
        if i != j: pairs.add((min(i,j), max(i,j)))
    if topology == 'chain':
        for i in range(n-1): add(i,i+1)
    elif topology == 'star':
        if not 0<=hub<n: raise ValueError('star hub must be a valid vertex index')
        for i in range(n): add(hub,i)
    elif topology == 'complete':
        for i in range(n):
            for j in range(i+1,n): add(i,j)
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


def load_graph(path):
    """Load a validated undirected Graph from a standard adjacency file.

    Supported: dense NumPy .npy, scipy.sparse.save_npz .npz, or .npz with
    an ``adjacency`` matrix. Also accepts the row/col/n/edge_weights keys of
    the existing numeric-only saved-system files.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix == '.npy':
        return Graph(np.load(path, allow_pickle=False))
    if path.suffix != '.npz':
        raise ValueError('graph must be .npy or .npz')
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        if {'data', 'indices', 'indptr', 'shape', 'format'} <= names:
            # SciPy sparse .npz: preserve sparse representation for large graphs.
            return Graph(sparse.load_npz(path))
        if 'adjacency' in names:
            return Graph(data['adjacency'])
        if {'n', 'row', 'col', 'edge_weights'} <= names:
            return from_edges(int(data['n']), data['row'], data['col'], data['edge_weights'])
    raise ValueError('unsupported .npz graph: expected sparse CSR, adjacency, or saved graph edges')