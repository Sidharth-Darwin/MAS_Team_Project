"""CSV/JSON reporting, portable model checkpoints, and publication-size figures."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distributed_consensus.graphs import from_edges
from distributed_consensus.control.plants import build_system


def to_jsonable(obj):
    if isinstance(obj,dict): return {str(k):to_jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [to_jsonable(x) for x in obj]
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.integer,np.floating)): return obj.item()
    if isinstance(obj,np.bool_): return bool(obj)
    return obj


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f: json.dump(to_jsonable(obj),f,indent=2,allow_nan=False)


def write_csv(path,rows):
    if not rows: return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()));writer.writeheader()
        writer.writerows([{k:(v if v is not None else '') for k,v in row.items()} for row in rows])


def save_checkpoint(path,system,graph,W,activation='identity',lam=0.):
    """Save numeric-only NPZ; loading never executes pickle or Python objects."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows,cols=graph.edges()
    weights=np.asarray(graph.adjacency[rows,cols]).reshape(-1)
    np.savez_compressed(path,A=system.A,B=system.B,K=system.K,
        target=system.target,W=np.asarray(W,np.float64),n=np.array(system.n),
        row=rows,col=cols,edge_weights=weights,activation=np.array(activation),lam=np.array(lam))


def load_checkpoint(path):
    """Returns System, Graph, W, activation, lambda with exact matching rechecked."""
    with np.load(path,allow_pickle=False) as z:
        system=build_system(z['A'],z['B'],z['target'])
        if not np.allclose(system.K,z['K'],atol=1e-10,rtol=1e-10):
            raise ValueError('checkpoint K differs from recomputed exact-matching K')
        graph=from_edges(int(z['n']),z['row'],z['col'],z['edge_weights'])
        W=z['W'].copy();activation=str(z['activation'].item());lam=float(z['lam'])
    if W.shape!=(2,1) or not np.isfinite(W).all() or activation not in ('identity','tanh'):
        raise ValueError('invalid checkpoint controller')
    return system,graph,W,activation,lam


def save_system(path,system,graph):
    """Persist EVERY inference system's physical matrices and sparse graph, numeric-only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows,cols=graph.edges()
    weights=np.asarray(graph.adjacency[rows,cols]).reshape(-1)
    np.savez_compressed(path,A=system.A,B=system.B,K=system.K,
         target=system.target,n=np.array(system.n),row=rows,col=cols,edge_weights=weights)
