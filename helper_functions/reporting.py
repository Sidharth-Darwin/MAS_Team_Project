"""CSV/JSON reporting, portable model checkpoints, and publication-size figures."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .graphs import from_edges
from .plants import build_system


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


def save_training_plot(path,history,name):
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    x=[r['epoch'] for r in history]
    for ax,label in zip(axes,('J_E','J_U')):
        ax.plot(x,[r[f'train_{label}'] for r in history],label='train')
        vv=[(r['epoch'],r[f'val_{label}']) for r in history if r[f'val_{label}'] is not None]
        ax.plot([v[0] for v in vv],[v[1] for v in vv],label='held-out validation')
        ax.set(xlabel='Epoch',ylabel=label,title=name);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(str(path)+'.png',dpi=160);fig.savefig(str(path)+'.svg');plt.close(fig)


def save_curve_plot(path,curves,label):
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(curves['baseline'],label='local-only')
    ax.plot(curves['learned'],label='learned correction')
    ax.set(xlabel='Physical timestep',ylabel='Mean disagreement 2-norm',title=label)
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(str(path)+'.png',dpi=160);fig.savefig(str(path)+'.svg');plt.close(fig)


def save_tradeoff_plot(path,rows):
    # Rows contain held-out metrics for different lambdas on the SAME small network.
    if not rows:return
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    labels=[f"{r['lambda']:.4g}" for r in rows]
    x=np.arange(len(rows))
    for ax,key,title in zip(axs,('mean_J_E','mean_J_U'),('Disagreement','Physical control energy')):
        ax.bar(x,[r[key] for r in rows]);ax.set_xticks(x,labels)
        ax.set(xlabel='lambda',ylabel=key,title=title);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=160)
    fig.savefig(str(path)+'.svg');plt.close(fig)


def save_system(path,system,graph):
    """Persist EVERY inference system's physical matrices and sparse graph, numeric-only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows,cols=graph.edges()
    weights=np.asarray(graph.adjacency[rows,cols]).reshape(-1)
    np.savez_compressed(path,A=system.A,B=system.B,K=system.K,
         target=system.target,n=np.array(system.n),row=rows,col=cols,edge_weights=weights)


def save_curve_csv(path,curves):
    write_csv(path,[{'k':k,'baseline_mean_disagreement':float(curves['baseline'][k]),
                     'learned_mean_disagreement':float(curves['learned'][k])}
                    for k in range(len(curves['baseline']))])
