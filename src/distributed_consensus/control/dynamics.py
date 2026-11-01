"""Physical simulation and portable public NumPy stepping/rollout interfaces."""
from dataclasses import dataclass
import numpy as np
import torch
from distributed_consensus.control.controllers import correction_numpy, activate


def step_numpy(states, system, graph, W=None, activation='identity', disturbance=None):
    """One PHYSICAL step. states:(N,2); W:(2,1) or None for baseline.

    Returns dict: states, disagreement (both before action), local, correction,
    total (N,1), next_states (N,2). No simulator/rendering dependency.
    """
    x = np.asarray(states,np.float64)
    if x.shape != (system.n,2) or graph.n != system.n:
        raise ValueError('states/graph/system sizes do not match')
    e = graph.error(x)
    local = np.einsum('nij,nj->ni',system.K,x)
    corr = np.zeros_like(local) if W is None else correction_numpy(e,W,activation)
    total = local + corr
    # External actuator-channel perturbation only at inference. Report commanded
    # physical energy using 'total', never add disturbance to the training objective.
    d = np.zeros_like(total) if disturbance is None else np.asarray(disturbance,np.float64)
    if d.shape != total.shape or not np.isfinite(d).all():
        raise ValueError('disturbance must be finite shape (N,1)')
    next_states = np.einsum('nij,nj->ni',system.A,x) + np.einsum('nij,nj->ni',system.B,total+d)
    return dict(states=x.copy(), disagreement=e, local=local, correction=corr,
                total=total, next_states=next_states)


def rollout_numpy(initial, system, graph, W=None, activation='identity', horizon=100, disturbances=None):
    """Return full histories with X,E length T+1 and actions length T."""
    if horizon < 1: raise ValueError('T>=1')
    x = np.asarray(initial,np.float64).copy()
    if disturbances is not None:
        disturbances=np.asarray(disturbances,np.float64)
        if disturbances.shape != (horizon,system.n,1) or not np.isfinite(disturbances).all():
            raise ValueError('disturbances must be finite shape (T,N,1)')
    xx, ee, ll, vv, uu = [x.copy()], [graph.error(x)], [], [], []
    for k in range(horizon):
        step = step_numpy(x,system,graph,W,activation,
                          None if disturbances is None else disturbances[k])
        ll.append(step['local']); vv.append(step['correction']); uu.append(step['total'])
        x = step['next_states']
        xx.append(x.copy()); ee.append(graph.error(x))
    return { 'states':np.stack(xx), 'disagreement':np.stack(ee),
             'local':np.stack(ll), 'correction':np.stack(vv),'total':np.stack(uu)}


@dataclass
class TorchWorld:
    """Differentiable batch simulator; keeps physical agents fixed, uses neighbor indices."""
    A: torch.Tensor
    B: torch.Tensor
    K: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    edge_weights: torch.Tensor
    degrees: torch.Tensor

    @classmethod
    def from_system(cls, system, graph, device='cpu'):
        if system.n != graph.n: raise ValueError('graph/system mismatch')
        dt = torch.float64
        rows,cols = graph.edges()
        a = graph.adjacency.tocsr()
        weights = np.asarray(a[rows,cols]).reshape(-1).astype(np.float64)
        return cls(*(torch.as_tensor(np.array(arr,copy=True),dtype=dt,device=device) for arr in
                     (system.A,system.B,system.K)),
                   torch.as_tensor(rows,device=device,dtype=torch.long),
                   torch.as_tensor(cols,device=device,dtype=torch.long),
                   torch.as_tensor(weights,device=device,dtype=dt),
                   torch.as_tensor(graph.degrees,device=device,dtype=dt))

    def error(self, X):
        """X:(batch,N,2), E same, using local neighborhood data only."""
        neighbors = X.index_select(1,self.cols) * self.edge_weights[None,:,None]
        summation = torch.zeros_like(X).index_add(1,self.rows,neighbors)
        return X - summation / self.degrees[None,:,None]

    def step(self, X, W=None, activation='identity'):
        """Physical dynamics with local action and optional learned correction."""
        e = self.error(X)
        local = torch.einsum('nij,bnj->bni',self.K,X)
        correction = torch.zeros_like(local) if W is None else activate(e @ W,activation)
        u = local + correction
        x_next = torch.einsum('nij,bnj->bni',self.A,X) + torch.einsum('nij,bnj->bni',self.B,u)
        return x_next,e,local,correction,u

    def costs(self, X0, W, horizon, activation='identity'):
        """Mean J_E,J_U using T action indices 0..T-1, and terminal norm separately."""
        X=X0; je=torch.zeros((),dtype=X.dtype,device=X.device); ju=je.clone()
        for _ in range(horizon):
            X,e,_,_,u=self.step(X,W,activation)
            je=je+e.square().sum(dim=(1,2)).mean()
            ju=ju+u.square().sum(dim=(1,2)).mean()
        terminal=self.error(X).square().sum(dim=(1,2)).mean()
        return je,ju,terminal