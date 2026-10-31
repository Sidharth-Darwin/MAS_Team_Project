"""Fixed-plant, differentiable projected-Adam optimization of shared W only."""
import numpy as np
import torch
from .controllers import project_spectral_
from .dynamics import TorchWorld
from .plant_gain import full_gain


def set_reproducible(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def baseline_scales(system,graph,horizon=70,samples=128,seed=291,device='cpu'):
    """Estimate train-system relative cost scales before choosing positive lambdas."""
    world=TorchWorld.from_system(system,graph,device)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    batch=torch.randn((samples,system.n,2),dtype=torch.float64,generator=gen).to(device)
    with torch.no_grad(): je,ju,_=world.costs(batch,None,horizon)
    return float(je.item()),float(ju.item())


def train(system,graph,lam=0.,activation='identity',epochs=200,horizon=80,batch=128,
          val_batch=128,lr=.002,eta=.95,seed=7,device='cpu',val_every=10):
    """Learn W:(2,1) only, with J_E+lambda*J_U, projected after EACH step.

    All matrices, graph and K_i fixed across iterations. Train and validation RNG
    streams are independent and seed-separated; repeated runs with same seed/lambda
    use identical initial weights and training samples.
    """
    if lam<0 or not np.isfinite(lam): raise ValueError('lambda must be >=0 and finite')
    if activation not in ('identity','tanh'): raise ValueError('activation')
    if epochs<1 or horizon<1 or batch<1: raise ValueError('invalid training sizes')
    gamma=full_gain(system,graph).gamma
    cap=eta/gamma
    set_reproducible(seed)
    world=TorchWorld.from_system(system,graph,device)
    generator=torch.Generator(device='cpu').manual_seed(seed+10001)
    valgen=torch.Generator(device='cpu').manual_seed(seed+20003)
    valX=torch.randn((val_batch,system.n,2),dtype=torch.float64,generator=valgen).to(device)
    W=torch.nn.Parameter(torch.zeros((2,1),dtype=torch.float64,device=device))
    opt=torch.optim.Adam([W],lr=lr)
    history=[]
    for epoch in range(1,epochs+1):
        initial=torch.randn((batch,system.n,2),dtype=torch.float64,generator=generator).to(device)
        opt.zero_grad(set_to_none=True)
        je,ju,_=world.costs(initial,W,horizon,activation)
        loss=je+lam*ju
        if not torch.isfinite(loss): raise ArithmeticError('nonfinite objective during training')
        loss.backward()
        opt.step()
        norm=project_spectral_(W,cap)
        if norm>cap+1e-10: raise AssertionError('spectral projection failed')
        record={'epoch':epoch,'train_J_E':float(je.item()),'train_J_U':float(ju.item()),
                'train_objective':float(loss.item()),'W_spectral_norm':norm,
                'cap':cap, 'val_J_E':None,'val_J_U':None,'val_objective':None}
        if epoch==1 or epoch%val_every==0 or epoch==epochs:
            with torch.no_grad():
                vje,vju,_=world.costs(valX,W,horizon,activation)
            record.update(val_J_E=float(vje.item()),val_J_U=float(vju.item()),
                          val_objective=float((vje+lam*vju).item()))
        history.append(record)
    return {'W':W.detach().cpu().numpy().copy(),'history':history,'cap':cap,
            'gamma_train':gamma,'lambda':float(lam),'activation':activation,
            'seed':seed,'train_seed':seed+10001,'validation_seed':seed+20003}
