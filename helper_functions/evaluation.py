"""Paired, finite-horizon evaluation; separate convergence and small-gain criteria."""
import numpy as np
from .dynamics import rollout_numpy


def trajectory_metrics(history,epsilon_fraction=.01):
    """T+1 disagreements, T physical actions; null crossing if never in threshold."""
    e=history['disagreement']; u=history['total']
    t=len(u)
    norms=np.linalg.norm(e.reshape(t+1,-1),axis=1)
    e0=norms[0]
    threshold=epsilon_fraction*e0
    if e0<=1e-12 and np.max(norms)<1e-9:
        cross=0; settle=0
    else:
        in_bound=norms<=threshold
        crossing=np.flatnonzero(in_bound)
        cross=int(crossing[0]) if crossing.size else None
        violating=np.flatnonzero(~in_bound)
        settle=int(violating[-1]+1) if violating.size and violating[-1]<t else (None if violating.size else 0)
    # Mean-centered metric is topology-independent: (1/N) sum_i ||x_i-xbar||^2.
    # T controls, with no terminal stage cost, as for the original J_E.
    x=history.get('states')
    if x is None:
        jpair=None; pairmax=None; pairterminal=None; pairsettle=None
        normalized=None; eper=None; uper=None
    else:
        n=x.shape[1]
        center=x-x.mean(axis=1,keepdims=True)
        paircurve=np.square(center).sum(axis=(1,2))/n
        jpair=float(paircurve[:-1].sum())
        pairmax=float(np.sqrt(paircurve.max()))
        pairterminal=float(np.sqrt(paircurve[-1]))
        pairnorm=np.sqrt(paircurve)
        if pairnorm[0]<=1e-12 and np.max(pairnorm)<1e-9:
            pairsettle=0
        else:
            violations=np.flatnonzero(pairnorm>epsilon_fraction*pairnorm[0])
            pairsettle=(None if violations.size and violations[-1]==t
                        else int(violations[-1]+1) if violations.size else 0)
        normalized=float(np.square(e[:-1]).sum() / max(e0**2,1e-14))
        eper=float(np.square(e[:-1]).sum()/n)
        uper=float(np.square(u).sum()/n)
    return {'J_pair':jpair,'J_E_per_agent':eper,'J_U_per_agent':uper,
            'J_E_initial_normalized':normalized,'max_centered_rms':pairmax,
            'terminal_centered_rms':pairterminal,'pair_settling_time':pairsettle,
            'J_E':float(np.square(e[:-1]).sum()),
            'J_U':float(np.square(u).sum()),
            'J_corr':float(np.square(history['correction']).sum()),
            'first_crossing':cross,'settling_time':settle,
            'settled':settle is not None,'max_E':float(norms.max()),
            'terminal_E':float(norms[-1]),'initial_E':float(e0),
            'threshold':float(threshold),'norm_curve':norms}


def evaluate_pair(system,graph,W,horizon=100,samples=24,seed=413,activation='identity',
                  initial_states=None,disturbances=None):
    """Compare baseline vs learned on identical x0, same physical matrices and graph.

    Returns per-trajectory paired records, averaged curves and representative
    full trajectory histories. Unsettled times are recorded as None (censored).
    """
    rng=np.random.default_rng(seed)
    if initial_states is not None:
        initial_states=np.asarray(initial_states,np.float64)
        if initial_states.shape != (samples,system.n,2):
            raise ValueError('initial_states must be (samples,N,2)')
    if disturbances is not None:
        disturbances=np.asarray(disturbances,np.float64)
        if disturbances.shape != (samples,horizon,system.n,1):
            raise ValueError('disturbances must be (samples,T,N,1)')
    records=[]; curves_baseline=[]; curves_learned=[]
    representative={}
    for s in range(samples):
        x0=rng.standard_normal((system.n,2)) if initial_states is None else initial_states[s]
        d=None if disturbances is None else disturbances[s]
        base=rollout_numpy(x0,system,graph,None,horizon=horizon,disturbances=d)
        learned=rollout_numpy(x0,system,graph,W,activation,horizon,disturbances=d)
        bm=trajectory_metrics(base); lm=trajectory_metrics(learned)
        if s==0: representative={'baseline':base,'learned':learned}
        row={'sample':s}
        for label,m in (('baseline',bm),('learned',lm)):
            for k,v in m.items():
                if k!='norm_curve': row[f'{label}_{k}']=v
        for k in ('J_E','J_U','J_pair','terminal_E'):
            denom=bm[k]
            row[f'{k}_improvement_pct']=100*(denom-lm[k])/denom if denom>1e-14 else None
        row['paired_win_J_E']=lm['J_E']<bm['J_E']
        records.append(row)
        curves_baseline.append(bm['norm_curve']); curves_learned.append(lm['norm_curve'])
    def describe(label):
        key=lambda k: np.array([r[f'{label}_{k}'] for r in records],np.float64)
        crossing=[r[f'{label}_first_crossing'] for r in records]
        settling=[r[f'{label}_settling_time'] for r in records]
        pair_settling=[r[f'{label}_pair_settling_time'] for r in records]
        return {'mean_J_pair':float(key('J_pair').mean()),
                'mean_J_E_per_agent':float(key('J_E_per_agent').mean()),
                'mean_J_U_per_agent':float(key('J_U_per_agent').mean()),
                'mean_J_E_initial_normalized':float(key('J_E_initial_normalized').mean()),
                'mean_max_centered_rms':float(key('max_centered_rms').mean()),
                'pair_settled_fraction':float(np.mean([v is not None for v in pair_settling])),
                'mean_pair_settling_if_settled':float(np.mean([v for v in pair_settling if v is not None])) if any(v is not None for v in pair_settling) else None,
                'mean_J_E':float(key('J_E').mean()),'mean_J_U':float(key('J_U').mean()),
                'mean_J_corr':float(key('J_corr').mean()),
                'mean_max_E':float(key('max_E').mean()),
                'mean_terminal_E':float(key('terminal_E').mean()),
                'crossed_fraction':float(np.mean([v is not None for v in crossing])),
                'settled_fraction':float(np.mean([v is not None for v in settling])),
                'mean_first_crossing_if_crossed':float(np.mean([v for v in crossing if v is not None])) if any(v is not None for v in crossing) else None,
                'mean_settling_if_settled':float(np.mean([v for v in settling if v is not None])) if any(v is not None for v in settling) else None}
    base_stats=describe('baseline'); learned_stats=describe('learned')
    gainpct=100*(base_stats['mean_J_E']-learned_stats['mean_J_E'])/base_stats['mean_J_E']
    energypct=100*(base_stats['mean_J_U']-learned_stats['mean_J_U'])/base_stats['mean_J_U']
    jpairpct=100*(base_stats['mean_J_pair']-learned_stats['mean_J_pair'])/max(base_stats['mean_J_pair'],1e-14)
    return {'baseline':base_stats,'learned':learned_stats,
            'J_pair_improvement_pct':float(jpairpct),
            'J_E_improvement_pct':float(gainpct),'J_U_reduction_pct':float(energypct),
            'paired_win_rate':float(np.mean([r['paired_win_J_E'] for r in records])),
            'records':records,'curves':{'baseline':np.mean(curves_baseline,axis=0),
                                      'learned':np.mean(curves_learned,axis=0)},
            'representative':representative}