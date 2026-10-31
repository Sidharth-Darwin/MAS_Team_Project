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
    return {'J_E':float(np.square(e[:-1]).sum()),
            'J_U':float(np.square(u).sum()),
            'J_corr':float(np.square(history['correction']).sum()),
            'first_crossing':cross,'settling_time':settle,
            'settled':settle is not None,'max_E':float(norms.max()),
            'terminal_E':float(norms[-1]),'initial_E':float(e0),
            'threshold':float(threshold),'norm_curve':norms}


def evaluate_pair(system,graph,W,horizon=100,samples=24,seed=413,activation='identity'):
    """Compare baseline vs learned on identical x0, same physical matrices and graph.

    Returns per-trajectory paired records, averaged curves and representative
    full trajectory histories. Unsettled times are recorded as None (censored).
    """
    rng=np.random.default_rng(seed)
    records=[]; curves_baseline=[]; curves_learned=[]
    representative={}
    for s in range(samples):
        x0=rng.standard_normal((system.n,2))
        base=rollout_numpy(x0,system,graph,None,horizon=horizon)
        learned=rollout_numpy(x0,system,graph,W,activation,horizon)
        bm=trajectory_metrics(base); lm=trajectory_metrics(learned)
        if s==0: representative={'baseline':base,'learned':learned}
        row={'sample':s}
        for label,m in (('baseline',bm),('learned',lm)):
            for k,v in m.items():
                if k!='norm_curve': row[f'{label}_{k}']=v
        for k in ('J_E','J_U','terminal_E'):
            denom=bm[k]
            row[f'{k}_improvement_pct']=100*(denom-lm[k])/denom if denom>1e-14 else None
        row['paired_win_J_E']=lm['J_E']<bm['J_E']
        records.append(row)
        curves_baseline.append(bm['norm_curve']); curves_learned.append(lm['norm_curve'])
    def describe(label):
        key=lambda k: np.array([r[f'{label}_{k}'] for r in records],np.float64)
        crossing=[r[f'{label}_first_crossing'] for r in records]
        settling=[r[f'{label}_settling_time'] for r in records]
        return {'mean_J_E':float(key('J_E').mean()),'mean_J_U':float(key('J_U').mean()),
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
    return {'baseline':base_stats,'learned':learned_stats,
            'J_E_improvement_pct':float(gainpct),'J_U_reduction_pct':float(energypct),
            'paired_win_rate':float(np.mean([r['paired_win_J_E'] for r in records])),
            'records':records,'curves':{'baseline':np.mean(curves_baseline,axis=0),
                                      'learned':np.mean(curves_learned,axis=0)},
            'representative':representative}
