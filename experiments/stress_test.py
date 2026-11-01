"""Frozen-controller robustness checks. Disturbance is an exogenous actuator-input perturbation.

The reported J_U measures commanded physical control, excluding the external perturbation.
This script never modifies the training method or modifies/rescales W.
"""
import argparse
from pathlib import Path
import numpy as np

from distributed_consensus.graphs import make_graph
from distributed_consensus.control.plants import generate_system
from distributed_consensus.control.plant_gain import small_gain
from distributed_consensus.evaluation.metrics import evaluate_pair
from distributed_consensus.evaluation.reporting import load_checkpoint,write_csv,write_json


def run_stress(W,activation='identity',sizes=(100,1000),samples=3,horizon=70,
               seed=21,disturbance_amplitude=3.,b_ranges=((.065,.14),(.06,.3))):
    W=np.asarray(W,np.float64);original=W.copy()
    rows=[]
    for n in sizes:
        for physical_case,b_range in enumerate(b_ranges):
            system=generate_system(n,seed=seed+17*n+physical_case,b_range=b_range)
            initial=np.random.default_rng(seed+200+n).standard_normal((samples,n,2))
            cases=[(topology,0) for topology in ('star','chain','ring','wide_ring','random')]
            # Vary which heterogeneous physical B_i occupies the high-degree star hub.
            cases += [('star',int(np.argmin(system.B[:,1,0]))),
                      ('star',int(np.argmax(system.B[:,1,0])))]
            for topology,hub in dict.fromkeys(cases):
                graph=make_graph(n,topology,seed=seed+900+n,hub=hub)
                sg=small_gain(system,graph,W)
                for disturbance_case in ('none','pulse'):
                    disturbances=np.zeros((samples,horizon,n,1))
                    if disturbance_case=='pulse':
                        left=min(horizon//2,25)
                        disturbances[:,left:min(left+5,horizon),0,0]=disturbance_amplitude
                    pair=evaluate_pair(system,graph,W,activation=activation,
                        horizon=horizon,samples=samples,initial_states=initial,
                        disturbances=disturbances)
                    rows.append({'N':n,'topology':topology,'star_hub':hub if topology=='star' else None,
                        'physical_case':physical_case,'b_min':b_range[0],'b_max':b_range[1],
                        'disturbance_case':disturbance_case,'disturbance_amplitude':disturbance_amplitude,
                        'gamma':sg['gamma'],'W_norm':sg['W_spectral_norm'],
                        'product':sg['loop_gain'],'small_gain_status':sg['status'],
                        'rigorously_certified_upper_bound':False,
                        'JE_improvement_pct':pair['J_E_improvement_pct'],
                        'Jpair_improvement_pct':pair['J_pair_improvement_pct'],
                        'physical_energy_reduction_pct':pair['J_U_reduction_pct'],
                        'paired_win_rate':pair['paired_win_rate'],
                        'baseline_pair':pair['baseline']['mean_J_pair'],
                        'learned_pair':pair['learned']['mean_J_pair'],
                        'baseline_energy_per_agent':pair['baseline']['mean_J_U_per_agent'],
                        'learned_energy_per_agent':pair['learned']['mean_J_U_per_agent'],
                        'baseline_settling':pair['baseline']['mean_settling_if_settled'],
                        'learned_settling':pair['learned']['mean_settling_if_settled']})
    if not np.array_equal(original,W): raise AssertionError('frozen W changed')
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--sizes',nargs='+',type=int,default=[100,1000])
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--horizon',type=int,default=70)
    p.add_argument('--seed',type=int,default=21)
    p.add_argument('--disturbance-amplitude',type=float,default=3.)
    p.add_argument('--out',default='results/stress_tests')
    a=p.parse_args(argv)
    _,_,W,activation,_=load_checkpoint(a.checkpoint)
    rows=run_stress(W,activation,a.sizes,a.samples,a.horizon,a.seed,a.disturbance_amplitude)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'stress_results.csv',rows)
    write_json(out/'stress_results.json',{'rows':rows,'source_checkpoint':a.checkpoint,
       'interpretation':'Not formally certified; gain >=1 implies inconclusive sufficient condition, not instability.'})
    print('Saved',out.resolve(),len(rows),'evaluations')

if __name__=='__main__':
    main()
