"""Paired graph-topology inference, with frozen checkpoint weights.

Run at project root: python experiments/compare_topologies.py --help
The same physical System and same initial states are reused at each N.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helper_functions.graphs import make_graph
from helper_functions.plants import generate_system
from helper_functions.plant_gain import full_gain, small_gain, spatial_norm
from helper_functions.evaluation import evaluate_pair
from helper_functions.reporting import load_checkpoint, write_csv, write_json

TOPOLOGIES=('star','chain','ring','wide_ring','complete','random')


def topology_comparison(W, activation='identity', sizes=(3,5,10,100,1000),
                        topologies=TOPOLOGIES, samples=3, horizon=60, seed=16,
                        include_large_complete=False):
    """Produce topology-comparable metrics; never mutate, rescale, or retrain W."""
    W=np.asarray(W,np.float64)
    before=W.copy()
    rows=[]
    for n in sizes:
        system=generate_system(n,seed=seed+10000+n)
        initial=np.random.default_rng(seed+900+n).standard_normal((samples,n,2))
        for topo in topologies:
            if topo=='complete' and n>=1000 and not include_large_complete:
                print('Skipping dense N>=1000 complete graph (enable --include-large-complete)')
                continue
            graph=make_graph(n,topo,seed=seed+37*n)
            gain=full_gain(system,graph)
            sg=small_gain(system,graph,W)
            pair=evaluate_pair(system,graph,W,horizon=horizon,samples=samples,
                               initial_states=initial,activation=activation)
            # Numeric ||L_rw|| uses b=1 for spatial norm; star has an analytic identity.
            lap_norm=spatial_norm(graph,np.ones(n))
            r={'N':n,'topology':topo,'graph_seed':seed+37*n,
               'physical_seed':seed+10000+n,'initial_seed':seed+900+n,
               'Lrw_norm':lap_norm,
               'star_formula_N_over_sqrt_Nminus1':n/np.sqrt(n-1) if topo=='star' else None,
               'degree_min':float(graph.degrees.min()),'degree_max':float(graph.degrees.max()),
               'gamma_plant':gain.gamma,'gamma0':gain.gamma0,'gain_method':gain.method,
               'rigorously_certified_upper_bound':False,
               'W_norm':sg['W_spectral_norm'],'loop_gain':sg['loop_gain'],
               'small_gain_status':sg['status'],
               'J_E_improvement_pct':pair['J_E_improvement_pct'],
               'J_pair_improvement_pct':pair['J_pair_improvement_pct'],
               'J_U_reduction_pct':pair['J_U_reduction_pct'],
               'paired_win_rate':pair['paired_win_rate']}
            for role in ('baseline','learned'):
                for k,v in pair[role].items():
                    r[f'{role}_{k}']=v
            rows.append(r)
            print(f'N={n:4d} {topo:10s} ||L||={lap_norm:.3f} '
                  f'gamma={gain.gamma:.4f} ||W||={sg["W_spectral_norm"]:.4f} '
                  f'gain product={sg["loop_gain"]:.4f} [{sg["status"]}] '
                  f'Jpair improvement={pair["J_pair_improvement_pct"]:+.2f}% '
                  f'JE improvement={pair["J_E_improvement_pct"]:+.2f}%')
    if not np.array_equal(W,before): raise AssertionError('inference modified W')
    return rows


def plot_topologies(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows: return
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    names=sorted({r['topology'] for r in rows})
    for topo in names:
        data=sorted((r for r in rows if r['topology']==topo),key=lambda r:r['N'])
        axs[0].plot([r['N'] for r in data],[r['loop_gain'] for r in data],'.-',label=topo)
        axs[1].plot([r['N'] for r in data],[r['J_pair_improvement_pct'] for r in data],'.-',label=topo)
    axs[0].axhline(.95,ls='--',color='grey',label='eta = 0.95')
    axs[0].axhline(1.,ls=':',color='black',label='gain = 1')
    axs[0].set(xlabel='Number of agents',ylabel='Gamma_N * ||W||_2',xscale='log')
    axs[1].axhline(0,color='grey',lw=1)
    axs[1].set(xlabel='Number of agents',ylabel='Mean-centered disagreement improvement (%)',xscale='log')
    for a in axs:a.grid(alpha=.2);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=150);fig.savefig(str(path)+'.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--sizes',nargs='+',type=int,default=[3,5,10,100,1000])
    p.add_argument('--topologies',nargs='+',choices=TOPOLOGIES,default=list(TOPOLOGIES))
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--horizon',type=int,default=60)
    p.add_argument('--seed',type=int,default=16)
    p.add_argument('--include-large-complete',action='store_true')
    p.add_argument('--out',default='results/topology_comparison')
    a=p.parse_args()
    _,_,W,activation,_=load_checkpoint(a.checkpoint)
    rows=topology_comparison(W,activation,a.sizes,a.topologies,a.samples,a.horizon,
                             a.seed,a.include_large_complete)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'topology_comparison.csv',rows)
    write_json(out/'topology_comparison.json',{'checkpoint':a.checkpoint,
       'note':'Floating-point candidate maxima and sparse singular values, not formally certified bounds. '
              'All graphs at a given N share physical agents and initial states. '
              'Above 1 means the sufficient small-gain theorem is inconclusive, not instability.',
       'rows':rows})
    plot_topologies(rows,out/'topology_comparison')
    print('Saved',out.resolve())

if __name__=='__main__':main()
