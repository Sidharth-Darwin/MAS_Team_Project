"""Paired graph-topology inference, with frozen checkpoint weights.

Run at project root: python experiments/compare_topologies.py --help
The same physical System and same initial states are reused at each N.
"""
import argparse
from pathlib import Path
import numpy as np


from distributed_consensus.graphs import make_graph, load_graph
from distributed_consensus.control.plants import generate_system, build_system
from distributed_consensus.control.plant_gain import full_gain, small_gain, spatial_norm
from distributed_consensus.evaluation.metrics import evaluate_pair
from distributed_consensus.evaluation.reporting import load_checkpoint, write_csv, write_json

TOPOLOGIES=('star','chain','ring','wide_ring','complete','random')


def topology_comparison(W, activation='identity', sizes=(3,5,10,100,1000),
                        topologies=TOPOLOGIES, samples=3, horizon=60, seed=16,
                        include_large_complete=False, custom_graph=None, custom_system=None):
    """Produce topology-comparable metrics; never mutate, rescale, or retrain W."""
    W=np.asarray(W,np.float64)
    before=W.copy()
    rows=[]
    for n in sizes:
        system=(custom_system if custom_system is not None else
                generate_system(n,seed=seed+10000+n))
        if system.n != n:
            raise ValueError(f'graph/system size mismatch: graph N={n}, system N={system.n}')
        initial=np.random.default_rng(seed+900+n).standard_normal((samples,n,2))
        for topo in topologies:
            if topo=='complete' and n>=1000 and not include_large_complete:
                print('Skipping dense N>=1000 complete graph (enable --include-large-complete)')
                continue
            graph=(custom_graph if topo=='custom' else make_graph(n,topo,seed=seed+37*n))
            gain=full_gain(system,graph)
            sg=small_gain(system,graph,W)
            pair=evaluate_pair(system,graph,W,horizon=horizon,samples=samples,
                               initial_states=initial,activation=activation)
            # Numeric ||L_rw|| uses b=1 for spatial norm; star has an analytic identity.
            lap_norm=spatial_norm(graph,np.ones(n))
            r={'N':n,'topology':topo,
               'graph_seed':(None if topo=='custom' else seed+37*n),
               'physical_seed':(None if custom_system is not None else seed+10000+n),
               'initial_seed':seed+900+n,
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


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--sizes',nargs='+',type=int,default=None)
    p.add_argument('--adjacency',default=None,
                   help='Optional .npy or .npz adjacency input; compare this graph only')
    p.add_argument('--system',default=None,
                   help='Optional numeric-only saved system .npz; use with --adjacency')
    p.add_argument('--topologies',nargs='+',choices=TOPOLOGIES,default=None)
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--horizon',type=int,default=60)
    p.add_argument('--seed',type=int,default=16)
    p.add_argument('--include-large-complete',action='store_true')
    p.add_argument('--out',default='results/topology_comparison')
    a=p.parse_args(argv)
    _,_,W,activation,_=load_checkpoint(a.checkpoint)
    if a.system and not a.adjacency:
        p.error('--system requires --adjacency')
    if a.adjacency:
        if a.topologies is not None or a.sizes is not None:
            p.error('--adjacency cannot be combined with --sizes/--topologies')
        graph=load_graph(a.adjacency)
        system=None
        if a.system:
            with np.load(a.system, allow_pickle=False) as z:
                system=build_system(z['A'],z['B'],z['target'])
        rows=topology_comparison(W,activation,[graph.n],['custom'],a.samples,a.horizon,
                                 a.seed,a.include_large_complete,graph,system)
    else:
        rows=topology_comparison(W,activation,
                                 a.sizes if a.sizes is not None else (3,5,10,100,1000),
                                 a.topologies if a.topologies is not None else TOPOLOGIES,
                                 a.samples,a.horizon,a.seed,a.include_large_complete)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'topology_comparison.csv',rows)
    write_json(out/'topology_comparison.json',{'checkpoint':a.checkpoint,
       'adjacency_file':a.adjacency,'system_file':a.system,
       'note':'Floating-point candidate maxima and sparse singular values, not formally certified bounds. '
              'All graphs at a given N share physical agents and initial states. '
              'Above 1 means the sufficient small-gain theorem is inconclusive, not instability.',
       'rows':rows})
    plot_topologies(rows,out/'topology_comparison')
    print('Saved',out.resolve())

if __name__=='__main__':
    main()
