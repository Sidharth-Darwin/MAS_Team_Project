"""Resume-safe standalone search over the EXISTING direct controller and objective.

Run from project root. Every candidate shares one fixed heterogeneous physical
system and independent fixed validation initial states. Candidate selection uses
one common held-out validation horizon, never held-out test data. Search lambdas
are TRAINING objective weights; candidates with different lambdas are selected
by a separate, explicitly defined, shared validation score:
  score = J_E/J_E(local) + selection_energy_weight * J_U/J_U(local).

No tuning framework is needed and no controller/dynamics/training logic is copied.
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helper_functions.plants import generate_system
from helper_functions.graphs import make_graph
from helper_functions.plant_gain import full_gain,small_gain
from helper_functions.training import train
from helper_functions.evaluation import evaluate_pair
from helper_functions.reporting import (save_checkpoint,load_checkpoint,write_json,
                                        write_csv,save_training_plot)


def config_id(cfg):
    raw=json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def candidates(a):
    keys=['lr','batch','epochs','horizon','activation','lam','seed']
    vals=[a.learning_rates,a.batch_sizes,a.epochs,a.horizons,
          a.activations,a.lambda_values,a.seeds]
    cfgs=[dict(zip(keys,combo)) for combo in itertools.product(*vals)]
    if a.search=='random':
        rng=np.random.default_rng(a.search_seed)
        order=rng.permutation(len(cfgs))
        cfgs=[cfgs[i] for i in order]
    if a.max_candidates>0: cfgs=cfgs[:a.max_candidates]
    return cfgs


def non_dominated(rows):
    """Pareto minimization in comparable validation JE and JU (not mixed objectives)."""
    return [r for r in rows if not any(
        (q['val_J_E']<=r['val_J_E'] and q['val_J_U']<=r['val_J_U']) and
        (q['val_J_E']<r['val_J_E'] or q['val_J_U']<r['val_J_U'])
        for q in rows if q['id']!=r['id'])]


def draw_plots(rows,out):
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for r in rows:
        axs[0].scatter(r['val_J_U'],r['val_J_E'],s=35,
            marker='o' if r['stage']=='coarse' else 'x')
    front=non_dominated(rows)
    front.sort(key=lambda x:x['val_J_U'])
    axs[0].plot([r['val_J_U'] for r in front],[r['val_J_E'] for r in front],
                ls='--',color='grey',label='Pareto frontier')
    axs[0].set(xlabel='Common validation J_U (physical energy)',
               ylabel='Common validation J_E',title='Disagreement–energy trade-off')
    axs[0].legend()
    order=sorted(rows,key=lambda r:r['selection_score'])
    axs[1].bar(range(len(order)),[r['selection_score'] for r in order])
    axs[1].set(xlabel='Candidates ordered by validation score',
               ylabel='JE/JE_local + weight · JU/JU_local',title='Common selection score')
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'search_summary.png',dpi=150)
    fig.savefig(out/'search_summary.svg');plt.close(fig)


def execute(a):
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    system=generate_system(a.train_size,seed=a.system_seed)
    graph=make_graph(a.train_size,a.train_topology,seed=a.graph_seed)
    gamma=full_gain(system,graph).gamma
    # Fixed INDEPENDENT validation data for all candidates regardless of seed,
    # batch size, horizon, activation or lambda.
    common_val=np.random.default_rng(a.validation_seed).standard_normal(
        (a.validation_samples,system.n,2))
    fingerprint={'N':a.train_size,'system_seed':a.system_seed,
                 'graph_seed':a.graph_seed,'topology':a.train_topology,
                 'validation_seed':a.validation_seed,'validation_samples':a.validation_samples,
                 'selection_horizon':a.selection_horizon,
                 'selection_energy_weight':a.selection_energy_weight,
                 'eta':a.eta,'val_every':a.val_every,
                 'patience':a.patience,'min_delta':a.min_delta}
    descriptor=out/'search_spec.json'
    if descriptor.exists():
        if json.loads(descriptor.read_text())!=fingerprint:
            raise ValueError('Search output belongs to a different fixed system or selection protocol')
    else:write_json(descriptor,fingerprint)
    base=evaluate_pair(system,graph,np.zeros((2,1)),a.selection_horizon,
          samples=a.validation_samples,initial_states=common_val)
    je_ref=base['baseline']['mean_J_E'];ju_ref=base['baseline']['mean_J_U']
    if je_ref<=0 or ju_ref<=0:raise ValueError('baseline reference cost is zero')

    def one(cfg,stage):
        cid=config_id(cfg)
        folder=out/'candidates'/cid
        resultpath=folder/'result.json'
        if resultpath.exists():
            old=json.loads(resultpath.read_text())
            if old.get('config')!=cfg:raise ValueError('candidate id collision')
            if not (folder/'best_validation_checkpoint.npz').exists():
                raise ValueError('Completed metadata found but checkpoint missing')
            print('RESUME/SKIP:',cid)
            return old
        folder.mkdir(parents=True,exist_ok=True)
        print('TRAIN:',stage,cid,cfg,flush=True)
        r=train(system,graph,lam=cfg['lam'],activation=cfg['activation'],
            epochs=cfg['epochs'],horizon=cfg['horizon'],batch=cfg['batch'],
            val_batch=len(common_val),lr=cfg['lr'],eta=a.eta,seed=cfg['seed'],
            device=a.device,val_every=a.val_every,
            early_stopping_patience=a.patience,min_delta=a.min_delta,
            validation_initial_states=common_val)
        save_checkpoint(folder/'final_checkpoint.npz',system,graph,r['W'],
                        cfg['activation'],cfg['lam'])
        save_checkpoint(folder/'best_validation_checkpoint.npz',system,graph,r['best_W'],
                        cfg['activation'],cfg['lam'])
        write_csv(folder/'training_history.csv',r['history'])
        save_training_plot(folder/'training',r['history'],f'{cid} {cfg["activation"]}')
        val=evaluate_pair(system,graph,r['best_W'],a.selection_horizon,
            samples=len(common_val),initial_states=common_val,
            activation=cfg['activation'])
        je=val['learned']['mean_J_E'];ju=val['learned']['mean_J_U']
        score=je/je_ref + a.selection_energy_weight*ju/ju_ref
        record={'id':cid,'stage':stage,'config':cfg,'gamma':gamma,
             'best_epoch':r['best_epoch'],'stopped_early':r['stopped_early'],
             'best_training_horizon_validation_objective':r['best_validation_objective'],
             'validation_selection_horizon':a.selection_horizon,
             'val_J_E':je,'val_J_U':ju,'val_J_pair':val['learned']['mean_J_pair'],
             'baseline_val_J_E':je_ref,'baseline_val_J_U':ju_ref,
             'selection_score':score,'selection_energy_weight':a.selection_energy_weight,
             'small_gain':small_gain(system,graph,r['best_W']),
             'checkpoint':str((folder/'best_validation_checkpoint.npz').relative_to(out))}
        # Mark complete only after every result/checkpoint is saved.
        write_json(resultpath,record)
        return record

    records={}
    for cfg in candidates(a):
        r=one(cfg,'coarse');records[r['id']]=r
        persist(records,out)
    if a.refine_top>0:
        best=sorted(records.values(),key=lambda r:r['selection_score'])[:a.refine_top]
        extra=[]
        for r in best:
            c=r['config']
            for factor in (.5,1.5):
                new=dict(c);new['lr']=float(f'{c["lr"]*factor:.12g}');extra.append(new)
                if c['lam']>0:
                    new=dict(c);new['lam']=float(f'{c["lam"]*factor:.12g}');extra.append(new)
        for cfg in extra:
            r=one(cfg,'refined');records[r['id']]=r
            persist(records,out)
    ranked=sorted(records.values(),key=lambda r:r['selection_score'])
    front={r['id'] for r in non_dominated(list(records.values()))}
    summary={'selection_rule':'minimize common validation JE/JE_baseline + '
              f'{a.selection_energy_weight} * JU/JU_baseline',
             'validation_not_test':True,'pareto_ids':sorted(front),
             'best_candidate_id':ranked[0]['id'], 'baseline_validation_JE':je_ref,
             'baseline_validation_JU':ju_ref,'N_train':system.n,'gamma_train':gamma}
    write_json(out/'selection.json',summary)
    draw_plots(list(records.values()),out)
    # Only finalists incur expensive large-network inference. Completed finalist
    # results are keyed by selected IDs and evaluation settings for safe resume.
    finalist_ids=[r['id'] for r in ranked[:a.finalists]]
    finalist_spec={'ids':finalist_ids,'sizes':a.finalist_sizes,'samples':a.finalist_samples,
                  'horizon':a.finalist_horizon}
    finalist_spec_file=out/'finalists_spec.json'
    finalist_results_file=out/'finalists_inference.json'
    if finalist_results_file.exists() and finalist_spec_file.exists() and \
            json.loads(finalist_spec_file.read_text())==finalist_spec:
        print('RESUME/SKIP completed finalist inference')
        return summary
    finalists=[]
    for r in ranked[:a.finalists]:
        _,_,W,act,_=load_checkpoint(out/r['checkpoint'] if not Path(r['checkpoint']).is_absolute() else r['checkpoint'])
        for n in a.finalist_sizes:
            large=generate_system(n,seed=a.system_seed+10000+n)
            initial=np.random.default_rng(a.validation_seed+999+n).standard_normal(
                (a.finalist_samples,n,2))  # Finalist evaluation, NEVER used in selection
            for topology in ('ring','star'):
                g=make_graph(n,topology,seed=a.graph_seed+n)
                sg=small_gain(large,g,W)
                metrics=evaluate_pair(large,g,W,a.finalist_horizon,
                    a.finalist_samples,activation=act,initial_states=initial)
                finalists.append({'candidate_id':r['id'],'N':n,'topology':topology,
                  'gamma':sg['gamma'],'W_norm':sg['W_spectral_norm'],
                  'loop_gain':sg['loop_gain'],'status':sg['status'],
                  'J_E_improvement_pct':metrics['J_E_improvement_pct'],
                  'J_pair_improvement_pct':metrics['J_pair_improvement_pct']})
    write_csv(out/'finalists_inference.csv',finalists)
    write_json(out/'finalists_inference.json',finalists)
    write_json(finalist_spec_file,finalist_spec)
    print('Completed:',len(records),'candidates; selected:',ranked[0]['id'])
    return summary


def persist(records,out):
    rows=[]
    for r in records.values():
        cfg=r['config'];sg=r['small_gain']
        rows.append({'id':r['id'],'stage':r['stage'],**cfg,
            'best_epoch':r['best_epoch'],'val_J_E':r['val_J_E'],
            'val_J_U':r['val_J_U'],'val_J_pair':r['val_J_pair'],
            'selection_score':r['selection_score'],'W_norm':sg['W_spectral_norm'],
            'gamma':sg['gamma'],'loop_gain':sg['loop_gain'],'status':sg['status']})
    write_csv(out/'candidates.csv',rows)
    write_json(out/'candidates.json',list(records.values()))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='results/hyperparameter_search')
    p.add_argument('--train-size',type=int,default=3)
    p.add_argument('--train-topology',default='chain',
      choices=['star','chain','path','ring','wide_ring','complete','random'])
    p.add_argument('--system-seed',type=int,default=316)
    p.add_argument('--graph-seed',type=int,default=16)
    p.add_argument('--validation-seed',type=int,default=90001)
    p.add_argument('--validation-samples',type=int,default=32)
    p.add_argument('--selection-horizon',type=int,default=70)
    p.add_argument('--selection-energy-weight',type=float,default=0.1,
      help='COMMON validation score weight across lambdas, not a training objective')
    p.add_argument('--learning-rates',type=float,nargs='+',default=[.002,.005])
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[64])
    p.add_argument('--epochs',type=int,nargs='+',default=[100])
    p.add_argument('--horizons',type=int,nargs='+',default=[70])
    p.add_argument('--activations',nargs='+',choices=['identity','tanh'],default=['identity'])
    p.add_argument('--lambda-values',type=float,nargs='+',default=[0.,.1])
    p.add_argument('--seeds',type=int,nargs='+',default=[7])
    p.add_argument('--search',choices=['grid','random'],default='grid')
    p.add_argument('--search-seed',type=int,default=99)
    p.add_argument('--max-candidates',type=int,default=0,help='0 means entire specified grid')
    p.add_argument('--refine-top',type=int,default=0)
    p.add_argument('--val-every',type=int,default=10)
    p.add_argument('--patience',type=int,default=None,
      help='Optional early stopping in validation checks, not epochs')
    p.add_argument('--min-delta',type=float,default=0.)
    p.add_argument('--eta',type=float,default=.95)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--finalists',type=int,default=1)
    p.add_argument('--finalist-sizes',type=int,nargs='+',default=[100,1000])
    p.add_argument('--finalist-samples',type=int,default=3)
    p.add_argument('--finalist-horizon',type=int,default=70)
    return p.parse_args()

if __name__=='__main__':execute(parse_args())
