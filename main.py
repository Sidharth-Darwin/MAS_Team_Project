"""Research experiments: run `python main.py --help` (quick and full suites)."""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from config import Config
from helper_functions.graphs import make_graph
from helper_functions.plants import generate_system,matching_residual,check_target
from helper_functions.plant_gain import full_gain,small_gain
from helper_functions.training import train,baseline_scales
from helper_functions.evaluation import evaluate_pair
from helper_functions.reporting import (write_json,write_csv,save_checkpoint,
        save_training_plot,save_curve_plot,save_tradeoff_plot,save_curve_csv,save_system)


def parse_args():
    p=argparse.ArgumentParser(description='Small-gain-constrained direct Laplacian controller')
    p.add_argument('--suite',choices=['quick','full'],default='quick')
    p.add_argument('--run-name',default=None)
    p.add_argument('--results-root',default='results')
    p.add_argument('--seed',type=int,default=16)
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--horizon',type=int,default=None)
    p.add_argument('--batch',type=int,default=None)
    p.add_argument('--eval-samples',type=int,default=None)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--early-stopping-patience',type=int,default=None,
                   help='Optional number of validation checks without improvement')
    p.add_argument('--val-every',type=int,default=10)
    p.add_argument('--min-delta',type=float,default=0.)
    p.add_argument('--lambda-values',type=float,nargs='*',default=None,
                   help='Explicit lambdas; otherwise derived from measured baseline J_E/J_U')
    p.add_argument('--inference-sizes',type=int,nargs='*',default=[100,1000])
    return p.parse_args()


def execute(args):
    cfg=Config(seed=args.seed,epochs=args.epochs or (65 if args.suite=='quick' else 300),
        horizon=args.horizon or (55 if args.suite=='quick' else 100),
        batch=args.batch or (64 if args.suite=='quick' else 256),
        eval_samples=args.eval_samples or (12 if args.suite=='quick' else 48),
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device,
        inference_sizes=tuple(args.inference_sizes))
    if cfg.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    run_name=args.run_name or datetime.now().strftime('consensus_%Y%m%d_%H%M%S')
    root=Path(args.results_root)/run_name
    if root.exists() and any(root.iterdir()): raise FileExistsError(f'results dir not empty: {root}')
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'configuration.json',vars(cfg)|{'suite':args.suite,'run_name':run_name})
    target=None
    sizes=[3] if args.suite=='quick' else [3,5,10]
    activations=['identity'] if args.suite=='quick' else ['identity','tanh']
    all_rows=[]
    try:
        for n_train in sizes:
            # Distinct experiments may use different networks; WITHIN one run never regenerate.
            system=generate_system(n_train,seed=cfg.seed+100*n_train)
            graph=make_graph(n_train,'path',seed=cfg.seed+100*n_train)
            gain=full_gain(system,graph)
            je0,ju0=baseline_scales(system,graph,cfg.horizon,min(128,cfg.batch*2),seed=cfg.seed+501,device=cfg.device)
            if ju0<=1e-12: raise ValueError('baseline physical energy scale too small for normalized tradeoff')
            ratio=je0/ju0
            lambdas=([0.,.02*ratio,.1*ratio] if args.lambda_values is None else args.lambda_values)
            print(f'TRAIN N={n_train} target_rho={check_target(system.target):.6f}, '
                  f'matching={matching_residual(system):.2e}, gamma={gain.gamma:.6f}, '
                  f'J_E0={je0:.3f} J_U0={ju0:.3f} lambda_balance={ratio:.5g}')
            for activation in activations:
                tradeoff=[]
                for idx,lam in enumerate(lambdas):
                    tag=f'trainN{n_train}_{activation}_lambda{idx}'
                    folder=root/tag;folder.mkdir()
                    result=train(system,graph,lam=lam,activation=activation,epochs=cfg.epochs,
                        horizon=cfg.horizon,batch=cfg.batch,val_batch=max(64,cfg.batch),
                        lr=cfg.lr,eta=cfg.eta,seed=cfg.seed+11,device=cfg.device,
                        val_every=args.val_every,
                        early_stopping_patience=args.early_stopping_patience,
                        min_delta=args.min_delta)
                    W=result['W']
                    write_csv(folder/'training_history.csv',result['history'])
                    save_training_plot(folder/'training',result['history'],f'N={n_train}, {activation}, lambda={lam:.4g}')
                    save_checkpoint(folder/'checkpoint.npz',system,graph,W,activation,lam)
                    save_checkpoint(folder/'final_checkpoint.npz',system,graph,W,activation,lam)
                    save_checkpoint(folder/'best_validation_checkpoint.npz',system,graph,
                                    result['best_W'],activation,lam)
                    local=evaluate_pair(system,graph,W,cfg.horizon,cfg.eval_samples,
                                        seed=cfg.seed+801,activation=activation)
                    write_csv(folder/'paired_small.csv',local['records'])
                    np.savez_compressed(folder/'representative_small.npz',
                        **{f'{label}_{key}':v for label,h in local['representative'].items() for key,v in h.items()})
                    save_curve_plot(folder/'disagreement_small',local['curves'],f'N={n_train}; {tag}')
                    save_curve_csv(folder/'disagreement_small.csv',local['curves'])
                    summary={'tag':tag,'N_train':n_train,'activation':activation,'lambda':lam,
                             'training_seed':result['seed'],'W':W,
                             'best_validation_epoch':result['best_epoch'],
                             'best_validation_objective':result['best_validation_objective'],
                             'stopped_early':result['stopped_early'],'gamma_train':gain.gamma,
                             'omega_peak':gain.omega_peak,'gamma0':gain.gamma0,
                             'matching_residual':matching_residual(system),'cap':result['cap'],
                             'norm':float(np.linalg.norm(W,2)),'train_baseline_J_E':je0,
                             'train_baseline_J_U':ju0,'small_gain_train':small_gain(system,graph,W),
                             'local_eval':{k:v for k,v in local.items() if k not in ('records','curves','representative')},
                             'inferences':[]}
                    tradeoff.append({'lambda':lam,'mean_J_E':local['learned']['mean_J_E'],
                                     'mean_J_U':local['learned']['mean_J_U']})
                    print(f'  {tag}: W={W.ravel().round(5)} norm={summary["norm"]:.5f} '
                          f'J_E gain={local["J_E_improvement_pct"]:+.2f}% '
                          f'J_U reduction={local["J_U_reduction_pct"]:+.2f}%')
                    for n in cfg.inference_sizes:
                        if n<2: raise ValueError('inference N >=2')
                        # Same sampled heterogeneous agents for both undirected topologies.
                        large_system=generate_system(n,seed=cfg.seed+10000+n)
                        for topology in ('ring','wide_ring'):
                            large_graph=make_graph(n,topology)
                            system_file=root/'inference_systems'/f'N{n}_{topology}.npz'
                            if not system_file.exists():
                                save_system(system_file,large_system,large_graph)
                            gain_n=full_gain(large_system,large_graph)
                            sg=small_gain(large_system,large_graph,W)
                            pair=evaluate_pair(large_system,large_graph,W,cfg.horizon,
                                cfg.eval_samples,seed=cfg.seed+801,activation=activation)
                            label=f'N{n}_{topology}'
                            write_csv(folder/f'paired_{label}.csv',pair['records'])
                            # Record representative states/actions for external inspection;
                            # plot only disagreement summary to keep reports manageable.
                            np.savez_compressed(folder/f'representative_{label}.npz',
                                **{f'{who}_{key}':v for who,h in pair['representative'].items() for key,v in h.items()})
                            save_curve_plot(folder/f'curve_{label}',pair['curves'],f'{tag} -> {label}')
                            save_curve_csv(folder/f'curve_{label}.csv',pair['curves'])
                            record={'network_size':n,'topology':topology,'gamma':gain_n.gamma,
                                    'system_file':str(system_file.relative_to(root)),
                                    'physical_seed':cfg.seed+10000+n,
                                    'evaluation_seed':cfg.seed+801,
                                    'gain_relation_to_train':'<=' if gain_n.gamma<=gain.gamma else '>',
                                    'small_gain':sg,
                                    'metrics':{k:v for k,v in pair.items() if k not in ('records','curves','representative')}}
                            summary['inferences'].append(record)
                            print(f'    {label}: gamma={gain_n.gamma:.4f} '
                                  f'g_N={sg["loop_gain"]:.4f} gain_relation={record["gain_relation_to_train"]} '
                                  f'J_E gain={pair["J_E_improvement_pct"]:+.2f}%')
                    write_json(folder/'summary.json',summary)
                    all_rows.append({'tag':tag,'N_train':n_train,'activation':activation,'lambda':lam,
                            'gamma_train':gain.gamma,'W_norm':summary['norm'],
                            'training_loop_gain':summary['small_gain_train']['loop_gain'],
                            'J_E_baseline':local['baseline']['mean_J_E'],
                            'J_E_learned':local['learned']['mean_J_E'],
                            'J_U_baseline':local['baseline']['mean_J_U'],
                            'J_U_learned':local['learned']['mean_J_U'],
                            'J_E_improvement_pct':local['J_E_improvement_pct'],
                            'J_U_reduction_pct':local['J_U_reduction_pct'],
                            'paired_win_rate':local['paired_win_rate']})
                save_tradeoff_plot(root/f'tradeoff_N{n_train}_{activation}',tradeoff)
    finally:
        write_csv(root/'overview.csv',all_rows)
    write_json(root/'run_summary.json',{'run_name':run_name,'completed_runs':len(all_rows),'rows':all_rows})
    print(f'RESULTS: {root.resolve()}')
    return root

if __name__=='__main__':
    execute(parse_args())
