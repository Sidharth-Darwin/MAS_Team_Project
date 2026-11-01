"""Train controller checkpoints and evaluate on the held-out training network.

External-network inference belongs to compare_topologies.py or stress_test.py.
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from distributed_consensus.config import Config
from distributed_consensus.graphs import make_graph, load_graph
from distributed_consensus.control.plants import generate_system,matching_residual,check_target
from distributed_consensus.control.plant_gain import full_gain,small_gain
from distributed_consensus.learning.training import train,baseline_scales
from distributed_consensus.evaluation.metrics import evaluate_pair
from distributed_consensus.evaluation.reporting import (write_json,write_csv,save_checkpoint,
        save_training_plot,save_curve_plot,save_tradeoff_plot,save_curve_csv)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Small-gain-constrained direct Laplacian controller')
    p.add_argument('--suite',choices=['quick','full'],default='quick',
                   help='Preset combinations of training sizes and activations')
    p.add_argument('--train-sizes',type=int,nargs='+',default=None,
                   help='Override training sizes selected by suite; supports any N>=2')
    p.add_argument('--train-adjacency',default=None,
                   help='Direct graph file (.npy dense or SciPy .npz sparse); requires one training size')
    p.add_argument('--run-name',default=None)
    p.add_argument('--train-topology',choices=('chain','star','ring','wide_ring','complete','random'),
                   default='chain', help='Connected undirected training graph')
    p.add_argument('--train-graph-seed',type=int,default=None,
                   help='Graph seed (default: seed + 100*N_train)')
    p.add_argument('--train-neighborhood',type=int,default=8,
                   help='Number of forward neighbors for the wide_ring generator')
    p.add_argument('--train-hub',type=int,default=0,
                   help='Hub index for star training graphs')
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
    return p.parse_args(argv)


def execute(args):
    cfg=Config(seed=args.seed,epochs=args.epochs if args.epochs is not None else (65 if args.suite=='quick' else 300),
        horizon=args.horizon if args.horizon is not None else (55 if args.suite=='quick' else 100),
        batch=args.batch if args.batch is not None else (64 if args.suite=='quick' else 256),
        eval_samples=args.eval_samples if args.eval_samples is not None else (12 if args.suite=='quick' else 48),
        device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if cfg.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if any(v < 1 for v in (cfg.epochs, cfg.horizon, cfg.batch, cfg.eval_samples)):
        raise ValueError('epochs, horizon, batch, eval_samples must be positive')
    sizes=list(args.train_sizes) if args.train_sizes is not None else ([3] if args.suite=='quick' else [3,5,10])
    if not sizes or any(n<2 for n in sizes):
        raise ValueError('training sizes must be >=2')
    if args.train_adjacency is not None and len(sizes)!=1:
        raise ValueError('--train-adjacency requires exactly one --train-sizes value')
    if args.lambda_values is not None and not args.lambda_values:
        raise ValueError('at least one --lambda-values value is required')
    run_name=args.run_name or datetime.now().strftime('consensus_%Y%m%d_%H%M%S')
    root=Path(args.results_root)/run_name
    if root.exists() and any(root.iterdir()): raise FileExistsError(f'results dir not empty: {root}')
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'configuration.json',vars(cfg)|{'suite':args.suite,'run_name':run_name,
                      'cli_arguments':vars(args),'evaluation_model':'best_validation'})
    activations=['identity'] if args.suite=='quick' else ['identity','tanh']
    all_rows=[]
    try:
        for n_train in sizes:
            # Distinct experiments may use different networks; WITHIN one run never regenerate.
            system=generate_system(n_train,seed=cfg.seed+100*n_train)
            graph_seed=(cfg.seed+100*n_train) if args.train_graph_seed is None else args.train_graph_seed
            graph=(load_graph(args.train_adjacency) if args.train_adjacency else
                   make_graph(n_train,args.train_topology,seed=graph_seed,
                              neighborhood=args.train_neighborhood,hub=args.train_hub))
            if graph.n != n_train:
                raise ValueError(f'input graph has {graph.n} nodes but training system has {n_train}')
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
                    # Use validation-selected weights for reported training-network evaluation.
                    # Preserve the actual final-epoch weights in a separate checkpoint.
                    final_W=result['W']
                    W=result['best_W']
                    if W is None:
                        raise RuntimeError('training did not produce best-validation weights')
                    write_csv(folder/'training_history.csv',result['history'])
                    save_training_plot(folder/'training',result['history'],f'N={n_train}, {activation}, lambda={lam:.4g}')
                    save_checkpoint(folder/'checkpoint.npz',system,graph,W,activation,lam)
                    save_checkpoint(folder/'final_checkpoint.npz',system,graph,final_W,activation,lam)
                    local=evaluate_pair(system,graph,W,cfg.horizon,cfg.eval_samples,
                                        seed=cfg.seed+801,activation=activation)
                    write_csv(folder/'paired_small.csv',local['records'])
                    np.savez_compressed(folder/'representative_small.npz',
                        **{f'{label}_{key}':v for label,h in local['representative'].items() for key,v in h.items()})
                    save_curve_plot(folder/'disagreement_small',local['curves'],f'N={n_train}; {tag}')
                    save_curve_csv(folder/'disagreement_small.csv',local['curves'])
                    summary={'tag':tag,'N_train':n_train,'activation':activation,'lambda':lam,
                             'training_topology':('custom' if args.train_adjacency else args.train_topology),
                             'training_adjacency_file':args.train_adjacency,
                             'training_graph_seed':(None if args.train_adjacency else graph_seed),
                             'evaluated_checkpoint':'checkpoint.npz',
                             'final_weight_norm':float(np.linalg.norm(final_W,2)),
                             'training_seed':result['seed'],'W':W,
                             'best_validation_epoch':result['best_epoch'],
                             'best_validation_objective':result['best_validation_objective'],
                             'stopped_early':result['stopped_early'],'gamma_train':gain.gamma,
                             'omega_peak':gain.omega_peak,'gamma0':gain.gamma0,
                             'matching_residual':matching_residual(system),'cap':result['cap'],
                             'norm':float(np.linalg.norm(W,2)),'train_baseline_J_E':je0,
                             'train_baseline_J_U':ju0,'small_gain_train':small_gain(system,graph,W),
                             'local_eval':{k:v for k,v in local.items() if k not in ('records','curves','representative')},
                             }
                    tradeoff.append({'lambda':lam,'mean_J_E':local['learned']['mean_J_E'],
                                     'mean_J_U':local['learned']['mean_J_U']})
                    print(f'  {tag}: W={W.ravel().round(5)} norm={summary["norm"]:.5f} '
                          f'J_E gain={local["J_E_improvement_pct"]:+.2f}% '
                          f'J_U reduction={local["J_U_reduction_pct"]:+.2f}%')
                    write_json(folder/'summary.json',summary)
                    all_rows.append({'tag':tag,'N_train':n_train,'activation':activation,'lambda':lam,
                            'training_topology':('custom' if args.train_adjacency else args.train_topology),
                            'evaluated_checkpoint':'checkpoint.npz',
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

def main(argv=None):
    return execute(parse_args(argv))


if __name__ == '__main__':
    main()