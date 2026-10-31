"""Regression coverage only for incremental topology, reporting and tuning additions."""
import json
import numpy as np
import pytest
from helper_functions.graphs import make_graph
from helper_functions.plants import generate_system
from helper_functions.plant_gain import small_gain,spatial_norm
from helper_functions.training import train
from helper_functions.dynamics import rollout_numpy
from helper_functions.evaluation import evaluate_pair,trajectory_metrics
from helper_functions.reporting import save_checkpoint,load_checkpoint
from experiments.compare_topologies import topology_comparison
from experiments.tune_hyperparameters import config_id,non_dominated


@pytest.mark.parametrize('n',[3,5,10,100,1000])
def test_star_formula(n):
    graph=make_graph(n,'star')
    assert np.isclose(spatial_norm(graph,np.ones(n)),n/np.sqrt(n-1),rtol=1e-9)
    assert graph.degrees.max()==n-1


@pytest.mark.parametrize('topo',['star','chain','ring','wide_ring','complete','random'])
def test_six_undirected_connected_topologies(topo):
    g=make_graph(10,topo,seed=23)
    assert (g.adjacency-g.adjacency.T).nnz==0
    assert np.allclose(g.error(np.ones((10,2))),0)
    assert g.n==10


def test_mean_centered_metric_and_original_costs():
    s=generate_system(5,seed=1);g=make_graph(5,'star')
    x=np.arange(10,dtype=float).reshape(5,2)/4
    h=rollout_numpy(x,s,g,None,horizon=4)
    m=trajectory_metrics(h)
    centered=h['states']-h['states'].mean(axis=1,keepdims=True)
    assert m['J_pair']==pytest.approx(np.sum(centered[:-1]**2)/5)
    assert m['J_E']==pytest.approx(np.sum(h['disagreement'][:-1]**2))
    assert m['J_U_per_agent']==pytest.approx(m['J_U']/5)


def test_frozen_across_graphs_and_shared_initial_states():
    s=generate_system(5,seed=4);rng=np.random.default_rng(15)
    initial=rng.normal(size=(3,5,2));W=np.array([[-.08],[-.1]])
    vals=[]
    for topology in ('star','ring'):
        g=make_graph(5,topology)
        result=evaluate_pair(s,g,W,5,3,initial_states=initial)
        vals.append(result)
        assert np.array_equal(result['representative']['baseline']['states'][0],initial[0])
    assert not np.isclose(vals[0]['baseline']['mean_J_E'],vals[1]['baseline']['mean_J_E'])
    assert np.array_equal(W,np.array([[-.08],[-.1]]))


def test_disturbance_input_physical_and_paired():
    s=generate_system(3,seed=31);g=make_graph(3,'star')
    x0=np.array([[1.,2.],[0.,1.],[-1.,2.]])
    d=np.zeros((5,3,1));d[1,0,0]=3.
    base=rollout_numpy(x0,s,g,horizon=5)
    disturbed=rollout_numpy(x0,s,g,horizon=5,disturbances=d)
    assert np.allclose(disturbed['states'][2]-base['states'][2],
                       s.B[:, :, 0]*d[1,:,0,None])
    initial=x0[None,:,:]; ds=d[None,:,:,:]
    pair=evaluate_pair(s,g,np.zeros((2,1)),horizon=5,samples=1,
                       initial_states=initial,disturbances=ds)
    assert pair['J_E_improvement_pct']==pytest.approx(0.,abs=1e-12)
    with pytest.raises(ValueError,match='disturbances'):
        rollout_numpy(x0,s,g,horizon=5,disturbances=np.zeros((4,3,1)))


def test_gain_categories_on_same_system():
    s=generate_system(100,seed=42);g=make_graph(100,'star')
    gamma=small_gain(s,g,np.array([[0.],[.01]]))['gamma']
    for product,cat in [(.94,'at_or_below_eta'),(.97,'between_eta_and_one'),
                        (1.02,'at_or_above_one')]:
        sg=small_gain(s,g,np.array([[0.],[product/gamma]]))
        assert sg['status']==cat
        assert sg['rigorously_certified_upper_bound'] is False


def test_training_best_final_and_margin():
    s=generate_system(3,seed=1);g=make_graph(3,'chain')
    r=train(s,g,epochs=7,horizon=4,batch=5,val_batch=6,
            seed=3,val_every=1,early_stopping_patience=2,lr=.003)
    vals=[h for h in r['history'] if h['val_objective'] is not None]
    assert r['best_epoch'] in [h['epoch'] for h in vals]
    assert r['best_validation_objective']==min(h['val_objective'] for h in vals)
    assert all(h['constraint_margin']>=-1e-10 for h in r['history'])
    assert r['best_W'].shape==r['W'].shape==(2,1)


def test_old_checkpoint_compatibility(tmp_path):
    s=generate_system(3,seed=3);g=make_graph(3,'star');w=np.array([[.1],[-.1]])
    save_checkpoint(tmp_path/'checkpoint.npz',s,g,w)
    assert np.array_equal(load_checkpoint(tmp_path/'checkpoint.npz')[2],w)


def test_topology_script_does_not_mutate_W():
    W=np.array([[.01],[-.05]])
    before=W.copy()
    rows=topology_comparison(W,sizes=[3],topologies=['star','ring'],samples=1,horizon=3)
    assert len(rows)==2 and np.array_equal(W,before)


def test_tuner_ids_and_pareto():
    cfg={'lr':.002,'lam':0.,'batch':8,'seed':1}
    assert config_id(cfg)==config_id(dict(reversed(list(cfg.items()))))
    rows=[{'id':'a','val_J_E':1.,'val_J_U':4.},
          {'id':'b','val_J_E':2.,'val_J_U':2.},
          {'id':'c','val_J_E':3.,'val_J_U':5.}]
    assert {r['id'] for r in non_dominated(rows)}=={'a','b'}


def test_star_hub_location_changes_actual_plant_gain():
    from helper_functions.plant_gain import full_gain
    s=generate_system(20,seed=4,b_range=(.04,.25))
    lo=int(np.argmin(s.B[:,1,0]));hi=int(np.argmax(s.B[:,1,0]))
    gl=make_graph(20,'star',hub=lo)
    gh=make_graph(20,'star',hub=hi)
    assert full_gain(s,gl).gamma!=pytest.approx(full_gain(s,gh).gamma)
    assert np.isclose(spatial_norm(gl,np.ones(20)),20/np.sqrt(19))
    with pytest.raises(ValueError,match='hub'):
        make_graph(20,'star',hub=20)
