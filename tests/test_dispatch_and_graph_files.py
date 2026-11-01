"""Focused regression checks for thin CLI and adjacency-file interoperability."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from distributed_consensus.graphs import Graph, load_graph, make_graph
from distributed_consensus.control.plants import generate_system
from distributed_consensus.evaluation.reporting import save_system, load_checkpoint


def test_load_dense_numpy_adjacency(tmp_path):
    adj=make_graph(5,'star').adjacency.toarray()
    path=tmp_path/'graph.npy'
    np.save(path,adj)
    assert np.array_equal(load_graph(path).adjacency.toarray(),adj)


def test_load_sparse_scipy_adjacency(tmp_path):
    g=make_graph(1000,'ring')
    path=tmp_path/'graph.npz'
    sparse.save_npz(path,g.adjacency)
    loaded=load_graph(path)
    assert loaded.n==1000
    assert loaded.adjacency.nnz==2000
    assert (loaded.adjacency!=g.adjacency).nnz==0


def test_load_dense_npz_and_existing_saved_system_graph(tmp_path):
    g=make_graph(5,'ring')
    path=tmp_path/'adj.npz'
    np.savez_compressed(path,adjacency=g.adjacency.toarray())
    assert (load_graph(path).adjacency!=g.adjacency).nnz==0
    system=generate_system(5,seed=22)
    saved=tmp_path/'system.npz'
    save_system(saved,system,g)
    assert (load_graph(saved).adjacency!=g.adjacency).nnz==0


def test_load_invalid_graph_rejected(tmp_path):
    p=tmp_path/'graph.npy'
    np.save(p, np.array([[0.,1.,0.],[0.,0.,1.],[0.,1.,0.]]))
    with pytest.raises(ValueError,match='undirected'):
        load_graph(p)
    with pytest.raises(ValueError,match='must be .npy or .npz'):
        load_graph(tmp_path/'adj.json')


def test_main_dispatches_to_existing_experiment(monkeypatch):
    import main
    from experiments import compare_topologies
    observed=[]
    def fake_main(argv):
        observed.append(argv)
    monkeypatch.setattr(compare_topologies, 'main', fake_main)
    main.main(['topologies', '--checkpoint', 'weights.npz'])
    assert observed == [['--checkpoint', 'weights.npz']]


def test_topology_comparison_accepts_direct_external_graph():
    from experiments.compare_topologies import topology_comparison
    g=Graph(np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]))
    system=generate_system(3,seed=5)
    w=np.array([[.01],[-.02]])
    rows=topology_comparison(w,sizes=[3],topologies=['custom'],samples=1,horizon=4,
                             custom_graph=g,custom_system=system)
    assert len(rows)==1
    assert rows[0]['topology']=='custom'
    assert rows[0]['N']==3
    assert rows[0]['W_norm']==pytest.approx(float(np.linalg.norm(w,2)))


def test_training_run_custom_adjacency_and_separate_inference(tmp_path):
    from experiments.train_controller import execute,parse_args
    graphfile=tmp_path/'my_graph.npy'
    np.save(graphfile,make_graph(3,'ring').adjacency.toarray())
    args=parse_args(['--suite','quick','--train-sizes','3',
        '--train-adjacency',str(graphfile),'--epochs','2','--horizon','4',
        '--batch','4','--eval-samples','2','--lambda-values','0',
        '--results-root',str(tmp_path),'--run-name','trained','--device','cpu'])
    out=execute(args)
    f=out/'trainN3_identity_lambda0'
    system,graph,W,act,lam=load_checkpoint(f/'checkpoint.npz')
    assert (graph.adjacency!=make_graph(3,'ring').adjacency).nnz==0
    assert (f/'final_checkpoint.npz').exists()
    assert not (out/'inference_systems').exists()
    summary=json.loads((f/'summary.json').read_text())
    assert summary['training_topology']=='custom'
    assert summary['evaluated_checkpoint']=='checkpoint.npz'


def test_dispatcher_tuner_cli_help():
    """The dispatcher calls tuner's normal main(argv) entry point."""
    import main
    with pytest.raises(SystemExit) as exc:
        main.main(['tune','--help'])
    assert exc.value.code == 0


def test_reject_even_tiny_asymmetry():
    # No hidden epsilon-based correction of a supposedly undirected graph.
    from distributed_consensus.graphs import Graph
    with pytest.raises(ValueError, match='undirected'):
        Graph(np.array([[0., 1.], [1. + 5e-13, 0.]]))


def test_main_requires_an_explicit_command():
    import main
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main.main(['--help']) == 0
    assert 'train|topologies|stress|tune' in buf.getvalue()
    with pytest.raises(SystemExit, match='Unknown command'):
        main.main(['--suite', 'quick'])