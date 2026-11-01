"""Run: python -m unittest discover -s simulation/tests -v (from repository root)."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np

from distributed_consensus.control.dynamics import step_numpy, rollout_numpy
from distributed_consensus.graphs import make_graph
from simulation.model import (
    DualSimulation, LocalWorld, get_system, initial_conditions, DEFAULT_CHECKPOINT,
    state_metrics, POSITION_TOL, VELOCITY_TOL, SPREAD_TOL, HOLD_STEPS,
)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system, cls.graph, cls.W, cls.activation = get_system(DEFAULT_CHECKPOINT)

    def test_checkpoint_weight_is_actual_saved_weight(self):
        self.assertEqual(self.W.shape, (2, 1))
        self.assertEqual(self.system.n, 10)
        self.assertEqual(self.activation, 'identity')
        self.assertAlmostEqual(self.W[0, 0], -0.6677979252969951)

    def test_local_only_and_learned_match_repository_numpy_reference(self):
        rng = np.random.default_rng(17)
        for topo in ('checkpoint', 'ring', 'complete', 'star', 'random'):
            system, graph, W, activation = get_system(topology=topo)
            x = rng.normal(size=(10, 2))
            d = rng.normal(size=(10, 1))
            for learned in (False, True):
                world = LocalWorld(system, graph, W, activation, learned, x)
                reference = step_numpy(x, system, graph,
                                       W if learned else None, activation=activation,
                                       disturbance=d)
                output = world.step(d)
                for field in ('disagreement', 'local', 'correction', 'total', 'next_states'):
                    np.testing.assert_allclose(output[field], reference[field], rtol=1e-12, atol=1e-12,
                                               err_msg=f'mismatch in {topo} {learned} {field}')

    def test_full_rollout_matches_repository_reference(self):
        sim = DualSimulation()
        x0 = sim.initial.copy()
        for _ in range(135):
            sim.step()
        for mode in ('baseline', 'learned'):
            expected = rollout_numpy(x0, sim.system, sim.graph,
                                     None if mode == 'baseline' else sim.W,
                                     activation=sim.activation, horizon=135)
            np.testing.assert_allclose(np.stack(sim.histories[mode]), expected['states'], atol=1e-12)
            np.testing.assert_allclose(np.stack([o['total'] for o in sim.actions[mode]]),
                                       expected['total'], atol=1e-12)

    def test_default_scenario_reproduces_saving_comparison(self):
        sim = DualSimulation()
        for _ in range(200):
            sim.step()
        self.assertEqual(sim.ready_step, {'baseline': 43, 'learned': 38})
        self.assertEqual(sim.error_step, {'baseline': 79, 'learned': 58})
        self.assertLess(sim.energy['learned'], sim.energy['baseline'])
        self.assertLess(sim.je['learned'], sim.je['baseline'])

    def test_operator_drag_applies_same_delta_to_both(self):
        sim = DualSimulation()
        for _ in range(90):
            sim.step()
        before = {mode: world.states.copy() for mode, world in sim.worlds.items()}
        sim.drag(2, 0.8)
        for mode, world in sim.worlds.items():
            expected = before[mode].copy()
            expected[2, 0] += .8
            np.testing.assert_array_equal(world.states, expected)
            self.assertIsNone(sim.ready_step[mode])
        self.assertEqual(sim.segment_start, 90)

    def test_input_pulse_matched_and_physical(self):
        sim = DualSimulation()
        sim.pulse(2, 8., 3)
        ds = [sim.step() for _ in range(5)]
        self.assertEqual([d[2, 0] for d in ds], [8., 8., 8., 0., 0.])
        self.assertTrue(all(np.count_nonzero(d) == (1 if i < 3 else 0) for i, d in enumerate(ds)))
        for mode in ('baseline', 'learned'):
            first = sim.actions[mode][0]
            expected = step_numpy(sim.initial, sim.system, sim.graph,
                                  None if mode == 'baseline' else sim.W,
                                  disturbance=ds[0])
            np.testing.assert_allclose(first['next_states'], expected['next_states'], atol=1e-12)

    def test_different_network_size_is_still_exact_matching(self):
        for n in (3, 5, 20):
            s, g, w, activation = get_system(n=n, topology='ring', seed=3)
            self.assertEqual(s.n, n)
            self.assertEqual(g.n, n)
            np.testing.assert_allclose(s.A + s.B @ s.K, np.broadcast_to(s.target, (n, 2, 2)), atol=1e-12)
            np.testing.assert_array_equal(w, self.W)  # Frozen checkpoint weights

    def test_reset_reproduces_identical_initial_conditions(self):
        sim = DualSimulation()
        original = sim.initial.copy()
        sim.pulse(2)
        sim.step()
        sim.drag(5, .5)
        sim.reset()
        np.testing.assert_array_equal(sim.worlds['baseline'].states, original)
        np.testing.assert_array_equal(sim.worlds['learned'].states, original)
        self.assertEqual(sim.k, 0)
        self.assertFalse(sim.events)

    def test_csv_reports_physical_controls_and_interventions(self):
        sim = DualSimulation()
        sim.step()
        sim.drag(2, 1.)
        sim.step()
        with tempfile.TemporaryDirectory() as tmp:
            out = sim.export_csv(Path(tmp) / 'audit.csv')
            with out.open() as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2 * 10 * 2)
        self.assertTrue(any(abs(float(row['position_jump_after_step']) - 1.) < 1e-12
                            for row in rows))
        for row in rows:
            self.assertAlmostEqual(float(row['physical_input']),
                                   float(row['local_input']) + float(row['correction_input']), places=10)


if __name__ == '__main__':
    unittest.main()
