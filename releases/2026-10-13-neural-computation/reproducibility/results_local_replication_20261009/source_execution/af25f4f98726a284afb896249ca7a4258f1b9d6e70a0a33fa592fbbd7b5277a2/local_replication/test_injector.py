"""Numerical checks on archived networks; historical results are read-only."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))
from experiments_v2.core.controller import UnifiedSNNController
from experiments_v2.core.encoder import ObservationEncoder
from experiments_v2.core.protocol import ControllerProtocol, TASK_SPECS
from experiments_v2.core.scaffold import NeuronScaffold
from _mediation.mc_injector import JExchanger, run_cell

INPUTS = ROOT / "results_local_replication_20261009/inputs"


def controller(task_name, variant=False):
    seed = 4000 if task_name == "CartPole" else 4012
    task, proto = TASK_SPECS[task_name], ControllerProtocol()
    if variant:
        path = INPUTS / "results_mc/variants" / f"var_{task_name}_seed{seed}_d0_r0.npy"
    else:
        path = INPUTS / "results_mc/topo" / task_name / f"F0_topology_{task_name}_seed{seed}.npy"
    enc = ObservationEncoder.load(INPUTS / "observation_stats" / f"observation_stats_{task_name}.json")
    return UnifiedSNNController(NeuronScaffold.from_seed(seed, task, proto), np.load(path), enc, proto)


class InjectorTests(unittest.TestCase):
    def observations(self, task):
        return np.random.RandomState(18).normal(size=(40, TASK_SPECS[task].obs_dim))

    def test_strength_including_zero_counts_and_ties(self):
        for mode in ("action_only", "action_centered", "energy_random"):
            for m in (2, 3, 4):
                dummy = type("Dummy", (), {"n_out": m})()
                for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                    for counts in (np.zeros(m), np.ones(m), np.arange(m)):
                        ex = JExchanger(dummy, dummy, mode=mode, alpha=alpha)
                        native, donor = np.linspace(-1, 2, m), np.linspace(2, -2, m)
                        delta = donor - native
                        actual = ex._mix(native, donor, delta, 0, counts) - native
                        np.testing.assert_allclose(np.linalg.norm(actual), alpha * np.linalg.norm(delta), atol=1e-12)
                        if mode == "action_only" and alpha:
                            self.assertEqual(np.argmax(actual), np.argmax(counts))
                        np.testing.assert_array_equal(ex._mix(native, native, np.zeros(m), 0, counts), native)

    def test_disabled_is_bit_identical(self):
        for task in ("CartPole", "Acrobot"):
            for mode, alpha in (("none", 1.0), ("self_exact", 0.0)):
                native = controller(task, True)
                ex = JExchanger(controller(task, True), controller(task), mode=mode, alpha=alpha, keep_trace=True)
                for obs in self.observations(task):
                    action, counts, _ = native.forward(obs, return_trace=True)
                    got_action, got_counts = ex.forward(obs)
                    self.assertEqual(action, got_action)
                    np.testing.assert_array_equal(counts[-native.n_out:], got_counts)
                    np.testing.assert_array_equal(native.v, ex.rec.v)
                self.assertEqual(ex._corr_inj_sum, 0.0)

    def test_exact_matches_donor(self):
        for task in ("CartPole", "Acrobot"):
            reference = controller(task)
            ex = JExchanger(controller(task, True), controller(task), mode="self_exact", alpha=1.0)
            for obs in self.observations(task):
                action, counts, _ = reference.forward(obs, return_trace=True)
                got_action, got_counts = ex.forward(obs)
                self.assertEqual(action, got_action)
                np.testing.assert_array_equal(counts[-reference.n_out:], got_counts)
                np.testing.assert_allclose(reference.v[-reference.n_out:], ex.rec.v[-reference.n_out:], atol=1e-10)

    def test_random_cadence_and_repeatability(self):
        ex = JExchanger(controller("CartPole", True), controller("CartPole"), mode="energy_random", rng=np.random.RandomState(21))
        directions, batches = [], []
        mix = ex._mix

        def record(*args):
            result = mix(*args)
            if ex._randn is not None:
                directions.append(ex._randn.copy())
            return result

        with patch.object(ex, "_mix", side_effect=record):
            for obs in self.observations("CartPole")[:3]:
                directions.clear()
                ex.forward(obs)
                self.assertTrue(directions)
                for direction in directions:
                    np.testing.assert_array_equal(direction, directions[0])
                batches.append(directions[0].copy())
        self.assertFalse(np.array_equal(batches[0], batches[1]))
        twin = JExchanger(controller("CartPole", True), controller("CartPole"), mode="energy_random", rng=np.random.RandomState(21))
        for i, obs in enumerate(self.observations("CartPole")[:3]):
            twin.forward(obs)
            np.testing.assert_array_equal(twin._randn, batches[i])
        np.testing.assert_array_equal(twin.rec.v, ex.rec.v)

    def test_faulty_strength_rejected(self):
        ex = JExchanger(controller("CartPole", True), controller("CartPole"), mode="action_only")
        with patch.object(ex, "_mix", side_effect=lambda native, *args: native.copy()):
            with self.assertRaises(AssertionError):
                for obs in self.observations("CartPole"):
                    ex.forward(obs)

    def test_headless_env_and_step_assertions(self):
        import gymnasium as gym
        for task in ("CartPole", "Acrobot"):
            for mode in ("action_only", "action_centered", "energy_random"):
                env = gym.make(TASK_SPECS[task].env_id)
                try:
                    row = run_cell(controller(task, True), controller(task), env, [86400], 0.5, mode, 73, keep_trace=True)[0]
                    self.assertEqual(row["energy_assertion"]["checks"], row["n_steps"] * 20)
                    self.assertLess(row["energy_assertion"]["max_abs_error"], 1e-10)
                    ta = row["trace_agg"]
                    self.assertAlmostEqual(ta["j_corr_inj_norm_mean"], 0.5 * ta["j_corr_exact_norm_mean"], places=10)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
