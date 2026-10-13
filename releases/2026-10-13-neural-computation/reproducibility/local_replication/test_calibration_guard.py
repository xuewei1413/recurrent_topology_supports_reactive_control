"""Checks for replay-window filtering and narrowly bounded legacy-result reuse."""
import copy
import json
import unittest

from local_replication.measure_association import (
    CALIBRATION_GUARD, LEGACY_WRAPPER_SHA256, WRAPPER_PATH,
    compatible_legacy_config_hash, filter_calibration, load_module,
)
from local_replication.run_current import INPUTS, OUT, canonical, digest, np


class CalibrationGuardTests(unittest.TestCase):
    def episode(self, length, json_keys=False):
        return {"seq": list(range(length)),
                "snaps": {(str(t) if json_keys else t): [t] for t in range(length)}}

    def configs(self):
        legacy = {"stage": "measurement", "calibration_states": 250,
                  "additional_sources": {WRAPPER_PATH: LEGACY_WRAPPER_SHA256,
                                         "controller.py": "unchanged"}}
        current = copy.deepcopy(legacy)
        current["additional_sources"][WRAPPER_PATH] = "guarded-runner"
        current["calibration_guard"] = copy.deepcopy(CALIBRATION_GUARD)
        return legacy, current

    def test_valid_samples_are_unchanged_and_in_order(self):
        episodes = [self.episode(40), self.episode(30, json_keys=True)]
        states = [(0, 8), (1, 7), (0, 5), (1, 20)]
        result, audit = filter_calibration(episodes, states, minimum_states=4)
        self.assertIs(result[0], episodes)
        self.assertIs(result[1], states)
        self.assertEqual(audit["dropped_states"], 0)

    def test_last_complete_window_is_retained(self):
        episodes = [self.episode(11)]
        states = [(0, 5), (0, 6)]
        result, audit = filter_calibration(episodes, states, minimum_states=1)
        self.assertEqual(result[1], [(0, 5)])
        self.assertEqual(states, [(0, 5), (0, 6)])
        self.assertEqual(audit["dropped_windows"][0]["time_index"], 6)

    def test_short_episode_is_not_padded_or_replaced(self):
        episodes = [self.episode(10), self.episode(40)]
        states = [(0, 5), (1, 8), (1, 5)]
        original = copy.deepcopy(episodes)
        result, audit = filter_calibration(episodes, states, minimum_states=2)
        self.assertEqual(result[1], [(1, 8), (1, 5)])
        self.assertEqual(episodes, original)
        self.assertEqual(audit["sampled_states"], 3)
        self.assertEqual(audit["retained_states"], 2)

    def test_insufficient_valid_samples_stop(self):
        with self.assertRaisesRegex(ValueError, "require 250"):
            filter_calibration([self.episode(10)], [(0, 5)])

    def test_corrupt_indices_or_snapshots_are_not_silently_dropped(self):
        episodes = [self.episode(40)]
        for state in ((-1, 5), (1, 5), (0, -1), (0, 5.0)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                filter_calibration(episodes, [state], minimum_states=1)
        del episodes[0]["snaps"][5]
        with self.assertRaisesRegex(ValueError, "snapshot"):
            filter_calibration(episodes, [(0, 5)], minimum_states=1)

    def test_known_legacy_wrapper_is_compatible(self):
        legacy, current = self.configs()
        self.assertEqual(compatible_legacy_config_hash(legacy, current), digest(canonical(legacy)))

    def test_scientific_parameter_or_source_change_is_rejected(self):
        legacy, current = self.configs()
        current["calibration_states"] = 249
        with self.assertRaises(ValueError):
            compatible_legacy_config_hash(legacy, current)
        legacy, current = self.configs()
        current["additional_sources"]["controller.py"] = "changed"
        with self.assertRaises(ValueError):
            compatible_legacy_config_hash(legacy, current)
        legacy, current = self.configs()
        current["additional_sources"]["extra.py"] = "added"
        with self.assertRaises(ValueError):
            compatible_legacy_config_hash(legacy, current)

    def test_unknown_legacy_runner_or_guard_policy_is_rejected(self):
        legacy, current = self.configs()
        legacy["additional_sources"][WRAPPER_PATH] = "unknown"
        with self.assertRaises(ValueError):
            compatible_legacy_config_hash(legacy, current)
        legacy, current = self.configs()
        current["calibration_guard"]["tau_max"] = 4
        with self.assertRaises(ValueError):
            compatible_legacy_config_hash(legacy, current)

    @unittest.skipUnless((OUT / "association/full/c2/measurements/attempts").exists(),
                         "Requires saved local calibration data")
    def test_saved_failed_calibration_keeps_267_samples(self):
        paths = list((OUT / "association/full/c2/measurements/attempts").glob(
            "seed4275_*/calibration.json"))
        self.assertTrue(paths)
        original = next(p for p in paths if len(json.loads(p.read_text(encoding="utf-8"))[1]) == 268)
        episodes, states = json.loads(original.read_text(encoding="utf-8"))
        result, audit = filter_calibration(episodes, states)
        self.assertEqual(len(result[1]), 267)
        self.assertEqual(audit["dropped_windows"], [{"episode_index": 4, "time_index": 5,
                                                   "episode_length": 10,
                                                   "reason": "Incomplete replay window"}])

    @unittest.skipUnless((OUT / "association/full/v3/measurements/local_measurement_seed4230.json").exists(),
                         "Requires a completed local measurement")
    def test_saved_valid_replay_is_bit_identical(self):
        payload = json.loads((OUT / "association/full/v3/measurements/local_measurement_seed4230.json").read_text(
            encoding="utf-8"))
        episodes, states = json.loads((OUT / payload["attempt"] / "calibration.json").read_text(encoding="utf-8"))
        for episode in episodes:
            episode["seq"] = [np.asarray(obs) for obs in episode["seq"]]
            episode["snaps"] = {int(t): tuple(np.asarray(part) for part in snapshot)
                                for t, snapshot in episode["snaps"].items()}
        sample = states[:4]
        retained, _ = filter_calibration(episodes, sample, minimum_states=4)
        module = load_module("_regime3_measure")
        task, protocol = module.TASK_SPECS["CartPole"], module.ControllerProtocol()
        scaffold = module.NeuronScaffold.from_seed(4230, task, protocol)
        encoder = module.ObservationEncoder.load(INPUTS / "observation_stats/observation_stats_CartPole.json")
        adjacency = np.load(OUT / "association/full/v3/evolution/F0_topology_r3_CartPole_seed4230.npy")
        group = next(g for g in module.fixed_groups(adjacency, task.n_input, task.n_output) if len(g))
        for lesion in (None, group[:3]):
            with self.subTest(lesioned=lesion is not None):
                original = module.replay_margins(adjacency, scaffold, encoder, protocol, episodes, sample, lesion)
                guarded = module.replay_margins(adjacency, scaffold, encoder, protocol, *retained, lesion)
                np.testing.assert_array_equal(original, guarded)


if __name__ == "__main__":
    unittest.main()
