"""CPU-only checks for protocol validation and paired scene bootstrapping."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.compare_metrics import REQUIRED_PROTOCOL_FIELDS, compare_results


ROOT = Path(__file__).resolve().parents[1]


def results():
    return {
        "metadata": {
            "dataset": "R2R",
            "split": "val_unseen",
            "feature_id": "test-features-v1",
            "base_checkpoint_sha256": "a" * 64,
            "protocol_sha256": "d" * 64,
            "upstream_commit": "b" * 40,
            "max_action_len": 15,
            "feedback": "argmax",
        },
        "episodes": [
            {"instr_id": "1_0", "scan_id": "scene-a", "success": 0, "spl": 0.0},
            {"instr_id": "2_0", "scan_id": "scene-a", "success": 1, "spl": 0.5},
            {"instr_id": "3_0", "scan_id": "scene-b", "success": 0, "spl": 0.0},
        ],
    }


class CompareMetricsTests(unittest.TestCase):
    def compare(self, baseline=None, method=None, **kwargs):
        if baseline is None:
            baseline = results()
        if method is None:
            method = copy.deepcopy(baseline)
        return compare_results(baseline, method, resamples=kwargs.pop("resamples", 200), **kwargs)

    def test_identical_outputs_keep_explicit_zero_values(self):
        result = self.compare()
        self.assertAlmostEqual(result["metrics"]["sr"]["baseline"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["spl"]["baseline"], 0.5 / 3)
        for metric in ("sr", "spl"):
            self.assertEqual(result["metrics"][metric]["delta_pp"], 0.0)
            self.assertEqual(result["metrics"][metric]["ci95_pp"], [0.0, 0.0])

    def test_explicit_false_is_valid_success(self):
        payload = results()
        payload["episodes"][0]["success"] = False
        self.assertAlmostEqual(self.compare(payload)["metrics"]["sr"]["baseline"], 1 / 3)

    def test_pairing_ignores_file_order_and_is_reproducible(self):
        baseline = results()
        method = copy.deepcopy(baseline)
        method["episodes"][0].update(success=1, spl=0.75)
        first = self.compare(baseline, method, seed=17)
        method["episodes"].reverse()
        self.assertEqual(first, self.compare(baseline, method, seed=17))
        self.assertAlmostEqual(first["metrics"]["sr"]["delta_pp"], 100 / 3)
        self.assertAlmostEqual(first["metrics"]["spl"]["delta_pp"], 25)

    def test_scene_bootstrap_keeps_whole_scenes_and_episode_weighting(self):
        baseline = results()
        baseline["episodes"] = [
            {"instr_id": str(index), "scan_id": "scene-a" if index < 20 else "scene-b",
             "success": 0, "spl": 0}
            for index in range(30)
        ]
        method = copy.deepcopy(baseline)
        for episode in method["episodes"][:20]:
            episode.update(success=1, spl=1)
        # Sampling two scenes gives AA=100pp, AB=66.67pp, or BB=0pp.
        # Incorrectly sampling 30 independent episodes would give a narrow CI.
        result = self.compare(baseline, method, resamples=2000, seed=1)
        self.assertAlmostEqual(result["metrics"]["sr"]["delta_pp"], 200 / 3)
        self.assertEqual(result["metrics"]["sr"]["ci95_pp"], [0, 100])
        self.assertEqual(result["n_episodes"], 30)
        self.assertEqual(result["n_scenes"], 2)

    def test_constant_paired_effect_has_exact_interval(self):
        baseline = results()
        method = copy.deepcopy(baseline)
        for before, after in zip(baseline["episodes"], method["episodes"]):
            before.update(success=0, spl=0)
            after.update(success=1, spl=0.25)
        result = self.compare(baseline, method)
        self.assertEqual(result["metrics"]["sr"]["ci95_pp"], [100, 100])
        self.assertEqual(result["metrics"]["spl"]["ci95_pp"], [25, 25])

    def test_harmful_missing_episode_is_rejected(self):
        baseline = results()
        method = copy.deepcopy(baseline)
        del method["episodes"][0]  # Removing a failure must not improve the score.
        with self.assertRaisesRegex(ValueError, "episode set mismatch"):
            self.compare(baseline, method)

    def test_extra_episode_is_rejected(self):
        baseline = results()
        method = copy.deepcopy(baseline)
        method["episodes"].append(dict(method["episodes"][0], instr_id="extra"))
        with self.assertRaisesRegex(ValueError, "extra in method"):
            self.compare(baseline, method)

    def test_duplicate_instruction_is_rejected_in_either_result(self):
        for side in (0, 1):
            with self.subTest(side=side):
                pair = [results(), results()]
                pair[side]["episodes"].append(copy.deepcopy(pair[side]["episodes"][0]))
                with self.assertRaisesRegex(ValueError, "duplicate instr_id"):
                    self.compare(*pair)

    def test_changed_scene_is_rejected(self):
        baseline = results()
        method = copy.deepcopy(baseline)
        method["episodes"][0]["scan_id"] = "another-scene"
        with self.assertRaisesRegex(ValueError, "scan_id mismatch"):
            self.compare(baseline, method)

    def test_every_required_protocol_field_must_exist_on_both_sides(self):
        for field in REQUIRED_PROTOCOL_FIELDS:
            for side in (0, 1):
                with self.subTest(field=field, side=side):
                    pair = [results(), results()]
                    del pair[side]["metadata"][field]
                    with self.assertRaisesRegex(ValueError, f"missing metadata.{field}"):
                        self.compare(*pair)

    def test_every_required_protocol_field_must_match(self):
        for field in REQUIRED_PROTOCOL_FIELDS:
            with self.subTest(field=field):
                baseline = results()
                method = copy.deepcopy(baseline)
                value = method["metadata"][field]
                method["metadata"][field] = (
                    value + 1 if field == "max_action_len"
                    else "c" * 64 if field in ("base_checkpoint_sha256", "protocol_sha256")
                    else value + "-different"
                )
                with self.assertRaisesRegex(ValueError, f"protocol mismatch for metadata.{field}"):
                    self.compare(baseline, method)

    def test_invalid_metric_values_are_rejected(self):
        for field in ("success", "spl"):
            for invalid in (None, "0", -0.01, 1.01, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid):
                    method = results()
                    method["episodes"][0][field] = invalid
                    with self.assertRaisesRegex(ValueError, field):
                        self.compare(results(), method)

    def test_missing_episode_field_does_not_default_to_zero(self):
        for field in ("success", "spl", "instr_id", "scan_id"):
            with self.subTest(field=field):
                method = results()
                del method["episodes"][0][field]
                with self.assertRaisesRegex(ValueError, f"missing {field}"):
                    self.compare(results(), method)

    def test_invalid_metadata_values_are_rejected(self):
        for field, value in (("max_action_len", True), ("max_action_len", 0),
                             ("feature_id", ""), ("dataset", None),
                             ("base_checkpoint_sha256", "not-a-digest"),
                             ("protocol_sha256", "not-a-digest")):
            with self.subTest(field=field, value=value):
                payload = results()
                payload["metadata"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self.compare(payload)

    def test_empty_episodes_and_one_scene_are_rejected(self):
        empty = results()
        empty["episodes"] = []
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            self.compare(empty)
        one_scene = results()
        for episode in one_scene["episodes"]:
            episode["scan_id"] = "only-scene"
        with self.assertRaisesRegex(ValueError, "at least two distinct scan_id"):
            self.compare(one_scene)

    def test_cli_writes_json_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            method = root / "method.json"
            output = root / "nested" / "comparison.json"
            baseline.write_text(json.dumps(results()), encoding="utf-8")
            method.write_text(json.dumps(results()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "compare_metrics.py"),
                 "--baseline", str(baseline), "--method", str(method),
                 "--resamples", "100", "--output", str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), json.loads(output.read_text(encoding="utf-8")))

    def test_cli_rejects_overwriting_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.json"
            original = json.dumps(results())
            input_path.write_text(original, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "compare_metrics.py"),
                 "--baseline", str(input_path), "--method", str(input_path),
                 "--resamples", "10", "--output", str(input_path)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("must not overwrite", completed.stderr)
            self.assertEqual(input_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
