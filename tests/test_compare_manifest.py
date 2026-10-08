"""Saved-output comparison, strict file checks, and group isolation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.eval import compare_manifest


class CompareManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.json"
        self.before = self.root / "before"
        self.after = self.root / "after"
        self.before.mkdir()
        self.after.mkdir()
        self.doc = {
            "components": {"U1": {"Name": "CHIP", "type": "box", "value": None, "bbox": [10, 10, 30, 30]}},
            "pins": {"U1": {"pin_1": {"pinname": "IN", "point": {"x": 5, "y": 20}}}},
            "nets": {"net_1": {"hyperGraph": "(U1.1)", "edges": {"edge_1": [{"x": 1, "y": 20}, {"x": 5, "y": 20}]}}},
        }
        self.write(self.root / "dev-gt.json", self.doc)
        self.write(self.root / "verification-gt.json", self.doc)
        self.samples = [{"case_id": "a", "gt": "dev-gt.json", "group": "dev"}, {"case_id": "b", "gt": "verification-gt.json", "group": "verification"}]
        self.write(self.manifest, {"samples": self.samples})
        for cid in ("a", "b"):
            self.write(self.before / f"{cid}_Pred.json", self.doc)
        changed = copy.deepcopy(self.doc)
        changed["pins"]["U1"]["pin_1"]["pinname"] = "WRONG"
        self.write(self.after / "a_Pred.json", changed)
        self.write(self.after / "b_Pred.json", self.doc)

    @staticmethod
    def write(path, doc):
        path.write_text(json.dumps(doc), encoding="utf-8")

    def test_two_arms_group_means_deltas_and_counts(self):
        report = compare_manifest.compare(self.manifest, [("before", self.before), ("after", self.after)], self_check=True)
        self.assertEqual(report["matching"]["backend"], "scipy_hungarian")
        self.assertEqual(report["predictions"]["before"]["aggregate"]["mean_score"], 100)
        self.assertEqual(report["predictions"]["after"]["aggregate"]["mean_score"], 87.5)
        self.assertEqual(report["predictions"]["after"]["by_group"]["dev"]["mean_score"], 75)
        self.assertEqual(report["predictions"]["after"]["by_group"]["verification"]["mean_score"], 100)
        self.assertEqual(report["comparisons"]["after"]["aggregate_delta"]["mean_score"], -12.5)
        self.assertEqual(report["comparisons"]["after"]["cases"][0]["delta"]["counts"]["pin"], {"tp": -1, "fp": 1, "fn": 1})
        self.assertEqual(report["predictions"]["after"]["aggregate"]["counts"]["pin"]["tp"], 1)

    def test_dev_group_never_loads_verification_gt_or_prediction(self):
        (self.root / "verification-gt.json").unlink()
        (self.before / "b_Pred.json").unlink()
        report = compare_manifest.compare(self.manifest, [("before", self.before)], group="dev")
        self.assertEqual(report["selection"]["cases"], 1)
        self.assertEqual(set(report["predictions"]["before"]["by_group"]), {"dev"})

    def test_missing_prediction_is_error_and_no_partial_report(self):
        (self.after / "a_Pred.json").unlink()
        out = self.root / "comparison.json"
        with self.assertRaisesRegex(ValueError, "Missing prediction.*after.*a"):
            compare_manifest.compare(self.manifest, [("before", self.before), ("after", self.after)], group="dev", output=out)
        self.assertFalse(out.exists())

    def test_cannot_overwrite_manifest_gt_prediction_or_existing_report(self):
        for path in (self.manifest, self.root / "dev-gt.json", self.before / "a_Pred.json"):
            original = path.read_bytes()
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "cannot overwrite an input"):
                compare_manifest.compare(self.manifest, [("before", self.before)], group="dev", output=path)
            self.assertEqual(path.read_bytes(), original)
        output = self.root / "comparison.json"
        output.write_text("KEEP", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            compare_manifest.compare(self.manifest, [("before", self.before)], group="dev", output=output)
        self.assertEqual(output.read_text(), "KEEP")

    def test_duplicate_ids_and_arm_names_are_errors(self):
        with self.assertRaisesRegex(ValueError, "duplicate prediction arm"):
            compare_manifest.compare(self.manifest, [("same", self.before), ("same", self.after)])
        self.write(self.manifest, {"samples": [self.samples[0], self.samples[0]]})
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            compare_manifest.compare(self.manifest, [("before", self.before)])

    def test_rejects_scipy_fallback_and_gt_hash_drift(self):
        with patch.object(compare_manifest.metrics, "_HAVE_SCIPY", False):
            with self.assertRaisesRegex(ValueError, "SciPy Hungarian matching is required"):
                compare_manifest.compare(self.manifest, [("before", self.before)])
        self.samples[0]["gt_sha256"] = "0" * 64
        self.write(self.manifest, {"samples": self.samples})
        with self.assertRaisesRegex(ValueError, "GT SHA256 mismatch"):
            compare_manifest.compare(self.manifest, [("before", self.before)], group="dev")

    def test_fresh_json_report_has_source_hashes(self):
        out = self.root / "comparison.json"
        compare_manifest.compare(self.manifest, [("before", self.before)], group="dev", output=out)
        saved = json.loads(out.read_text(encoding="utf-8"))
        row = saved["predictions"]["before"]["cases"][0]
        self.assertEqual(row["prediction_sha256"], hashlib.sha256((self.before / "a_Pred.json").read_bytes()).hexdigest())
        self.assertEqual(saved["manifest"]["sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
