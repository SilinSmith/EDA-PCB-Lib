"""Default IC output publishes observed numbers, never positional guesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from stage2 import assemble


ROOT = Path(__file__).resolve().parents[1]


class BoxPinOutputPolicyTests(unittest.TestCase):
    @staticmethod
    def build(numbers, *, ctype="box", string_keys=False):
        points = [[80, 60], [80, 90], [80, 130]]
        stage = dict(image=dict(w=320, h=220),
                     components=[dict(id="component", type=ctype,
                                      bbox_img=[100, 40, 180, 150],
                                      bbox_json=[100, 70, 180, 180])],
                     pins={"component": points},
                     wire_detail=dict(segments=[[point, [30, point[1]]]
                                               for point in points]),
                     nets=[dict(pins=[["component", i]], seg_index=[i])
                           for i in range(3)])
        reads = {str(i) if string_keys else i:
                 dict(number=None if number is None else dict(text=number, conf=.95),
                      name=dict(text=name, conf=.95))
                 for i, (number, name) in enumerate(zip(numbers, ("LABEL_A", "LABEL_B", "LABEL_C")))}
        assigned = dict(comps={"component": dict(designator=dict(text="U1"), pins=reads)})
        # Isolate the output policy; neither OCR/name eligibility nor the
        # legacy whole-component number fallback is under test here.
        with patch.object(assemble, "BOX_PIN_PARTIAL", True), \
                patch.object(assemble, "BOX_PIN_NEED_NAME", False):
            return assemble.build_doc(stage, assigned, "synthetic")

    @staticmethod
    def imported_policy(mode, numbers, ctype="box"):
        """Check real environment parsing in a fresh interpreter."""
        env = os.environ.copy()
        env.pop("STAGE2_DBG_PINS", None)
        if mode is None:
            env.pop("STAGE2_BOX_PIN_DROP_UNREAD", None)
        else:
            env["STAGE2_BOX_PIN_DROP_UNREAD"] = mode
        code = """
import json, sys
sys.path.insert(0, 'tests')
from test_box_pin_output_policy import BoxPinOutputPolicyTests
from stage2 import assemble
print(json.dumps(dict(mode=assemble._DROP_MODE,
    doc=BoxPinOutputPolicyTests.build(json.loads(sys.argv[1]), ctype=sys.argv[2]))))
"""
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code,
                                 json.dumps(numbers), ctype], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf8", check=True)
        return json.loads(result.stdout)

    def test_unset_default_all_keeps_component_but_no_unknown_pins_or_network_refs(self):
        result = self.imported_policy(None, [None, "not_a_number", ""])
        self.assertEqual(result["mode"], "all")
        doc = result["doc"]
        self.assertIn("U1", doc["components"])
        self.assertEqual(doc["pins"]["U1"], {})
        self.assertEqual(doc["__pin_index"], {})
        self.assertEqual(doc["nets"], {})

    def test_sparse_real_numbers_keep_original_indices_and_only_their_networks(self):
        for string_keys in (False, True):
            with self.subTest(string_keys=string_keys), \
                    patch.object(assemble, "BOX_PIN_DROP_UNREAD", True), \
                    patch.object(assemble, "BOX_PIN_DROP_MIXED_ONLY", False), \
                    patch.object(assemble, "BOX_PIN_DROP_CAP", False):
                doc = self.build(["391", None, "68"], string_keys=string_keys)
                self.assertEqual(set(doc["pins"]["U1"]), {"pin_391", "pin_68"})
                self.assertEqual(doc["__pin_index"],
                                 {"component": {"0": "pin_391", "2": "pin_68"}})
                self.assertEqual(doc["pins"]["U1"]["pin_391"]["point"], {"x": 80., "y": 160.})
                self.assertEqual({net["hyperGraph"] for net in doc["nets"].values()},
                                 {"(U1.391)", "(U1.68)"})

    def test_unknown_ic_reference_is_removed_without_erasing_other_component_net(self):
        stage = dict(image=dict(w=320, h=220), components=[
            dict(id="ic", type="box", bbox_img=[100, 40, 180, 150],
                 bbox_json=[100, 70, 180, 180]),
            dict(id="r", type="r", bbox_img=[45, 110, 65, 130],
                 bbox_json=[45, 90, 65, 110])],
            pins={"ic": [[80, 120]], "r": [[50, 120]]},
            wire_detail=dict(segments=[[[50, 120], [80, 120]]]),
            nets=[dict(pins=[["ic", 0], ["r", 0]], seg_index=[0])])
        assigned = dict(comps={"ic": dict(designator=dict(text="U1"),
            pins={0: dict(number=None, name=dict(text="LABEL_A", conf=.95))}),
            "r": dict(designator=dict(text="R1"))})
        with patch.object(assemble, "BOX_PIN_PARTIAL", True), \
                patch.object(assemble, "BOX_PIN_DROP_UNREAD", True), \
                patch.object(assemble, "BOX_PIN_DROP_MIXED_ONLY", False), \
                patch.object(assemble, "BOX_PIN_DROP_CAP", False):
            doc = assemble.build_doc(stage, assigned, "synthetic")
        self.assertEqual(doc["pins"]["U1"], {})
        self.assertNotIn("ic", doc["__pin_index"])
        self.assertEqual(len(doc["nets"]), 1)
        net = next(iter(doc["nets"].values()))
        self.assertIn("R1.", net["hyperGraph"])
        self.assertNotIn("U1.", net["hyperGraph"])
        self.assertTrue(net["edges"])

    def test_nonconsecutive_descending_and_large_read_numbers_are_not_renumbered(self):
        doc = self.imported_policy(None, ["999", "4", "39"])["doc"]
        self.assertEqual(set(doc["pins"]["U1"]), {"pin_999", "pin_4", "pin_39"})
        self.assertEqual(doc["__pin_index"],
                         {"component": {"0": "pin_999", "1": "pin_4", "2": "pin_39"}})
        self.assertEqual({net["hyperGraph"] for net in doc["nets"].values()},
                         {"(U1.999)", "(U1.4)", "(U1.39)"})

    def test_nonbox_positional_numbering_is_unchanged_by_default_and_legacy_modes(self):
        default = self.imported_policy(None, [None, None, None], ctype="connector")["doc"]
        legacy = self.imported_policy("off", [None, None, None], ctype="connector")["doc"]
        self.assertEqual(default, legacy)
        self.assertEqual(set(default["pins"]["U1"]), {"pin_1", "pin_2", "pin_3"})
        self.assertEqual(default["__pin_index"],
                         {"component": {"0": "pin_1", "1": "pin_2", "2": "pin_3"}})
        self.assertEqual(len(default["nets"]), 3)

    def test_explicit_mixed_and_off_legacy_modes_still_allow_all_unknown_positions(self):
        for mode in ("mixed", "off"):
            with self.subTest(mode=mode):
                result = self.imported_policy(mode, [None, None, None])
                self.assertEqual(result["mode"], mode)
                self.assertEqual(set(result["doc"]["pins"]["U1"]),
                                 {"pin_1", "pin_2", "pin_3"})
                self.assertEqual(len(result["doc"]["nets"]), 3)

    def test_explicit_cap_mode_preserves_legacy_gap_guess_while_default_drops_it(self):
        numbers = ["8", None, "3"]
        default = self.imported_policy(None, numbers)["doc"]
        capped = self.imported_policy("cap", numbers)["doc"]
        self.assertEqual(set(default["pins"]["U1"]), {"pin_8", "pin_3"})
        self.assertEqual(set(capped["pins"]["U1"]), {"pin_8", "pin_1", "pin_3"})
        self.assertNotIn("1", default["__pin_index"]["component"])
        self.assertEqual(capped["__pin_index"]["component"]["1"], "pin_1")


if __name__ == "__main__":
    unittest.main()
