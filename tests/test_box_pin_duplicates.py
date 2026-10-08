"""Conflicting OCR numbers must retain the right contact and network mapping."""
import unittest
from unittest.mock import patch

from stage2 import assemble


class BoxPinDuplicateTests(unittest.TestCase):
    @staticmethod
    def read(number, name, nc=.99, nn=.99):
        return {"number": {"text": number, "conf": nc},
                "name": {"text": name, "conf": nn}}

    def build(self, reads, enabled=True, string_keys=False, partial=True):
        points = [[212, 120], [212, 180]]
        stage1 = {
            "image": {"w": 300, "h": 300},
            "components": [{"id": "ic", "type": "box",
                            "bbox_img": [100, 100, 200, 200],
                            "bbox_json": [100, 100, 200, 200]}],
            "pins": {"ic": points},
            "wire_detail": {"segments": [[p, [280, p[1]]] for p in points]},
            "nets": [{"pins": [["ic", k]], "seg_index": [k]} for k in range(2)],
        }
        assigned = {"comps": {"ic": {"designator": {"text": "U1"},
                         "pins": {str(k) if string_keys else k: r
                                  for k, r in enumerate(reads)}}}}
        with patch.object(assemble, "BOX_PIN_DUPLICATE_BEST", enabled), \
                patch.object(assemble, "BOX_PIN_PARTIAL", partial), \
                patch.object(assemble, "BOX_PIN_DROP_UNREAD", True), \
                patch.object(assemble, "BOX_PIN_DROP_MIXED_ONLY", True), \
                patch.object(assemble, "BOX_PIN_DROP_CAP", False), \
                patch.object(assemble, "BOX_PIN_NEED_NAME", True):
            return assemble.build_doc(stage1, assigned, "synthetic")

    def test_stronger_complete_duplicate_keeps_own_point_and_only_own_net(self):
        for string_keys in (False, True):
            with self.subTest(string_keys=string_keys):
                doc = self.build([self.read("8", "VOUT", .47, .95),
                                  self.read("8", "GND", .96, .99)],
                                 string_keys=string_keys)
                self.assertEqual(doc["pins"]["U1"], {
                    "pin_8": {"pinname": "GND", "point": {"x": 212., "y": 120.}}})
                self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})
                self.assertEqual(len(doc["nets"]), 1)
                net = next(iter(doc["nets"].values()))
                self.assertEqual(net["hyperGraph"], "(U1.8)")
                self.assertEqual(next(iter(net["edges"].values()))[0]["y"], 120.)

    def test_name_quality_also_matters_when_numbers_are_confident(self):
        doc = self.build([self.read("2", "R1", .999, .51),
                          self.read("2", "2", .998, .999)])
        self.assertEqual(doc["pins"]["U1"]["pin_2"]["pinname"], "2")

    def test_unnamed_incumbent_is_not_promoted_into_new_output(self):
        doc = self.build([self.read("8", "", .4, .4), self.read("8", "GND")])
        self.assertEqual(doc["pins"]["U1"], {})
        self.assertEqual(doc["__pin_index"], {})

    def test_unnamed_later_duplicate_cannot_replace_complete_read(self):
        doc = self.build([self.read("8", "GND", .4, .4), self.read("8", "")])
        self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_equal_unknown_and_invalid_confidence_preserve_first(self):
        for invalid in (None, "bad", float("nan"), float("inf"), -1, 2):
            with self.subTest(conf=invalid):
                doc = self.build([self.read("8", "GND", invalid),
                                  self.read("8", "VOUT")])
                self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})
        doc = self.build([self.read("8", "GND"), self.read("8", "VOUT")])
        self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_disabled_switch_preserves_first_read(self):
        doc = self.build([self.read("8", "VOUT", .4, .4), self.read("8", "GND")],
                         enabled=False)
        self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_unique_numbers_are_unchanged(self):
        reads = [self.read("6", "VOUT", .4, .4), self.read("8", "GND")]
        self.assertEqual(self.build(reads), self.build(reads, enabled=False))

    def test_disabled_partial_number_policy_is_unchanged(self):
        reads = [self.read("8", "VOUT", .4, .4), self.read("8", "GND")]
        self.assertEqual(self.build(reads, partial=False),
                         self.build(reads, enabled=False, partial=False))


if __name__ == "__main__":
    unittest.main()
