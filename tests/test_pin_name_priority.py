"""Binding must preserve a qualified provider's per-contact hard priority."""
import copy
import os
import unittest
from unittest.mock import patch

from stage2.assign import assign


class PinNamePriorityTests(unittest.TestCase):
    def setUp(self):
        self.components = [{"id": "ic", "type": "box", "bbox_img": [100, 100, 300, 300]}]
        self.pins = {"ic": [[90, 180]]}
        self.evidence = {"ic": [{"point_img": [90, 180], "anchor_img": [100, 180],
                                "edge": "left", "trust": "trusted", "source": "red_green_transition",
                                "frame_confirmed": True}]}

    def legacy(self, box, text="OLD", confidence=.95):
        return {"bbox": box, "reads": {"pin_name": (text, confidence)}, "name_only": True}

    def preferred(self, box, confidence=.95):
        return {"bbox": box, "reads": {"pin_name": ("REAL", confidence)},
                "name_only": True, "number_only": False, "source": "color_pin_name_ppocr",
                "pin_target": {"component_id": "ic", "pin_index": 0, "edge": "left",
                               "point_img": [90, 180], "anchor_img": [100, 180]},
                "pin_name_ocr": {"backend": "ppocr_v4", "accepted": True, "flags": []}}

    def bind(self, texts, midpoint=False):
        with patch.dict(os.environ, {"STAGE2_PIN_NAME_EDGE_MIDPOINT": "1" if midpoint else "0"}):
            return assign(self.components, self.pins, texts,
                          pin_evidence=self.evidence)["comps"]["ic"]["pins"]

    def test_ineligible_closer_legacy_does_not_eliminate_preferred_name(self):
        old = self.legacy([103, 175, 113, 185])
        new = self.preferred([120, 175, 145, 185])
        self.assertEqual(self.bind([old])[0]["name"]["text"], "OLD")
        self.assertEqual(self.bind([new])[0]["name"]["text"], "REAL")
        self.assertEqual(self.bind([old, new])[0]["name"]["text"], "REAL")

    def test_edge_midpoint_cannot_bypass_preferred_hard_eligibility(self):
        old = self.legacy([111, 175, 130, 185], confidence=.99)
        new = self.preferred([112, 175, 131, 185], confidence=.91)
        self.assertEqual(self.bind([old, new], midpoint=True)[0]["name"]["text"], "REAL")

    def test_priority_is_local_to_its_current_contact(self):
        self.pins["ic"].append([90, 220])
        self.evidence["ic"].append({"point_img": [90, 220], "anchor_img": [100, 220],
                                   "edge": "left", "trust": "trusted", "source": "red_green_transition",
                                   "frame_confirmed": True})
        old = self.legacy([103, 175, 113, 185])
        new = self.preferred([120, 175, 145, 185])
        other = self.legacy([110, 215, 145, 225], text="OTHER")
        got = self.bind([old, new, other])
        self.assertEqual(got[0]["name"]["text"], "REAL")
        self.assertEqual(got[1]["name"]["text"], "OTHER")
        self.assertIsNone(got[0]["number"])
        self.assertIsNone(got[1]["number"])

    def test_held_provider_cannot_disable_legacy_observation(self):
        old = self.legacy([103, 175, 113, 185])
        new = self.preferred([120, 175, 145, 185])
        new["pin_name_ocr"].update(accepted=False, held_reason="unresolved_partner")
        new["reads"] = {}
        before = copy.deepcopy([old, new])
        got = self.bind([old, new])
        self.assertEqual(got[0]["name"]["text"], "OLD")
        self.assertEqual([old, new], before)


if __name__ == "__main__":
    unittest.main()
