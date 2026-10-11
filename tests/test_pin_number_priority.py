"""A verified number override is local, current and geometrically external."""
import copy
import unittest

from stage2.assign import assign


class PinNumberPriorityTests(unittest.TestCase):
    def setUp(self):
        self.components = [{"id": "ic", "type": "box", "bbox_img": [100, 100, 300, 300]}]
        self.pins = {"ic": [[90, 180]]}
        self.evidence = {"ic": [{"point_img": [90, 180], "anchor_img": [100, 180],
                                "edge": "left", "trust": "trusted",
                                "source": "red_green_transition", "frame_confirmed": True}]}
        self.old = {"bbox": [88, 167, 98, 179], "number_only": True,
                    "reads": {"pin_number": ("8", .999)}}
        self.new = {"bbox": [89, 168, 97, 179], "geometry_bbox": [89, 168, 97, 179],
                    "number_only": True, "name_only": False,
                    "reads": {"pin_number": ("6", .94)}, "source": "color_pin_number_ppocr",
                    "pin_target": {"component_id": "ic", "pin_index": 0, "edge": "left",
                                   "point_img": [90, 180], "anchor_img": [100, 180]},
                    "pin_number_ocr": {"backend": "ppocr_v4_direct", "accepted": True,
                        "flags": [], "selected_views": [
                            {"padding": 2, "text": "6", "confidence": .94},
                            {"padding": 4, "text": "6", "confidence": .96}]}}

    def bind(self, records):
        return assign(self.components, self.pins, records,
                      pin_evidence=self.evidence)["comps"]["ic"]["pins"]

    def test_qualified_views_replace_high_confidence_legacy_only_on_own_row(self):
        self.assertEqual(self.bind([self.old])[0]["number"]["text"], "8")
        self.assertEqual(self.bind([self.old, self.new])[0]["number"]["text"], "6")
        self.assertEqual(self.bind([self.new, self.old])[0]["number"]["text"], "6")

    def test_disagreement_or_low_view_cannot_override(self):
        for change in ({"text": "8"}, {"confidence": .89}, {"confidence": float("nan")},
                       {"padding": 8}, {"confidence": "bad"}):
            with self.subTest(change=change):
                record = copy.deepcopy(self.new)
                record["pin_number_ocr"]["selected_views"][1].update(change)
                self.assertEqual(self.bind([self.old, record])[0]["number"]["text"], "8")

    def test_stale_point_anchor_and_wrong_edge_cannot_override(self):
        for change in ({"point_img": [90, 181]}, {"anchor_img": [101, 180]},
                       {"edge": "right"}, {"pin_index": True}, {"component_id": "other"}):
            with self.subTest(change=change):
                record = copy.deepcopy(self.new)
                record["pin_target"].update(change)
                self.assertEqual(self.bind([self.old, record])[0]["number"]["text"], "8")

    def test_held_or_incomplete_record_cannot_suppress_legacy(self):
        for change in ({"accepted": False}, {"held_reason": "ambiguous"},
                       {"flags": ["glyph_integrity_unresolved"]}):
            record = copy.deepcopy(self.new)
            record["pin_number_ocr"].update(change)
            self.assertEqual(self.bind([self.old, record])[0]["number"]["text"], "8")

    def test_internal_or_far_from_wire_row_cannot_override(self):
        for box in ([110, 168, 120, 179], [89, 130, 97, 140],
                    [89, 168, 104, 179]):
            record = copy.deepcopy(self.new)
            record["bbox"] = record["geometry_bbox"] = box
            self.assertEqual(self.bind([self.old, record])[0]["number"]["text"], "8")

    def test_other_real_row_and_names_survive_numeric_override(self):
        self.pins["ic"].append([90, 220])
        self.evidence["ic"].append(dict(self.evidence["ic"][0], point_img=[90, 220], anchor_img=[100, 220]))
        other = {"bbox": [89, 207, 97, 219], "reads": {"pin_number": ("8", .98)}}
        names = [{"bbox": [110, y-5, 138, y+5], "reads": {"pin_name": (name, .97)}}
                 for y, name in ((180, "EN"), (220, "FB"))]
        records = [self.old, other, self.new] + names
        before = copy.deepcopy(records)
        got = self.bind(records)
        self.assertEqual((got[0]["number"]["text"], got[1]["number"]["text"]), ("6", "8"))
        self.assertEqual((got[0]["name"]["text"], got[1]["name"]["text"]), ("EN", "FB"))
        self.assertEqual(records, before)

    def test_cross_row_glyph_cannot_gain_priority(self):
        self.pins["ic"].append([90, 190])
        self.evidence["ic"].append(dict(self.evidence["ic"][0], point_img=[90, 190], anchor_img=[100, 190]))
        record = copy.deepcopy(self.new)
        record["bbox"] = record["geometry_bbox"] = [89, 168, 97, 194]
        self.assertEqual(self.bind([self.old, record])[0]["number"]["text"], "8")


if __name__ == "__main__":
    unittest.main()
