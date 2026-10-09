"""Real wire rows must retain their observed number when OCR adds glyph contacts."""
import copy
import importlib
import json
import unittest
from unittest.mock import patch

from stage2 import structmatch

assign = importlib.import_module("stage2.assign")


def contact(point, edge="left", trust="trusted", anchor=None):
    return {"point_img": list(point), "edge": edge, "trust": trust,
            "anchor_img": list(anchor or [100, point[1]]),
            "frame_confirmed": True, "source": "red_green_transition"}


def text(rect, **reads):
    return {"bbox": list(rect), "reads": {role: (value, .97)
                                           for role, value in reads.items()}}


class PinNumberRowBindingTests(unittest.TestCase):
    box = [100, 100, 220, 240]

    def assigned(self, points, texts, records):
        comps = [{"id": "ic", "type": "box", "bbox_img": self.box}]
        with patch.object(assign, "PIN_NUMBER_TRUST_BIND", True):
            return assign.assign(comps, {"ic": points}, copy.deepcopy(texts),
                                 pin_evidence={"ic": records} if records is not None else None)

    def test_real_named_rows_win_over_closer_numeric_glyph_contacts(self):
        points = [[90, 140], [90, 180], [90, 132], [90, 172]]
        records = [contact(p, trust="trusted" if i < 2 else "uncertain")
                   for i, p in enumerate(points)]
        texts = [text([110, 140, 135, 150], pin_name="GND"),
                 text([110, 180, 135, 190], pin_name="VCC"),
                 text([88, 129, 97, 139], pin_number="7"),
                 text([89, 130, 98, 140], pin_number="7"),
                 text([88, 169, 97, 179], pin_number="8"),
                 text([89, 170, 98, 180], pin_number="8")]
        pins = self.assigned(points, texts, records)["comps"]["ic"]["pins"]
        self.assertEqual((pins[0]["name"]["text"], pins[0]["number"]["text"]), ("GND", "7"))
        self.assertEqual((pins[1]["name"]["text"], pins[1]["number"]["text"]), ("VCC", "8"))
        self.assertIsNone(pins[2]["number"])
        self.assertIsNone(pins[3]["number"])

    def test_crossing_number_can_be_read_without_stealing_a_numeric_pin_name(self):
        points = [[90, 140], [90, 180]]
        records = [contact(p) for p in points]
        for numeric_name in ("0", "1"):
            with self.subTest(name=numeric_name):
                texts = [text([93, 129, 113, 140], pin_number="1", pin_name="1"),
                         text([110, 140, 138, 150], pin_name="GND"),
                         text([110, 180, 118, 190], pin_name=numeric_name),
                         text([89, 170, 98, 180], pin_number="8")]
                pins = self.assigned(points, texts, records)["comps"]["ic"]["pins"]
                self.assertEqual(pins[0]["number"]["text"], "1")
                self.assertEqual(pins[0]["name"]["text"], "GND")
                self.assertEqual(pins[1]["number"]["text"], "8")
                self.assertEqual(pins[1]["name"]["text"], numeric_name)

    def test_fully_internal_numeric_name_is_not_reused_as_a_number(self):
        points = [[90, 140]]
        for numeric_name in ("0", "1", "12"):
            pins = self.assigned(points, [text([110, 137, 130, 147],
                    pin_name=numeric_name, pin_number=numeric_name)],
                    [contact(points[0])])["comps"]["ic"]["pins"]
            self.assertEqual(pins[0]["name"]["text"], numeric_name)
            self.assertIsNone(pins[0]["number"])

    def test_number_does_not_jump_an_entire_row_to_fill_missing_reads(self):
        points = [[90, 140], [90, 180]]
        records = [contact(p) for p in points]
        texts = [text([110, 140, 135, 150], pin_name="A"),
                 text([110, 180, 135, 190], pin_name="B"),
                 text([89, 170, 98, 180], pin_number="8")]
        pins = self.assigned(points, texts, records)["comps"]["ic"]["pins"]
        self.assertIsNone(pins[0]["number"])
        self.assertEqual(pins[1]["number"]["text"], "8")

    def test_nonmonotonic_drawn_numbers_remain_legal(self):
        points = [[90, y] for y in (125, 155, 185, 215)]
        texts = [text([89, p[1]-10, 98, p[1]], pin_number=n)
                 for p, n in zip(points, ("9", "2", "7", "1"))]
        pins = self.assigned(points, texts, [contact(p) for p in points])["comps"]["ic"]["pins"]
        self.assertEqual([pins[i]["number"]["text"] for i in range(4)], ["9", "2", "7", "1"])

    def test_distinct_same_number_boxes_on_different_rows_are_not_merged(self):
        points = [[90, 140], [90, 180]]
        texts = [text([89, p[1]-10, 98, p[1]], pin_number="7") for p in points]
        pins = self.assigned(points, texts, [contact(p) for p in points])["comps"]["ic"]["pins"]
        self.assertEqual([pins[i]["number"]["text"] for i in range(2)], ["7", "7"])

    def test_number_binding_is_unchanged_when_only_name_anchor_moves(self):
        points = [[90, 140], [90, 180]]
        texts = [text([89, p[1]-10, 98, p[1]], pin_number=n)
                 for p, n in zip(points, ("7", "8"))]
        before = self.assigned(points, texts, [contact(p) for p in points])
        after = self.assigned(points, texts, [contact(p, anchor=[100, p[1]+3]) for p in points])
        self.assertEqual(before, after)

    def test_stale_or_wholly_uncertain_evidence_keeps_original_behavior(self):
        points = [[90, 140], [90, 180]]
        texts = [text([89, 170, 98, 180], pin_number="8")]
        before = self.assigned(points, texts, None)
        for records in ([contact([90, 141]), contact([90, 181])],
                        [contact(p, trust="uncertain") for p in points], []):
            with self.subTest(records=records):
                after = self.assigned(points, texts, records)
                self.assertEqual(json.dumps(before, sort_keys=True), json.dumps(after, sort_keys=True))

    def test_monochrome_frame_trust_keeps_original_number_path(self):
        points = [[90, 140], [90, 132]]
        texts = [text([89, 129, 98, 139], pin_number="7")]
        records = [contact(points[0]), contact(points[1], trust="uncertain")]
        for rec in records:
            rec.pop("source")
        before = self.assigned(points, texts, None)
        after = self.assigned(points, texts, records)
        self.assertEqual(before, after)

    def test_switch_off_keeps_original_number_path(self):
        points = [[90, 140], [90, 132]]
        texts = [text([89, 129, 98, 139], pin_number="7")]
        comps = [{"id": "ic", "type": "box", "bbox_img": self.box}]
        records = {"ic": [contact(points[0]), contact(points[1], trust="uncertain")]}
        with patch.object(assign, "PIN_NUMBER_TRUST_BIND", False):
            before = assign.assign(comps, {"ic": points}, texts)
            after = assign.assign(comps, {"ic": points}, texts, pin_evidence=records)
        self.assertEqual(before, after)

    def test_top_number_is_not_bound_to_left_pin_at_same_corner(self):
        points = [[90, 115], [115, 90]]
        records = [contact(points[0]), contact(points[1], edge="top", anchor=[115, 100])]
        texts = [text([110, 82, 120, 94], pin_number="3")]
        pins = self.assigned(points, texts, records)["comps"]["ic"]["pins"]
        self.assertIsNone(pins[0]["number"])
        self.assertEqual(pins[1]["number"]["text"], "3")


if __name__ == "__main__":
    unittest.main()
