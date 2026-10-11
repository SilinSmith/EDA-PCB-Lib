"""Same-terminal contacts must retain the same pin number during assembly."""
import unittest
from unittest.mock import patch

from stage2 import assemble


class PinStampAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.vertical = {
            "id": "c0", "type": "r", "bbox_json": [30, 40, 50, 60],
            "bbox_img": [30, 40, 50, 60], "bbox_raw_img": [37, 40, 43, 60],
        }

    def stamp(self, points, ctype="r", component=None):
        with patch.object(assemble, "PIN_STAMP_ON", True):
            return assemble.stamp_pins(component or self.vertical, ctype, points, 100)

    def test_two_contacts_below_vertical_symbol_are_both_pin1(self):
        stamped = self.stamp([(39, 70), (41, 73)])
        self.assertEqual(stamped, [("1", (40, 30)), ("1", (40, 30))])

    def test_three_contacts_above_vertical_symbol_are_all_pin2(self):
        stamped = self.stamp([(39, 30), (40, 28), (41, 26)])
        self.assertEqual(stamped, [("2", (40, 70))] * 3)

    def test_reordering_contacts_does_not_change_their_physical_pin(self):
        points = [(39, 70), (41, 73), (40, 30), (39, 28)]
        self.assertEqual(self.stamp(points[::-1]), self.stamp(points)[::-1])

    def test_opposite_terminals_remain_distinct(self):
        self.assertEqual(self.stamp([(40, 70), (40, 30)]),
                         [("1", (40, 30)), ("2", (40, 70))])

    def test_single_observed_contact_does_not_invent_opposite_terminal(self):
        self.assertEqual(self.stamp([(40, 70)]), [("1", (40, 30))])

    def test_horizontal_contacts_on_left_are_both_pin1(self):
        horizontal = dict(self.vertical, bbox_raw_img=[30, 47, 50, 53])
        self.assertEqual(self.stamp([(20, 49), (18, 51)], component=horizontal),
                         [("1", (20, 50)), ("1", (20, 50))])

    def test_diode_polarity_convention_is_preserved(self):
        self.assertEqual(self.stamp([(40, 30), (39, 28)], ctype="d"),
                         [("1", (40, 70)), ("1", (40, 70))])

    def test_crystal_perpendicular_axis_is_preserved(self):
        self.assertEqual(self.stamp([(20, 50), (18, 50)], ctype="crystal_2pin"),
                         [("1", (20, 50)), ("1", (20, 50))])

    def build(self, points):
        stage1 = {
            "image": {"w": 100, "h": 100}, "components": [self.vertical],
            "pins": {"c0": points},
            "wire_detail": {"segments": [[point, (0, point[1])] for point in points]},
            "nets": [{"id": f"raw{k}", "pins": [["c0", k]], "seg_index": [k]}
                     for k in range(len(points))],
        }
        assigned = {"comps": {"c0": {"designator": {"text": "R1"},
                                      "value": {"text": "10K"}}}, "net_labels": []}
        with patch.object(assemble, "PIN_STAMP_ON", True):
            return assemble.build_doc(stage1, assigned, "synthetic")

    def test_duplicate_contacts_publish_one_pin_and_preserve_all_index_aliases(self):
        doc = self.build([(40, 70), (40, 72)])
        self.assertEqual(set(doc["pins"]["R1"]), {"pin_1"})
        self.assertEqual(doc["__pin_index"]["c0"], {"0": "pin_1", "1": "pin_1"})
        self.assertEqual(len(doc["nets"]), 1)
        net = next(iter(doc["nets"].values()))
        self.assertEqual(net["hyperGraph"], "(R1.1)")
        self.assertEqual(len(net["edges"]), 2)

    def test_opposite_contact_networks_are_not_merged(self):
        doc = self.build([(40, 70), (40, 30)])
        self.assertEqual(set(doc["pins"]["R1"]), {"pin_1", "pin_2"})
        self.assertEqual({net["hyperGraph"] for net in doc["nets"].values()},
                         {"(R1.1)", "(R1.2)"})


if __name__ == "__main__":
    unittest.main()
