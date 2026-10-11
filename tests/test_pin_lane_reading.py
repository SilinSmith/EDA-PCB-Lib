import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stage2.pin_lane_reading import read_pin_lanes


class PinLaneReadingTests(unittest.TestCase):
    def fixture(self, role="pin_name", vertical=False):
        image = np.full((50, 60, 3), 255, np.uint8)
        image[20:29, 24:38] = 0
        target = {"component_id": "ic", "pin_index": 0,
                  "edge": "bottom" if vertical else "left",
                  "anchor_img": [30, 35] if vertical else [20, 25],
                  "point_img": [30, 45] if vertical else [10, 25]}
        candidate = {"bbox": [24, 20, 38, 29], "role": role,
                     "vertical": vertical, "region": [22, 18, 40, 31],
                     "pin_target": target, "flags": [],
                     "cleanup_evidence": {"source_boxes": []}}
        doc = {"components": [], "pins": {}}
        return image, candidate, doc

    def run_read(self, image, candidates, doc, baseline, predictions, **kwargs):
        original = [{"bbox": [2, 2, 5, 5], "reads": {"pin_name": ("KEEP", .9)}}]
        before = copy.deepcopy(original)
        clean = image[..., 0].copy()
        with patch("stage2.pin_lane_reading.pin_lane_boxes", return_value=(candidates, clean)), \
             patch("stage2.pin_lane_reading.assign", return_value=baseline), \
             patch("stage2.pin_lane_reading.read_isolated_many", return_value=predictions) as helper:
            out = read_pin_lanes(doc, original, SimpleNamespace(recognizer=lambda: object()), image, **kwargs)
        self.assertEqual(original, before)
        return out, helper

    def predictions(self, values):
        return [{"text": text, "confidence": conf, "sampling": {"strategy": "single"}}
                for text, conf in values]

    def test_missing_name_recovered_without_mutating_original_reads(self):
        image, candidate, doc = self.fixture()
        out, helper = self.run_read(image, [candidate], doc, {"comps": {}},
                                    self.predictions([("EN", .91), ("EN", .92)]))
        self.assertEqual(out[0]["reads"], {"pin_name": ("EN", .91)})
        self.assertEqual(out[0]["pin_target"], candidate["pin_target"])
        self.assertTrue(out[0]["name_only"])
        self.assertEqual(helper.call_count, 1)

    def test_new_boundary_warning_does_not_replace_intact_old_word(self):
        image, candidate, doc = self.fixture()
        candidate["flags"] = ["glyph_touches_search_boundary", "color_contamination"]
        baseline = {"comps": {"ic": {"pins": {0: {
            "name": {"text": "OUT", "conf": .6, "box": candidate["bbox"]}}}}}}
        out, helper = self.run_read(image, [candidate], doc, baseline, [])
        self.assertEqual(out, [])
        helper.assert_not_called()

    def test_discarded_dirty_alternative_cannot_displace_clean_bound_source(self):
        image, candidate, doc = self.fixture()
        candidate["cleanup_evidence"]["source_boxes"] = [
            {"bbox": [18, 18, 40, 31], "color_contamination_ratio": .8, "crosses_frame": True}]
        baseline = {"comps": {"ic": {"pins": {0: {
            "name": {"text": "OUT", "box": candidate["bbox"]}}}}}}
        out, _ = self.run_read(image, [candidate], doc, baseline, [])
        self.assertEqual(out, [])

    def test_colored_name_uses_clean_view_not_highest_confidence(self):
        image, candidate, doc = self.fixture()
        image[20, 24] = [0, 0, 180]
        out, _ = self.run_read(image, [candidate], doc, {"comps": {}},
                               self.predictions([("CND", .99), ("GND", .85)]))
        self.assertEqual(out[0]["reads"], {"pin_name": ("GND", .85)})
        self.assertEqual(out[0]["pin_lane_crop"]["view"], "light")

    def test_number_orientation_is_independent_of_missing_name(self):
        image, candidate, doc = self.fixture("pin_number", vertical=True)
        out, _ = self.run_read(image, [candidate], doc, {"comps": {}},
                               self.predictions([("2", .99), ("2", .99), ("Z", .98), ("Z", .98)]))
        self.assertEqual(out[0]["reads"], {"pin_number": ("2", .99)})

    def test_directionless_six_nine_is_not_guessed_by_number_order(self):
        image, candidate, doc = self.fixture("pin_number", vertical=True)
        out, _ = self.run_read(image, [candidate], doc, {"comps": {}},
                               self.predictions([("6", .99), ("6", .99), ("9", .98), ("9", .98)]))
        self.assertEqual(out, [])

    def test_invalid_or_weak_number_is_not_added(self):
        image, candidate, doc = self.fixture("pin_number")
        for text, confidence in (("X1", .99), ("0", .99), ("1000", .99), ("6", .4)):
            out, _ = self.run_read(image, [copy.deepcopy(candidate)], doc, {"comps": {}},
                                   self.predictions([(text, confidence), (text, confidence)]))
            self.assertEqual(out, [])

    def test_unresolved_geometry_is_not_sent_for_replacement(self):
        image, candidate, doc = self.fixture()
        candidate["flags"] = ["glyph_integrity_unresolved"]
        out, helper = self.run_read(image, [candidate], doc, {"comps": {}}, [])
        self.assertEqual(out, [])
        helper.assert_not_called()

    def test_new_name_cannot_activate_dirty_old_number_without_confirmation(self):
        image, candidate, doc = self.fixture()
        number = copy.deepcopy(candidate)
        number.update(role="pin_number", bbox=[10, 15, 19, 23])
        number["cleanup_evidence"]["source_boxes"] = [
            {"bbox": [8, 14, 20, 24], "color_contamination_ratio": .7}]
        baseline = {"comps": {"ic": {"pins": {0: {
            "name": None, "number": {"text": "25", "box": [8, 14, 20, 24]}}}}}}
        out, _ = self.run_read(image, [candidate, number], doc, baseline,
                               self.predictions([("NC", .95), ("NC", .95)]), include_numbers=False)
        self.assertEqual(out[0]["reads"], {})
        self.assertEqual(out[0]["pin_lane_crop"]["proposed_read"], {"pin_name": ("NC", .95)})
        self.assertEqual(out[0]["pin_lane_crop"]["held_reason"],
                         "unresolved_symbol_contaminated_partner_number")

    def test_unknown_cleanup_policy_rejected(self):
        with self.assertRaises(ValueError):
            read_pin_lanes({}, [], None, None, cleanup="aggressive")

    def test_missing_digit_crop_still_checks_old_number_pixels_and_frame(self):
        image, candidate, doc = self.fixture()
        baseline = {"comps": {"ic": {"pins": {0: {
            "name": None, "number": {"text": "25", "box": [8, 14, 22, 24]}}}}}}
        out, _ = self.run_read(image, [candidate], doc, baseline,
                               self.predictions([("NC", .99), ("NC", .99)]))
        self.assertEqual(out[0]["reads"], {})
        self.assertIn("held_reason", out[0]["pin_lane_crop"])

    def test_filtered_number_cannot_confirm_high_confidence_name(self):
        image, candidate, doc = self.fixture()
        number = copy.deepcopy(candidate)
        number.update(role="pin_number", bbox=[10, 15, 19, 23])
        number["cleanup_evidence"]["source_boxes"] = [
            {"bbox": [8, 14, 20, 24], "color_contamination_ratio": .7}]
        baseline = {"comps": {"ic": {"pins": {0: {
            "name": None, "number": {"text": "25", "box": [8, 14, 20, 24], "conf": .98}}}}}}
        rebound = {"comps": {"ic": {"pins": {0: {
            "name": {"text": "NC", "box": candidate["bbox"], "conf": .99},
            "number": {"text": "6", "box": number["bbox"], "conf": .7}}}}}}
        with patch("stage2.pin_lane_reading.pin_lane_boxes", return_value=([candidate, number], image[..., 0])), \
             patch("stage2.pin_lane_reading.assign", side_effect=[baseline, rebound]), \
             patch("stage2.pin_lane_reading.read_isolated_many", return_value=self.predictions(
                 [("NC", .99), ("NC", .99), ("6", .7), ("6", .7)])):
            out = read_pin_lanes(doc, [], SimpleNamespace(recognizer=lambda: object()), image, min_conf=.95)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["reads"], {})
        self.assertIn("held_reason", out[0]["pin_lane_crop"])

    def test_min_confidence_is_applied_before_deciding_whether_role_exists(self):
        from stage2.assign import assign as actual_assign
        image, candidate, doc = self.fixture()
        target = candidate["pin_target"]
        doc = {"components": [{"id": "ic", "type": "box", "bbox_img": [20, 10, 50, 40]}],
               "pins": {"ic": [target["point_img"]]},
               "pin_evidence": {"ic": [{"edge": "left", "anchor_img": target["anchor_img"],
                   "point_img": target["point_img"], "trust": "trusted", "frame_confirmed": True,
                   "source": "red_green_transition"}]}}
        old = [{"bbox": candidate["bbox"], "reads": {"pin_name": ("F", .6)}}]
        with patch("stage2.pin_lane_reading.pin_lane_boxes", return_value=([candidate], image[..., 0])), \
             patch("stage2.pin_lane_reading.assign", wraps=actual_assign), \
             patch("stage2.pin_lane_reading.read_isolated_many", return_value=self.predictions(
                 [("FB", .99), ("FB", .99)])):
            out = read_pin_lanes(doc, old, SimpleNamespace(recognizer=lambda: object()), image, min_conf=.95)
        self.assertEqual(out[0]["reads"], {"pin_name": ("FB", .99)})
        self.assertEqual(old[0]["reads"], {"pin_name": ("F", .6)})


if __name__ == "__main__":
    unittest.main()
