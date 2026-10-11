import copy
import unittest

import numpy as np

from stage2.pin_lane_crops import pin_lane_boxes


class PinLaneCropTests(unittest.TestCase):
    def fixture(self, edge="left", rows=(50,)):
        image = np.full((130, 140, 3), 255, np.uint8)
        frame = [40, 25, 110, 105]
        points, evidence = [], []
        for row in rows:
            if edge == "left":
                point, anchor = [20, row], [40, row]
            elif edge == "right":
                point, anchor = [130, row], [110, row]
            elif edge == "top":
                point, anchor = [row, 5], [row, 25]
            else:
                point, anchor = [row, 125], [row, 105]
            points.append(point)
            evidence.append({"source": "red_green_transition", "trust": "trusted",
                             "frame_confirmed": True, "point_img": point,
                             "edge": edge, "anchor_img": anchor})
        return image, {"components": [{"id": "ic", "type": "box", "bbox_img": frame}],
                       "pins": {"ic": points}, "pin_evidence": {"ic": evidence}}

    def numbers(self, doc, image, reads=None):
        return [c for c in pin_lane_boxes(doc, image, reads)[0] if c["role"] == "pin_number"]

    def test_name_missing_does_not_prevent_external_number(self):
        image, doc = self.fixture()
        image[38:49, 29:34] = 0
        candidates = pin_lane_boxes(doc, image)[0]
        self.assertEqual([c["role"] for c in candidates], ["pin_number"])
        self.assertEqual(candidates[0]["glyph_bbox"], [29, 38, 34, 49])
        self.assertIn("missing_role_box", candidates[0]["flags"])

    def test_mixed_non_numeric_old_ocr_is_a_reason_not_a_gate(self):
        image, doc = self.fixture()
        image[38:49, 29:34] = 0
        image[39:48, 44:65] = 0
        reads = [{"bbox": [27, 36, 68, 51], "reads": {"pin_number": ("1ND", .8),
                                                            "pin_name": ("1AIO", .8)}}]
        original = copy.deepcopy(reads)
        numbers = self.numbers(doc, image, reads)
        self.assertEqual(len(numbers), 1)
        self.assertIn("mixed_frame_box", numbers[0]["flags"])
        self.assertIn("non_numeric_existing_read", numbers[0]["flags"])
        self.assertEqual(reads, original)

    def test_multidigit_broken_one_keeps_every_fragment(self):
        image, doc = self.fixture()
        image[38:41, 25:27] = 0
        image[44:49, 25:27] = 0
        image[38:49, 30:36] = 0
        number = self.numbers(doc, image)[0]
        self.assertEqual(number["glyph_bbox"], [25, 38, 36, 49])
        self.assertEqual(number["cleanup_evidence"]["glyph_component_count"], 3)
        self.assertEqual(number["cleanup_evidence"]["glyph_pixels"], 82)
        self.assertEqual(number["cleanup_evidence"]["black_components_removed"], 0)

    def test_four_sides_remain_outside_actual_frame_contact(self):
        for edge in ("left", "right", "top", "bottom"):
            with self.subTest(edge=edge):
                image, doc = self.fixture(edge, rows=(60,))
                if edge == "left": image[48:59, 29:34] = 0
                elif edge == "right": image[48:59, 116:121] = 0
                elif edge == "top": image[10:20, 55:61] = 0
                else: image[111:121, 55:61] = 0
                number = self.numbers(doc, image)[0]
                box, anchor = number["glyph_bbox"], number["pin_target"]["anchor_img"]
                self.assertEqual(number["vertical"], edge in ("top", "bottom"))
                if edge == "left": self.assertLessEqual(box[2], anchor[0])
                elif edge == "right": self.assertGreaterEqual(box[0], anchor[0])
                elif edge == "top": self.assertLessEqual(box[3], anchor[1])
                else: self.assertGreaterEqual(box[1], anchor[1])

    def test_nearby_horizontal_rows_do_not_share_digits(self):
        image, doc = self.fixture(rows=(50, 62))
        image[42:49, 28:34] = 0
        image[54:61, 27:35] = 0
        numbers = self.numbers(doc, image)
        self.assertEqual([c["glyph_bbox"] for c in numbers], [[28, 42, 34, 49], [27, 54, 35, 61]])
        self.assertLess(numbers[0]["region"][3], numbers[1]["region"][1])

    def test_nearby_rotated_rows_do_not_share_digits(self):
        image, doc = self.fixture("top", rows=(58, 66))
        image[12:20, 55:61] = 0
        image[12:20, 64:70] = 0
        numbers = self.numbers(doc, image)
        self.assertEqual([c["glyph_bbox"] for c in numbers], [[55, 12, 61, 20], [64, 12, 70, 20]])

    def test_touching_high_baseline_digit_expands_without_using_previous_row(self):
        image, doc = self.fixture(rows=(30, 50))
        image[18:27, 28:34] = 0
        image[33:45, 29:34] = 0
        numbers = self.numbers(doc, image)
        self.assertEqual(numbers[1]["glyph_bbox"], [29, 33, 34, 45])
        self.assertGreater(numbers[1]["region"][1], 30)
        self.assertEqual(numbers[1]["cleanup_evidence"]["glyph_pixels"], 60)

    def test_outer_vertical_name_completes_connected_strokes_to_blank(self):
        image, doc = self.fixture("top", rows=(60, 80))
        image[30:48, 80:92] = 0
        image[32:42, 97:103] = 0  # a separate word is never glued on
        names = [c for c in pin_lane_boxes(doc, image)[0] if c["role"] == "pin_name"]
        target = next(c for c in names if c["pin_target"]["pin_index"] == 1)
        self.assertEqual(target["glyph_bbox"], [80, 30, 92, 48])
        self.assertIn("initial_glyph_touches_search_boundary", target["flags"])
        self.assertIn("glyph_completed_from_boundary", target["flags"])
        self.assertNotIn("glyph_touches_search_boundary", target["flags"])
        self.assertNotIn("glyph_integrity_unresolved", target["flags"])

    def test_vertical_number_completes_strokes_on_other_tangent_side(self):
        image, doc = self.fixture("top", rows=(60, 80))
        image[10:16, 70:81] = 0
        number = next(c for c in self.numbers(doc, image) if c["pin_target"]["pin_index"] == 1)
        self.assertEqual(number["glyph_bbox"], [70, 10, 81, 16])
        self.assertIn("glyph_completed_from_boundary", number["flags"])
        self.assertEqual(number["cleanup_evidence"]["glyph_pixels"], 66)
        self.assertLess(number["glyph_bbox"][3], 25)

    def test_vertical_completion_still_limited_by_neighbour_marks_uncertainty(self):
        image, doc = self.fixture("top", rows=(60, 80))
        image[30:48, 60:72] = 0
        image[30:48, 80:92] = 0
        name = next(c for c in pin_lane_boxes(doc, image)[0]
                    if c["role"] == "pin_name" and c["pin_target"]["pin_index"] == 0)
        self.assertLessEqual(name["glyph_bbox"][2], 70)
        self.assertIn("glyph_integrity_unresolved", name["flags"])
        self.assertTrue(name["cleanup_evidence"]["glyph_completion"]["integrity_unresolved"])

    def test_detached_outside_fragment_is_not_added_and_marks_uncertainty(self):
        image, doc = self.fixture("top", rows=(60, 80))
        image[30:48, 80:92] = 0
        image[35:37, 94:95] = 0
        name = next(c for c in pin_lane_boxes(doc, image)[0]
                    if c["role"] == "pin_name" and c["pin_target"]["pin_index"] == 1)
        self.assertEqual(name["glyph_bbox"], [80, 30, 92, 48])
        self.assertIn("glyph_integrity_unresolved", name["flags"])
        completion = name["cleanup_evidence"]["glyph_completion"]
        self.assertEqual(completion["nearby_unattached_components"][0]["bbox"], [94, 35, 95, 37])

    def test_source_integrity_is_recorded_for_each_old_box_separately(self):
        image, doc = self.fixture()
        image[38:49, 29:34] = 0
        reads = [{"bbox": [28, 37, 35, 50], "reads": {"pin_number": ("1", .95)}},
                 {"bbox": [27, 37, 66, 51], "reads": {"pin_number": ("1ND", .8)}}]
        sources = self.numbers(doc, image, reads)[0]["cleanup_evidence"]["source_boxes"]
        self.assertEqual([s["crosses_frame"] for s in sources], [False, True])
        self.assertEqual([s["truncates_glyph"] for s in sources], [False, False])
        self.assertEqual([s["pin_row_count"] for s in sources], [1, 1])

    def test_untrusted_stale_or_unconfirmed_observations_open_no_lanes(self):
        image, doc = self.fixture()
        image[38:49, 29:34] = 0
        for update in ({"trust": "uncertain"}, {"frame_confirmed": False},
                       {"source": "grayscale"}, {"point_img": [21, 50]},
                       {"anchor_img": [60, 50]}, {"point_img": [float("nan"), 50]}):
            changed = copy.deepcopy(doc)
            changed["pin_evidence"]["ic"][0].update(update)
            self.assertEqual(pin_lane_boxes(changed, image)[0], [])

    def test_actual_anchor_not_detector_box_boundary_splits_roles(self):
        image, doc = self.fixture()
        doc["components"][0]["bbox_img"][0] = 36
        image[38:49, 35:39] = 0
        number = self.numbers(doc, image)[0]
        self.assertEqual(number["glyph_bbox"], [35, 38, 39, 49])
        self.assertEqual(number["region"][2], 40)

    def test_colored_symbol_does_not_erase_black_number(self):
        image, doc = self.fixture()
        image[38:49, 29:34] = 0
        image[37:48, 23:27] = [0, 160, 0]
        before = image.copy()
        numbers = self.numbers(doc, image)
        self.assertEqual(numbers[0]["glyph_bbox"], [29, 38, 34, 49])
        np.testing.assert_array_equal(image, before)

    def test_opposite_side_name_is_not_assigned_to_an_unnamed_pin(self):
        image, doc = self.fixture()
        image[45:55, 95:106] = 0
        image[38:49, 29:34] = 0
        self.assertEqual([c["role"] for c in pin_lane_boxes(doc, image)[0]], ["pin_number"])

    def test_name_and_number_windows_are_both_generated_on_four_sides(self):
        for edge in ("left", "right", "top", "bottom"):
            with self.subTest(edge=edge):
                image, doc = self.fixture(edge, rows=(60,))
                if edge == "left":
                    image[55:64, 44:63] = 0
                    image[48:59, 29:34] = 0
                elif edge == "right":
                    image[55:64, 85:106] = 0
                    image[48:59, 116:121] = 0
                elif edge == "top":
                    image[29:46, 56:63] = 0
                    image[10:20, 55:61] = 0
                else:
                    image[83:101, 56:63] = 0
                    image[111:121, 55:61] = 0
                self.assertEqual([c["role"] for c in pin_lane_boxes(doc, image)[0]],
                                 ["pin_name", "pin_number"])

    def test_invalid_input_is_safe(self):
        image, doc = self.fixture()
        for bad in (None, image[:, :, 0], np.zeros((5, 5, 4), np.uint8)):
            self.assertEqual(pin_lane_boxes(doc, bad), ([], None))


if __name__ == "__main__":
    unittest.main()
