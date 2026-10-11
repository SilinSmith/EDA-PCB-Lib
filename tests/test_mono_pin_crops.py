import copy
import unittest

import cv2
import numpy as np

from stage2.mono_pin_crops import mono_pin_boxes, _complete_group


def fixture(rows=(50, 80), edges=("left",)):
    image = np.full((120, 220, 3), 255, np.uint8)
    frame = [80, 10, 180, 110]
    cv2.rectangle(image, (80, 10), (180, 110), (0, 0, 0), 1)
    points, evidence = [], []
    for edge in edges:
        for row in rows:
            anchor = [frame[0] if edge == "left" else frame[2], row]
            point = [anchor[0] - 12 if edge == "left" else anchor[0] + 12, row]
            points.append(point)
            evidence.append({"point_img": point.copy(), "anchor_img": anchor,
                             "edge": edge, "trust": "trusted", "confidence": .88,
                             "reason": "frame_attached_normal_stroke",
                             "frame_confirmed": True, "frame_coverage": .99,
                             "normal_ink_run": 22, "normal_ink_hits": 22,
                             "first_ink_distance": 1, "text_overlap_fraction": 0.,
                             "branching_fraction": .10})
            cv2.line(image, (40 if edge == "left" else 180, row),
                     (80 if edge == "left" else 215, row), (0, 0, 0), 1)
    doc = {"components": [{"id": "ic", "type": "box", "bbox_img": frame}],
           "pins": {"ic": points}, "pin_evidence": {"ic": evidence}}
    return doc, image


class MonoPinCropTests(unittest.TestCase):
    def get(self, candidates, role, index=0):
        return next(c for c in candidates if c["role"] == role and c["pin_target"]["pin_index"] == index)

    def test_diagonal_ink_continuation_across_every_cell_side_is_incomplete(self):
        # Each selected stroke leaves only through an 8-neighbour diagonal;
        # same-column or same-row tests would miss the clipped continuation.
        seeds = {
            0: [(12 + t, 10 - t) for t in range(8)],
            1: [(10 - t, 12 + t) for t in range(8)],
            2: [(12 + t, 19 + t) for t in range(8)],
            3: [(19 + t, 12 + t) for t in range(8)],
        }
        for side, points in seeds.items():
            with self.subTest(side=side):
                mask = np.zeros((40, 40), dtype=bool)
                for y, x in points:
                    mask[y, x] = True
                region = [5, 5, 25, 25]
                _, evidence = _complete_group(mask, region, {"anchor_img": [4, 12]}, "pin_name", 30)
                self.assertTrue(evidence["integrity_unresolved"])
                self.assertIn(side, evidence["connected_ink_beyond_region_sides"])

    def test_reported_bottom_diagonal_reproduction_remains_held(self):
        mask = np.zeros((25, 35), dtype=bool)
        for y in range(9, 17):
            mask[y, y + 1] = True
        glyph, evidence = _complete_group(mask, [5, 5, 25, 15],
                                          {"anchor_img": [4, 12]}, "pin_name", 30)
        self.assertEqual(glyph, [10, 9, 16, 15])
        self.assertTrue(evidence["integrity_unresolved"])
        self.assertIn(3, evidence["connected_ink_beyond_region_sides"])

    def test_complete_repeated_narrow_ones_beyond_old_pin_segment(self):
        doc, image = fixture()
        # Three disconnected narrow 1s, including their horizontal caps. Each
        # gap is larger than a tight word heuristic, but within one row word.
        for x in (31, 44, 57):
            image[31:46, x:x + 2] = 0
            image[31:33, x - 2:x + 2] = 0
        before_doc, before_image = copy.deepcopy(doc), image.copy()
        candidates, gray = mono_pin_boxes(doc, image)
        number = self.get(candidates, "pin_number")
        self.assertEqual(number["bbox"], [29, 31, 59, 46])
        self.assertFalse(number["flags"])
        self.assertEqual(number["cleanup_evidence"]["glyph_component_count"], 3)
        self.assertLess(number["bbox"][0], doc["pins"]["ic"][0][0])
        self.assertEqual(doc, before_doc)
        np.testing.assert_array_equal(image, before_image)
        np.testing.assert_array_equal(gray, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def test_near_wire_one_cap_and_digit_horizontal_bars_are_unchanged(self):
        doc, image = fixture()
        image[32:48, 58:60] = 0
        image[32:34, 55:60] = 0
        image[46:48, 55:62] = 0
        boxes, gray = mono_pin_boxes(doc, image)
        number = self.get(boxes, "pin_number")
        self.assertEqual(number["bbox"], [55, 32, 62, 48])
        self.assertFalse(number["flags"])
        self.assertEqual(number["cleanup_evidence"]["black_pixels_removed"], 0)
        np.testing.assert_array_equal(gray[32:48, 55:62], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[32:48, 55:62])

    def test_digit_touching_stroke_is_held_rather_than_erased_or_invented(self):
        doc, image = fixture()
        image[32:51, 58:60] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        number = self.get(boxes, "pin_number")
        self.assertIn("glyph_integrity_unresolved", number["flags"])
        self.assertIn(3, number["cleanup_evidence"]["connected_ink_beyond_region_sides"])
        self.assertEqual(number["cleanup_evidence"]["black_pixels_removed"], 0)

    def test_same_row_opposite_names_do_not_merge(self):
        doc, image = fixture(edges=("left", "right"))
        image[43:57, 90:107] = 0
        image[43:57, 155:171] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        self.assertEqual(self.get(boxes, "pin_name", 0)["bbox"], [90, 43, 107, 57])
        self.assertEqual(self.get(boxes, "pin_name", 2)["bbox"], [155, 43, 171, 57])

    def test_neighbor_cells_are_independent_and_clipping_stays_unresolved(self):
        doc, image = fixture(rows=(50, 70))
        image[43:58, 90:107] = 0
        image[63:77, 90:108] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        self.assertEqual(self.get(boxes, "pin_name", 0)["bbox"], [90, 43, 107, 58])
        self.assertEqual(self.get(boxes, "pin_name", 1)["bbox"], [90, 63, 108, 77])
        image[55:65, 95:100] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        self.assertIn("glyph_integrity_unresolved", self.get(boxes, "pin_name", 0)["flags"])

    def test_old_ocr_text_cannot_determine_new_crop(self):
        doc, image = fixture()
        image[43:57, 90:107] = 0
        initial, _ = mono_pin_boxes(doc, image)
        altered, _ = mono_pin_boxes(doc, image, reads=[{"bbox": [0, 0, 200, 100], "reads": {
            "pin_name": ("FAKE_EXPECTED_WORD", .999), "pin_number": ("999", .999)}}])
        self.assertEqual(initial, altered)

    def test_stale_weak_unknown_and_color_evidence_cannot_open_target(self):
        doc, image = fixture(rows=(50,))
        image[43:57, 90:107] = 0
        for key, value in (("point_img", [67, 50]), ("anchor_img", [150, 50]),
                           ("trust", "uncertain"), ("frame_confirmed", False),
                           ("normal_ink_hits", 2), ("frame_coverage", .3),
                           ("source", "unknown_source"), ("source", "red_green_transition"),
                           ("edge", "top")):
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(doc)
                changed["pin_evidence"]["ic"][0][key] = value
                self.assertEqual(mono_pin_boxes(changed, image)[0], [])

    def test_missing_source_is_supported_but_absent_stroke_in_image_is_not(self):
        doc, image = fixture(rows=(50,))
        image[43:57, 90:107] = 0
        self.assertTrue(mono_pin_boxes(doc, image)[0])
        image[46:55, 40:80] = 255
        self.assertEqual(mono_pin_boxes(doc, image)[0], [])

    def test_colored_pixels_inside_a_mono_crop_are_held(self):
        doc, image = fixture(rows=(50,))
        image[43:57, 90:107] = 0
        image[47:50, 97:100] = (0, 0, 200)
        boxes, _ = mono_pin_boxes(doc, image)
        name = self.get(boxes, "pin_name")
        self.assertIn("glyph_integrity_unresolved", name["flags"])
        self.assertEqual(name["cleanup_evidence"]["unresolved_reason"], "non_monochrome_ink_inside_original_crop")


if __name__ == "__main__":
    unittest.main()
