import copy
import unittest

import cv2
import numpy as np

from stage2.mono_pin_crops import mono_pin_boxes, _complete_group, _frame_inner_bounds


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

    def test_dense_number_cell_starts_after_actual_previous_stroke_without_cutting_its_top(self):
        doc,image=fixture(rows=(50,62),edges=("right",))
        cv2.line(image,(180,50),(215,50),(0,0,0),2)
        image[53:61,187:191]=0
        image[53:61,197:202]=0
        original=image.copy()
        boxes,_=mono_pin_boxes(doc,image)
        number=self.get(boxes,"pin_number",1)
        self.assertEqual(number["bbox"],[187,53,202,61])
        self.assertFalse(number["flags"])
        prior=number["cleanup_evidence"]["stroke_probe"]["previous_current_stroke_limit"]
        self.assertEqual(prior["stroke_bottom"],52)
        self.assertEqual(prior["crop_lower_limit"],53)
        np.testing.assert_array_equal(image,original)

    def test_intact_number_cell_does_not_import_previous_stroke_residual(self):
        doc,image=fixture(rows=(50,85),edges=("right",))
        image[52,182:184]=0
        image[74:82,187:203]=0
        boxes,_=mono_pin_boxes(doc,image)
        number=self.get(boxes,"pin_number",1)
        self.assertEqual(number["bbox"],[187,74,203,82])
        self.assertFalse(number["flags"])

    def test_measured_thick_frame_inner_edge_preserves_complete_narrow_name(self):
        doc, image = fixture(rows=(50,))
        cv2.rectangle(image, (80,10), (180,110), (0,0,0), 5)
        image[42:58, 89:91] = 0  # narrow I, including every original pixel
        image[43:58, 96:106] = 0
        source = image.copy()
        boxes, gray = mono_pin_boxes(doc, image)
        name = self.get(boxes, "pin_name")
        self.assertEqual(name["bbox"], [89,42,106,58])
        self.assertFalse(name["flags"])
        measured = name["cleanup_evidence"]["stroke_probe"]["measured_frame_border"]
        self.assertTrue(any(b["side"] == 0 and b["persistent_band"][1] >= 84 for b in measured))
        self.assertEqual(name["cleanup_evidence"]["black_pixels_removed"], 0)
        np.testing.assert_array_equal(image, source)
        np.testing.assert_array_equal(gray, image[:,:,0])

    def test_short_near_frame_text_is_not_measured_as_frame_border(self):
        doc, image = fixture(rows=(50,))
        image[42:58, 84:87] = 0
        ob={"frame":doc["components"][0]["bbox_img"],"anchor_img":[80,50],"edge":"left"}
        bounds, evidence = _frame_inner_bounds(image[:,:,0]<190,image[:,:,0]<245,ob)
        self.assertEqual(next(b for b in evidence if b["side"]==0)["persistent_band"],[80,81])
        self.assertLessEqual(bounds[0], 82)
        name=self.get(mono_pin_boxes(doc,image)[0],"pin_name")
        self.assertEqual(name["bbox"],[84,42,87,58])

    def test_offset_suffix_groups_without_merging_adjacent_independent_label(self):
        doc, image = fixture(rows=(50,))
        image[41:57, 90:101] = 0
        for x in (103,114,125,136):
            image[51:66, x:x+8] = 0
        # Independent name begins just two pixels to the right but uses an
        # upper baseline. Its row projection overlaps the suffix by one row.
        image[27:43,146:157]=0
        image[38:53,159:167]=0
        boxes,_=mono_pin_boxes(doc,image)
        name=self.get(boxes,"pin_name")
        self.assertEqual(name["bbox"],[90,41,144,66])
        self.assertFalse(name["flags"])

    def test_opposite_name_one_pixel_closer_to_row_does_not_win(self):
        doc,image=fixture(rows=(50,),edges=("left","right"))
        image[43:57,90:107]=0
        image[44:58,151:171]=0
        # Left target row 51 prefers the left label, despite the opposite
        # label's center lying exactly on its row.
        doc["pins"]["ic"][0][1]=51
        doc["pin_evidence"]["ic"][0]["point_img"][1]=51
        doc["pin_evidence"]["ic"][0]["anchor_img"][1]=51
        cv2.line(image,(40,51),(80,51),(0,0,0),1)
        boxes,_=mono_pin_boxes(doc,image)
        self.assertEqual(self.get(boxes,"pin_name",0)["bbox"],[90,43,107,57])

    def test_overlapping_letter_envelopes_and_side_sign_are_preserved(self):
        doc,image=fixture(rows=(50,))
        # Separate slanted glyphs whose rectangles overlap by one column.
        cv2.line(image,(90,42),(99,57),(0,0,0),1)
        cv2.line(image,(99,42),(108,57),(0,0,0),1)
        image[50:52,112:118]=0  # a real baseline minus sign
        boxes,_=mono_pin_boxes(doc,image)
        name=self.get(boxes,"pin_name")
        self.assertEqual(name["bbox"],[90,42,118,58])
        self.assertFalse(name["flags"])

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

    def test_complete_digit_adjacent_only_to_measured_outer_fringe_is_not_clipped(self):
        doc,image=fixture()
        image[32:49,58:60]=0
        image[49,40:81]=210  # measured long upper raster fringe
        image[49,58:60]=20   # local raster darkening directly beneath the digit
        before=image.copy()
        boxes,gray=mono_pin_boxes(doc,image)
        number=self.get(boxes,"pin_number")
        self.assertEqual(number["bbox"],[58,32,60,49])
        self.assertFalse(number["flags"])
        e=number["cleanup_evidence"]
        self.assertTrue(e["nearby_measured_horizontal_fringe_only"])
        self.assertEqual(e["black_pixels_removed"],0)
        np.testing.assert_array_equal(image,before)
        np.testing.assert_array_equal(gray[32:49,58:60],image[32:49,58:60,0])

    def test_overhanging_numeral_cap_on_removed_fringe_is_still_held(self):
        doc,image=fixture()
        image[32:49,58:60]=0
        image[49,40:81]=210
        image[49,55:63]=0  # may be a bottom cap; cannot call it only line fringe
        boxes,_=mono_pin_boxes(doc,image)
        number=self.get(boxes,"pin_number")
        self.assertIn("glyph_integrity_unresolved",number["flags"])
        self.assertTrue(number["cleanup_evidence"]["removed_line_touches_retained_glyph"])
        self.assertFalse(number["cleanup_evidence"]["nearby_measured_horizontal_fringe_only"])

    def test_removed_horizontal_fringe_inside_word_bbox_stays_held(self):
        # Two letters/numerals have unequal bottoms; removal crosses the
        # longer observed part, even if the shorter part has a white margin.
        mask=np.zeros((40,40),bool)
        mask[10:23,10:12]=True
        mask[10:25,18:20]=True
        original=mask.copy();original[23,5:35]=True
        known=np.zeros_like(mask);known[23,5:35]=True
        cleaned=mask & ~known
        _,e=_complete_group(cleaned,[5,5,30,28],{"anchor_img":[31,25]},"pin_number",30,
                            known_lines=known,original_mask=original,
                            stroke={"stroke_top":24},line_evidence=[{"axis":"horizontal","row":23,"span":[5,35]}])
        self.assertTrue(e["integrity_unresolved"])
        self.assertFalse(e["nearby_measured_horizontal_fringe_only"])

    def test_detached_digit_cap_is_retained_with_its_complete_body(self):
        doc, image = fixture()
        image[25:28, 55:60] = 0
        image[32:46, 58:60] = 0
        boxes, gray = mono_pin_boxes(doc, image)
        number = self.get(boxes, "pin_number")
        self.assertEqual(number["bbox"], [55, 25, 60, 46])
        self.assertFalse(number["flags"])
        self.assertEqual(number["cleanup_evidence"]["glyph_component_count"], 2)
        np.testing.assert_array_equal(gray[25:28, 55:60], image[25:28, 55:60, 0])

    def test_detached_name_overbar_and_underscore_stay_in_source_crop(self):
        for small_row in (35, 62):
            with self.subTest(small_row=small_row):
                doc, image = fixture(rows=(50,))
                image[43:57, 90:107] = 0
                image[small_row:small_row+1, 91:106] = 0
                # A second independent full text row must not cause the small
                # semantic bar to vanish when the nearest body row is chosen.
                image[65:70, 90:107] = 0
                boxes, gray = mono_pin_boxes(doc, image)
                name = self.get(boxes, "pin_name")
                self.assertLessEqual(name["bbox"][1], small_row)
                self.assertGreater(name["bbox"][3], small_row)
                self.assertEqual(name["cleanup_evidence"]["black_pixels_removed"], 0)
                np.testing.assert_array_equal(gray[small_row:small_row+1, 91:106], image[small_row:small_row+1, 91:106, 0])

    def test_far_detached_small_name_symbol_cannot_be_silently_discarded(self):
        doc, image = fixture(rows=(60,))
        image[53:67, 90:107] = 0
        image[41:42, 91:106] = 0
        image[45:48, 90:107] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        name = self.get(boxes, "pin_name")
        self.assertIn("glyph_integrity_unresolved", name["flags"])
        self.assertTrue(name["cleanup_evidence"]["text_row_selection"]["ambiguous_detached_small_ink_rows"])

    def test_vertical_wire_and_raster_fringe_are_measured_beyond_the_cell(self):
        doc, image = fixture()
        image[31:46, 58:60] = 0
        image[12:80, 28:29] = 0
        image[12:80, 29:30] = 210
        cv2.line(image, (28, 50), (80, 50), (0, 0, 0), 1)
        before = image.copy()
        boxes, gray = mono_pin_boxes(doc, image)
        number = self.get(boxes, "pin_number")
        self.assertEqual(number["bbox"], [58, 31, 60, 46])
        self.assertFalse(number["flags"])
        cleanup = number["cleanup_evidence"]
        self.assertGreater(cleanup["excluded_region_pixel_count"], 0)
        self.assertLess(cleanup["excluded_region_ink_bbox"][0], number["bbox"][0])
        self.assertIn("vertical_raster_fringe", [line["axis"] for line in cleanup["measured_line_evidence"]])
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(gray, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def test_tall_isolated_one_is_not_removed_as_an_unattached_wire(self):
        doc, image = fixture(rows=(50, 60))
        image[15:46, 58:60] = 0
        image[15:17, 55:60] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        number = self.get(boxes, "pin_number")
        self.assertEqual(number["cleanup_evidence"]["black_pixels_removed"], 0)
        self.assertFalse(any(line["axis"] == "vertical" for line in number["cleanup_evidence"]["measured_line_evidence"]))
        self.assertIn("glyph_integrity_unresolved", number["flags"])

    def test_removed_wire_cannot_hide_a_leading_digit_cap_attached_to_it(self):
        doc, image = fixture()
        image[12:80, 28:30] = 0
        cv2.line(image, (28, 50), (80, 50), (0, 0, 0), 1)
        image[31:46, 29:31] = 0
        image[31:33, 28:34] = 0
        image[31:46, 44:49] = 0
        image[31:46, 58:63] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        number = self.get(boxes, "pin_number")
        self.assertIn("glyph_integrity_unresolved", number["flags"])
        self.assertTrue(number["cleanup_evidence"]["removed_line_touches_retained_glyph"])
        self.assertGreater(number["cleanup_evidence"]["excluded_region_pixel_count"], 0)

    def test_small_cap_on_measured_stub_side_is_not_erased_as_a_marker(self):
        doc, image = fixture()
        image[1:110, 70:71] = 0
        image[31:34, 68:74] = 0
        image[31:46, 55:60] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        number = self.get(boxes, "pin_number")
        self.assertIn("glyph_integrity_unresolved", number["flags"])
        self.assertTrue(number["cleanup_evidence"]["removed_line_touches_retained_glyph"])
        # The lateral cap pixels remain observed glyph ink. Sharing a line
        # cannot prove that they are a port marker rather than part of a 1.
        self.assertGreaterEqual(number["bbox"][2], 74)
        self.assertFalse(any(x in (68, 73) and y == 31 for x, y in number["cleanup_evidence"]["known_line_exclusions"]))

    def test_separate_neighbor_word_row_and_far_side_symbols_do_not_pollute_target(self):
        doc, image = fixture(rows=(50,))
        image[32:40, 90:110] = 0
        image[43:57, 90:110] = 0
        # A small symbol on the opposite side shares a raster row but never
        # belongs to the nearest left-side word's integrity audit.
        image[30:31, 160:175] = 0
        image[43:57, 160:175] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        name = self.get(boxes, "pin_name")
        self.assertEqual(name["bbox"], [90, 43, 110, 57])
        self.assertFalse(name["flags"])

    def test_same_row_opposite_names_do_not_merge(self):
        doc, image = fixture(edges=("left", "right"))
        image[43:57, 90:107] = 0
        image[43:57, 155:171] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        self.assertEqual(self.get(boxes, "pin_name", 0)["bbox"], [90, 43, 107, 57])
        self.assertEqual(self.get(boxes, "pin_name", 2)["bbox"], [155, 43, 171, 57])

    def test_detached_letter_foot_bridges_its_real_interletter_gap(self):
        doc, image = fixture(rows=(50,))
        image[43:54, 90:93] = 0
        image[52:54, 95:98] = 0
        image[43:54, 99:105] = 0
        image[43:53, 106:109] = 0
        boxes, gray = mono_pin_boxes(doc, image)
        name = self.get(boxes, "pin_name")
        self.assertEqual(name["bbox"], [90, 43, 109, 54])
        self.assertFalse(name["flags"])
        self.assertEqual(name["cleanup_evidence"]["black_pixels_removed"], 0)
        np.testing.assert_array_equal(gray[43:54, 90:109], image[43:54, 90:109, 0])

    def test_underscore_below_letters_bridges_complete_word_on_both_sides(self):
        for edge in ("left", "right"):
            with self.subTest(edge=edge):
                doc, image = fixture(rows=(50, 71), edges=(edge,))
                image[46:53, 100:114] = 0
                image[54:55, 114:120] = 0  # one white row below letters
                image[46:53, 120:132] = 0
                source = image.copy()
                boxes, gray = mono_pin_boxes(doc, image)
                name = self.get(boxes, "pin_name")
                self.assertEqual(name["bbox"], [100, 46, 132, 55])
                self.assertFalse(name["flags"])
                self.assertEqual(name["cleanup_evidence"]["black_pixels_removed"], 0)
                np.testing.assert_array_equal(gray[46:55, 100:132], source[46:55, 100:132, 0])
                np.testing.assert_array_equal(image, source)

    def test_underscore_does_not_bridge_opposite_names_across_real_blank_gap(self):
        doc, image = fixture(rows=(50, 71), edges=("left", "right"))
        image[46:53, 90:102] = 0
        image[54:55, 102:108] = 0
        image[46:53, 108:116] = 0
        image[46:53, 151:164] = 0
        image[54:55, 164:170] = 0
        image[46:53, 170:176] = 0
        boxes, _ = mono_pin_boxes(doc, image)
        self.assertEqual(self.get(boxes, "pin_name", 0)["bbox"], [90,46,116,55])
        self.assertEqual(self.get(boxes, "pin_name", 2)["bbox"], [151,46,176,55])

    def test_next_word_detached_serif_cannot_hold_the_current_complete_word(self):
        doc, image = fixture(rows=(50,))
        image[45:53, 90:104] = 0
        image[45:53, 105:112] = 0
        image[62:70, 90:104] = 0
        image[62:70, 105:112] = 0
        image[71:72, 90:92] = 0  # observed serif of the other baseline
        boxes, _ = mono_pin_boxes(doc, image)
        name = self.get(boxes, "pin_name")
        self.assertEqual(name["bbox"], [90,45,112,53])
        self.assertFalse(name["flags"])

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
