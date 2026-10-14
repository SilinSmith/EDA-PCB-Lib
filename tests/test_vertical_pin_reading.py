import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage2.assign import assign, _same_mono_stroke_duplicate
from stage2.mono_pin_reading import (read_mono_pin_texts, _guard_recovered_names,
                                   _old_number_geometry_reasons)
from stage2.vertical_pin_reading import ClockwisePinView, read_vertical_pin_texts


def vertical_fixture():
    horizontal = np.full((120, 220, 3), 255, np.uint8)
    cv2.rectangle(horizontal, (80, 10), (180, 110), (0, 0, 0), 1)
    cv2.line(horizontal, (180, 50), (215, 50), (0, 0, 0), 1)
    horizontal[42:55, 150:170] = 80
    horizontal[39:46, 187:203] = 40
    image = cv2.rotate(horizontal, cv2.ROTATE_90_COUNTERCLOCKWISE)
    point, anchor = [50, 27], [50, 39]
    evidence = dict(point_img=point[:], anchor_img=anchor, edge="top", trust="trusted",
                    frame_confirmed=True, frame_coverage=.99, normal_ink_run=22,
                    normal_ink_hits=22, first_ink_distance=1, text_overlap_fraction=0.,
                    branching_fraction=.1, reason="frame_attached_normal_stroke")
    doc = dict(components=[dict(id="ic", type="box", bbox_img=[10, 39, 110, 139])],
               pins={"ic": [point]}, pin_evidence={"ic": [evidence]},
               pincol={"ic": [dict(idx=0, cid="ic", x=50, y=27, side="T", axis="x",
                                    in_win=[42, 41, 57, 71], out_win=[42, 10, 57, 37])]})
    return doc, image, horizontal


class Provider:
    def __init__(self, values):
        self.values, self.inputs = iter(values), []

    def read_glyph(self, gray):
        self.inputs.append(gray.copy())
        text, certainty = next(self.values)
        return dict(text=text, confidence=certainty)


class VerticalPinReadingTests(unittest.TestCase):
    def test_pixel_points_and_half_open_rectangles_round_trip_independently(self):
        view = ClockwisePinView(11, 17)
        for point in ([0, 0], [16, 10], [3.5, 7.5]):
            self.assertEqual(view.inverse_point(view.point(point)), point)
        for rect in ([0, 0, 17, 11], [2, 3, 8, 9], [2.5, 3.5, 8.5, 9.5]):
            self.assertEqual(view.inverse_rectangle(view.rectangle(rect)), rect)
        self.assertEqual(view.frame([2, 3, 8, 9]), [1, 2, 7, 8])
        self.assertEqual(view.rectangle([2, 3, 8, 9]), [2, 2, 8, 8])

    def test_rectangle_mapping_selects_exactly_the_same_rotated_source_pixels(self):
        source = np.arange(11*17, dtype=np.uint8).reshape(11, 17)
        view = ClockwisePinView(*source.shape)
        rotated = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
        for rect in ([0, 0, 17, 11], [2, 3, 8, 9], [16, 10, 17, 11]):
            x0, y0, x1, y1 = rect
            a, b, c, d = view.rectangle(rect)
            np.testing.assert_array_equal(rotated[b:d, a:c],
                cv2.rotate(source[y0:y1, x0:x1], cv2.ROTATE_90_CLOCKWISE))

    def test_document_view_preserves_indices_and_rotates_pin_windows_without_mutation(self):
        doc, image, _ = vertical_fixture()
        doc["pin_mono"] = {"ic": {"stroke_rows": [dict(edge="top", tangent_band=[49, 51],
            tangent_center=50., anchor_img=[50, 39], point_img=[50, 27], frame_outer_img=39.)]}}
        before = copy.deepcopy(doc)
        view = ClockwisePinView(*image.shape[:2]);directional = view.document(doc)
        self.assertEqual(doc, before)
        self.assertEqual(directional["pins"]["ic"], [[192, 50]])
        self.assertEqual(directional["pin_evidence"]["ic"][0]["anchor_img"], [180, 50])
        self.assertEqual(directional["pincol"]["ic"][0]["side"], "R")
        self.assertEqual(directional["pincol"]["ic"][0]["axis"], "y")
        self.assertEqual(directional["pincol"]["ic"][0]["idx"], 0)
        self.assertEqual(directional["pin_mono"]["ic"]["stroke_rows"][0]["tangent_band"], [49, 51])

    def test_complete_vertical_views_bind_to_original_index_and_keep_pixels(self):
        doc, image, horizontal = vertical_fixture()
        before = copy.deepcopy(doc);source = image.copy()
        provider = Provider([("ABCD", .99), ("ABCD", .98), ("41", .99), ("41", .97)])
        with patch("stage2.mono_pin_reading.get_pin_name_ocr", return_value=provider):
            records = read_mono_pin_texts(doc, [], None, image)
        self.assertEqual(len(records), 2)
        expected = ([150, 42, 170, 55], [187, 39, 203, 46])
        view = ClockwisePinView(*image.shape[:2])
        for record, box in zip(records, expected):
            self.assertEqual(record["bbox"], view.inverse_rectangle(box))
            self.assertEqual(record["pin_target"]["edge"], "top")
            self.assertEqual(record["pin_target"]["pin_index"], 0)
            self.assertEqual(record["pin_target"]["point_img"], doc["pins"]["ic"][0])
            self.assertEqual(record["mono_pin_ocr"]["rotation"], 90)
        for box, samples in zip(expected, (provider.inputs[:2], provider.inputs[2:])):
            a, b, c, d = box
            for margin, sample in zip((2, 4), samples):
                np.testing.assert_array_equal(sample,
                    np.pad(horizontal[b:d, a:c, 0], margin, constant_values=255))
        bound = assign(doc["components"], doc["pins"], records, pin_evidence=doc["pin_evidence"])
        pin = bound["comps"]["ic"]["pins"][0]
        self.assertEqual(pin["number"]["text"], "41")
        self.assertEqual(pin["name"]["text"], "ABCD")
        self.assertTrue(pin["number_independently_read"])
        self.assertEqual(doc, before);np.testing.assert_array_equal(image, source)

    def test_missing_stale_or_bare_vertical_evidence_cannot_load_provider(self):
        for change in ("missing", "stale", "bare", "hollow"):
            doc, image, _ = vertical_fixture()
            if change == "missing":doc.pop("pin_evidence")
            elif change == "stale":doc["pin_evidence"]["ic"][0]["point_img"][0] += 20
            elif change == "bare":doc["pin_evidence"]["ic"][0].pop("normal_ink_run")
            else:doc["pin_evidence"]["ic"][0]["source"] = "frame_hollow_marker"
            with patch("stage2.mono_pin_reading.get_pin_name_ocr") as loader:
                self.assertEqual(read_mono_pin_texts(doc, [], None, image), [])
            loader.assert_not_called()

    def test_current_attached_line_absence_does_not_open_vertical_crop(self):
        doc, image, horizontal = vertical_fixture()
        horizontal[50, 181:216] = 255
        image = cv2.rotate(horizontal, cv2.ROTATE_90_COUNTERCLOCKWISE)
        with patch("stage2.mono_pin_reading.get_pin_name_ocr") as loader:
            self.assertEqual(read_mono_pin_texts(doc, [], None, image), [])
        loader.assert_not_called()

    def test_directional_view_does_not_release_dirty_number_when_old_name_exists(self):
        doc, image, _ = vertical_fixture()
        view = ClockwisePinView(*image.shape[:2]);d = view.document(doc)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        old = dict(comps={"ic": {"pins": {0: {"name": dict(text="OLD", conf=.99, box=[150,42,170,55]),
                    "number": dict(text="1", conf=.99, box=[177,39,190,53])}}}})
        target = dict(component_id="ic", pin_index=0, edge="right", point_img=[192,50], anchor_img=[180,50])
        record = dict(bbox=[150,42,170,55], name_only=True, pin_target=target,
                      reads={"pin_name": ("ABCD", .99)}, mono_pin_ocr={"accepted": True})
        candidate = dict(bbox=record["bbox"], role="pin_name", pin_target=target,
                         cleanup_evidence={"physical_search_region": [82,35,178,64]})
        with patch("stage2.assign.assign", return_value=old):
            _guard_recovered_names(d, [], [record], [candidate], rotated)
        self.assertFalse(record["reads"])
        self.assertEqual(record["mono_pin_ocr"]["held_reason"], "unresolved_dirty_partner_number_geometry")

    def test_a_newly_bound_dirty_legacy_digit_is_guarded_even_without_a_previous_number(self):
        doc, image, _ = vertical_fixture()
        view = ClockwisePinView(*image.shape[:2]);d = view.document(doc)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        empty = dict(comps={"ic": {"pins": {0: {"name": None, "number": None}}}})
        rebound = dict(comps={"ic": {"pins": {0: {"name": dict(text="ABCD"),
                      "number": dict(text="88", conf=.99, box=[177,39,190,53])}}}})
        target = dict(component_id="ic", pin_index=0, edge="right", point_img=[192,50], anchor_img=[180,50])
        record = dict(bbox=[150,42,170,55], name_only=True, pin_target=target,
                      reads={"pin_name": ("ABCD", .99)}, mono_pin_ocr={"accepted": True})
        with patch("stage2.assign.assign", side_effect=[empty, rebound]):
            _guard_recovered_names(d, [], [record], [], rotated)
        self.assertFalse(record["reads"])
        guard = record["mono_pin_ocr"]["partner_guard"]
        self.assertEqual(guard["winning_number_after_rebinding"]["text"], "88")
        self.assertIn("winning_number_box_crosses_actual_frame", guard["winning_old_number_geometry_reasons"])

    def test_disagreeing_vertical_ocr_views_keep_the_read_held(self):
        doc, image, _ = vertical_fixture()
        provider = Provider([("ABCD", .99), ("ABCE", .99)])
        with patch("stage2.mono_pin_reading.get_pin_name_ocr", return_value=provider):
            records = read_mono_pin_texts(doc, [], None, image, include_numbers=False)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["reads"])
        self.assertEqual(records[0]["mono_pin_ocr"]["held_reason"], "white_margin_views_disagree_or_below_threshold")

    def test_old_digit_cell_guard_checks_visible_ink_and_ignores_white_padding(self):
        doc, image, _ = vertical_fixture()
        view = ClockwisePinView(*image.shape[:2]);d = view.document(doc)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        target = dict(component_id="ic", pin_index=0, edge="right", point_img=[192,50], anchor_img=[180,50])
        number = dict(text="41", conf=.99, box=[185,37,205,48])
        candidate = dict(bbox=[187,39,203,46], region=[185,37,208,48],
                         cleanup_evidence={"physical_search_region": [187,39,203,46]})
        reason = "winning_number_has_source_ink_outside_current_physical_cell"
        self.assertNotIn(reason, _old_number_geometry_reasons(d, number, candidate, target, rotated))
        candidate["cleanup_evidence"]["physical_search_region"] = [187,39,198,46]
        self.assertIn(reason, _old_number_geometry_reasons(d, number, candidate, target, rotated))

    def test_vertical_duplicate_check_requires_measured_orientation_and_separate_row_stays_separate(self):
        meta = dict(rotation=90, cleanup_evidence={"stroke_probe": dict(stroke_top=50,
                    stroke_bottom=52, tangent_axis="x", rotation=90)})
        pins = [[50.5, 27], [53, 27], [80, 27]]
        self.assertTrue(_same_mono_stroke_duplicate(meta, pins, 0, 1, "top"))
        self.assertFalse(_same_mono_stroke_duplicate(meta, pins, 0, 2, "top"))
        self.assertFalse(_same_mono_stroke_duplicate({}, pins, 0, 1, "top"))


if __name__ == "__main__":
    unittest.main()
