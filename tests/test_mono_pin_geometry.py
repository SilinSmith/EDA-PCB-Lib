"""Black pin rows require real attached stroke evidence, never a pitch prior."""
import copy
import json
import unittest

import numpy as np

from stage1.pin_evidence import build_pin_evidence, recover_frame_pins


class MonoPinGeometryTests(unittest.TestCase):
    box = [40., 40., 120., 120.]

    def frame(self):
        image = np.full((180, 180, 3), 255, dtype=np.uint8)
        image[40, 40:121] = image[120, 40:121] = 0
        image[40:121, 40] = image[40:121, 120] = 0
        return image

    def recover(self, image, points=(), *, box=None, **kwargs):
        components = [dict(id="ic", type="box", bbox_img=box or self.box)]
        return recover_frame_pins(components, {"ic": list(points)}, image, **kwargs)

    def test_real_short_floating_stroke_on_each_side_needs_no_wire(self):
        image = self.frame()
        image[80, 34:41] = image[80, 120:127] = 0
        image[34:41, 80] = image[120:127, 80] = 0
        pins, mapping, audit = self.recover(image)
        self.assertEqual(set(map(tuple, pins["ic"])),
                         {(28., 80.), (132., 80.), (80., 28.), (80., 132.)})
        self.assertEqual(mapping, {"ic": {}})
        for record in audit["ic"]["added"]:
            self.assertTrue(record["estimated_endpoint"])
            self.assertEqual(record["continuous_outward_pixels"], 6)
            self.assertEqual(record["evidence"]["trust"], "trusted")
            self.assertEqual(record["evidence"]["anchor_img"], record["anchor_img"])
        self.assertEqual(json.loads(json.dumps(audit))["ic"]["new_count"], 4)

    def test_actual_blank_rows_split_nonuniform_pin_rows_and_keep_old_false_point(self):
        image = self.frame()
        for row in (61, 82, 111):
            image[row, 120:146] = 0
        # Parallel ink outside the side cannot bridge actual separate strokes.
        image[59:114, 145] = 0
        old = [[132., 96.]]
        pins, mapping, audit = self.recover(image, old)
        self.assertEqual(pins["ic"][0], old[0])
        self.assertEqual(mapping, {"ic": {0: 0}})
        self.assertEqual([p[1] for p in pins["ic"][1:]], [61., 82., 111.])
        self.assertFalse(audit["ic"]["calibrated"])
        ink = np.any(image < 170, axis=2)
        evidence = build_pin_evidence([dict(id="ic", type="box", bbox_img=self.box)],
                                      pins, ink)["ic"]
        self.assertEqual(evidence[0]["trust"], "uncertain")

    def test_complete_but_detached_strokes_and_title_glyphs_do_not_create_pins(self):
        image = self.frame()
        image[80, 123:139] = 0  # two missing pixels after the right frame
        for x in (55, 64, 73, 82, 91, 100):
            image[25:38, x:x + 2] = 0  # a title with a real gap above the frame
        pins, _, audit = self.recover(image, [[132., 80.]],
                                      texts=[[52, 25, 104, 37]])
        self.assertEqual(pins["ic"], [[132., 80.]])
        self.assertFalse(audit["ic"]["added"])
        self.assertFalse(audit["ic"]["calibrated"])

    def test_missing_frame_background_and_outdated_box_do_not_invent_rows(self):
        blank = np.full((180, 180, 3), 255, dtype=np.uint8)
        blank[80, 120:145] = 0
        self.assertEqual(self.recover(blank)[0]["ic"], [])
        image = self.frame()
        image[80, 120:145] = 0
        self.assertEqual(self.recover(image, box=[51, 40, 131, 120])[0]["ic"], [])
        self.assertEqual(self.recover(image, box=[-1, 40, 120, 120])[0]["ic"], [])

    def test_existing_true_candidate_calibrates_only_inside_its_measured_row(self):
        image = self.frame()
        image[79:82, 120:145] = 0
        points = [[133., 81.5], [132., 95.]]
        before = copy.deepcopy(points)
        pixels_before = image.copy()
        pins, mapping, audit = self.recover(image, points)
        self.assertEqual(pins["ic"], [[132., 80.], [132., 95.]])
        self.assertEqual(mapping, {"ic": {0: 0, 1: 1}})
        self.assertEqual(audit["ic"]["calibrated"][0]["previous_point_img"], points[0])
        self.assertEqual(points, before)
        np.testing.assert_array_equal(image, pixels_before)

    def test_two_old_candidates_on_one_row_keep_both_and_do_not_duplicate(self):
        image = self.frame()
        image[80, 120:145] = 0
        points = [[132., 79.], [132., 81.]]
        pins, _, audit = self.recover(image, points)
        self.assertEqual(pins["ic"], points)
        self.assertEqual(audit["ic"]["held"][0]["reason"],
                         "multiple_old_candidates_on_stroke")

    def test_text_noise_cannot_borrow_neighboring_real_stroke_for_recovery(self):
        image = self.frame()
        image[70, 120:148] = 0
        # Glyph top bar touches the frame, but its shaft is not a pin. A +/-3
        # anchor search would wrongly borrow the adjacent long real row.
        image[73, 120:128] = 0
        image[73:84, 126:128] = 0
        pins, _, audit = self.recover(image, [[132., 70.]],
                                      texts=[[121, 73, 128, 84]])
        self.assertEqual(pins["ic"], [[132., 70.]])
        self.assertTrue(any(record["reason"] == "recovery_stroke_not_trusted"
                            for record in audit["ic"]["held"]))

    def test_uncertain_old_candidate_is_not_promoted_by_nearby_better_stroke(self):
        image = self.frame()
        image[80, 120:127] = 0
        image[83:89, 120:141] = 0
        texts = [[121, 83, 141, 89]]
        pins, _, audit = self.recover(image, [[132., 83.]], texts=texts)
        self.assertEqual(pins["ic"], [[132., 83.]])
        self.assertTrue(any(r["reason"] == "old_candidate_not_trusted"
                            for r in audit["ic"]["held"]))

    def test_thick_frame_and_thick_stroke_measure_actual_outer_boundary(self):
        image = self.frame()
        image[40:121, 119:123] = 0
        image[79:83, 120:132] = 0
        pins, _, audit = self.recover(image)
        self.assertEqual(pins["ic"], [[132.5, 80.5]])
        record = audit["ic"]["added"][0]
        self.assertEqual(record["frame_outer_img"], 122.)
        self.assertEqual(record["stroke_width"], 4)

    def test_explicit_color_sides_are_skipped_and_color_ink_cannot_create_pins(self):
        image = self.frame()
        image[80, 120:145] = 0
        image[80, 25:41] = [0, 200, 0]
        pins, _, audit = self.recover(image, skip_sides={"ic": {"right"}},
                                      ink=np.any(image < 170, axis=2))
        self.assertEqual(pins["ic"], [])
        self.assertEqual(audit["ic"]["skipped_sides"], ["right"])

    def test_existing_colored_contact_is_preserved_even_without_color_ledger(self):
        image = self.frame()
        image[80, 120:145] = 0
        image[79:82, 132:135] = [0, 0, 200]
        pins, _, audit = self.recover(image, [[134., 80.]])
        self.assertEqual(pins["ic"], [[134., 80.]])
        self.assertEqual(audit["ic"]["held"][0]["reason"], "protected_existing_candidate")

    def test_mask_cannot_make_bright_gray_or_colored_pixels_into_strokes(self):
        image = self.frame()
        image[80, 120:145] = 210
        self.assertEqual(self.recover(image, ink=np.ones((180, 180)))[0]["ic"], [])
        image[80, 120:145] = [0, 0, 150]
        self.assertEqual(self.recover(image, ink=np.ones((180, 180)))[0]["ic"], [])

    def test_input_shape_and_out_distance_are_validated(self):
        for kwargs in (dict(ink=np.ones((180, 181))), dict(out_dist=float("nan")),
                       dict(out_dist=0)):
            with self.assertRaises(ValueError):
                self.recover(self.frame(), **kwargs)
        with self.assertRaises(ValueError):
            self.recover(np.zeros((180, 180)))

    def test_non_ic_pins_and_identity_indices_are_untouched(self):
        components = [dict(id="r", type="r", bbox_img=self.box)]
        old = {"r": [[20., 80.], [140., 80.]]}
        pins, mapping, audit = recover_frame_pins(components, old, self.frame())
        self.assertEqual(pins, old)
        self.assertIsNot(pins["r"], old["r"])
        self.assertEqual(mapping, {"r": {0: 0, 1: 1}})
        self.assertEqual(audit, {})

    def test_substring_types_and_invalid_old_points_are_not_reinterpreted(self):
        image = self.frame()
        image[80, 120:145] = 0
        components = [dict(id="unknown", type="textbox", bbox_img=self.box),
                      dict(id="ic", type="box", bbox_img=self.box)]
        old = {"unknown": [], "ic": [None, [float("nan"), 20]]}
        pins, mapping, audit = recover_frame_pins(components, old, image)
        self.assertEqual(pins["unknown"], [])
        self.assertNotIn("unknown", audit)
        self.assertIsNone(pins["ic"][0])
        self.assertEqual(pins["ic"][2], [132., 80.])
        self.assertEqual(mapping["ic"], {0: 0, 1: 1})

    def test_shape_only_closed_rectangle_does_not_gain_semantic_pin_claims(self):
        image = self.frame()
        image[80, 120:145] = 0
        component = dict(id="shape", type="box", bbox_img=self.box, source="cv:bigic")
        original = {"shape": [[132., 81.], [80., 132.]]}
        pins, mapping, audit = recover_frame_pins([component], original, image)
        self.assertEqual(pins, original)
        self.assertEqual(mapping, {"shape": {0: 0, 1: 1}})
        self.assertEqual(audit["shape"]["held"][0]["reason"],
                         "shape_only_component_without_semantic_confirmation")
        self.assertFalse(audit["shape"]["added"])
        self.assertFalse(audit["shape"]["calibrated"])


if __name__ == "__main__":
    unittest.main()
