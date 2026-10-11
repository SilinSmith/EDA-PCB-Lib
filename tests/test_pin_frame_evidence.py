"""Frame evidence must suppress character ink without erasing uncertain pins."""
import json
import unittest

import numpy as np

from stage1.pin_evidence import build_pin_evidence


class PinFrameEvidenceTests(unittest.TestCase):
    box = [40.0, 40.0, 120.0, 120.0]

    def image(self, frame=True):
        ink = np.zeros((180, 180), dtype=np.uint8)
        if frame:
            ink[40, 40:121] = ink[120, 40:121] = 1
            ink[40:121, 40] = ink[40:121, 120] = 1
        return ink

    def evidence(self, ink, points, texts=(), box=None):
        component = dict(id="ic", type="box", bbox_img=box or self.box)
        return build_pin_evidence([component], {"ic": points}, ink, texts)["ic"]

    def test_short_unconnected_stubs_on_all_sides_are_valid(self):
        ink = self.image()
        ink[80, 34:41] = ink[80, 120:127] = 1
        ink[34:41, 80] = ink[120:127, 80] = 1
        points = [[28, 80], [132, 80], [80, 28], [80, 132]]
        evidence = self.evidence(ink, points)
        self.assertEqual([r["edge"] for r in evidence], ["left", "right", "top", "bottom"])
        self.assertEqual([r["trust"] for r in evidence], ["trusted"] * 4)
        self.assertEqual([r["point_img"] for r in evidence], points)

    def test_two_pixel_scan_gap_does_not_reject_real_stub(self):
        ink = self.image()
        ink[80, 123:133] = 1  # two missing pixels immediately after the frame
        record = self.evidence(ink, [[132, 80]])[0]
        self.assertEqual(record["trust"], "trusted")
        self.assertEqual(record["first_ink_distance"], 3)

    def test_two_pixel_gap_stub_inside_oversized_ocr_box_is_uncertain(self):
        ink = self.image()
        ink[123:133, 80] = 1
        record = self.evidence(ink, [[80, 132]], [[52, 123, 103, 132]])[0]
        self.assertEqual(record["trust"], "uncertain")
        self.assertEqual(record["first_ink_distance"], 3)
        self.assertTrue(record["external_word_band"])
        self.assertFalse(record["external_word_ink_confirmed"])
        self.assertEqual(json.loads(json.dumps(record))["trust"], "uncertain")

    def test_oversized_ocr_stub_rule_rotates_with_side(self):
        ink = self.image()
        ink[80, 123:133] = 1
        record = self.evidence(ink, [[132, 80]], [[123, 52, 132, 103]])[0]
        self.assertEqual(record["trust"], "uncertain")
        self.assertEqual(record["edge"], "right")
        self.assertFalse(record["external_word_ink_confirmed"])

    def test_scan_speckles_in_oversized_ocr_box_do_not_prove_word(self):
        ink = self.image()
        ink[123:133, 80] = 1
        ink[130, 58] = ink[127, 98] = 1
        record = self.evidence(ink, [[80, 132]], [[52, 123, 103, 132]])[0]
        self.assertEqual(record["trust"], "uncertain")
        self.assertFalse(record["external_word_ink_confirmed"])

    def test_word_shafts_after_two_pixel_gap_are_not_real_bottom_pins(self):
        ink = self.image()
        # Word-like OCR band: individual I/1/8 shafts can each look straight.
        for x in (57, 65, 73, 81, 89, 97):
            ink[123:133, x:x + 2] = 1
        texts = [[52, 123, 103, 132]]
        evidence = self.evidence(ink, [[57, 132], [73, 132], [89, 132]], texts)
        self.assertTrue(all(r["trust"] == "rejected" for r in evidence))
        self.assertTrue(all(r["reason"] == "detached_word_ink_outside_frame"
                            for r in evidence))
        self.assertTrue(all(r["external_word_ink_confirmed"] for r in evidence))
        self.assertTrue(all(r["trust"] == "rejected"
                            for r in json.loads(json.dumps(evidence))))

    def test_single_i_glyph_is_ambiguous_not_hard_rejected(self):
        ink = self.image()
        ink[123:133, 80] = 1
        record = self.evidence(ink, [[80, 132]], [[78, 123, 82, 132]])[0]
        self.assertEqual(record["trust"], "uncertain")

    def test_word_box_over_real_long_line_does_not_erase_line(self):
        ink = self.image()
        ink[120:160, 80] = 1
        record = self.evidence(ink, [[80, 132]], [[52, 123, 103, 132]])[0]
        self.assertEqual(record["trust"], "trusted")

    def test_short_real_stub_overlapping_text_is_preserved_as_uncertain(self):
        ink = self.image()
        ink[120:128, 80] = 1
        record = self.evidence(ink, [[80, 126]], [[52, 121, 103, 130]])[0]
        self.assertIn(record["trust"], ("trusted", "uncertain"))

    def test_short_stub_beside_long_character_stroke_is_not_rejected(self):
        ink = self.image()
        ink[80, 120:127] = 1
        ink[82, 127:144] = 1
        record = self.evidence(ink, [[132, 80]], [[127, 81, 144, 85]])[0]
        self.assertIn(record["trust"], ("trusted", "uncertain"))

    def test_missing_actual_frame_keeps_candidate_uncertain(self):
        ink = self.image(frame=False)
        ink[80, 120:145] = 1
        record = self.evidence(ink, [[132, 80]])[0]
        self.assertEqual(record["trust"], "uncertain")
        self.assertFalse(record["frame_confirmed"])

    def test_nearby_actual_frame_calibrates_anchor_without_moving_output(self):
        ink = self.image()
        ink[80, 120:145] = 1
        point = [134.14, 80.03]
        record = self.evidence(ink, [point], box=[42, 42, 122, 122])[0]
        self.assertEqual(record["trust"], "trusted")
        self.assertEqual(record["anchor_img"][0], 120.0)
        self.assertEqual(record["point_img"], [134.1, 80.0])

    def test_distant_frame_is_not_used_for_calibration(self):
        ink = self.image()
        record = self.evidence(ink, [[143, 80]], box=[51, 40, 131, 120])[0]
        self.assertFalse(record["frame_confirmed"])
        self.assertEqual(record["trust"], "uncertain")

    def test_final_merged_pin_order_and_non_ic_components_are_preserved(self):
        ink = self.image()
        ink[80, 34:41] = 1
        components = [dict(id="ic", type="bigic", bbox_img=self.box),
                      dict(id="r", type="r", bbox_img=[5, 5, 10, 10])]
        pins = {"ic": [[80.08, 132.02], [28.01, 80.05]], "r": [[5, 0]]}
        result = build_pin_evidence(components, pins, ink)
        self.assertEqual(set(result), {"ic"})
        self.assertEqual([r["point_img"] for r in result["ic"]],
                         [[80.1, 132.0], [28.0, 80.0]])
        self.assertEqual(pins["ic"][0], [80.08, 132.02])

    def test_invalid_mask_is_explicit(self):
        with self.assertRaises(ValueError):
            build_pin_evidence([], {}, np.zeros((20, 20, 3)))


if __name__ == "__main__":
    unittest.main()
