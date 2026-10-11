import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from stage2.mono_pin_reading import read_mono_pin_texts, _guard_recovered_names


class Provider:
    def __init__(self, values):
        self.values, self.inputs = iter(values), []

    def read_glyph(self, gray):
        self.inputs.append(gray.copy())
        text, conf = next(self.values)
        return {"text": text, "confidence": conf, "sampling": {
            "original_gray_input_shape": list(gray.shape),
            "original_gray_input_sha256": hashlib.sha256(gray.tobytes()).hexdigest()}}


class MonoPinReadingTests(unittest.TestCase):
    def run_reader(self, values=None, role="pin_number", flags=None, provider_missing=False, exclusions=None, **kwargs):
        gray = np.full((80, 90), 255, np.uint8)
        gray[26:35, 12:18] = np.arange(54, dtype=np.uint8).reshape(9, 6)
        bgr = np.repeat(gray[:, :, None], 3, axis=2)
        target = {"component_id": "ic", "pin_index": 0, "edge": "left",
                  "point_img": [10, 30], "anchor_img": [20, 30]}
        candidate = {"bbox": [12, 26, 18, 35], "role": role, "vertical": False,
                     "region": [10, 24, 20, 37], "pin_target": target,
                     "flags": flags or [], "cleanup_evidence": {"black_pixels_removed": 0}}
        if exclusions is not None:
            candidate["cleanup_evidence"]["known_line_exclusions"] = exclusions
        reads, doc = [{"bbox": [12, 26, 18, 35], "reads": {"pin_number": ("13", .999)}}], {"pins": {"ic": [[10, 30]]}}
        old_reads, old_doc, old_candidate = copy.deepcopy(reads), copy.deepcopy(doc), copy.deepcopy(candidate)
        provider = Provider(values or [("113", .98), ("113", .96)])
        with patch("stage2.mono_pin_reading.mono_pin_boxes", return_value=([candidate], gray)), \
             patch("stage2.mono_pin_reading.get_pin_name_ocr", return_value=None if provider_missing else provider) as loader:
            out = read_mono_pin_texts(doc, reads, None, bgr, **kwargs)
        self.assertEqual(reads, old_reads)
        self.assertEqual(doc, old_doc)
        self.assertEqual(candidate, old_candidate)
        return out, provider, loader, gray

    def test_two_fixed_white_margin_views_use_complete_original_pixels(self):
        out, provider, loader, gray = self.run_reader()
        for padding, sample in zip((2, 4), provider.inputs):
            np.testing.assert_array_equal(sample, np.pad(gray[26:35, 12:18], padding, constant_values=255))
        self.assertEqual(out[0]["reads"], {"pin_number": ("113", .96)})
        self.assertTrue(out[0]["number_only"])
        self.assertFalse(out[0]["name_only"])
        self.assertEqual(out[0]["source"], "mono_pin_number_ppocr")
        self.assertEqual(out[0]["geometry_bbox"], [12, 26, 18, 35])
        self.assertEqual(len(out[0]["mono_pin_ocr"]["raw_reads"]), 2)
        loader.assert_called_once_with("auto")

    def test_name_output_is_role_limited_and_no_dictionary_repair_occurs(self):
        out, _, _, _ = self.run_reader([("VBAT", .99), ("VBAT", .98)], role="pin_name")
        self.assertEqual(out[0]["reads"], {"pin_name": ("VBAT", .98)})
        self.assertTrue(out[0]["name_only"])
        self.assertFalse(out[0]["number_only"])
        self.assertEqual(out[0]["source"], "mono_pin_name_ppocr")
        out, _, _, _ = self.run_reader([("VBA1", .99), ("VBAT", .99)], role="pin_name")
        self.assertEqual(out[0]["reads"], {})

    def test_only_recorded_measured_line_pixels_are_whitened_in_recognition_copy(self):
        exclusions = [[12, 26], [13, 27], [1, 1], [90, 80]]
        _, provider, _, gray = self.run_reader(exclusions=exclusions)
        expected = gray[26:35, 12:18].copy()
        expected[0, 0] = 255
        expected[1, 1] = 255
        for padding, sample in zip((2, 4), provider.inputs):
            np.testing.assert_array_equal(sample, np.pad(expected, padding, constant_values=255))
        # The shared grayscale/source and untouched numeral pixels keep their
        # exact values; cleanup is confined to the independent provider copy.
        self.assertEqual(int(gray[26, 12]), 0)
        self.assertEqual(int(gray[27, 13]), 7)

    def test_both_agreed_views_must_pass_fixed_legal_and_confidence_gate(self):
        for values in ([("113", .999), ("13", .99)], [("113", .99), ("113", .89)],
                       [("0", .99), ("0", .99)], [("1000", .99), ("1000", .99)],
                       [("1o", .99), ("1o", .99)], [("113", float("nan")), ("113", .99)]):
            with self.subTest(values=values):
                out, _, _, _ = self.run_reader(values)
                self.assertEqual(out[0]["reads"], {})
                self.assertFalse(out[0]["mono_pin_ocr"]["accepted"])

    def test_incomplete_crop_is_recorded_without_ocr(self):
        out, provider, _, _ = self.run_reader(flags=["glyph_integrity_unresolved"])
        self.assertFalse(provider.inputs)
        self.assertEqual(out[0]["reads"], {})
        self.assertEqual(out[0]["mono_pin_ocr"]["held_reason"], "glyph_integrity_unresolved")

    def test_off_and_disabled_roles_perform_zero_optional_runtime_loads(self):
        for options in ({"mode": "off"}, {"include_names": False, "include_numbers": False},
                        {"include_numbers": False}):
            with self.subTest(options=options):
                out, provider, loader, _ = self.run_reader(**options)
                self.assertEqual(out, [])
                loader.assert_not_called()
                self.assertFalse(provider.inputs)
        out, _, loader, _ = self.run_reader(role="pin_name", include_names=False)
        self.assertEqual(out, [])
        loader.assert_not_called()

    def test_missing_auto_provider_returns_empty_and_on_is_passed_explicitly(self):
        self.assertEqual(self.run_reader(provider_missing=True)[0], [])
        _, _, loader, _ = self.run_reader(mode="on")
        loader.assert_called_once_with("on")
        with patch("stage2.mono_pin_reading.mono_pin_boxes", return_value=([{"role": "pin_name"}], None)), \
             patch("stage2.mono_pin_reading.get_pin_name_ocr", side_effect=RuntimeError("missing")):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                read_mono_pin_texts({}, [], None, np.zeros((3, 3, 3), np.uint8), mode="on")

    def test_min_confidence_applies_to_each_view_and_bad_inputs_fail(self):
        out, _, _, _ = self.run_reader([("113", .96), ("113", .94)], min_conf=.95)
        self.assertEqual(out[0]["reads"], {})
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                read_mono_pin_texts({}, [], None, None, min_conf=value)
        with self.assertRaises(ValueError):
            read_mono_pin_texts({}, [], None, None, mode="typo")

    def partner(self, old_box, *, previous_name=None, confirmed=False, candidates=True,
                initial_reads=None, flags=None, rebound_number=None, rebound_marker=True):
        image = np.full((330, 350, 3), 255, np.uint8)
        image[163:177, 79:96] = 0
        image[180, 60:101] = 0
        target = {"component_id": "ic", "pin_index": 0, "edge": "left",
                  "point_img": [90, 180], "anchor_img": [100, 180]}
        evidence = []
        for row in (180, 220):
            evidence.append({"point_img": [90, row], "anchor_img": [100, row],
                             "edge": "left", "trust": "trusted",
                             "reason": "frame_attached_normal_stroke", "frame_confirmed": True,
                             "frame_coverage": 1., "normal_ink_run": 22, "normal_ink_hits": 22,
                             "first_ink_distance": 1, "text_overlap_fraction": 0., "branching_fraction": .1})
        doc = {"components": [{"id": "ic", "type": "box", "bbox_img": [100, 100, 300, 300]}],
               "pins": {"ic": [[90, 180], [90, 220]]}, "pin_evidence": {"ic": evidence}}
        name = {"name_only": True, "number_only": False, "pin_target": target,
                "reads": {"pin_name": ("VBAT", .99)}, "mono_pin_ocr": {"accepted": True}}
        output = [name]
        if confirmed:
            output.append({"number_only": True, "pin_target": target,
                           "bbox": [79, 163, 96, 177], "reads": {"pin_number": ("113", .99)},
                           "mono_pin_ocr": {"accepted": True}})
        old_number = {"text": "13", "conf": .99, "box": old_box}
        original = {"comps": {"ic": {"pins": {0: {"name": previous_name, "number": old_number}}}}}
        digit = {"role": "pin_number", "pin_target": target,
                 "bbox": [79, 163, 96, 177], "region": [65, 150, 98, 179],
                 "flags": ["glyph_integrity_unresolved"] if flags is None else flags,
                 "cleanup_evidence": {"stroke_probe": {"stroke_top": 180, "stroke_bottom": 181}}}
        initial_reads = initial_reads or []
        snapshot = copy.deepcopy((doc, original, digit, initial_reads))
        before_pixels = image.copy()
        rebound = {"comps": {"ic": {"pins": {0: {
            "number_independently_read": rebound_marker,
            "number": rebound_number or {"text": "113", "box": [79, 163, 96, 177]}}}}}}
        with patch("stage2.assign.assign", side_effect=[original, rebound] if confirmed else [original]):
            _guard_recovered_names(doc, initial_reads, output, [digit] if candidates else [], image)
        self.assertEqual((doc, original, digit, initial_reads), snapshot)
        np.testing.assert_array_equal(image, before_pixels)
        return name

    def test_recovered_name_cannot_activate_observed_bad_winning_number_geometry(self):
        cases = [([79, 164, 108, 178], "winning_number_box_crosses_actual_frame"),
                 ([77, 164, 98, 230], "winning_number_box_contains_multiple_physical_rows"),
                 ([79, 164, 96, 182], "attached_pin_stroke_enters_winning_number_box"),
                 ([86, 163, 96, 177], "winning_number_box_cuts_observed_connected_glyph")]
        for box, reason in cases:
            with self.subTest(reason=reason):
                name = self.partner(box)
                self.assertEqual(name["reads"], {})
                meta = name["mono_pin_ocr"]
                self.assertFalse(meta["accepted"])
                self.assertEqual(meta["held_reason"], "unresolved_dirty_partner_number_geometry")
                self.assertIn(reason, meta["partner_guard"]["winning_old_number_geometry_reasons"])

    def test_low_confidence_disagreement_or_held_new_crop_does_not_make_clean_old_digit_dirty(self):
        for candidate, flags in ((True, ["glyph_integrity_unresolved"]), (True, []), (False, [])):
            with self.subTest(candidate=candidate, flags=flags):
                name = self.partner([77, 160, 98, 179], candidates=candidate, flags=flags)
                self.assertEqual(name["reads"], {"pin_name": ("VBAT", .99)})
                self.assertTrue(name["mono_pin_ocr"]["accepted"])
                self.assertEqual(name["mono_pin_ocr"]["partner_guard"]["winning_old_number_geometry_reasons"], [])

    def test_confirmed_complete_new_number_allows_recovered_name_despite_old_bad_geometry(self):
        name = self.partner([79, 164, 108, 182], confirmed=True)
        self.assertEqual(name["reads"], {"pin_name": ("VBAT", .99)})
        self.assertTrue(name["mono_pin_ocr"]["partner_guard"]["new_number_confirmed"])

    def test_rebound_number_must_have_marker_and_match_observed_text_and_box(self):
        for actual, marker in (({"text": "114", "box": [79, 163, 96, 177]}, True),
                               ({"text": "113", "box": [79, 163, 95, 177]}, True),
                               ({"text": "113", "box": [79, 163, 96, 177]}, False)):
            with self.subTest(actual=actual, marker=marker):
                name = self.partner([79, 164, 108, 182], confirmed=True,
                                    rebound_number=actual, rebound_marker=marker)
                self.assertEqual(name["reads"], {})
                self.assertFalse(name["mono_pin_ocr"]["partner_guard"]["new_number_confirmed"])

    def test_previous_name_is_preserved_for_number_winner_evidence_and_guard_is_missing_name_only(self):
        previous = {"text": "VBA1", "conf": .94, "box": [111, 174, 151, 188]}
        name = self.partner([79, 164, 108, 182], previous_name=previous)
        self.assertEqual(name["mono_pin_ocr"]["previous_name"], previous)
        self.assertEqual(name["reads"], {"pin_name": ("VBAT", .99)})
        self.assertNotIn("partner_guard", name["mono_pin_ocr"])

    def test_discarded_dirty_old_read_does_not_poison_selected_clean_old_number(self):
        discarded = {"bbox": [70, 150, 110, 240], "reads": {"pin_number": ("25", .95)}}
        name = self.partner([77, 160, 98, 179], initial_reads=[discarded])
        self.assertEqual(name["reads"], {"pin_name": ("VBAT", .99)})
        self.assertEqual(name["mono_pin_ocr"]["previous_number"]["box"], [77, 160, 98, 179])

    def test_accepted_digit_rejected_by_actual_assignment_cannot_release_dirty_partner(self):
        from stage2.assign import assign
        image = np.full((330, 350, 3), 255, np.uint8)
        image[163:177, 79:96] = 0
        image[180, 60:101] = 0
        evidence = {"point_img": [90, 180], "anchor_img": [100, 180], "edge": "left",
                    "trust": "trusted", "reason": "frame_attached_normal_stroke",
                    "frame_confirmed": True, "frame_coverage": 1., "normal_ink_run": 22,
                    "normal_ink_hits": 22, "first_ink_distance": 1,
                    "text_overlap_fraction": 0., "branching_fraction": .1}
        doc = {"components": [{"id": "ic", "type": "box", "bbox_img": [100, 100, 300, 300]}],
               "pins": {"ic": [[90, 180]]}, "pin_evidence": {"ic": [evidence]}}
        target = {"component_id": "ic", "pin_index": 0, "edge": "left",
                  "point_img": [90, 180], "anchor_img": [100, 180]}
        def observation(role, text, box):
            return {"bbox": box, "geometry_bbox": box, "pin_target": target,
                    "name_only": role == "pin_name", "number_only": role == "pin_number",
                    "source": "mono_" + role + "_ppocr", "reads": {role: (text, .99)},
                    "mono_pin_ocr": {"backend": "ppocr_v4_direct", "accepted": True, "flags": [],
                                     "selected_views": [{"text": text, "confidence": .99, "padding": p}
                                                        for p in (2, 4)]}}
        old = {"bbox": [79, 167, 96, 182], "number_only": True, "reads": {"pin_number": ("13", .99)}}
        name = observation("pin_name", "VBAT", [111, 174, 151, 188])
        # It claims the correct target, but its actual box is far above the
        # physical row. The real assignment rejects the observation.
        rejected_digit = observation("pin_number", "113", [79, 105, 96, 117])
        rebound = assign(doc["components"], doc["pins"], [old, name, rejected_digit],
                         pin_evidence=doc["pin_evidence"])["comps"]["ic"]["pins"][0]
        self.assertEqual(rebound["number"]["text"], "13")
        self.assertIsNot(rebound.get("number_independently_read"), True)
        digit_crop = {"role": "pin_number", "bbox": [79, 163, 96, 177],
                      "region": [65, 150, 98, 179], "pin_target": target,
                      "cleanup_evidence": {"stroke_probe": {"stroke_top": 180, "stroke_bottom": 181}}}
        _guard_recovered_names(doc, [old], [name, rejected_digit], [digit_crop], image)
        self.assertEqual(name["reads"], {})
        self.assertFalse(name["mono_pin_ocr"]["partner_guard"]["new_number_confirmed"])
        self.assertEqual(name["mono_pin_ocr"]["held_reason"], "unresolved_dirty_partner_number_geometry")

    def test_name_below_pipeline_min_confidence_cannot_bypass_dirty_partner_guard(self):
        image = np.full((330, 350, 3), 255, np.uint8)
        image[163:177, 79:96] = 0
        image[180, 60:101] = 0
        image[174:188, 111:151] = 0
        evidence = {"point_img": [90, 180], "anchor_img": [100, 180], "edge": "left",
                    "trust": "trusted", "reason": "frame_attached_normal_stroke",
                    "frame_confirmed": True, "frame_coverage": 1., "normal_ink_run": 22,
                    "normal_ink_hits": 22, "first_ink_distance": 1,
                    "text_overlap_fraction": 0., "branching_fraction": .1}
        doc = {"components": [{"id": "ic", "type": "box", "bbox_img": [100, 100, 300, 300]}],
               "pins": {"ic": [[90, 180]]}, "pin_evidence": {"ic": [evidence]}}
        target = {"component_id": "ic", "pin_index": 0, "edge": "left",
                  "point_img": [90, 180], "anchor_img": [100, 180]}
        old = [{"bbox": [79, 167, 96, 182], "number_only": True, "reads": {"pin_number": ("13", .95)}},
               {"bbox": [111, 174, 151, 188], "name_only": True, "reads": {"pin_name": ("VBA1", .80)}}]
        original = copy.deepcopy(old)
        candidates = [{"role": "pin_name", "bbox": [111, 174, 151, 188], "region": [102, 170, 298, 190],
                       "pin_target": target, "flags": [], "cleanup_evidence": {}},
                      {"role": "pin_number", "bbox": [79, 163, 96, 177], "region": [65, 150, 98, 179],
                       "pin_target": target, "flags": ["glyph_integrity_unresolved"],
                       "cleanup_evidence": {"stroke_probe": {"stroke_top": 180, "stroke_bottom": 181}}}]
        provider = Provider([("VBAT", .96), ("VBAT", .96)])
        with patch("stage2.mono_pin_reading.mono_pin_boxes", return_value=(candidates, image[:, :, 0])), \
             patch("stage2.mono_pin_reading.get_pin_name_ocr", return_value=provider):
            output = read_mono_pin_texts(doc, old, None, image, min_conf=.90)
        self.assertEqual(output[0]["reads"], {})
        self.assertIsNone(output[0]["mono_pin_ocr"]["previous_name"])
        self.assertEqual(output[0]["mono_pin_ocr"]["held_reason"], "unresolved_dirty_partner_number_geometry")
        self.assertEqual(old, original)


if __name__ == "__main__":
    unittest.main()
