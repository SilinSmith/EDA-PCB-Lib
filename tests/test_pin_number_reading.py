import copy
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from stage2.pin_number_reading import read_pin_numbers, _agreement


class FakeProvider:
    def __init__(self, readings):
        self.readings = iter(readings)
        self.inputs = []

    def read_glyph(self, gray):
        self.inputs.append(gray.copy())
        text, confidence = next(self.readings)
        return {"text": text, "confidence": confidence, "sampling": {
            "original_gray_input_shape": list(gray.shape),
            "original_gray_input_sha256": hashlib.sha256(gray.tobytes()).hexdigest()}}


class PinNumberReadingTests(unittest.TestCase):
    def fixture(self, vertical=False):
        bgr = np.full((80, 90, 3), 255, np.uint8)
        # Nonsymmetric source verifies direction and exact original-pixel use.
        cleaned = np.full((80, 90), 255, np.uint8)
        cleaned[26:35, 12:18] = np.arange(54, dtype=np.uint8).reshape(9, 6)
        edge = "bottom" if vertical else "left"
        point, anchor = ([30, 78], [30, 70]) if vertical else ([10, 30], [20, 30])
        target = {"component_id": "ic", "pin_index": 0, "edge": edge,
                  "point_img": point, "anchor_img": anchor}
        candidate = {"bbox": [12, 26, 18, 35], "role": "pin_number",
                     "vertical": vertical, "region": [10, 24, 20, 37],
                     "pin_target": target, "flags": [], "cleanup_evidence": {}}
        doc = {"components": [{"id": "ic", "type": "box", "bbox_img": [20, 10, 70, 70]}],
               "pins": {"ic": [point]}, "pin_evidence": {"ic": [{
                   "edge": edge, "point_img": point, "anchor_img": anchor,
                   "trust": "trusted", "frame_confirmed": True,
                   "source": "red_green_transition"}]}}
        return bgr, cleaned, candidate, doc

    def run_reader(self, values=None, *, vertical=False, reads=None, old=None,
                   verification=None, candidates=None, provider_missing=False, **kwargs):
        bgr, cleaned, candidate, doc = self.fixture(vertical)
        reads = reads or []
        candidates = [candidate] if candidates is None else candidates
        before_reads, before_doc = copy.deepcopy(reads), copy.deepcopy(doc)
        before_candidates = copy.deepcopy(candidates)
        before_bgr, before_cleaned = bgr.copy(), cleaned.copy()
        provider = FakeProvider(values or [("6", .98), ("6", .96)])
        engine = SimpleNamespace(recognizer=Mock(return_value=object()))
        baseline = {"comps": {"ic": {"pins": {0: {"number": old}}}}}
        verification = verification or {"text": "o", "confidence": .3, "sampling": {}}
        with patch("stage2.pin_number_reading.number_repair_boxes", return_value=(candidates, cleaned)), \
             patch("stage2.pin_number_reading.get_pin_name_ocr", return_value=None if provider_missing else provider) as loader, \
             patch("stage2.pin_number_reading.assign", return_value=baseline), \
             patch("stage2.pin_number_reading.read_isolated_many", return_value=[verification]) as verifier:
            result = read_pin_numbers(doc, reads, engine, bgr, **kwargs)
        self.assertEqual(reads, before_reads)
        self.assertEqual(doc, before_doc)
        self.assertEqual(candidates, before_candidates)
        np.testing.assert_array_equal(bgr, before_bgr)
        np.testing.assert_array_equal(cleaned, before_cleaned)
        return result, provider, loader, verifier

    def direction(self, *, stale=False):
        _, _, candidate, _ = self.fixture(vertical=True)
        target = copy.deepcopy(candidate["pin_target"])
        if stale:
            target["point_img"][0] += 1
        return {"bbox": [26, 40, 32, 64], "name_only": True,
                "pin_target": target, "reads": {"pin_name": ("GND", .99)},
                "pin_text_crop": {"rotation": -90, "role": "pin_name"}}

    def test_original_glyph_two_white_margins_and_input_immutability(self):
        result, provider, _, _ = self.run_reader()
        glyph = self.fixture()[1][26:35, 12:18]
        self.assertEqual(len(provider.inputs), 2)
        for padding, seen in zip((2, 4), provider.inputs):
            np.testing.assert_array_equal(seen, np.pad(glyph, padding, constant_values=255))
            self.assertEqual(seen.dtype, np.uint8)
            self.assertNotEqual(seen.shape[0], 32)
        self.assertEqual(result[0]["bbox"], [12, 26, 18, 35])
        self.assertEqual(result[0]["reads"], {"pin_number": ("6", .96)})
        self.assertTrue(result[0]["number_only"])
        self.assertFalse(result[0]["name_only"])
        self.assertEqual(result[0]["source"], "color_pin_number_ppocr")

    def test_disagreement_cannot_choose_the_more_confident_view(self):
        result, _, _, verifier = self.run_reader([("6", .999), ("8", .95)])
        self.assertEqual(result[0]["reads"], {})
        self.assertEqual(result[0]["pin_number_ocr"]["held_reason"], "white_margin_views_disagree_or_below_threshold")
        verifier.assert_not_called()

    def test_two_views_must_each_pass_confidence_and_legal_number_gate(self):
        for values in ([("6", .99), ("6", .89)], [("0", .99), ("0", .99)],
                       [("o", .99), ("o", .99)], [("1000", .99), ("1000", .99)],
                       [("6", float("nan")), ("6", .99)]):
            with self.subTest(values=values):
                result, _, _, _ = self.run_reader(values)
                self.assertEqual(result[0]["reads"], {})

    def test_min_confidence_applies_to_both_views(self):
        result, _, _, _ = self.run_reader([("6", .96), ("6", .94)], min_conf=.95)
        self.assertEqual(result[0]["reads"], {})
        self.assertEqual(result[0]["pin_number_ocr"]["acceptance_threshold"], .95)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                read_pin_numbers({}, [], None, np.zeros((2, 2, 3), np.uint8), min_conf=value)

    def test_strong_clean_crnn_disagreement_preserves_old_observation(self):
        old = {"text": "4", "conf": .99, "box": [12, 26, 18, 35]}
        result, _, _, verifier = self.run_reader([("3", .99), ("3", .97)], old=old,
            verification={"text": "4", "confidence": .994, "sampling": {}})
        self.assertEqual(result[0]["reads"], {})
        self.assertEqual(result[0]["pin_number_ocr"]["held_reason"], "strong_clean_crnn_disagreement")
        self.assertEqual(result[0]["pin_number_ocr"]["previous_number"], old)
        verifier.assert_called_once()

    def test_weak_or_non_number_crnn_does_not_veto_agreed_number(self):
        for verification in ({"text": "8", "confidence": .89}, {"text": "o", "confidence": .99},
                             {"text": "6", "confidence": .99}):
            result, _, _, _ = self.run_reader(verification=dict(verification, sampling={}))
            self.assertEqual(result[0]["reads"]["pin_number"][0], "6")

    def test_confirmation_does_not_rerank_an_unchanged_number(self):
        old = {"text": "6", "conf": .6, "box": [12, 26, 18, 35]}
        result, _, _, verifier = self.run_reader(old=old)
        self.assertEqual(result[0]["reads"], {})
        self.assertEqual(result[0]["pin_number_ocr"]["held_reason"], "unchanged_bound_number")
        verifier.assert_not_called()

    def test_unknown_direction_disagreeing_6_9_is_held(self):
        result, provider, _, _ = self.run_reader(
            [("6", .99), ("6", .98), ("9", .99), ("9", .98)], vertical=True)
        self.assertEqual(len(provider.inputs), 4)
        self.assertEqual(result[0]["reads"], {})
        self.assertEqual(result[0]["pin_number_ocr"]["held_reason"], "unresolved_number_direction")

    def test_only_one_accepted_6_9_direction_cannot_guess_orientation(self):
        for number in ("6", "9"):
            result, _, _, _ = self.run_reader(
                [(number, .99), (number, .99), ("?", .99), ("?", .99)], vertical=True)
            self.assertEqual(result[0]["reads"], {})
            self.assertEqual(result[0]["pin_number_ocr"]["held_reason"], "unresolved_number_direction")

    def test_one_accepted_non_6_9_direction_is_usable(self):
        result, _, _, _ = self.run_reader(
            [("?", .99), ("?", .99), ("2", .99), ("2", .98)], vertical=True)
        self.assertEqual(result[0]["reads"]["pin_number"][0], "2")
        self.assertEqual(result[0]["pin_number_ocr"]["rotation"], -90)

    def test_fresh_name_direction_orients_original_glyph(self):
        result, provider, _, _ = self.run_reader(vertical=True, reads=[self.direction()])
        glyph = np.rot90(self.fixture(True)[1][26:35, 12:18], -1)
        self.assertEqual(len(provider.inputs), 2)
        np.testing.assert_array_equal(provider.inputs[0], np.pad(glyph, 2, constant_values=255))
        self.assertEqual(result[0]["pin_number_ocr"]["rotation"], -90)

    def test_stale_name_direction_cannot_resolve_6_9(self):
        result, provider, _, _ = self.run_reader(
            [("6", .99), ("6", .98), ("9", .99), ("9", .98)],
            vertical=True, reads=[self.direction(stale=True)])
        self.assertEqual(len(provider.inputs), 4)
        self.assertEqual(result[0]["reads"], {})

    def test_missing_optional_provider_keeps_existing_path(self):
        result, provider, loader, verifier = self.run_reader(provider_missing=True)
        self.assertEqual(result, [])
        loader.assert_called_once_with("auto")
        self.assertEqual(provider.inputs, [])
        verifier.assert_not_called()

    def test_unresolved_or_absent_candidates_do_not_load_provider(self):
        candidate = self.fixture()[2]
        for candidates in ([], [dict(candidate, flags=["glyph_integrity_unresolved"])]):
            result, provider, loader, verifier = self.run_reader(candidates=candidates)
            self.assertEqual(result, [])
            loader.assert_not_called()
            self.assertEqual(provider.inputs, [])
            verifier.assert_not_called()

    def test_invalid_image_does_not_load_provider(self):
        with patch("stage2.pin_number_reading.get_pin_name_ocr", side_effect=AssertionError("must not load")):
            for image in (None, np.zeros((2, 2), np.uint8)):
                self.assertEqual(read_pin_numbers({}, [], None, image), [])

    def test_agreement_requires_exact_predefined_views(self):
        view = {"text": "6", "confidence": .99}
        for variants in ([], [dict(view, padding=2)], [dict(view, padding=4), dict(view, padding=2)],
                         [dict(view, padding=2), dict(view, padding=8)]):
            self.assertIsNone(_agreement(variants, .90))


if __name__ == "__main__":
    unittest.main()
