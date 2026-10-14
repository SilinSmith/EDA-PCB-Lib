"""Complete names may retain source fringes; unrelated ink stays protected."""
import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np

from stage2.name_fringe import antialiased_name_candidate, qualified_fringe_view
from stage2.mono_pin_reading import read_mono_pin_texts


class Provider:
    def __init__(self, values):
        self.values, self.inputs = iter(values), []

    def read_glyph(self, pixels):
        self.inputs.append(pixels.copy())
        text, confidence = next(self.values)
        return dict(text=text, confidence=confidence)


class NameFringeTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((70, 100), 255, np.uint8)
        self.gray[25:34, 30:51] = 60
        self.gray[25:34, 29] = 210
        self.gray[34, 30:51] = 220
        self.image = np.repeat(self.gray[:, :, None], 3, 2)
        self.candidate = dict(bbox=[30,25,51,34], region=[25,20,65,40], role='pin_name', flags=[],
            pin_target=dict(component_id='ic', pin_index=0, edge='left', point_img=[10,30], anchor_img=[20,30]),
            cleanup_evidence=dict(physical_search_region=[22,20,75,40], integrity_unresolved=False))

    def test_pale_source_edge_is_retained_without_altering_any_pixel_or_target(self):
        before = copy.deepcopy(self.candidate)
        pixels = self.image.copy()
        result = antialiased_name_candidate(self.candidate, self.gray, self.image)
        self.assertEqual(result['bbox'], [29,25,51,35])
        self.assertEqual(result['pin_target'], self.candidate['pin_target'])
        self.assertEqual(self.candidate, before)
        np.testing.assert_array_equal(self.image, pixels)
        evidence = result['cleanup_evidence']['name_fringe']
        self.assertGreater(evidence['added_pale_pixels'], 0)
        meta = dict(role='pin_name', view=result['recognition_view'], cleanup_evidence=result['cleanup_evidence'])
        self.assertTrue(qualified_fringe_view(meta, result['bbox']))
        self.assertFalse(qualified_fringe_view(meta, [28,25,51,35]))

    def test_new_dark_body_unresolved_crop_or_ownership_edge_cannot_retry(self):
        for change in ('dark','flags','boundary','cleanup'):
            with self.subTest(change=change):
                self.setUp()
                if change == 'dark': self.image[25,29] = 0; self.gray[25,29] = 0
                elif change == 'flags': self.candidate['flags']=['glyph_integrity_unresolved']
                elif change == 'boundary': self.candidate['cleanup_evidence']['physical_search_region'][0]=29
                else: self.candidate['cleanup_evidence']['known_line_exclusions']=[[30,25]]
                self.assertIsNone(antialiased_name_candidate(self.candidate, self.gray, self.image))

    def read(self, values):
        provider = Provider(values)
        with patch('stage2.mono_pin_reading.mono_pin_boxes', return_value=([self.candidate], self.gray)), \
             patch('stage2.mono_pin_reading.get_pin_name_ocr', return_value=provider):
            result = read_mono_pin_texts({}, [], None, self.image, mode='on')
        return result[0], provider

    def test_original_complete_fringe_replaces_held_symbolic_name_from_two_agreeing_views(self):
        result, provider = self.read([('IWAKE',.88),('/WAKE',.97),('/WAKE',.98),('/WAKE',.88),('/WAKE',.99)])
        self.assertEqual(result['reads']['pin_name'], ('/WAKE',.98))
        self.assertEqual(result['mono_pin_ocr']['paddings'], [0,2])
        self.assertEqual(result['bbox'], [29,25,51,35])
        self.assertEqual(len(provider.inputs), 5)
        np.testing.assert_array_equal(provider.inputs[-1], self.gray[25:35,29:51])
        self.assertTrue(qualified_fringe_view(result['mono_pin_ocr'], result['bbox']))
        assign = importlib.import_module('stage2.assign')
        self.assertIsNotNone(assign._qualified_mono_views(result, 'pin_name'))
        stale = copy.deepcopy(result)
        stale['mono_pin_ocr']['cleanup_evidence'].pop('name_fringe')
        self.assertIsNone(assign._qualified_mono_views(stale, 'pin_name'))

    def test_confident_contradiction_on_complete_word_is_held(self):
        result, _ = self.read([('WAKE',.89),('W',.99),('/WAKE',.98),('IWAKE',.97),('/WAKE',.99)])
        self.assertFalse(result['reads'])
        self.assertFalse(result['mono_pin_ocr']['accepted'])

    def test_accepted_original_word_never_opens_extra_sampling(self):
        result, provider = self.read([('READY',.97),('READY',.98)])
        self.assertEqual(result['reads']['pin_name'], ('READY',.97))
        self.assertEqual(result['bbox'], self.candidate['bbox'])
        self.assertEqual(len(provider.inputs), 2)


if __name__ == '__main__':
    unittest.main()
