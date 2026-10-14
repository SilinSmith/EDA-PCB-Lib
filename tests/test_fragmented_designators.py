import copy
import unittest
from unittest.mock import patch

import numpy as np

from stage2.fragmented_text import recover_fragmented_boxes
from stage2.text import merge_boxes, merge_boxes_both
from stage2.designator_reading import caption_candidates, read_designator_captions, _pixel_caption_boxes, PADDINGS


class Provider:
    def __init__(self, values):
        self.values = iter(values)
        self.inputs = []

    def read_glyph(self, gray):
        self.inputs.append(gray.copy())
        text, confidence = next(self.values)
        return {'text': text, 'confidence': confidence, 'sampling': {'purpose': 'fake-test'}}


class FragmentedWordsTests(unittest.TestCase):
    def test_full_same_row_survives_while_old_area_gate_stays_strict(self):
        old = [(2, 2, 6, 13), (23, 3, 28, 13)]
        complete = [(0, 1, 28, 13)]
        self.assertEqual(merge_boxes(old, complete), old)
        self.assertEqual(recover_fragmented_boxes(old, complete), [tuple(map(float, complete[0]))])
        self.assertIn(complete[0], merge_boxes_both(old, complete))
        self.assertTrue(all(b in merge_boxes_both(old, complete) for b in old))

    def test_large_or_multiline_or_single_piece_det_is_not_added(self):
        cases = [([(0, 0, 4, 10)], [(0, 0, 28, 10)]),
                 ([(0, 0, 4, 10), (22, 12, 28, 22)], [(0, 0, 28, 22)]),
                 ([(0, 0, 4, 10), (23, 0, 28, 10)], [(0, 0, 100, 10)]),
                 ([(0, 0, 4, 10), (3, 0, 6, 10)], [(0, 0, 28, 10)]),
                 ([(0, 0, 4, 10), (90, 0, 94, 10)], [(0, 0, 94, 10)])]
        for old, det in cases:
            with self.subTest(old=old, det=det):
                self.assertEqual(recover_fragmented_boxes(old, det), [])

    def test_output_limit_and_invalid_boxes(self):
        old = [(0, 0, 4, 10), (22, 0, 28, 10), (50, 0, 54, 10), (72, 0, 78, 10)]
        det = [(0, 0, 28, 10), (50, 0, 78, 10), (0, 0, float('nan'), 10)]
        self.assertEqual(len(recover_fragmented_boxes(old, det, limit=1)), 1)
        self.assertEqual(recover_fragmented_boxes(old, det, limit=0), [])


class CaptionReadingTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((100, 120), 255, np.uint8)
        self.gray[10:20, 20:48] = np.arange(280, dtype=np.uint8).reshape(10, 28)
        self.components = [{'id': 'ic', 'type': 'box', 'bbox_img': [20, 25, 100, 90]}]
        self.reads = [{'bbox': [20, 10, 48, 20], 'reads': {'designator': ('U', .9)}}]

    def test_preserves_observed_separator_suffix_and_original_pixels(self):
        for value in ('U_A', 'U_B', 'U21', 'U21A'):
            provider = Provider([(value, .98), (value, .95)])
            old = copy.deepcopy(self.reads)
            output, audit = read_designator_captions(self.gray, self.components, self.reads, provider=provider)
            self.assertEqual(output[0]['reads'], {'designator': (value, .95)})
            self.assertTrue(audit[0]['accepted'])
            self.assertEqual(self.reads, old)
            for padding, actual in zip(PADDINGS, provider.inputs):
                np.testing.assert_array_equal(actual, np.pad(self.gray[10:20, 20:48], padding, constant_values=255))

    def test_no_spelling_digit_or_separator_repair_and_no_confidence_relaxation(self):
        for pair in ((('UA', .99), ('UA', .99)), (('U_A', .99), ('U_B', .99)),
                     (('U_A', .99), ('U_A', .89)), (('IR2184', .99), ('IR2184', .99)),
                     (('U', .99), ('U', .99)), (('U_A', float('nan')), ('U_A', .99))):
            with self.subTest(pair=pair):
                output, audit = read_designator_captions(self.gray, self.components, self.reads, provider=Provider(pair+pair))
                self.assertEqual(output, [])
                self.assertFalse(audit[0]['accepted'])

    def test_inside_distant_and_ambiguous_captions_are_held(self):
        for box in ([20, 30, 48, 40], [20, -40, 48, -30], [102, 10, 120, 20]):
            with self.subTest(box=box):
                self.assertEqual(caption_candidates(self.components, [{'bbox': box}]), [])
        duplicate = self.components + [dict(self.components[0], id='second')]
        self.assertEqual(caption_candidates(duplicate, self.reads), [])

    def test_missing_provider_does_not_change_old_reads(self):
        with patch('stage2.designator_reading.get_pin_name_ocr', return_value=None) as load:
            self.assertEqual(read_designator_captions(self.gray, self.components, self.reads), ([], []))
            load.assert_called_once_with('auto')

    def test_reliable_existing_numeric_designator_is_not_reread(self):
        for value in ('U21', 'U21A', 'U_A'):
            reads = [{'bbox': self.reads[0]['bbox'], 'reads': {'designator': (value, .95)}}]
            with patch('stage2.designator_reading.get_pin_name_ocr') as load:
                self.assertEqual(read_designator_captions(self.gray, self.components, reads), ([], []))
                load.assert_not_called()

    def test_missing_separator_old_read_does_not_block_actual_reread(self):
        for value in ('UA', 'UB'):
            reads = [{'bbox': self.reads[0]['bbox'], 'reads': {'designator': (value, .99)}}]
            provider = Provider([('U_A', .99), ('U_A', .98)])
            output, _ = read_designator_captions(self.gray, self.components, reads, provider=provider)
            self.assertEqual(output[0]['reads']['designator'], ('U_A', .98))

    def test_off_mode_has_no_extra_provider_observations(self):
        self.assertEqual(read_designator_captions(self.gray, self.components, self.reads, mode='off'), ([], []))

    def test_restored_complete_word_is_reread_even_when_crnn_confidently_guesses_number(self):
        reads = [{'bbox': [20, 10, 48, 20], 'reads': {'designator': ('U3', .99)}},
                 {'bbox': [20, 10, 24, 20], 'reads': {'designator': ('1', .99)}},
                 {'bbox': [41, 10, 48, 20], 'reads': {'designator': ('1', .99)}}]
        provider = Provider([('U_B', .99), ('U_B', .98)])
        output, audit = read_designator_captions(self.gray, self.components, reads, provider=provider)
        self.assertEqual(output[0]['reads']['designator'], ('U_B', .98))
        self.assertTrue(audit[0]['complete_word_supported_by_fragments'])

    def test_bottom_caption_and_compact_two_character_caption_are_considered(self):
        components = [{'id':'ic','type':'box','bbox_img':[20,25,100,70]}]
        for box in ([50,72,68,84], [50,13,62,23]):
            reads = [{'bbox':box,'reads':{'designator':('U',.99)}}]
            output, _ = read_designator_captions(self.gray,components,reads,
                provider=Provider([('IC2',.99),('IC2',.98)]))
            self.assertEqual(output[0]['reads']['designator'],('IC2',.98))
            self.assertEqual(output[0]['designator_target'],'ic')

    def test_measured_caption_glyphs_require_text_support_and_clear_outside_lane(self):
        gray=np.full((160,150),255,np.uint8)
        components=[{'id':'ic','type':'box','source':'yolo','bbox_img':[20,30,120,100]}]
        gray[105:117,60:64]=0;gray[105:117,67:73]=0;gray[105:117,76:83]=0
        reads=[{'bbox':[60,105,64,117]}]
        boxes=_pixel_caption_boxes(gray,components,reads)
        self.assertEqual([b['bbox'] for b in boxes],[[59,104,84,118]])
        self.assertEqual(_pixel_caption_boxes(gray,components,[]),[])
        self.assertEqual(_pixel_caption_boxes(gray,[dict(components[0],source='cv:bigic')],reads),[])
        evidence={'ic':[{'edge':'bottom','trust':'trusted','frame_confirmed':True}]}
        self.assertEqual(_pixel_caption_boxes(gray,components,reads,evidence),[])
        gray[100:106,60:64]=0
        self.assertEqual(_pixel_caption_boxes(gray,components,reads),[])

    def test_repeated_identity_is_reread_and_wrong_prefix_never_creates_a_new_identity(self):
        components=[{'id':'ic','type':'box','bbox_img':[20,25,100,90]},
                    {'id':'next','type':'box','bbox_img':[20,65,100,99]}]
        reads=[{'bbox':[20,10,48,20],'reads':{'designator':('U1',.99)}},
               {'bbox':[20,50,48,60],'reads':{'designator':('U1',.99)}}]
        # Make ownership unique while keeping identical reads at different words.
        components[1]['bbox_img']=[55,65,115,99]
        reads[1]['bbox']=[65,50,93,60]
        output,audit=read_designator_captions(self.gray,components,reads,
            provider=Provider([('J1',.99),('J1',.98),('U1',.99),('U1',.98)]))
        self.assertEqual(len(output),2)
        self.assertTrue(audit[0]['rejected_designator_prefix'])
        self.assertFalse(audit[1]['rejected_designator_prefix'])

    def test_lowercase_existing_caption_needs_independent_case_confirmation(self):
        reads=[{'bbox':self.reads[0]['bbox'],'reads':{'designator':('u1',.99)}}]
        output,audit=read_designator_captions(self.gray,self.components,reads,
            provider=Provider([('u1',.99),('u1',.98)]))
        self.assertEqual(output[0]['reads']['designator'],('u1',.98))
        self.assertTrue(audit[0]['literal_case_reread'])

    def test_ambiguous_caption_can_only_reject_a_duplicated_wrong_prefix(self):
        components=self.components+[dict(self.components[0],id='duplicate')]
        reads=[{'bbox':self.reads[0]['bbox'],'reads':{'designator':('U1',.99)}},
               {'bbox':[20,92,48,99],'reads':{'designator':('U1',.99)}}]
        # A real identity cannot be assigned to either overlapping owner.
        output,audit=read_designator_captions(self.gray,components,reads,
            provider=Provider([('U1',.99),('U1',.98),('U1',.99),('U1',.98)]))
        self.assertEqual(output,[])
        output,audit=read_designator_captions(self.gray,components,reads,
            provider=Provider([('J1',.99),('J1',.98),('J2',.99),('J2',.98)]))
        self.assertTrue(output[0]['designator_ocr']['rejection_only'])
        self.assertTrue(output[0]['designator_ocr']['rejected_designator_prefix'])
        self.assertEqual(output[0]['designator_ocr']['owner_candidates'],['duplicate','ic'])


if __name__ == '__main__':
    unittest.main()
