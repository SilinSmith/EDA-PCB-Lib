"""Printed case needs token evidence and an independent original-digit read."""
import copy
import importlib
import unittest
from types import SimpleNamespace

import numpy as np

from stage2.pin_name_ocr import _ctc_token_reads
from stage2.segmented_caption import read_segmented_caption, qualified_segmented_caption


class Provider:
    def __init__(self, values=(('4',.98),('4',.97))):
        self.values, self.inputs = iter(values), []

    def read_glyph(self, pixels):
        self.inputs.append(pixels.copy())
        text, confidence = next(self.values)
        return dict(text=text, confidence=confidence)


class SegmentedCaptionTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((100, 120), 255, np.uint8)
        self.gray[20:27,20:27] = 0
        self.gray[15:27,30:34] = 0
        self.image = np.repeat(self.gray[:, :, None], 3, 2)
        self.box = [19,14,35,28]
        self.body = [18,40,90,95]
        self.views = [dict(padding=p, text='q'+tail, confidence=.8,
                           ctc_tokens=[dict(text='q',confidence=.99),dict(text=tail,confidence=.6)])
                      for p, tail in ((4,'4'),(6,'A'))]

    def recover(self, provider=None):
        return read_segmented_caption(self.gray, self.image, self.box, self.body,
                                      self.views, provider or Provider(), {'Q'})

    def test_prefix_case_and_numeric_body_have_separate_verified_evidence(self):
        provider=Provider()
        before=copy.deepcopy(self.views)
        pixels=self.image.copy()
        result=self.recover(provider)
        self.assertEqual(result['prefix']+result['suffix'], 'q4')
        self.assertEqual(result['confidence'], .97)
        self.assertEqual(self.views,before)
        np.testing.assert_array_equal(self.image,pixels)
        x0,y0,x1,y1=result['suffix_bbox']
        for padding, received in zip((2,4),provider.inputs):
            np.testing.assert_array_equal(received,np.pad(self.gray[y0:y1,x0:x1],padding,constant_values=255))
        meta=dict(bbox=self.box, raw_reads=self.views, segmented_caption=result)
        self.assertEqual(qualified_segmented_caption(meta,'q4',.97), 'q4')
        gray_only = read_segmented_caption(self.gray, self.gray, self.box, self.body,
                                           self.views, Provider(), {'Q'})
        self.assertEqual(gray_only['prefix']+gray_only['suffix'], 'q4')
        for field, value in (('prefix','Q'),('suffix','8'),('suffix_bbox',[19,13,36,29])):
            bad=copy.deepcopy(meta);bad['segmented_caption'][field]=value
            self.assertIsNone(qualified_segmented_caption(bad,'q4',.97))

    def test_weak_or_conflicting_prefix_cannot_force_letter_case(self):
        for variant in ('missing_tokens','low_prefix','different_case'):
            with self.subTest(variant=variant):
                self.setUp()
                if variant=='missing_tokens': self.views[0].pop('ctc_tokens')
                elif variant=='low_prefix': self.views[0]['ctc_tokens'][0]['confidence']=.89
                else:
                    self.views[1]['text']='QA';self.views[1]['ctc_tokens'][0]['text']='Q'
                provider=Provider()
                self.assertIsNone(self.recover(provider))
                self.assertFalse(provider.inputs)

    def test_a_confident_disagreeing_whole_suffix_or_digit_views_stays_held(self):
        self.views[1]['ctc_tokens'][1]['confidence']=.95
        self.assertIsNone(self.recover())
        self.setUp()
        self.assertIsNone(self.recover(Provider((('4',.98),('9',.98)))))
        self.assertIsNone(self.recover(Provider((('4',.89),('4',.98)))))

    def test_joined_glyphs_or_frame_ink_never_open_numeric_resampling(self):
        for variant in ('joined','body','outside_ink'):
            with self.subTest(variant=variant):
                self.setUp()
                if variant=='joined': self.gray[23,26:31]=0
                elif variant=='body': self.body[1]=28
                else: self.gray[13,32]=0
                self.image=np.repeat(self.gray[:,:,None],3,2)
                provider=Provider()
                self.assertIsNone(self.recover(provider))
                self.assertFalse(provider.inputs)

    def test_assignment_accepts_literal_combined_evidence_and_rejects_stale_parts(self):
        assign=importlib.import_module('stage2.assign')
        self.views=[dict(v,text='u'+v['text'][1:],ctc_tokens=[dict(text='u',confidence=.99),v['ctc_tokens'][1]]) for v in self.views]
        split=read_segmented_caption(self.gray,self.image,self.box,self.body,self.views,Provider(),{'U'})
        meta=dict(bbox=self.box,raw_reads=self.views,segmented_caption=split,accepted=True,
                  unchanged_source_pixels=True,owner_candidates=['ic'])
        record=dict(bbox=self.box,source='ic_caption_ppocr',designator_target='ic',designator_only=True,
                    reads={'designator':('u4',.97)},designator_ocr=meta)
        self.assertEqual(assign._caption_consensus(record,'ic'),'u4')
        corrupted=copy.deepcopy(record);corrupted['designator_ocr']['segmented_caption']['suffix_reads'][1]['text']='9'
        self.assertIsNone(assign._caption_consensus(corrupted,'ic'))
        record['designator_target']='other'
        self.assertIsNone(assign._caption_consensus(record,'ic'))

    def test_ctc_evidence_preserves_repeated_tokens_after_blank_and_refuses_decoder_mismatch(self):
        decoder=SimpleNamespace(character=['','q','4'],get_ignored_tokens=lambda:[0])
        probabilities=np.zeros((1,5,3),np.float32)
        for position,index in enumerate((1,1,0,2,2)): probabilities[0,position,index]=.96
        tokens=_ctc_token_reads(probabilities,decoder,'q4')
        self.assertEqual([t['text'] for t in tokens],['q','4'])
        self.assertEqual(_ctc_token_reads(probabilities,decoder,'Q4'),[])
        self.assertEqual(_ctc_token_reads(probabilities,lambda:None,'q4'),[])


if __name__ == '__main__':
    unittest.main()
