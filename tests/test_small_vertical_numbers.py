import unittest
import numpy as np
from stage2.mono_pin_reading import _small_vertical_number_retry

class Reader:
    def __init__(self,words):self.words=iter(words);self.images=[]
    def read_glyph(self,image):
        self.images.append(image.copy());text,conf=next(self.words)
        return {'text':text,'confidence':conf}

class SmallVerticalNumberTests(unittest.TestCase):
    def run_read(self,words,initial=None):
        glyph=np.full((7,9),255,np.uint8);glyph[1:6,2:7]=0;before=glyph.copy()
        reader=Reader(words)
        result,trials=_small_vertical_number_retry(glyph,[0],reader,initial or [],.90)
        np.testing.assert_array_equal(glyph,before)
        self.assertEqual(len(trials),2)
        self.assertEqual([a.shape for a in reader.images],[(18,22),(22,26),(25,31),(29,35)])
        return result
    def test_four_views_release_only_consistent_decimal(self):
        result=self.run_read([('18',.97),('18',.96),('18',.98),('18',.95)])
        self.assertEqual(result[:2],('18',.95))
    def test_one_weak_view_is_not_replaced_by_scale_search(self):
        self.assertIsNone(self.run_read([('18',.97),('18',.89),('18',.98),('18',.95)]))
    def test_strong_original_conflict_cannot_be_overridden(self):
        initial=[[{'text':'8','confidence':.95,'padding':2}]]
        self.assertIsNone(self.run_read([('18',.97)]*4,initial))
    def test_disagreement_between_scales_stays_held(self):
        self.assertIsNone(self.run_read([('18',.97),('18',.97),('13',.97),('13',.97)]))
    def test_nondecimal_retries_do_not_guess_a_decimal(self):
        self.assertIsNone(self.run_read([('B18',.97)]*4))

if __name__=='__main__':unittest.main()
