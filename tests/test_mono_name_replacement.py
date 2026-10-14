"""Strong old words are protected only by same-cell, observed missing ink."""
import copy
import unittest
import numpy as np
from stage2.mono_pin_reading import _incomplete_name_replacement, _ambiguous_complete_name_replacement


class MonoNameReplacementTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 150, 3), 255, np.uint8)
        self.image[30:38, 60:74] = 0
        self.image[39, 74:80] = 0
        self.image[30:38, 80:92] = 0
        self.old = {"text": "GVDD_AB", "conf": .99, "box": [60,30,92,40]}
        self.observation = {"reads": {"pin_name": ("AB", .999)}}
        self.candidate = {"bbox": [80,30,92,38], "cleanup_evidence": {
            "physical_search_region": [45,25,100,45], "word_gap_pixels": 5}}

    def test_adjacent_source_prefix_and_mark_prevent_suffix_replacement(self):
        source = self.image.copy()
        reason = _incomplete_name_replacement(self.old,self.observation,self.candidate,self.image)
        self.assertIsNotNone(reason)
        self.assertGreater(reason["adjacent_excluded_source_pixels"], 0)
        np.testing.assert_array_equal(self.image,source)

    def test_white_old_rectangle_does_not_prove_a_missing_prefix(self):
        self.image[:,:80] = 255
        self.assertIsNone(_incomplete_name_replacement(self.old,self.observation,self.candidate,self.image))

    def test_weak_other_row_and_noncontained_old_boxes_do_not_hold_read(self):
        for change in ({"conf": .89}, {"box": [60,10,92,20]}, {"box": [5,30,70,40]},
                       {"conf": float("nan")}, {"conf": 1.1}):
            with self.subTest(change=change):
                old=dict(self.old,**change)
                self.assertIsNone(_incomplete_name_replacement(old,self.observation,self.candidate,self.image))

    def test_complete_corrected_read_is_allowed_without_dictionary_guessing(self):
        observed=copy.deepcopy(self.observation)
        observed["reads"]["pin_name"]=("GVDD_AC",.99)
        self.assertIsNone(_incomplete_name_replacement(self.old,observed,self.candidate,self.image))
        observed["reads"]["pin_name"]=("GVDD_AB_EXTRA",.99)
        self.assertIsNone(_incomplete_name_replacement(self.old,observed,self.candidate,self.image))

    def test_distant_opposite_name_does_not_protect_overwide_old_rectangle(self):
        self.image[:,:80]=255
        self.image[30:38,47:55]=0
        self.old["box"]=[47,30,92,40]
        self.assertIsNone(_incomplete_name_replacement(self.old,self.observation,self.candidate,self.image))

    def test_same_complete_strong_word_ambiguity_is_held_even_after_pin_index_changes(self):
        old={'text':'P10','conf':.96,'box':[80,30,92,38]}
        observed={'reads':{'pin_name':('P1O',.99)}}
        guard=_ambiguous_complete_name_replacement(old,observed,self.candidate)
        self.assertEqual(guard['previous_word'],'P10')
        for changed in ({'conf':.89},{'box':[80,10,92,18]},{'text':'D10'},
                        {'box':[75,30,92,38]}):
            self.assertIsNone(_ambiguous_complete_name_replacement(dict(old,**changed),observed,self.candidate))


if __name__ == "__main__":
    unittest.main()
