"""Secondary caption evidence cannot guess identifiers or consume IC text."""
import copy
import unittest
from unittest.mock import Mock,patch

import numpy as np

from stage2 import non_ic_caption_reading as caption


class NonIcCaptionReadingTests(unittest.TestCase):
    def fixture(self, old='R13', typ='r', confidence=.72):
        image=np.full((100,140,3),255,np.uint8)
        image[25:34,35:40]=0
        doc={'components':[{'id':'p','type':typ,'bbox_img':[28,50,65,65]}]}
        assigned={'comps':{'p':{'designator':{'text':old,'conf':confidence,'box':[30,20,65,38]}}}}
        return doc,assigned,image

    def read(self,doc,assigned,image,views=None,**kwargs):
        provider=Mock()
        provider.read_glyph.side_effect=views or [{'text':'R18','confidence':.99},{'text':'R18','confidence':.98}]
        with patch.object(caption,'get_pin_name_ocr',return_value=provider)as loader:
            extra,audit=caption.read_non_ic_captions(doc,assigned,image,**kwargs)
        return extra,audit,provider,loader

    def test_agreeing_high_confidence_views_add_a_real_read_only(self):
        doc,a,im=self.fixture();frozen=copy.deepcopy(a)
        extra,audit,provider,_=self.read(doc,a,im)
        self.assertEqual(extra[0]['reads'],{'designator':('R18',.98)})
        self.assertEqual(extra[0]['bbox'],[30.,20.,65.,38.])
        self.assertEqual(audit[0]['glyph_bbox'],[28,18,67,40])
        self.assertTrue(audit[0]['accepted'])
        self.assertEqual(provider.read_glyph.call_count,2)
        self.assertEqual(a,frozen)

    def test_source_pixels_are_preserved_before_synthetic_margin_views(self):
        doc,a,im=self.fixture()
        _,_,provider,_=self.read(doc,a,im)
        first=provider.read_glyph.call_args_list[0].args[0]
        second=provider.read_glyph.call_args_list[1].args[0]
        self.assertEqual(first.shape,(26,43))
        self.assertEqual(second.shape,(30,47))
        np.testing.assert_array_equal(first[2:-2,2:-2],im[18:40,28:67,0])
        self.assertTrue(np.all(first[:2]==255))

    def test_disagreeing_views_do_not_change_the_designator(self):
        doc,a,im=self.fixture()
        extra,audit,_,_=self.read(doc,a,im,[{'text':'R18','confidence':.99},{'text':'R19','confidence':.99}])
        self.assertEqual(extra,[])
        self.assertFalse(audit[0]['accepted'])

    def test_low_confidence_view_is_not_averaged_upward(self):
        doc,a,im=self.fixture()
        extra,audit,_,_=self.read(doc,a,im,[{'text':'R18','confidence':1.},{'text':'R18','confidence':.96}])
        self.assertEqual(extra,[])
        self.assertFalse(audit[0]['accepted'])

    def test_wrong_component_prefix_and_combined_caption_are_rejected(self):
        doc,a,im=self.fixture()
        for text in ['U18','C18','R18 10K','R','R18A']:
            with self.subTest(text=text):
                extra,_,_,_=self.read(doc,a,im,[{'text':text,'confidence':1.}]*2)
                self.assertEqual(extra,[])

    def test_nonfinite_backend_scores_are_rejected(self):
        doc,a,im=self.fixture()
        for value in [float('nan'),float('inf'),1.1]:
            with self.subTest(value=value):
                extra,_,_,_=self.read(doc,a,im,[{'text':'R18','confidence':value}]*2)
                self.assertEqual(extra,[])

    def test_strong_primary_does_not_load_the_optional_backend(self):
        doc,a,im=self.fixture(confidence=.9)
        extra,audit,_,loader=self.read(doc,a,im)
        self.assertEqual((extra,audit),([],[]))
        loader.assert_not_called()

    def test_matching_primary_does_not_create_a_duplicate_candidate(self):
        doc,a,im=self.fixture(old='R18')
        extra,audit,_,_=self.read(doc,a,im)
        self.assertEqual(extra,[])
        self.assertEqual(audit[0]['held_reason'],'primary_already_agrees')

    def test_ic_components_and_crops_inside_an_ic_are_excluded(self):
        doc,a,im=self.fixture(typ='box')
        extra,audit,_,loader=self.read(doc,a,im)
        self.assertEqual((extra,audit),([],[]));loader.assert_not_called()
        doc,a,im=self.fixture();doc['components'].append({'id':'ic','type':'box','bbox_img':[20,15,80,45]})
        extra,audit,_,loader=self.read(doc,a,im)
        self.assertEqual((extra,audit),([],[]));loader.assert_not_called()

    def test_crop_padding_cannot_enter_a_nearby_ic(self):
        doc,a,im=self.fixture();doc['components'].append({'id':'ic','type':'box','bbox_img':[66,18,80,40]})
        extra,audit,_,loader=self.read(doc,a,im)
        self.assertEqual((extra,audit),([],[]));loader.assert_not_called()

    def test_clipped_glyph_is_held_without_reading_a_partial_identifier(self):
        doc,a,im=self.fixture();im[18,35:40]=0
        extra,audit,provider,loader=self.read(doc,a,im)
        self.assertEqual(extra,[])
        self.assertEqual(audit[0]['held_reason'],'source_glyph_touches_crop_border')
        provider.read_glyph.assert_not_called();loader.assert_not_called()

    def test_optional_backend_failure_preserves_existing_assignment(self):
        doc,a,im=self.fixture()
        with patch.object(caption,'get_pin_name_ocr',return_value=None):
            extra,audit=caption.read_non_ic_captions(doc,a,im)
        self.assertEqual(extra,[])
        self.assertEqual(audit[0]['held_reason'],'optional_provider_unavailable')
        self.assertEqual(a['comps']['p']['designator']['text'],'R13')

    def test_crop_recognition_error_is_audited_and_keeps_the_primary(self):
        doc,a,im=self.fixture()
        provider=Mock();provider.read_glyph.side_effect=ValueError('invalid tensor')
        with patch.object(caption,'get_pin_name_ocr',return_value=provider):
            extra,audit=caption.read_non_ic_captions(doc,a,im)
        self.assertEqual(extra,[])
        self.assertEqual(audit[0]['held_reason'],'secondary_recognition_failed')

    def test_off_mode_and_missing_caption_do_not_load_optional_dependencies(self):
        doc,a,im=self.fixture()
        for kwargs,assigned in [({'mode':'off'},a),({}, {'comps':{}})]:
            with self.subTest(kwargs=kwargs):
                extra,audit,_,loader=self.read(doc,assigned,im,**kwargs)
                self.assertEqual((extra,audit),([],[]));loader.assert_not_called()

    def unassigned_fixture(self):
        doc,_,im=self.fixture()
        doc['texts']=[[30,20,65,38]]
        return doc,{'comps':{'p':{'designator':None}}},im

    def test_missing_caption_with_one_geometric_owner_adds_source_read(self):
        doc,a,im=self.unassigned_fixture()
        frozen=copy.deepcopy(doc)
        extra,audit,_,_=self.read(doc,a,im)
        self.assertEqual(extra[0]['reads'],{'designator':('R18',.98)})
        self.assertEqual(audit[0]['ownership_source'],'unique_nearest_caption_geometry')
        self.assertEqual(doc,frozen)
        self.assertIsNone(a['comps']['p']['designator'])

    def test_equal_distance_or_two_pixel_margin_is_not_unique_ownership(self):
        doc,a,im=self.unassigned_fixture()
        for top in [50,52]:
            with self.subTest(top=top):
                doc['components'].append({'id':'other','type':'r','bbox_img':[28,top,65,top+15]})
                extra,audit,_,loader=self.read(doc,a,im)
                self.assertEqual((extra,audit),([],[]));loader.assert_not_called()
                doc['components'].pop()

    def test_nearer_ground_or_connector_blocks_unassigned_caption_read(self):
        doc,a,im=self.unassigned_fixture()
        for typ in ['gnd','v','connector','box']:
            with self.subTest(typ=typ):
                doc['components'].append({'id':'other','type':typ,'bbox_img':[28,39,65,46]})
                extra,audit,_,loader=self.read(doc,a,im)
                self.assertEqual((extra,audit),([],[]));loader.assert_not_called()
                doc['components'].pop()

    def test_centre_and_rectangle_must_agree_on_the_owner(self):
        doc,a,im=self.unassigned_fixture()
        doc['texts']=[[25,20,80,38]]
        doc['components']=[{'id':'p','type':'r','bbox_img':[0,40,28,60]},
                           {'id':'other','type':'gnd','bbox_img':[49,45,56,60]}]
        extra,audit,_,loader=self.read(doc,a,im)
        self.assertEqual((extra,audit),([],[]));loader.assert_not_called()

    def test_ownership_uses_actual_assignment_box_instead_of_raw_detector_box(self):
        doc,a,im=self.unassigned_fixture()
        doc['components'][0]['bbox_raw_img']=[100,80,120,99]
        extra,audit,_,_=self.read(doc,a,im)
        self.assertTrue(audit[0]['accepted']);self.assertEqual(len(extra),1)

    def test_same_row_digit_fragment_is_merged_and_read_once(self):
        doc,a,im=self.unassigned_fixture()
        doc['texts']=[{'bbox_img':[30,20,50,38]},[52,20,65,38]]
        im[25:34,54:58]=0
        extra,audit,provider,_=self.read(doc,a,im)
        self.assertEqual(extra[0]['bbox'],[30.,20.,65.,38.])
        self.assertEqual(len(audit),1);self.assertEqual(provider.read_glyph.call_count,2)
        np.testing.assert_array_equal(provider.read_glyph.call_args_list[0].args[0][2:-2,2:-2],im[18:40,28:67,0])

    def test_separate_lines_are_not_merged_and_disagreeing_numbers_are_held(self):
        doc,a,im=self.unassigned_fixture()
        doc['texts']=[[30,20,65,32],[30,35,65,47]]
        im[:]=255;im[23:29,35:40]=0;im[38:44,35:40]=0
        views=[{'text':text,'confidence':.99}for text in ['R18','R18','R19','R19']]
        extra,audit,provider,_=self.read(doc,a,im,views)
        self.assertEqual(extra,[]);self.assertEqual(provider.read_glyph.call_count,4)
        self.assertEqual(len(audit),2)
        self.assertTrue(all(r['held_reason']=='multiple_caption_regions_disagree'for r in audit))

    def test_merged_word_cannot_cross_ownership_boundary(self):
        doc,a,im=self.unassigned_fixture()
        doc['components']=[{'id':'p','type':'r','bbox_img':[15,50,40,65]},
                           {'id':'other','type':'gnd','bbox_img':[62,50,90,65]}]
        doc['texts']=[[30,20,50,38],[52,20,65,38]]
        extra,audit,_,loader=self.read(doc,a,im)
        self.assertEqual((extra,audit),([],[]));loader.assert_not_called()

    def test_distant_invalid_or_absent_text_boxes_do_not_load_the_backend(self):
        doc,a,im=self.unassigned_fixture()
        for boxes in [[],[[30,-100,65,-90]],[[1,2,3]],[[1,2,float('nan'),9]],[[30,30,20,40]]]:
            with self.subTest(boxes=boxes):
                doc['texts']=boxes
                extra,audit,_,loader=self.read(doc,a,im)
                self.assertEqual((extra,audit),([],[]));loader.assert_not_called()

    def test_unassigned_caption_inside_an_ic_is_not_read(self):
        doc,a,im=self.unassigned_fixture()
        doc['components'].append({'id':'ic','type':'box','bbox_img':[20,15,80,45]})
        extra,audit,_,loader=self.read(doc,a,im)
        self.assertEqual((extra,audit),([],[]));loader.assert_not_called()

    def test_read_is_assigned_to_the_same_owner_by_real_assigner(self):
        from stage2.assign import assign
        doc,a,im=self.unassigned_fixture()
        doc['components'].append({'id':'other','type':'r','bbox_img':[100,70,125,85]})
        extra,_,_,_=self.read(doc,a,im)
        result=assign(doc['components'],{},extra)
        self.assertEqual(result['comps']['p']['designator']['canon'],'R18')
        self.assertIsNone(result['comps']['other']['designator'])

    def small_caption(self):
        doc,a,im=self.fixture(old='C15',typ='c',confidence=.5)
        im[:]=255
        im[25:35,35:39]=(120,0,0)
        im[25:35,44:49]=(120,0,0)
        return doc,a,im

    def test_small_caption_retries_trim_blank_pixels_without_changing_the_identifier(self):
        doc,a,im=self.small_caption();frozen=im.copy()
        views=[{'text':'C6','confidence':score}for score in [.96,.94,.99,.98]]
        extra,audit,provider,_=self.read(doc,a,im,views)
        self.assertEqual(extra[0]['reads'],{'designator':('C6',.98)})
        self.assertEqual(audit[0]['accepted_view_family'],'tight_original_pixels')
        self.assertEqual(len(audit[0]['views']),2)
        self.assertEqual(len(audit[0]['retry_views']),2)
        self.assertNotIn('held_reason',audit[0])
        self.assertEqual(provider.read_glyph.call_count,4)
        expected=caption.cv2.cvtColor(im[25:35,35:49],caption.cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(provider.read_glyph.call_args_list[2].args[0][2:-2,2:-2],expected)
        np.testing.assert_array_equal(im,frozen)

    def test_scale_retry_requires_two_high_confidence_views_plus_original_agreement(self):
        doc,a,im=self.small_caption()
        views=[{'text':'C9','confidence':score}for score in [.963,.941,.961,.937,.977,.981]]
        extra,audit,provider,_=self.read(doc,a,im,views)
        self.assertEqual(extra[0]['reads'],{'designator':('C9',.977)})
        self.assertEqual(audit[0]['accepted_view_family'],'scale_normalized_caption')
        self.assertEqual([v['target_foreground_height']for v in audit[0]['retry_views'][-2:]],[32,48])
        self.assertEqual([v.args[0].shape[0]for v in provider.read_glyph.call_args_list[-2:]],[36,52])
        self.assertEqual(provider.read_glyph.call_count,6)

    def test_changed_retry_read_is_rejected_even_when_its_score_is_higher(self):
        doc,a,im=self.small_caption()
        views=[{'text':'C9','confidence':.94}]*2+[{'text':'C8','confidence':1.}]*2
        extra,audit,provider,_=self.read(doc,a,im,views)
        self.assertEqual(extra,[])
        self.assertEqual(audit[0]['held_reason'],'small_caption_retry_disagrees')
        self.assertEqual(provider.read_glyph.call_count,4)

    def test_conflicting_original_reads_cannot_be_repaired_by_retry_votes(self):
        doc,a,im=self.small_caption()
        views=[{'text':'C9','confidence':.96},{'text':'C8','confidence':.96}]
        extra,_,provider,_=self.read(doc,a,im,views)
        self.assertEqual(extra,[]);self.assertEqual(provider.read_glyph.call_count,2)

    def test_a_single_high_confidence_scaled_view_does_not_pass_the_old_threshold(self):
        doc,a,im=self.small_caption()
        views=[{'text':'C9','confidence':score}for score in [.94,.95,.96,.95,1.,.96]]
        extra,audit,provider,_=self.read(doc,a,im,views)
        self.assertEqual(extra,[])
        self.assertFalse(audit[0]['accepted']);self.assertEqual(provider.read_glyph.call_count,6)

    def test_very_weak_or_nonfinite_original_scores_do_not_trigger_more_views(self):
        doc,a,im=self.small_caption()
        for score in [.89,float('nan'),float('inf'),1.1]:
            with self.subTest(score=score):
                views=[{'text':'C9','confidence':score}]*2
                extra,_,provider,_=self.read(doc,a,im,views)
                self.assertEqual(extra,[]);self.assertEqual(provider.read_glyph.call_count,2)

    def test_parameter_text_and_other_symbol_types_do_not_enter_capacitor_retry(self):
        doc,a,im=self.small_caption()
        for typ,text in [('c','100nF'),('r','R9'),('led','LED9'),('d','D9')]:
            with self.subTest(typ=typ):
                doc['components'][0]['type']=typ
                extra,_,provider,_=self.read(doc,a,im,[{'text':text,'confidence':.94}]*2)
                self.assertEqual(extra,[]);self.assertEqual(provider.read_glyph.call_count,2)

    def test_faint_border_ink_is_not_cropped_away_by_normalization(self):
        doc,a,im=self.small_caption();im[18,35:40]=(200,175,175)
        views=[{'text':'C9','confidence':.94}]*2
        extra,audit,provider,_=self.read(doc,a,im,views)
        self.assertEqual(extra,[]);self.assertNotIn('retry_views',audit[0])
        self.assertEqual(provider.read_glyph.call_count,2)

    def test_large_caption_stays_on_the_established_original_pixel_path(self):
        doc,a,im=self.small_caption()
        a['comps']['p']['designator']['box']=[30,5,65,42]
        im[8:40,35:39]=(120,0,0)
        extra,audit,provider,_=self.read(doc,a,im,[{'text':'C9','confidence':.94}]*2)
        self.assertEqual(extra,[]);self.assertNotIn('retry_views',audit[0])
        self.assertEqual(provider.read_glyph.call_count,2)

    def test_retry_recognition_error_is_audited_without_altering_primary_assignment(self):
        doc,a,im=self.small_caption();frozen=copy.deepcopy(a)
        provider=Mock();provider.read_glyph.side_effect=[{'text':'C9','confidence':.94}]*2+[RuntimeError('unavailable tensor')]
        with patch.object(caption,'get_pin_name_ocr',return_value=provider):
            extra,audit=caption.read_non_ic_captions(doc,a,im)
        self.assertEqual(extra,[]);self.assertEqual(a,frozen)
        self.assertEqual(audit[0]['held_reason'],'secondary_recognition_failed')

    def test_retry_cannot_bypass_a_stricter_caller_threshold(self):
        doc,a,im=self.small_caption()
        views=[{'text':'C9','confidence':score}for score in [.94,.95,.98,.98,.98,.98]]
        extra,_,provider,_=self.read(doc,a,im,views,min_conf=.99)
        self.assertEqual(extra,[]);self.assertEqual(provider.read_glyph.call_count,6)

    def test_neutral_or_red_ink_does_not_define_a_blue_caption_region(self):
        doc,a,im=self.small_caption()
        for value in [(0,0,0),(0,0,120)]:
            with self.subTest(value=value):
                im[25:35,35:39]=value;im[25:35,44:49]=value
                extra,audit,provider,_=self.read(doc,a,im,[{'text':'C9','confidence':.94}]*2)
                self.assertEqual(extra,[]);self.assertNotIn('retry_views',audit[0])
                self.assertEqual(provider.read_glyph.call_count,2)

    def test_blue_noise_inside_mostly_neutral_text_is_not_a_complete_blue_caption(self):
        doc,a,im=self.small_caption();im[25:35,44:49]=0
        extra,audit,provider,_=self.read(doc,a,im,[{'text':'C9','confidence':.94}]*2)
        self.assertEqual(extra,[]);self.assertNotIn('retry_views',audit[0])
        self.assertEqual(provider.read_glyph.call_count,2)


if __name__=='__main__':unittest.main()
