import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from stage2 import color_crops
from stage2.assign import _best_read,fit_pinname,fit_pinnum


class ColorNumberCropsTests(unittest.TestCase):
    def fixture(self):
        image=np.full((80,120,3),255,np.uint8)
        image[27:35,44:48]=0  # external digit
        image[27:35,53:60]=0  # internal pin-1 circle must be excluded
        image[20:61,50]=[20,20,140]
        doc={"components":[{"id":"ic","type":"box"}],"pins":{"ic":[[30,38]]},
             "pin_evidence":{"ic":[{"source":"red_green_transition","trust":"trusted",
             "frame_confirmed":True,"point_img":[30,38],"edge":"left","anchor_img":[50,38]}]}}
        reads=[{"bbox":[42,25,62,36],"reads":{"pin_number":("13",.8)}}]
        return image,doc,reads

    def test_cross_frame_digit_crop_excludes_body_circle_and_colored_frame(self):
        image,doc,reads=self.fixture()
        boxes,gray=color_crops.color_number_boxes(doc,reads,image)
        self.assertEqual(len(boxes),1)
        self.assertLessEqual(boxes[0]["bbox"][2],50)
        self.assertTrue(np.all(gray[20:61,50]==255))
        self.assertEqual(boxes[0]["original_bbox"],reads[0]["bbox"])

    def test_read_is_role_limited_and_original_reads_are_unchanged(self):
        image,doc,reads=self.fixture();original=copy.deepcopy(reads)
        recognizer=Mock();recognizer.read_many.return_value=[("1",.98)]
        result=color_crops.read_color_numbers(doc,reads,SimpleNamespace(recognizer=lambda:recognizer),image)
        self.assertEqual(reads,original)
        self.assertEqual(result[0]["reads"],{"pin_number":("1",.98)})
        self.assertTrue(result[0]["number_only"])
        self.assertIsNone(_best_read(result[0]["reads"],"pin_name",fit_pinname))
        self.assertEqual(_best_read(result[0]["reads"],"pin_number",fit_pinnum)[1],"1")

    def test_valid_numeric_names_from_name_role_are_preserved(self):
        self.assertEqual(_best_read({"pin_name":("1",.98)},"pin_name",fit_pinname)[1],"1")

    def test_monochrome_stale_and_unconfirmed_evidence_do_not_trigger_recrops(self):
        image,doc,reads=self.fixture()
        for change in ({"source":"grayscale"},{"point_img":[30,30]}, {"frame_confirmed":False}):
            modified=copy.deepcopy(doc);modified['pin_evidence']['ic'][0].update(change)
            self.assertEqual(color_crops.color_number_boxes(modified,reads,image)[0],[])

    def test_identical_windows_are_deduplicated(self):
        image,doc,reads=self.fixture()
        self.assertEqual(len(color_crops.color_number_boxes(doc,reads*2,image)[0]),1)

    def test_geometry_tracks_actual_ink_not_sampling_padding(self):
        image,doc,reads=self.fixture()
        recognizer=Mock();recognizer.read_many.return_value=[("1",.98)]
        result=color_crops.read_color_numbers(doc,reads,SimpleNamespace(recognizer=lambda:recognizer),image)
        self.assertEqual(result[0]["geometry_bbox"],[44,27,48,35])
        self.assertEqual(result[0]["glyph_bbox"],result[0]["geometry_bbox"])
        self.assertEqual(result[0]["bbox"],[42,25,50,37])
        self.assertEqual(result[0]["color_crop"]["glyph_bbox"],[44,27,48,35])
        self.assertEqual(recognizer.read_many.call_args[0][1], [([42,25,50,37],"pin_number")])

    def test_nc_cross_cleanup_preserves_all_digits_and_records_pollution(self):
        image,doc,reads=self.fixture()
        image[:]=255
        image[27:35,41:44]=0
        image[27:35,46:49]=0  # two distinct digits, never take only rightmost one
        for dy in range(-4,5):
            image[30+dy,34+dy]=[0,150,0]
            image[30+dy,34-dy]=[0,150,0]
        reads[0]["bbox"]=[29,25,50,36]
        boxes,gray=color_crops.color_number_boxes(doc,reads,image)
        self.assertEqual(boxes[0]["geometry_bbox"],[41,27,49,35])
        self.assertEqual(np.count_nonzero(gray[25:36,29:39]<190),0)
        self.assertGreater(boxes[0]["color_contamination_ratio"],.1)
        self.assertGreater(boxes[0]["source_boxes"][0]["colored_pixels"],0)
        self.assertEqual(boxes[0]["glyph_pixels"],48)

    def test_overlap_sources_of_same_glyph_keep_provenance(self):
        image,doc,reads=self.fixture()
        second=copy.deepcopy(reads[0]);second["bbox"]=[43,26,61,36]
        boxes,_=color_crops.color_number_boxes(doc,reads+[second],image)
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0]["original_bboxes"],[reads[0]["bbox"],second["bbox"]])
        self.assertEqual([item["read_index"] for item in boxes[0]["source_boxes"]],[0,1])
        self.assertTrue(boxes[0]["entity_id"].startswith("ic:0:"))

    def test_malformed_metadata_and_ocr_windows_are_ignored(self):
        image,doc,reads=self.fixture()
        for field,value in (("point_img",[30]),("point_img",[float("nan"),38]),
                            ("anchor_img","wrong"),("source",[])):
            bad=copy.deepcopy(doc);bad['pin_evidence']['ic'][0][field]=value
            self.assertEqual(color_crops.color_number_boxes(bad,reads,image)[0],[])
        for malformed in ({"bbox":[42,25,float("inf"),36],"reads":reads[0]["reads"]},
                          {"bbox":"no", "reads":reads[0]["reads"]},
                          {"bbox":reads[0]["bbox"],"reads":None},None):
            self.assertEqual(color_crops.color_number_boxes(doc,[malformed],image)[0],[])

    def test_edge_clipping_is_separate_from_complete_multidigit_ink(self):
        image,doc,reads=self.fixture()
        image[:]=255
        image[27:35,39:42]=0
        image[27:35,46:49]=0
        image[27:35,53:60]=0  # internal marker excluded
        boxes,_=color_crops.color_number_boxes(doc,reads,image)
        # Expand original OCR window to include both external digits.
        reads[0]["bbox"]=[38,25,62,36]
        boxes,_=color_crops.color_number_boxes(doc,reads,image)
        self.assertEqual(boxes[0]["geometry_bbox"],[39,27,49,35])
        self.assertEqual(boxes[0]["bbox"][2],50)
        self.assertTrue(boxes[0]["frame_clipped"])
        self.assertFalse(boxes[0]["glyph_touches_search_boundary"])

    def test_nearest_actual_row_owns_glyph_even_when_searches_overlap(self):
        image,doc,reads=self.fixture()
        doc["pins"]["ic"].append([30,58])
        peer=copy.deepcopy(doc["pin_evidence"]["ic"][0])
        peer["point_img"]=[30,58];peer["anchor_img"]=[50,58]
        doc["pin_evidence"]["ic"].append(peer)
        boxes,_=color_crops.color_number_boxes(doc,reads,image)
        self.assertEqual([box["pin_index"] for box in boxes],[0])

    def test_complete_digit_not_cut_at_pin_row_minus_one(self):
        image,doc,reads=self.fixture()
        image[35:39,44:48]=0
        reads[0]["bbox"]=[42,25,49,39]
        boxes,_=color_crops.color_number_boxes(doc,reads,image)
        self.assertEqual(boxes[0]["geometry_bbox"],[44,27,48,39])
        self.assertEqual(boxes[0]["bbox"][3],41)

    def test_jpeg_color_fringe_removed_only_near_color_stroke(self):
        image,doc,reads=self.fixture()
        image[36,42]=[0,160,0]
        image[36,43]=[130,144,168]  # faint NC/wire compression fringe
        image[28,45]=[110,130,140]  # distant antialiased dark text preserved
        boxes,cleaned=color_crops.color_number_boxes(doc,reads,image)
        self.assertEqual(cleaned[36,43],255)
        self.assertLess(cleaned[28,45],190)
        self.assertEqual(boxes[0]["geometry_bbox"],[44,27,48,35])

    def test_isolated_l_shaped_three_pixel_noise_is_removed_beside_digit(self):
        ink=np.zeros((16,24),bool)
        ink[1:11,15:21]=True
        ink[12,2]=ink[13,2]=ink[13,3]=True
        result,removed=color_crops._remove_numeric_specks(ink)
        self.assertEqual(np.count_nonzero(removed),3)
        self.assertEqual(np.count_nonzero(result),60)

    def test_narrow_one_and_all_multidigit_bodies_survive(self):
        ink=np.zeros((16,24),bool)
        ink[2:8,3:5]=True  # 2x6 digit 1, never a 2x2 noise blob
        ink[2:11,14:18]=True  # second digit body must also remain
        ink[13,21]=True
        result,removed=color_crops._remove_numeric_specks(ink)
        self.assertTrue(np.all(result[2:8,3:5]))
        self.assertTrue(np.all(result[2:11,14:18]))
        self.assertEqual(np.count_nonzero(removed),1)

    def test_tiny_low_resolution_number_without_body_is_unchanged(self):
        ink=np.zeros((8,8),bool)
        ink[2,2]=ink[3,2]=ink[3,3]=True
        result,removed=color_crops._remove_numeric_specks(ink)
        self.assertTrue(np.array_equal(result,ink))
        self.assertFalse(np.any(removed))

    def test_noise_reread_is_local_and_does_not_mutate_base_gray(self):
        image,doc,reads=self.fixture()
        image[36,42]=image[37,42]=image[37,43]=0
        reads[0]["bbox"]=[40,25,62,39]
        recognizer=Mock();recognizer.read_many.side_effect=[[('11',.7)],[('1',.99)]]
        result=color_crops.read_color_numbers(doc,reads,SimpleNamespace(recognizer=lambda:recognizer),image)
        self.assertEqual(result[0]["reads"]["pin_number"],('1',.99))
        self.assertEqual(result[0]["color_crop"]["isolated_pixels_removed"],3)
        first_gray=recognizer.read_many.call_args_list[0][0][0]
        second_gray=recognizer.read_many.call_args_list[1][0][0]
        self.assertEqual(first_gray[36,42],0)
        self.assertEqual(second_gray[36,42],255)
        self.assertEqual(image[36,42,0],0)
        self.assertNotIn('_noise_pixels_img',result[0]["color_crop"])


if __name__=="__main__":
    unittest.main()
