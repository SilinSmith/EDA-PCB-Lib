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


if __name__=="__main__":
    unittest.main()
