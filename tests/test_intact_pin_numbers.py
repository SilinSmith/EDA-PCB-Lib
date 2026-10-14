import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage2.intact_pin_numbers import intact_number_line_box
from stage2.mono_pin_crops import mono_pin_boxes
from stage2.mono_pin_reading import read_mono_pin_texts
from stage2.vertical_pin_reading import ClockwisePinView
from test_vertical_pin_reading import vertical_fixture, Provider


def touching_fixture():
    doc, _, horizontal = vertical_fixture()
    horizontal[39:46,187:203] = 255
    horizontal[43:50,187:203] = 40
    image = cv2.rotate(horizontal, cv2.ROTATE_90_COUNTERCLOCKWISE)
    view = ClockwisePinView(*image.shape[:2])
    candidates, gray = mono_pin_boxes(view.document(doc), horizontal)
    candidate = next(c for c in candidates if c["role"] == "pin_number")
    return doc, image, horizontal, candidate, gray


class IntactPinNumberTests(unittest.TestCase):
    def test_attached_core_and_all_original_digit_pixels_are_kept_without_mutation(self):
        _, _, _, candidate, gray = touching_fixture()
        before=copy.deepcopy(candidate);source=gray.copy()
        self.assertEqual(candidate["cleanup_evidence"]["connected_ink_beyond_region_sides"], [3])
        opened=intact_number_line_box(candidate,gray)
        self.assertEqual(opened["bbox"], [187,43,203,51])
        self.assertFalse(opened["flags"])
        self.assertEqual(opened["cleanup_evidence"]["black_pixels_removed"],0)
        self.assertEqual(opened["cleanup_evidence"]["known_line_exclusions"],[])
        self.assertTrue(opened["cleanup_evidence"]["prior_glyph_integrity"]["integrity_unresolved"])
        self.assertEqual(candidate,before);np.testing.assert_array_equal(gray,source)

    def test_stale_core_deleted_character_ink_top_cut_or_marker_stays_held(self):
        for kind in ("stale_pixels","stale_evidence","erased_ink","top_cut","marker","no_measured_line"):
            _, _, _, candidate, gray=touching_fixture()
            e=candidate["cleanup_evidence"]
            if kind=="stale_pixels":gray[50,180:216]=255
            elif kind=="stale_evidence":e["physical_evidence"]["point_img"][1]+=20
            elif kind=="erased_ink":e["black_pixels_removed"]=1
            elif kind=="top_cut":e["connected_ink_beyond_region_sides"]=[1,3]
            elif kind=="marker":e["physical_evidence"]["source"]="frame_hollow_marker"
            else:e["measured_line_evidence"]=[]
            self.assertIsNone(intact_number_line_box(candidate,gray),kind)

    def test_provider_receives_exact_source_digit_and_attached_wire_in_each_view(self):
        doc,image,horizontal,_,_=touching_fixture();source=image.copy()
        provider=Provider([("ABCD",.99),("ABCD",.99),("42",.99),("42",.98)])
        with patch("stage2.mono_pin_reading.get_pin_name_ocr",return_value=provider):
            records=read_mono_pin_texts(doc,[],None,image)
        number=next(r for r in records if r["number_only"])
        self.assertEqual(number["reads"]["pin_number"],("42",.98))
        self.assertEqual(number["mono_pin_ocr"]["view"],"original_numeric_glyph_with_measured_attached_line")
        for margin,sample in zip((2,4),provider.inputs[2:]):
            np.testing.assert_array_equal(sample,np.pad(horizontal[43:51,187:203,0],margin,constant_values=255))
        np.testing.assert_array_equal(image,source)

    def test_uncertain_views_and_alphanumeric_raw_footer_do_not_override_identifier(self):
        for values in ([("42",.99),("42",.89)],[("42",.99),("43",.99)],[("P9",.99),("P9",.99)]):
            doc,image,_,_,_=touching_fixture()
            provider=Provider(values)
            with patch("stage2.mono_pin_reading.get_pin_name_ocr",return_value=provider):
                records=read_mono_pin_texts(doc,[],None,image,include_names=False)
            self.assertFalse(records[0]["reads"])
            self.assertFalse(records[0]["mono_pin_ocr"]["accepted"])


if __name__ == "__main__":
    unittest.main()
