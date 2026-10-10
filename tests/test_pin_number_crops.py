import copy
import unittest

import numpy as np

from stage2.pin_lane_crops import pin_lane_boxes
from stage2.pin_number_crops import number_repair_boxes


class PinNumberCropRepairTests(unittest.TestCase):
    def fixture(self, rows=(61, 80.5, 100)):
        image = np.full((140, 140, 3), 255, np.uint8)
        points = [[x, 125] for x in rows]
        evidence = [{"source": "red_green_transition", "trust": "trusted",
                     "frame_confirmed": True, "point_img": point,
                     "edge": "bottom", "anchor_img": [point[0], 105]}
                    for point in points]
        doc = {"components": [{"id": "ic", "type": "box", "bbox_img": [40, 25, 110, 105]}],
               "pins": {"ic": points}, "pin_evidence": {"ic": evidence}}
        return image, doc

    def target(self, candidates, index=1):
        return next(c for c in candidates if c["pin_target"]["pin_index"] == index)

    def test_one_pixel_clipped_connected_digit_completes_to_white(self):
        image, doc = self.fixture()
        image[110:116, 70:80] = 0
        old = self.target([c for c in pin_lane_boxes(doc, image)[0] if c["role"] == "pin_number"])
        self.assertIn("glyph_integrity_unresolved", old["flags"])
        new = self.target(number_repair_boxes(doc, image)[0])
        self.assertEqual(new["bbox"], [70, 110, 80, 116])
        self.assertNotIn("glyph_integrity_unresolved", new["flags"])
        self.assertEqual(new["pin_target"], old["pin_target"])

    def test_tiny_contact_residue_removed_without_erasing_digit(self):
        image, doc = self.fixture()
        image[110:116, 70:80] = 0
        image[123:130, 79:84] = [0, 160, 0]
        image[124:127, 80:82] = 0
        before = image.copy()
        new, clean = number_repair_boxes(doc, image)
        number = self.target(new)
        self.assertEqual(number["bbox"], [70, 110, 80, 116])
        self.assertEqual(number["number_crop_repair"]["contact_residue_removed"][0]["pixels"], 6)
        self.assertTrue(np.all(clean[124:127, 80:82] == 255))
        np.testing.assert_array_equal(image, before)

    def test_digit_beyond_one_pixel_allowance_remains_unresolved(self):
        image, doc = self.fixture()
        image[110:116, 68:80] = 0
        new = self.target(number_repair_boxes(doc, image)[0])
        self.assertIn("glyph_integrity_unresolved", new["flags"])
        self.assertTrue(new["number_crop_repair"]["completion"]["connected_ink_beyond_allowance"])

    def test_connected_neighbour_digit_is_not_borrowed(self):
        image, doc = self.fixture()
        image[110:116, 60:81] = 0
        new = self.target(number_repair_boxes(doc, image)[0])
        self.assertIn("glyph_integrity_unresolved", new["flags"])
        self.assertGreaterEqual(new["bbox"][0], 71)

    def test_disconnected_number_fragments_away_from_contact_are_retained(self):
        image, doc = self.fixture()
        image[110:116, 75:80] = 0
        image[119:122, 78:80] = 0
        new = self.target(number_repair_boxes(doc, image)[0])
        self.assertEqual(new["bbox"], [75, 110, 80, 122])
        self.assertEqual(new["number_crop_repair"]["contact_residue_removed"], [])

    def test_wider_existing_ownership_does_not_add_an_unselected_word(self):
        image, doc = self.fixture(rows=(60, 80))
        image[110:116, 80:92] = 0
        image[110:116, 97:103] = 0
        new = self.target(number_repair_boxes(doc, image)[0])
        self.assertEqual(new["bbox"], [80, 110, 92, 116])

    def test_long_narrow_digit_at_contact_is_preserved(self):
        image, doc = self.fixture()
        image[110:116, 74:79] = 0
        image[121:128, 80:81] = 0
        image[121:128, 81:84] = [0, 160, 0]
        new, clean = number_repair_boxes(doc, image)
        number = self.target(new)
        self.assertEqual(number["number_crop_repair"]["contact_residue_removed"], [])
        self.assertTrue(np.all(clean[121:127, 80] == 0))

    def test_no_ocr_text_or_input_mutation(self):
        image, doc = self.fixture()
        image[110:116, 70:80] = 0
        reads = [{"bbox": [70, 110, 80, 116], "reads": {"pin_number": ("92", .9)}}]
        original_doc, original_reads = copy.deepcopy(doc), copy.deepcopy(reads)
        first = self.target(number_repair_boxes(doc, image, reads)[0])
        reads[0]["reads"]["pin_number"] = ("6", .01)
        second = self.target(number_repair_boxes(doc, image, reads)[0])
        self.assertEqual(first["bbox"], second["bbox"])
        self.assertEqual(doc, original_doc)
        self.assertEqual(original_reads[0]["reads"]["pin_number"], ("92", .9))

    def test_untrusted_or_monochrome_no_new_number_lanes(self):
        image, doc = self.fixture()
        image[110:116, 70:80] = 0
        for ev in doc["pin_evidence"]["ic"]:
            ev["source"] = "grayscale"
        self.assertEqual(number_repair_boxes(doc, image)[0], [])

    def test_invalid_input_is_safe(self):
        image, doc = self.fixture()
        for bad in (None, image[:, :, 0], np.zeros((2, 2, 4), np.uint8)):
            self.assertEqual(number_repair_boxes(doc, bad), ([], None))


if __name__ == "__main__":
    unittest.main()
