import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from stage2.pin_text_crops import pin_text_boxes, read_pin_texts


class PinTextCropTests(unittest.TestCase):
    def fixture(self, edge="bottom"):
        image = np.full((130, 100, 3), 255, np.uint8)
        # Two separated rotated glyphs, not an existing OCR detection box.
        image[58:65, 36:44] = 0
        image[69:78, 36:44] = 0
        image[18:87, 30] = [20, 20, 160]  # colored frame
        image[87, 30:71] = [20, 20, 160]
        image[93:102, 38] = [20, 20, 160]  # colored pin must disappear
        image[93:97, 32:36] = 0  # actual external rotated number
        image[43:55, 76:91] = [160, 20, 20]  # external blue part name
        point, anchor = [38, 109], [38, 87]
        if edge == "top":
            image = image[::-1].copy()
            point, anchor = [38, 20], [38, 42]
        frame = [30, 42 if edge == "top" else 18, 71, 112 if edge == "top" else 87]
        doc = {"components": [{"id": "ic", "type": "box", "bbox_img": frame}],
               "pins": {"ic": [point]}, "pin_evidence": {"ic": [{
                   "source": "red_green_transition", "trust": "trusted",
                   "frame_confirmed": True, "point_img": point,
                   "edge": edge, "anchor_img": anchor}]}}
        return image, doc

    def test_whole_vertical_word_is_recovered_without_existing_text_boxes(self):
        image, doc = self.fixture()
        boxes, gray = pin_text_boxes(doc, image)
        names = [r for r in boxes if r["role"] == "pin_name"]
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0]["bbox"], [36, 58, 44, 78])
        self.assertEqual(names[0]["pin_target"]["pin_index"], 0)
        self.assertTrue(np.all(gray[18:87, 30] == 255))
        self.assertTrue(np.all(gray[43:55, 76:91] == 255))

    def test_top_edge_word_stays_inside_confirmed_frame(self):
        image, doc = self.fixture("top")
        boxes, _ = pin_text_boxes(doc, image)
        name = next(r for r in boxes if r["role"] == "pin_name")
        number = next(r for r in boxes if r["role"] == "pin_number")
        self.assertGreater(name["bbox"][1], 42)
        self.assertLess(number["bbox"][3], 42)

    def test_roles_original_geometry_and_reading_direction_are_separate(self):
        image, doc = self.fixture()
        original = [{"bbox": [36, 69, 44, 78], "reads": {"pin_name": ("D", .8)}}]
        before = copy.deepcopy(original)
        recognizer = Mock()
        recognizer.read_many.return_value = [("LINC", .3), ("GND", .99),
                                               ("7", .99), ("4", .96)]
        extra = read_pin_texts(doc, original, SimpleNamespace(recognizer=lambda: recognizer), image)
        self.assertEqual(original, before)
        self.assertEqual(extra[0]["bbox"], [36, 58, 44, 78])
        self.assertEqual(extra[0]["reads"], {"pin_name": ("GND", .99)})
        self.assertTrue(extra[0]["name_only"])
        self.assertEqual(extra[1]["reads"], {"pin_number": ("4", .96)})
        self.assertTrue(extra[1]["number_only"])
        self.assertEqual(extra[0]["pin_text_crop"]["rotation"], -90)
        self.assertEqual(recognizer.read_many.call_count, 1)
        atlas, items = recognizer.read_many.call_args.args
        self.assertEqual(len(items), 4)
        # Name OCR sees a wider than tall, padded and rotated image.
        first = items[0][0]
        self.assertGreater(first[2] - first[0], first[3] - first[1])
        self.assertEqual(atlas[0, 0], 255)

    def test_missing_model_and_unsupported_evidence_add_nothing(self):
        image, doc = self.fixture()
        self.assertEqual(read_pin_texts(doc, [], SimpleNamespace(recognizer=lambda: None), image), [])
        for change in ({"source": "grayscale"}, {"source": []}, {"trust": "uncertain"},
                       {"frame_confirmed": False}, {"point_img": [39, 109]},
                       {"anchor_img": [38, 120]}, {"point_img": [float("nan"), 109]}):
            altered = copy.deepcopy(doc)
            altered["pin_evidence"]["ic"][0].update(change)
            self.assertEqual(pin_text_boxes(altered, image)[0], [])

    def test_weak_rotated_number_cannot_replace_an_existing_numeric_read(self):
        image, doc = self.fixture()
        original = [{"bbox": [32, 93, 36, 97], "reads": {"pin_number": ("3", .58)}}]
        before = copy.deepcopy(original)
        recognizer = Mock()
        recognizer.read_many.return_value = [("TUO", .2), ("OUT", .98),
                                             ("F", .23), ("4", .74)]
        extra = read_pin_texts(doc, original, SimpleNamespace(recognizer=lambda: recognizer), image)
        self.assertEqual(original, before)
        self.assertEqual([r["reads"] for r in extra], [{"pin_name": ("OUT", .98)}])

    def test_horizontal_name_is_not_rotated_by_default(self):
        image, doc = self.fixture()
        image[58:78, 36:44] = 255
        image[70:78, 33:56] = 0
        self.assertEqual(pin_text_boxes(doc, image)[0], [])

    def test_adjacent_pin_lanes_do_not_merge_two_vertical_words(self):
        image, doc = self.fixture()
        image[58:65, 57:65] = 0
        image[69:78, 57:65] = 0
        ev = copy.deepcopy(doc["pin_evidence"]["ic"][0])
        ev.update(point_img=[61, 109], anchor_img=[61, 87])
        doc["pins"]["ic"].append([61, 109])
        doc["pin_evidence"]["ic"].append(ev)
        names = [r for r in pin_text_boxes(doc, image)[0] if r["role"] == "pin_name"]
        self.assertEqual(len(names), 2)
        self.assertLess(names[0]["bbox"][2], names[1]["bbox"][0])

    def test_region_does_not_cross_image_boundary(self):
        image, doc = self.fixture()
        doc["components"][0]["bbox_img"] = [-50, -30, 71, 87]
        for rec in pin_text_boxes(doc, image)[0]:
            x0, y0, x1, y1 = rec["bbox"]
            self.assertTrue(0 <= x0 < x1 <= image.shape[1])
            self.assertTrue(0 <= y0 < y1 <= image.shape[0])

    def test_top_and_bottom_names_can_have_opposite_reading_directions(self):
        image, doc = self.fixture()
        image[25:40, 36:44] = 0
        ev = copy.deepcopy(doc["pin_evidence"]["ic"][0])
        ev.update(point_img=[38, 0], anchor_img=[38, 18], edge="top")
        doc["pins"]["ic"].append([38, 0])
        doc["pin_evidence"]["ic"].append(ev)
        recognizer = Mock()
        recognizer.read_many.return_value = [("LING", .3), ("GND", .99),
                                               ("7", .99), ("4", .96),
                                               ("IN", .98), ("NI", .2)]
        extra = read_pin_texts(doc, [], SimpleNamespace(recognizer=lambda: recognizer), image)
        names = [r for r in extra if r["name_only"]]
        self.assertEqual([r["pin_text_crop"]["rotation"] for r in names], [-90, 90])
        self.assertEqual(extra[1]["reads"], {"pin_number": ("4", .96)})

    def test_opt_in_horizontal_single_letter_does_not_steal_opposite_word(self):
        image, doc = self.fixture()
        image[18:87, 31:71] = 255
        image[34:44, 34:40] = 0
        image[34:44, 56:68] = 0
        doc["pins"]["ic"] = [[10, 35]]
        doc["pin_evidence"]["ic"][0].update(
            edge="left", point_img=[10, 35], anchor_img=[30, 35])
        self.assertEqual(pin_text_boxes(doc, image)[0], [])
        candidates = pin_text_boxes(doc, image, include_horizontal=True)[0]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["bbox"], [34, 34, 40, 44])
        self.assertFalse(candidates[0]["vertical"])

    def test_opt_in_horizontal_baseline_recrop_keeps_trailing_underscore(self):
        image, doc = self.fixture()
        image[18:87, 31:71] = 255
        image[33:42, 34:42] = 0
        image[33:42, 50:58] = 0
        image[42:44, 42:50] = 0  # baseline lower than wire-centered lane
        doc["pins"]["ic"] = [[10, 35], [10, 55]]
        first = doc["pin_evidence"]["ic"][0]
        first.update(edge="left", point_img=[10, 35], anchor_img=[30, 35])
        second = copy.deepcopy(first)
        second.update(point_img=[10, 55], anchor_img=[30, 55])
        doc["pin_evidence"]["ic"].append(second)
        candidates = pin_text_boxes(doc, image, include_horizontal=True)[0]
        self.assertEqual(candidates[0]["bbox"], [34, 33, 58, 44])


if __name__ == "__main__":
    unittest.main()
