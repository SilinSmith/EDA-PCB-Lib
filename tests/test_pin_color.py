"""Behaviour tests for measured colour endpoints, independent of any GT file."""
import unittest

import cv2
import numpy as np

from stage1.pincolor import extract_color_pins


RED = (20, 25, 115)  # BGR, intentionally dark red like compressed schematic exports.
GREEN = (20, 130, 25)
BOX = [50.0, 50.0, 100.0, 100.0]
COMP = {"id": "ic", "type": "box", "bbox_img": BOX}


def image():
    im = np.full((150, 150, 3), 250, dtype=np.uint8)
    cv2.rectangle(im, (50, 50), (100, 100), RED, 1)
    return im


class TestPinColor(unittest.TestCase):
    def pins(self, im, component=None):
        return extract_color_pins(im, [component or COMP]).get("ic", [])

    def test_all_four_sides_use_measured_colour_transition(self):
        im = image()
        for start, tip, end in [((50, 75), (35, 75), (15, 75)),
                                ((100, 75), (112, 75), (135, 75)),
                                ((75, 50), (75, 32), (75, 12)),
                                ((75, 100), (75, 124), (75, 145))]:
            cv2.line(im, start, tip, RED, 1)
            cv2.line(im, tip, end, GREEN, 1)
        pins = {p["edge"]: p for p in self.pins(im)}
        self.assertEqual(set(pins), {"left", "right", "top", "bottom"})
        for edge, expected in [("left", (35.5, 75)), ("right", (111.5, 75)),
                                ("top", (75, 32.5)), ("bottom", (75, 123.5))]:
            self.assertTrue(pins[edge]["connected"])
            np.testing.assert_allclose(pins[edge]["point_img"], expected, atol=0.01)
            self.assertEqual(pins[edge]["source"], "red_green_transition")

    def test_dark_red_and_antialiased_line(self):
        im = image()
        cv2.line(im, (100, 74), (116, 74), RED, 1, cv2.LINE_AA)
        cv2.line(im, (116, 74), (141, 74), GREEN, 1, cv2.LINE_AA)
        pins = self.pins(im)
        self.assertEqual(len(pins), 1)
        self.assertTrue(pins[0]["connected"])
        np.testing.assert_allclose(pins[0]["point_img"], (116, 74), atol=1.2)

    def test_floating_red_stub_keeps_its_real_tip(self):
        im = image()
        cv2.line(im, (50, 75), (31, 75), RED, 1)
        pins = self.pins(im)
        self.assertEqual(len(pins), 1)
        self.assertFalse(pins[0]["connected"])
        self.assertEqual(pins[0]["point_img"], [31.0, 75.0])
        self.assertEqual(pins[0]["source"], "red_stub_endpoint")

    def test_green_no_connect_cross_is_not_a_wire(self):
        im = image()
        cv2.line(im, (50, 75), (35, 75), RED, 1)
        cv2.line(im, (28, 71), (36, 79), GREEN, 1)
        cv2.line(im, (28, 79), (36, 71), GREEN, 1)
        pins = self.pins(im)
        self.assertEqual(len(pins), 1)
        self.assertFalse(pins[0]["connected"])
        self.assertEqual(pins[0]["point_img"][1], 75.0)

    def test_thick_antialiased_green_cross_cannot_pass_as_six_pixel_wire(self):
        im = image()
        cv2.line(im, (50, 75), (35, 75), RED, 1)
        cv2.line(im, (23, 67), (39, 83), GREEN, 3, cv2.LINE_AA)
        cv2.line(im, (23, 83), (39, 67), GREEN, 3, cv2.LINE_AA)
        pins = self.pins(im)
        self.assertEqual(len(pins), 1)
        self.assertIs(pins[0]["connected"], False)
        self.assertTrue(pins[0]["color_evidence"]["green_marker_near_tip"])

    def test_detached_red_word_cannot_create_a_pin(self):
        im = image()
        cv2.putText(im, "CHIP", (56, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
        self.assertEqual(self.pins(im), [])

    def test_number_above_stub_does_not_shift_or_delete_pin(self):
        im = image()
        cv2.line(im, (100, 75), (115, 75), RED, 1)
        cv2.line(im, (115, 75), (140, 75), GREEN, 1)
        cv2.putText(im, "6", (101, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1)
        pins = self.pins(im)
        self.assertEqual(len(pins), 1)
        self.assertTrue(pins[0]["connected"])
        np.testing.assert_allclose(pins[0]["point_img"], (114.5, 75), atol=0.01)

    def test_bbox_offset_calibrates_to_actual_red_frame(self):
        im = image()
        cv2.line(im, (100, 75), (114, 75), RED, 1)
        cv2.line(im, (114, 75), (140, 75), GREEN, 1)
        component = dict(COMP, bbox_img=[54.8, 50.0, 104.8, 100.0])
        pins = self.pins(im, component)
        self.assertEqual(len(pins), 1)
        np.testing.assert_allclose(pins[0]["anchor_img"], (100, 75), atol=0.01)
        np.testing.assert_allclose(pins[0]["point_img"], (113.5, 75), atol=0.01)

    def test_different_stub_lengths_are_not_forced_to_twelve_pixels(self):
        for tip in (106, 121, 132):
            im = image()
            cv2.line(im, (100, 75), (tip, 75), RED, 1)
            cv2.line(im, (tip, 75), (145, 75), GREEN, 1)
            pin = self.pins(im)[0]
            self.assertEqual(pin["point_img"], [tip - 0.5, 75.0])

    def test_only_box_class_and_confirmed_red_frame_are_eligible(self):
        im = image()
        cv2.line(im, (100, 75), (115, 75), RED, 1)
        cv2.line(im, (115, 75), (140, 75), GREEN, 1)
        self.assertEqual(extract_color_pins(im, [dict(COMP, type="block")]), {})
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        monochrome = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.assertEqual(self.pins(monochrome), [])

    def test_two_adjacent_real_pins_remain_separate(self):
        im = image()
        for y in (72, 78):
            cv2.line(im, (100, y), (112, y), RED, 1)
            cv2.line(im, (112, y), (140, y), GREEN, 1)
        self.assertEqual([p["point_img"][1] for p in self.pins(im)], [72.0, 78.0])

    def test_green_diagonal_touching_stub_is_not_straight_wire(self):
        im = image()
        cv2.line(im, (100, 75), (115, 75), RED, 1)
        cv2.line(im, (116, 75), (135, 94), GREEN, 1)
        pin = self.pins(im)[0]
        self.assertIsNone(pin["connected"])
        self.assertEqual(pin["connection_reason"], "short_green_stroke_connection_uncertain")

    def test_green_wire_that_turns_immediately_is_uncertain_not_disconnected(self):
        im = image()
        cv2.line(im, (100, 75), (115, 75), RED, 1)
        cv2.line(im, (116, 75), (118, 75), GREEN, 1)
        cv2.line(im, (118, 75), (118, 105), GREEN, 1)
        pin = self.pins(im)[0]
        self.assertIsNone(pin["connected"])
        self.assertFalse(pin["color_evidence"]["green_marker_near_tip"])

    def test_frame_report_is_available_when_no_stub_is_found(self):
        frames = {}
        pins = extract_color_pins(image(), [COMP], frame_report=frames)
        self.assertEqual(pins["ic"], [])
        self.assertTrue(frames["ic"]["all_edges_confirmed"])
        self.assertEqual(set(frames["ic"]["confirmed_edges"]), {"top", "right", "bottom", "left"})

    def test_invalid_input_is_ignored_without_fabricating_pins(self):
        self.assertEqual(extract_color_pins(None, [COMP]), {})
        self.assertEqual(extract_color_pins(image(), [dict(COMP, bbox_img=[0, 0, 1, 1])]), {})
        self.assertEqual(extract_color_pins(image(), [dict(COMP, bbox_img=[0, 0, float("nan"), 99])]), {})


if __name__ == "__main__":
    unittest.main()
