"""Mixed-colour raster links must not bridge white, green or blue gaps."""
import unittest

import cv2
import numpy as np

import test_passive_pins as fixtures
from stage1.passive_pins import _blend_bridges, _blended_red_green_pixels, refine_pins


class MixedColourLinkTests(unittest.TestCase):
    def linked_junction(self, gap_colour=(0, 100, 100), gap=3):
        image, comp, old = fixtures.PassivePinTests().resistor()
        image[111:181, 98:104] = 255
        cv2.line(image, (100, 110), (100, 125), (0, 0, 200), 1)
        centre = 131 + gap
        cv2.circle(image, (100, centre), 5, (0, 0, 200), -1)
        image[126:126 + gap, 100] = gap_colour
        cv2.line(image, (100, centre + 6), (100, 180), (0, 140, 0), 1)
        cv2.line(image, (106, centre), (180, centre), (0, 140, 0), 1)
        return image, comp, old, centre

    def test_three_continuous_mixed_pixels_attach_a_red_junction(self):
        image, comp, old, centre = self.linked_junction()
        result, report = refine_pins([comp], old, image)
        self.assertEqual(len(result['r']), 2)
        np.testing.assert_allclose(result['r'][1], (100, centre + 5.5))
        self.assertEqual(report['r']['connection_evidence'][1]['red_body_link']['kind'],
                         'short_red_green_blend')

    def test_white_gap_does_not_attach_the_nearby_red_junction(self):
        image, comp, old, _ = self.linked_junction(gap_colour=(255, 255, 255))
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 74.5)])

    def test_blue_text_pixels_do_not_attach_the_nearby_red_junction(self):
        image, comp, old, _ = self.linked_junction(gap_colour=(200, 0, 0))
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 74.5)])

    def test_pure_green_wire_is_not_promoted_to_red_body(self):
        image, comp, old, centre = self.linked_junction(gap_colour=(0, 140, 0))
        result, _ = refine_pins([comp], old, image)
        self.assertFalse(any(np.linalg.norm(np.asarray(p) - (100, centre + 5.5)) < .1
                             for p in result['r']))
        self.assertFalse(_blended_red_green_pixels(image)[126:129,100].any())

    def test_four_mixed_pixels_are_not_a_local_body_link(self):
        image, comp, old, _ = self.linked_junction(gap=4)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 74.5)])

    def test_mixed_colour_link_survives_quarter_turn(self):
        image, comp, old, centre = self.linked_junction()
        rotate = lambda p: (p[1], 219 - p[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp = dict(comp, bbox_img=[*corners.min(axis=0), *corners.max(axis=0)])
        comp['bbox_raw_img'] = comp['bbox_img']
        result, _ = refine_pins([comp], old, np.rot90(image))
        np.testing.assert_allclose(sorted(result['r']), sorted(map(rotate, [(100,74.5),(100,centre+5.5)])))

    def test_partial_mixed_gap_with_one_white_pixel_stays_disconnected(self):
        red = np.zeros((5,10),bool)
        red[2,2] = red[2,6] = True
        blend = np.zeros_like(red)
        blend[2,3:6] = True
        blend[2,4] = False
        self.assertFalse(_blend_bridges(red, blend).any())


if __name__ == '__main__':
    unittest.main()
