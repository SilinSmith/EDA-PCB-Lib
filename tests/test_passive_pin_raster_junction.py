"""Raster sampling, antialiased bars, and compact junction regressions."""
import unittest

import cv2
import numpy as np

import test_passive_pins as fixtures
import test_passive_pin_observed_geometry as geometry
from stage1.passive_pins import refine_pins


class RasterAndJunctionTests(unittest.TestCase):
    def resistor(self):
        return fixtures.PassivePinTests().resistor()

    def two_pixel_leads(self):
        image, comp, old = self.resistor()
        image[20:91, 99:103] = 255
        image[110:181, 99:103] = 255
        image[75:91, 100:102] = (0, 0, 200)
        image[110:126, 100:102] = (0, 0, 200)
        image[20:75, 100:102] = (0, 140, 0)
        image[126:181, 100:102] = (0, 140, 0)
        return image, comp, old

    def test_half_pixel_centre_samples_both_rows_of_a_two_pixel_lead(self):
        image, comp, old = self.two_pixel_leads()
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100.5, 74.5), (100.5, 125.5)])

    def test_one_pixel_translation_does_not_skip_alternate_raster_rows(self):
        image, comp, old = self.two_pixel_leads()
        first, _ = refine_pins([comp], old, image)
        shifted = np.full_like(image, 255)
        shifted[1:, 1:] = image[:-1, :-1]
        comp = dict(comp, bbox_img=[v + 1 for v in comp['bbox_img']],
                    bbox_raw_img=[v + 1 for v in comp['bbox_raw_img']])
        result, _ = refine_pins([comp], old, shifted)
        np.testing.assert_allclose(result['r'], np.asarray(first['r']) + 1)

    def test_antialiased_rows_of_one_ground_bar_are_grouped_together(self):
        image, comp, old = geometry.ObservedColourGeometryTests().ground()
        cv2.line(image, (90, 105), (110, 105), (0, 0, 200), 1)
        cv2.line(image, (86, 107), (114, 107), (0, 0, 200), 1)
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 74.5)])
        rotate = lambda p: (p[1], 219 - p[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp = dict(comp, bbox_img=[*corners.min(axis=0), *corners.max(axis=0)])
        comp['bbox_raw_img'] = comp['bbox_img']
        result, _ = refine_pins([comp], old, np.rot90(image))
        np.testing.assert_allclose(result['r'], [rotate((100, 74.5))])

    def junction(self, radius=5):
        image, comp, old = self.resistor()
        cv2.circle(image, (100, 125), radius, (0, 0, 200), -1)
        cv2.line(image, (106, 125), (180, 125), (0, 140, 0), 1)
        return image, comp, old

    def test_narrow_ground_stem_does_not_merge_into_the_longest_bar(self):
        image, comp, old = geometry.ObservedColourGeometryTests().ground()
        cv2.line(image, (100, 75), (100, 100), (0, 0, 200), 2)
        comp.update(bbox_img=[92, 106, 108, 121], bbox_raw_img=[92, 106, 108, 121])
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 73.5)])

    def test_direct_red_lead_uses_the_first_filled_junction_centre(self):
        image, comp, old = self.junction()
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(len(result['r']), 2)
        self.assertAlmostEqual(result['r'][1][0], 100, places=1)
        self.assertAlmostEqual(result['r'][1][1], 125, places=1)

    def test_large_filled_patch_is_not_accepted_by_relaxing_contact_area(self):
        image, comp, old = self.resistor()
        cv2.rectangle(image, (90, 125), (110, 145), (0, 0, 200), -1)
        image[146:181, 90:111] = (0, 140, 0)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 74.5)])

    def test_bulb_without_a_continuous_inward_lead_is_not_a_terminal(self):
        image, comp, old = self.junction()
        image[113:119, 99:102] = 255
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 74.5)])


if __name__ == '__main__':
    unittest.main()
