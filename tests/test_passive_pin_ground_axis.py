"""Observed ground bars supply a stem body and a normal, not a side wire."""
import unittest

import cv2
import numpy as np

from stage1.passive_pins import refine_pins


class GroundBodyAxisTests(unittest.TestCase):
    def ground_beside_wire(self, stem=True):
        image = np.full((240, 240, 3), 255, np.uint8)
        for x, half_width in ((150, 16), (144, 11), (138, 6)):
            cv2.line(image, (x, 110-half_width), (x, 110+half_width), (0, 0, 200), 1)
        if stem:
            cv2.line(image, (150, 110), (165, 110), (0, 0, 200), 1)
            cv2.line(image, (166, 110), (220, 110), (0, 140, 0), 1)
        cv2.line(image, (40, 91), (164, 91), (0, 140, 0), 1)
        cv2.line(image, (165, 91), (190, 91), (0, 0, 200), 1)
        comp = dict(id='g', type='gnd', bbox_img=[135, 92, 154, 128], bbox_raw_img=[135, 92, 154, 128])
        return image, comp, {'g': [(144, 90)]}

    def test_ground_bar_end_beside_another_wire_is_not_the_pin(self):
        image, comp, old = self.ground_beside_wire()
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['g'], [(165.5, 110)])
        self.assertEqual(report['g']['coordinate_rule'], 'red_green_pixel_boundary')

    def test_ground_without_its_own_wire_keeps_candidates_despite_nearby_wire(self):
        image, comp, old = self.ground_beside_wire(stem=False)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_secondary_ground_bar_end_cannot_borrow_a_parallel_wire(self):
        image, comp, old = self.ground_beside_wire()
        cv2.line(image, (40, 97), (145, 97), (0, 140, 0), 1)
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['g'], [(165.5, 110)])

    def test_ground_axis_and_body_seeds_survive_a_quarter_turn(self):
        image, comp, old = self.ground_beside_wire()
        rotate = lambda point: (point[1], 239-point[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp['bbox_img'] = comp['bbox_raw_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        result, _ = refine_pins([comp], {'g': list(map(rotate, old['g']))}, np.rot90(image))
        np.testing.assert_allclose(result['g'], [rotate((165.5, 110))])

    def test_ground_bars_do_not_reorient_another_component_type(self):
        image, comp, old = self.ground_beside_wire()
        comp['type'] = 'v'
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})


if __name__ == '__main__':
    unittest.main()
