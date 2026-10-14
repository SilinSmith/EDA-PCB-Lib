"""Small filled wire nodes need a real core and two independent wire arms."""
import unittest

import cv2
import numpy as np

import test_passive_pin_junction_wires as fixtures
from stage1.passive_pins import refine_pins


class SmallWireNodeTests(unittest.TestCase):
    def circuit(self, branch=True, cross=False):
        image, comp, old = fixtures.JunctionWireTests().circuit(branch=False)
        image[94:107, 126:139] = 255
        cv2.line(image, (126, 100), (127, 100), (0, 140, 0), 1)
        cv2.line(image, (134, 100), (220, 100), (0, 140, 0), 1)
        for y, width in enumerate((2, 4, 6, 6, 4, 2), 98):
            start = 128 + (6-width)//2
            image[y, start:start+width] = (0, 0, 200)
        if cross:
            image[98:104, 128:134] = 255
            image[100:102, 128:134] = (0, 0, 200)
            image[98:104, 130:132] = (0, 0, 200)
        if branch:
            cv2.line(image, (130, 20), (130, 97), (0, 140, 0), 1)
        return image, comp, old

    def test_six_pixel_filled_node_retains_the_component_boundary(self):
        image, comp, old = self.circuit()
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100), (125.5, 100)])
        proof = report['r']['connection_evidence'][1]
        self.assertEqual(proof['kind'], 'short_green_wire_to_red_junction')
        np.testing.assert_allclose(proof['wire_junction_img'], (130.5, 100.5))
        self.assertGreaterEqual(len(proof['junction_wire_arms']), 2)

    def test_two_pixel_thick_cross_with_same_bounds_is_not_a_filled_node(self):
        image, comp, old = self.circuit(cross=True)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100), (125.5, 100)])
        self.assertNotIn('red_junction_img', report['r']['connection_evidence'][1])

    def test_small_filled_node_still_needs_two_other_wire_arms(self):
        image, comp, old = self.circuit(branch=False)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100), (125.5, 100)])
        self.assertNotIn('junction_wire_arms', report['r']['connection_evidence'][1])

    def test_white_break_in_short_wire_is_not_filled(self):
        image, comp, old = self.circuit()
        image[100, 127] = 255
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100)])

    def test_detached_arm_is_not_counted(self):
        image, comp, old = self.circuit()
        image[97, 130] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100), (125.5, 100)])
        self.assertNotIn('junction_wire_arms', report['r']['connection_evidence'][1])

    def test_small_filled_node_survives_a_quarter_turn(self):
        image, comp, old = self.circuit()
        rotate = lambda point: (point[1], 239-point[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp['bbox_img'] = comp['bbox_raw_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        result, _ = refine_pins([comp], {cid:list(map(rotate,points))for cid,points in old.items()}, np.rot90(image))
        np.testing.assert_allclose(sorted(result['r']), sorted(map(rotate, [(74.5, 100), (125.5, 100)])))


if __name__ == '__main__':
    unittest.main()
