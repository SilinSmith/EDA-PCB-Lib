"""Short wires reaching red junctions need independent observed green arms."""
import unittest

import cv2
import numpy as np

from stage1.passive_pins import refine_pins


class JunctionWireTests(unittest.TestCase):
    def circuit(self, branch=True):
        image = np.full((240, 240, 3), 255, np.uint8)
        cv2.rectangle(image, (90, 96), (110, 104), (0, 0, 200), 1)
        cv2.line(image, (75, 100), (90, 100), (0, 0, 200), 1)
        cv2.line(image, (110, 100), (125, 100), (0, 0, 200), 1)
        cv2.line(image, (20, 100), (74, 100), (0, 140, 0), 1)
        cv2.line(image, (126, 100), (129, 100), (0, 140, 0), 1)
        cv2.circle(image, (133, 100), 3, (0, 0, 200), -1)
        cv2.line(image, (137, 100), (220, 100), (0, 140, 0), 1)
        if branch:
            cv2.line(image, (133, 20), (133, 96), (0, 140, 0), 1)
        comp = dict(id='r', type='r', bbox_img=[90, 96, 110, 104], bbox_raw_img=[90, 96, 110, 104])
        return image, comp, {'r': [(80, 100), (120, 100)]}

    def test_short_wire_to_observed_red_node_retains_own_boundary(self):
        image, comp, old = self.circuit()
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(74.5, 100), (125.5, 100)])
        proof = report['r']['connection_evidence'][1]
        self.assertEqual(proof['kind'], 'short_green_wire_to_red_junction')
        self.assertEqual(len(proof['junction_wire_arms']), 2)
        np.testing.assert_allclose(proof['wire_junction_img'], (133, 100))

    def test_one_outgoing_wire_does_not_turn_a_red_patch_into_a_junction(self):
        image, comp, old = self.circuit(branch=False)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100), (125.5, 100)])
        self.assertEqual(report['r']['connection_evidence'][1]['kind'], 'short_observed_green_wire')

    def test_white_break_in_the_incoming_green_wire_is_not_bridged(self):
        image, comp, old = self.circuit()
        image[100, 128] = 255
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100)])

    def test_detached_transverse_wire_does_not_supply_an_arm(self):
        image, comp, old = self.circuit()
        image[95:97, 133] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100), (125.5, 100)])
        self.assertNotIn('junction_wire_arms', report['r']['connection_evidence'][1])

    def test_thin_red_cross_is_not_a_filled_node(self):
        image, comp, old = self.circuit()
        image[97:104, 130:137] = 255
        cv2.line(image, (130, 100), (136, 100), (0, 0, 200), 1)
        cv2.line(image, (133, 97), (133, 103), (0, 0, 200), 1)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100), (125.5, 100)])
        self.assertNotIn('red_junction_img', report['r']['connection_evidence'][1])

    def test_nc_cross_is_not_a_short_wire_even_with_a_red_node_beyond_it(self):
        image, comp, old = self.circuit()
        image[96:105, 124:130] = 255
        cv2.line(image, (123, 96), (131, 104), (0, 140, 0), 1)
        cv2.line(image, (123, 104), (131, 96), (0, 140, 0), 1)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(74.5, 100)])

    def test_junction_evidence_survives_a_quarter_turn(self):
        image, comp, old = self.circuit()
        rotate = lambda point: (point[1], 239-point[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp['bbox_img'] = comp['bbox_raw_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        result, _ = refine_pins([comp], {'r': list(map(rotate, old['r']))}, np.rot90(image))
        np.testing.assert_allclose(sorted(result['r']), sorted(map(rotate, [(74.5, 100), (125.5, 100)])))


if __name__ == '__main__':
    unittest.main()
