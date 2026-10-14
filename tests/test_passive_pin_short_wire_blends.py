"""Actual mixed pixels may join a short wire to its peer; white gaps cannot."""
import unittest

import numpy as np

import test_passive_pin_short_wires as fixtures
from stage1.passive_pins import refine_pins


class ShortWireBlendTests(unittest.TestCase):
    def pair(self):
        return fixtures.ShortWireTests().pair()

    def test_observed_mixed_pixel_connects_the_short_wire_to_its_peer(self):
        image, comps, old = self.pair()
        image[119, 100] = (30, 130, 134)
        result, report = refine_pins(comps, old, image)
        np.testing.assert_allclose(result['a'], [(100, 64.5), (100, 115.5)])
        proof = report['a']['connection_evidence'][1]
        self.assertEqual(proof['blend_pixels_img'], [[100, 119]])
        self.assertEqual(proof['peer_id'], 'b')

    def test_three_mixed_pixels_are_observed_not_filled_in(self):
        image, comps, old = self.pair()
        image[118:121, 100] = (30, 130, 134)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])
        self.assertEqual(len(report['a']['connection_evidence'][1]['blend_pixels_img']), 3)

    def test_white_at_peer_facing_boundary_is_not_called_a_blend(self):
        image, comps, old = self.pair()
        image[119, 100] = 255
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5)])

    def test_grey_pixel_at_peer_facing_boundary_is_not_a_red_green_blend(self):
        image, comps, old = self.pair()
        image[119, 100] = (130, 130, 130)
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5)])

    def test_four_mixed_pixels_exceed_the_observed_transition_limit(self):
        image, comps, old = self.pair()
        image[118:122, 100] = (30, 130, 134)
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5)])

    def test_mixture_inside_the_green_wire_does_not_bridge_a_break(self):
        image, comps, old = self.pair()
        image[117, 100] = (30, 130, 134)
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5)])

    def test_mixed_peer_transition_survives_a_quarter_turn(self):
        image, comps, old = self.pair()
        image[119, 100] = (30, 130, 134)
        rotate = lambda point: (point[1], 239-point[0])
        for comp in comps:
            corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
            comp['bbox_img'] = comp['bbox_raw_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        result, _ = refine_pins(comps, {cid:list(map(rotate, points)) for cid,points in old.items()}, np.rot90(image))
        np.testing.assert_allclose(sorted(result['a']), sorted(map(rotate, [(100, 64.5), (100, 115.5)])))


if __name__ == '__main__':
    unittest.main()
