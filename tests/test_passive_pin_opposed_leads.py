"""Offset LED axes require an independent, attached opposite red stem."""
import unittest
import numpy as np
import test_passive_pin_observed_geometry as fixtures
from stage1.passive_pins import refine_pins


class OpposedLEDLeadTests(unittest.TestCase):
    def single_contact(self):
        image, comp, old = fixtures.ObservedColourGeometryTests().offset_led()
        image[126:181, 99:102] = 255
        return image, comp, old

    def test_opposed_connected_red_stem_supports_the_clean_green_contact(self):
        image, comp, old = self.single_contact()
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 74.5)])
        evidence = report['r']['connection_evidence'][0]
        self.assertEqual(evidence['axis_support'], 'opposed_continuous_red_lead')
        self.assertEqual(len(evidence['opposed_red_lead_img']), 6)

    def test_white_gap_inside_the_opposite_stem_is_not_axis_evidence(self):
        image, comp, old = self.single_contact()
        image[113, 99:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_detached_opposite_stem_is_not_axis_evidence(self):
        image, comp, old = self.single_contact()
        image[110, 99:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_one_red_stem_cannot_decide_a_square_detector_axis(self):
        image, comp, old = self.single_contact()
        comp['bbox_raw_img'] = [90, 90, 110, 110]
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_other_types_keep_their_offset_pair_requirement(self):
        image, comp, old = self.single_contact()
        comp['type'] = 'd'
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_opposed_stem_evidence_survives_rotation(self):
        image, comp, old = self.single_contact()
        rotate = lambda p: (p[1], 219 - p[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp.update(bbox_img=[*corners.min(axis=0), *corners.max(axis=0)])
        comp['bbox_raw_img'] = comp['bbox_img']
        result, _ = refine_pins([comp], {'r': [rotate(p) for p in old['r']]}, np.rot90(image))
        np.testing.assert_allclose(result['r'], [rotate((100, 74.5))])


if __name__ == '__main__':
    unittest.main()
