"""Inductor terminals enter a continuous curve; a white break is not a lead."""
import unittest

import cv2
import numpy as np

from stage1.passive_pins import refine_pins


class CurvedInductorLeadTests(unittest.TestCase):
    def coil(self):
        image = np.full((220, 220, 3), 255, np.uint8)
        for cx in (95, 105, 115, 125):
            cv2.ellipse(image, (cx, 100), (5, 5), 0, 180, 360, (0, 0, 200), 1)
        cv2.line(image, (88, 100), (90, 100), (0, 0, 200), 1)
        cv2.line(image, (130, 100), (132, 100), (0, 0, 200), 1)
        cv2.line(image, (50, 100), (87, 100), (0, 140, 0), 1)
        cv2.line(image, (133, 100), (180, 100), (0, 140, 0), 1)
        comp = dict(id='l', type='l', bbox_img=[90, 94, 130, 102], bbox_raw_img=[90, 94, 130, 102])
        return image, comp, {'l': [(85, 98), (135, 98)]}

    def test_short_stems_enter_the_observed_coil_at_both_ends(self):
        image, comp, old = self.coil()
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['l'], [(87.5, 100), (132.5, 100)])
        self.assertTrue(all(r['red_lead']['kind'] == 'curved_red_body_path'
                            for r in report['l']['connection_evidence']))

    def test_curved_path_cannot_cross_a_white_column(self):
        image, comp, old = self.coil()
        image[:, 93] = 255
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['l'], [(132.5, 100)])

    def test_other_component_types_keep_their_straight_lead_requirement(self):
        image, comp, old = self.coil()
        comp['type'] = 'r'
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_curved_lead_survives_a_quarter_turn(self):
        image, comp, old = self.coil()
        rotate = lambda p: (p[1], 219 - p[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp.update(bbox_img=[*corners.min(axis=0), *corners.max(axis=0)])
        comp['bbox_raw_img'] = comp['bbox_img']
        result, _ = refine_pins([comp], {'l': [rotate(p) for p in old['l']]}, np.rot90(image))
        np.testing.assert_allclose(result['l'], sorted([rotate((87.5, 100)), rotate((132.5, 100))], key=lambda p:p[1]))

    def test_monochrome_coil_keeps_previous_candidates(self):
        image, comp, old = self.coil()
        grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        result, report = refine_pins([comp], old, grey)
        self.assertEqual(result, old)
        self.assertEqual(report, {})


if __name__ == '__main__':
    unittest.main()
