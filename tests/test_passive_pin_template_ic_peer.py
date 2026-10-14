"""Measured non-IC templates can end at an IC without changing its own pins."""
import copy
import unittest

import cv2
import numpy as np

from stage1.passive_pins import refine_pins
from stage1 import passive_terminal_geometry as g, passive_terminal_templates as t

RED, GREEN = (0, 0, 190), (0, 140, 0)


def component(cid, typ, box):
    return dict(id=cid, type=typ, pending=False, bbox_img=list(box), bbox_raw_img=list(box))


def scene():
    image = np.full((280, 280, 3), 255, np.uint8)
    comps, pins = [], {}
    for cid, y in [('target', 100), ('reference', 200)]:
        for x, half in [(200, 10), (204, 6), (208, 2)]:
            cv2.line(image, (x, y - half), (x, y + half), RED, 1)
        comps.append(component(cid, 'gnd', [198, y - 13, 211, y + 13]))
        pins[cid] = [(181, y)]
    cv2.line(image, (135, 100), (199, 100), RED, 1)
    cv2.line(image, (180, 200), (199, 200), RED, 1)
    cv2.line(image, (135, 200), (179, 200), GREEN, 1)
    cv2.rectangle(image, (115, 75), (135, 125), RED, 1)
    comps.append(component('ic', 'box', [114, 74, 136, 126]))
    pins['ic'] = [(136, 100)]
    return image, comps, pins


class TemplateICPeerTests(unittest.TestCase):
    def test_a_far_edge_of_a_direct_node_is_not_a_pin_length_reference(self):
        image = np.full((160, 220, 3), 255, np.uint8)
        cv2.rectangle(image, (96, 86), (116, 94), RED, 1)
        cv2.line(image, (84, 90), (95, 90), RED, 1)
        cv2.line(image, (30, 90), (80, 90), GREEN, 1)
        cv2.circle(image, (84, 90), 3, RED, -1)
        comp = component('r', 'r', [94, 84, 118, 96])
        kinds = g._pigments(image)
        geo = g._axial_body(kinds != 0, comp['bbox_raw_img'], 'r', False)
        bad = dict(point=(80.5, 90), side=-1, source='body_attached_red_green_transition', wire={})
        good = dict(point=(128.5, 90), side=1, source='body_attached_red_green_transition', wire={})
        rows = t._library([comp], kinds, {'r': [geo]}, {'r': [bad, good]})
        self.assertEqual([r['point'] for r in rows], [good['point']])

    def test_rotated_half_pixel_contact_still_visits_every_body_row(self):
        image, comps, pins = scene()
        rotate = lambda pt: (pt[1], 279 - pt[0])
        for comp in comps:
            x0, y0, x1, y1 = comp['bbox_raw_img']
            box = [y0, 279 - x1, y1, 279 - x0]
            comp.update(bbox_img=box, bbox_raw_img=box)
        pins = {cid: [rotate(pt) for pt in points] for cid, points in pins.items()}
        result, report = refine_pins(comps, pins, np.rot90(image))
        self.assertEqual(result['target'], [(100.0, 99.5)])
        self.assertEqual(result['ic'], pins['ic'])
        self.assertEqual(report['target']['sources'], ['same_symbol_terminal_length'])

    def test_ground_length_does_not_require_the_later_wire_to_remain_straight(self):
        image, comps, pins = scene()
        image[70:130, 110:180] = 255
        cv2.line(image, (164, 100), (199, 100), RED, 1)
        cv2.line(image, (164, 100), (164, 160), RED, 1)
        comps = [c for c in comps if c['id'] != 'ic']
        pins.pop('ic')
        result, report = refine_pins(comps, pins, image)
        self.assertEqual(result['target'], [(179.5, 100.0)])
        proof = report['target']['connection_evidence'][0]
        self.assertEqual(proof['connection_validation'], 'observed_own_stem')
        self.assertIsNone(proof['peer_id'])

    def test_reference_ground_length_locates_the_non_ic_end(self):
        image, comps, pins = scene()
        result, report = refine_pins(comps, pins, image)
        self.assertEqual(result['target'], [(179.5, 100.0)])
        self.assertEqual(report['target']['sources'], ['same_symbol_terminal_length'])
        self.assertEqual(report['target']['connection_evidence'][0]['peer_id'], 'ic')
        self.assertEqual(result['ic'], pins['ic'])
        self.assertNotIn('ic', report)

    def test_missing_reference_does_not_invent_a_ground_length(self):
        image, comps, pins = scene()
        cv2.line(image, (135, 200), (199, 200), RED, 1)
        _, report = refine_pins(comps, pins, image)
        self.assertNotIn('same_symbol_terminal_length', report.get('target', {}).get('sources', []))

    def test_white_break_cannot_supply_a_facing_ic_connection(self):
        image, comps, pins = scene()
        image[98:103, 153:157] = 255
        _, report = refine_pins(comps, pins, image)
        self.assertNotIn('same_symbol_terminal_length', report.get('target', {}).get('sources', []))

    def test_a_real_earlier_wire_boundary_keeps_priority(self):
        image, comps, pins = scene()
        cv2.line(image, (135, 100), (189, 100), GREEN, 1)
        result, report = refine_pins(comps, pins, image)
        self.assertEqual(result['target'], [(189.5, 100.0)])
        self.assertNotEqual(report['target']['sources'], ['same_symbol_terminal_length'])
        self.assertEqual(result['ic'], pins['ic'])

    def test_two_overlapping_ic_detections_keep_peer_ownership_ambiguous(self):
        image, comps, pins = scene()
        comps.append(component('ic-duplicate', 'box', [114, 74, 136, 126]))
        pins['ic-duplicate'] = [(137, 100)]
        result, report = refine_pins(comps, pins, image)
        self.assertNotIn('same_symbol_terminal_length', report.get('target', {}).get('sources', []))
        self.assertEqual(result['ic-duplicate'], pins['ic-duplicate'])

    def test_inputs_and_ic_coordinates_are_preserved(self):
        image, comps, pins = scene()
        before = copy.deepcopy((image, comps, pins))
        refine_pins(comps, pins, image)
        np.testing.assert_array_equal(image, before[0])
        self.assertEqual(comps, before[1])
        self.assertEqual(pins, before[2])


if __name__ == '__main__':
    unittest.main()
