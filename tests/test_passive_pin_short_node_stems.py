"""Raster junctions and thick ground bars must not hide real short stems."""
import copy
import unittest

import cv2
import numpy as np

from stage1 import passive_pins as p, passive_terminal_geometry as g

RED, BLUE, GREEN = (0, 0, 190), (180, 0, 0), (0, 140, 0)


def component(cid, typ, box):
    return dict(id=cid, type=typ, pending=False, bbox_img=box, bbox_raw_img=box)


def short_node_scene():
    image = np.full((170, 240, 3), 255, np.uint8)
    cv2.rectangle(image, (92, 86), (112, 94), RED, 1)
    cv2.line(image, (65, 90), (91, 90), RED, 1)
    cv2.line(image, (25, 90), (64, 90), GREEN, 1)
    cv2.line(image, (113, 90), (117, 90), RED, 1)
    image[89:93, 119:122] = BLUE
    image[93, 120] = BLUE  # A raster fringe does not change the junction role.
    image[90:92, 118] = (150, 65, 115)
    cv2.line(image, (122, 90), (195, 90), GREEN, 1)
    cv2.line(image, (120, 94), (120, 125), GREEN, 1)
    comp = component('r', 'r', [90, 84, 114, 96])
    return image, [comp], {'r': [(65, 90), (119, 90)]}


class ShortNodeStemTests(unittest.TestCase):
    def test_peer_text_cannot_hide_a_two_cell_mixed_span(self):
        image, comps, pins = short_node_scene()
        image[88:128, 118:197] = 255
        image[90, 118:120] = (30, 90, 95)
        cv2.line(image, (120, 90), (180, 90), RED, 1)
        image[84:90, 121:126] = 0
        result, report = p.refine_pins(comps, pins, image)
        self.assertEqual(result['r'][-1], (117.5, 90.0))
        self.assertEqual(report['r']['sources'][-1], 'mixed_colour_span_boundary')

    def test_tinted_ground_bar_fringe_does_not_hide_the_actual_stem(self):
        kinds = np.zeros((160, 180), np.uint8)
        kinds[100:104, 50:71] = 1
        kinds[99, 51:70] = 2
        kinds[80:99, 60] = 6
        geo = dict(axis_y=True, low=101, high=102, tangent=60, only_side=-1,
                   kind='tapering_ground_bars', body_bands_img=[dict(first=100, last=103, width=21)])
        self.assertEqual([s['pos'] for s in g._trace(kinds, geo, -1, 4)], [98, 97, 96, 95])
        kinds[99] = 0
        self.assertEqual(g._trace(kinds, geo, -1, 4), [])

    def test_differently_coloured_text_does_not_hide_the_ground_wire_boundary(self):
        image = np.full((170, 240, 3), 255, np.uint8)
        for y, half in [(124, 10), (128, 6), (132, 2)]:
            cv2.line(image, (100 - half, y), (100 + half, y), RED, 1)
        cv2.line(image, (100, 106), (100, 123), (130, 0, 130), 1)
        cv2.line(image, (100, 50), (100, 105), RED, 1)
        image[100:105, 101:112] = BLUE
        ground = component('g', 'gnd', [88, 122, 112, 135])
        result, report = p.refine_pins([ground], {'g': [(100, 109)]}, image)
        self.assertEqual(result['g'], [(100.0, 105.5)])
        self.assertEqual(report['g']['sources'], ['lead_wire_colour_boundary'])
        proof = report['g']['connection_evidence'][0]
        self.assertEqual(proof['lead_pigment'], 'purple')
        self.assertTrue(all(tuple(image[y, x]) == RED for x, y in proof['wire_pixels_img']))

    def test_thick_ground_starts_outside_the_longest_bar(self):
        kinds = np.zeros((140, 180), np.uint8)
        kinds[40:61, 100:104] = 1
        kinds[50, 104:123] = 1
        geo = dict(axis_y=False, low=101, high=102, tangent=50, only_side=1,
                   kind='tapering_ground_bars', body_bands_img=[
                       dict(first=100, last=103, width=21),
                       dict(first=107, last=108, width=13),
                       dict(first=112, last=113, width=5)])
        trace = g._trace(kinds, geo, 1, 4)
        self.assertEqual([s['pos'] for s in trace], [104, 105, 106, 107])

    def test_short_lead_reaches_a_blue_node_through_its_painted_fringe(self):
        image, comps, pins = short_node_scene()
        result, report = p.refine_pins(comps, pins, image)
        self.assertEqual(report['r']['sources'][-1], 'first_filled_junction_center')
        self.assertAlmostEqual(result['r'][-1][0], 120, delta=.01)
        self.assertAlmostEqual(result['r'][-1][1], 90.5, delta=.01)

    def test_short_node_search_does_not_cross_a_white_cell(self):
        image, comps, pins = short_node_scene()
        image[90, 116] = 255
        _, report = p.refine_pins(comps, pins, image)
        self.assertNotIn('first_filled_junction_center', report.get('r', {}).get('sources', []))

    def test_an_earlier_wire_colour_still_owns_the_terminal(self):
        image, comps, pins = short_node_scene()
        image[90, 115:119] = GREEN
        result, report = p.refine_pins(comps, pins, image)
        self.assertNotIn('first_filled_junction_center', report.get('r', {}).get('sources', []))
        self.assertNotEqual(result['r'][-1][0], 120)

    def test_a_blue_stroke_without_an_external_wire_is_not_a_junction(self):
        image, comps, pins = short_node_scene()
        image[88:128, 122:197] = 255
        image[94:128, 119:122] = 255
        _, report = p.refine_pins(comps, pins, image)
        self.assertNotIn('first_filled_junction_center', report.get('r', {}).get('sources', []))

    def test_a_monochrome_stem_with_coloured_text_remains_deferred(self):
        image, comps, pins = short_node_scene()
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cv2.putText(image, 'R1', (90, 65), cv2.FONT_HERSHEY_SIMPLEX, .5, BLUE, 1)
        self.assertEqual(p.refine_pins(comps, pins, image), (pins, {}))

    def test_refinement_keeps_original_inputs(self):
        image, comps, pins = short_node_scene()
        original = copy.deepcopy((image, comps, pins))
        p.refine_pins(comps, pins, image)
        np.testing.assert_array_equal(image, original[0])
        self.assertEqual(comps, original[1])
        self.assertEqual(pins, original[2])


if __name__ == '__main__':
    unittest.main()
