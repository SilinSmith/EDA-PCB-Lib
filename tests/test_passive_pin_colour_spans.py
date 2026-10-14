"""Colour evidence, separate span boundaries and observed peer-body entry."""
import unittest

import cv2
import numpy as np

import test_passive_pin_short_wires as short
from stage1.passive_pins import refine_pins
from stage2 import assemble


class ColourSpanTests(unittest.TestCase):
    def pair(self, color=(90, 130, 138), length=2):
        image, comps, old = short.ShortWireTests().pair()
        image[116:120, 100] = (0, 0, 200)
        image[116:116 + length, 100] = color
        return image, comps, old

    def test_two_mixed_pixels_use_the_boundary_next_to_each_lead(self):
        image, comps, old = self.pair()
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'][-1], (100, 115.5))
        self.assertEqual(result['b'][0], (100, 117.5))
        self.assertEqual(report['a']['sources'][-1], 'mixed_colour_span_boundary')
        self.assertEqual(report['b']['sources'][0], 'mixed_colour_span_boundary')
        self.assertTrue(report['a']['connection_evidence'][-1]['marker_axis_y'])
        self.assertEqual(report['a']['connection_evidence'][-1]['outward_direction_img'], [0, 1])
        self.assertEqual(report['b']['connection_evidence'][0]['outward_direction_img'], [0, -1])

    def test_red_dominant_mark_is_confirmed_by_colour_contrast_and_both_bodies(self):
        image, comps, old = self.pair((90, 110, 158))
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'][-1], (100, 115.5))
        self.assertEqual(result['b'][0], (100, 117.5))
        self.assertEqual(report['a']['connection_evidence'][-1]['marker_color'], 'red_dominant_blend')

    def test_missing_peer_cannot_confirm_a_red_dominant_mark(self):
        image, comps, old = self.pair((90, 110, 158))
        _, report = refine_pins(comps[:1], old, image)
        self.assertNotIn('mixed_colour_span_boundary', report['a']['sources'])

    def test_green_and_mixed_cells_across_one_row_are_one_shared_marker(self):
        image, comps, old = self.pair(length=1)
        image[101:140, 101] = (0, 0, 200)
        image[116, 100] = (0, 140, 0)
        image[116, 101] = (90, 130, 138)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'][-1], (100.5, 116))
        self.assertEqual(result['b'][0], (100.5, 116))
        for record in report.values():
            self.assertEqual(len(record['after']), 2)

    def test_white_break_on_either_facing_lead_prevents_span_confirmation(self):
        for gap in (110, 130):
            with self.subTest(gap=gap):
                image, comps, old = self.pair((90, 110, 158))
                image[gap, 99:102] = 255
                _, report = refine_pins(comps, old, image)
                for record in report.values():
                    self.assertNotIn('mixed_colour_span_boundary', record['sources'])

    def test_fractional_peer_edge_stops_at_the_observed_body_cell(self):
        image, comps, old = self.pair(length=1)
        comps[1]['bbox_img'][1] = comps[1]['bbox_raw_img'][1] = 140.13
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'][-1], (100, 116))
        self.assertEqual(report['a']['connection_evidence'][-1]['peer_id'], 'b')

    def test_duplicate_peer_detections_keep_a_mixed_span_ambiguous(self):
        image, comps, old = self.pair()
        comps.append(dict(comps[1], id='duplicate'))
        old['duplicate'] = old['b']
        _, report = refine_pins(comps, old, image)
        self.assertNotIn('mixed_colour_span_boundary', report['a']['sources'])

    def test_unchanging_warm_stroke_does_not_create_a_colour_boundary(self):
        image, comps, old = self.pair((90, 110, 158))
        image[101:140, 100] = (90, 110, 158)
        _, report = refine_pins(comps, old, image)
        for record in report.values():
            self.assertNotIn('mixed_colour_span_boundary', record['sources'])

    def test_span_boundaries_survive_rotation(self):
        image, comps, old = self.pair((90, 110, 158))
        rotate = lambda point: (point[1], 239 - point[0])
        for comp in comps:
            corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
            comp['bbox_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
            comp['bbox_raw_img'] = comp['bbox_img']
        result, _ = refine_pins(comps, {cid: list(map(rotate, pts)) for cid, pts in old.items()}, np.rot90(image))
        self.assertIn(rotate((100, 115.5)), result['a'])
        self.assertIn(rotate((100, 117.5)), result['b'])

    def net_port(self):
        image = np.full((180, 180, 3), 255, np.uint8)
        red = (0, 0, 200)
        cv2.polylines(image, [np.array([(30, 73), (50, 73), (60, 80), (50, 87), (30, 87)])], True, red, 1)
        cv2.rectangle(image, (100, 75), (110, 85), red, 1)
        cv2.line(image, (60, 80), (130, 80), red, 1)
        cv2.line(image, (131, 80), (160, 80), (0, 140, 0), 1)
        image[80, 80:82] = (90, 110, 158)
        comps = [dict(id='f', type='fuse', bbox_img=[100, 75, 110, 85], bbox_raw_img=[100, 75, 110, 85]),
                 dict(id='port', type='net_input', bbox_img=[30, 73, 45, 87], bbox_raw_img=[30, 73, 45, 87])]
        return image, comps, {'f': [(90, 80), (130, 80)], 'port': [(60, 80)]}

    def test_continuous_net_port_outline_can_support_an_inset_peer_box(self):
        image, comps, old = self.net_port()
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['f'][0], (81.5, 80))
        self.assertEqual(report['f']['connection_evidence'][0]['peer_id'], 'port')
        self.assertEqual(result['port'], old['port'])

    def test_net_port_outline_does_not_allow_a_white_break_on_its_lead(self):
        image, comps, old = self.net_port()
        image[:, 65] = 255
        _, report = refine_pins(comps, old, image)
        self.assertNotIn('mixed_colour_span_boundary', report.get('f', {}).get('sources', []))

    def test_assembly_preserves_observed_span_boundaries(self):
        points = [(100, 115.5), (100, 117.5)]
        stage = {'passive_pin_report': {'p': dict(type='r', after=points,
                 coordinate_rule='image_color_terminal_rules', sources=['mixed_colour_span_boundary'] * 2)}}
        self.assertEqual(assemble._measured_passive_points(stage, 'p', 'r', points), dict(enumerate(points)))


if __name__ == '__main__':
    unittest.main()
