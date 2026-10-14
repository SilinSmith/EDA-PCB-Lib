"""User rules for first-node centres, short spans, and shared colour pixels."""
import unittest

import cv2
import numpy as np

import test_passive_pin_short_wires as short
import test_passive_pin_raster_junction as raster
from stage1.passive_pins import refine_pins
from stage2 import assemble


class UserCoordinateRulesTests(unittest.TestCase):
    def pair(self, mixed=False):
        image, comps, old = short.ShortWireTests().pair()
        image[117:120, 100] = (0, 0, 200)
        if mixed:
            image[116, 100] = (90, 130, 138)
        return image, comps, old

    def test_one_green_pixel_is_the_same_coordinate_for_both_pins(self):
        image, comps, old = self.pair()
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 116)])
        self.assertEqual(result['b'], [(100, 116), (100, 180.5)])
        self.assertEqual(report['a']['sources'][1], 'shared_single_green_pixel')
        self.assertEqual(report['b']['sources'][0], 'shared_single_green_pixel')

    def test_observed_mixed_green_pixel_is_shared_without_needing_pure_green(self):
        image, comps, old = self.pair(mixed=True)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'][-1], result['b'][0])
        self.assertEqual(result['a'][-1], (100, 116))
        self.assertEqual(report['a']['connection_evidence'][1]['marker_color'], 'red_green_blend')

    def test_two_axial_green_pixels_keep_two_separate_boundaries(self):
        image, comps, old = self.pair()
        image[117, 100] = (0, 140, 0)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'][-1], (100, 115.5))
        self.assertEqual(result['b'][0], (100, 117.5))
        self.assertNotIn('shared_single_green_pixel', report['a']['sources'])

    def test_white_and_grey_between_pins_are_not_shared_colour_pixels(self):
        for color in ((255, 255, 255), (120, 120, 120)):
            image, comps, old = self.pair()
            image[116, 100] = color
            result, report = refine_pins(comps, old, image)
            self.assertEqual(result['a'], [(100, 64.5)])
            self.assertNotIn('shared_single_green_pixel', report['a']['sources'])

    def test_shared_pixel_requires_distinct_peer_detection(self):
        image, comps, old = self.pair()
        _, report = refine_pins(comps[:1], old, image)
        self.assertNotIn('shared_single_green_pixel', report['a']['sources'])

    def test_shared_pixel_needs_a_continuous_path_all_the_way_to_its_own_body(self):
        image, comps, old = self.pair(mixed=True)
        image[106, 99:102] = 255
        _, report = refine_pins(comps, old, image)
        self.assertNotIn('shared_single_green_pixel', report['a']['sources'])

    def test_wide_capacitor_box_does_not_override_its_observed_vertical_pins(self):
        image, comps, old = self.pair(mixed=True)
        image[80:101, 96:105] = 255
        cv2.line(image, (87, 80), (113, 80), (0, 0, 200), 1)
        cv2.line(image, (87, 100), (113, 100), (0, 0, 200), 1)
        comps[0].update(type='c', bbox_img=[86, 80, 114, 100], bbox_raw_img=[86, 80, 114, 100])
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 116)])
        self.assertTrue(report['a']['axis_y'])

    def test_two_red_contact_fragments_publish_one_boundary_after_a_mixed_pixel(self):
        image, comps, old = short.ShortWireTests().pair()
        image[114, 100] = (90, 130, 138)
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])

    def test_one_pixel_contact_with_short_red_stems_still_is_shared(self):
        image, comps, old = self.pair()
        comps[0]['bbox_img'] = comps[0]['bbox_raw_img'] = [96, 80, 104, 115]
        cv2.line(image, (96, 100), (96, 115), (0, 0, 200), 1)
        cv2.line(image, (104, 100), (104, 115), (0, 0, 200), 1)
        cv2.line(image, (96, 115), (104, 115), (0, 0, 200), 1)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'][-1], (100, 116))
        self.assertEqual(result['b'][0], (100, 116))

    def test_shared_pixel_and_node_rules_survive_rotation(self):
        image, comps, old = self.pair(mixed=True)
        rotate = lambda point: (point[1], 239 - point[0])
        for comp in comps:
            corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
            comp['bbox_img'] = comp['bbox_raw_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        result, _ = refine_pins(comps, {cid: list(map(rotate, points)) for cid, points in old.items()}, np.rot90(image))
        self.assertIn(rotate((100, 116)), result['a'])
        self.assertIn(rotate((100, 116)), result['b'])

    def test_direct_red_lead_uses_node_centre(self):
        image, comp, old = raster.RasterAndJunctionTests().junction()
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'][-1], (100, 125))
        self.assertEqual(report['r']['sources'][-1], 'first_red_junction_center')

    def test_offset_led_node_keeps_its_independently_observed_opposite_stem(self):
        image, comp, old = raster.RasterAndJunctionTests().junction()
        image[20:75, 100] = 255
        comp.update(type='led', bbox_img=[90, 90, 98, 110], bbox_raw_img=[90, 90, 98, 110])
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 125)])
        self.assertEqual(report['r']['sources'], ['first_red_junction_center'])
        self.assertIn('opposed_red_lead_img', report['r']['connection_evidence'][0])

    def test_first_of_two_nodes_on_one_red_lead_wins(self):
        image, comp, old = raster.RasterAndJunctionTests().junction(radius=3)
        cv2.line(image, (100, 126), (100, 147), (0, 0, 200), 1)
        cv2.circle(image, (100, 150), 3, (0, 0, 200), -1)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'][-1], (100, 125))

    def test_a_blue_wire_can_confirm_the_shape_of_a_red_junction(self):
        image, comp, old = raster.RasterAndJunctionTests().junction()
        image[np.all(image == (0, 140, 0), axis=2)] = (200, 0, 0)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 74.5), (100, 125)])
        self.assertEqual(report['r']['sources'], ['lead_wire_colour_boundary', 'first_red_junction_center'])

    def test_a_thick_power_triangle_tip_is_not_a_wire_junction(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        cv2.polylines(image, [np.array(((100, 60), (86, 90), (114, 90)))], True, (0, 0, 200), 3)
        cv2.line(image, (100, 90), (100, 135), (0, 0, 200), 2)
        comp = dict(id='v', type='v', bbox_img=[96, 68, 104, 80], bbox_raw_img=[96, 68, 104, 80])
        _, report = refine_pins([comp], {'v': [(100, 135)]}, image)
        self.assertNotIn('first_red_junction_center', report.get('v', {}).get('sources', []))

    def test_a_digit_overlapping_a_stem_does_not_turn_into_a_junction(self):
        image, comps, old = self.ground_pair()
        image[20:150, :] = 255
        cv2.line(image, (100, 90), (100, 150), (0, 0, 200), 2)
        cv2.putText(image, '2', (89, 119), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 200), 2)
        _, report = refine_pins(comps[1:], old, image)
        self.assertNotIn('first_red_junction_center', report.get('g', {}).get('sources', []))

    def test_thick_triangle_base_and_pin_t_joint_are_not_a_filled_node(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        cv2.polylines(image, [np.array(((101, 86), (126, 101), (101, 116)))], True, (0, 0, 200), 2)
        image[100:103, 70:103] = (0, 0, 200)
        image[100:103, 20:70] = (0, 140, 0)
        image[100:103, 126:146] = (0, 0, 200)
        image[100:103, 146:200] = (0, 140, 0)
        comp = dict(id='d', type='d', bbox_img=[103, 95, 124, 112], bbox_raw_img=[103, 95, 124, 112])
        result, report = refine_pins([comp], {'d': [(70, 101), (146, 101)]}, image)
        self.assertEqual(result['d'][0], (69.5, 101))
        self.assertNotIn('first_red_junction_center', report['d']['sources'])

    def test_offset_led_box_cannot_borrow_its_triangle_bottom_corner_as_a_node(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        cv2.polylines(image, [np.array(((100, 85), (126, 100), (100, 115)))], True, (0, 0, 200), 2)
        cv2.line(image, (70, 100), (100, 100), (0, 0, 200), 2)
        cv2.line(image, (20, 100), (69, 100), (0, 140, 0), 2)
        cv2.line(image, (126, 100), (146, 100), (0, 0, 200), 2)
        cv2.line(image, (147, 100), (200, 100), (0, 140, 0), 2)
        comp = dict(id='led', type='led', bbox_img=[103, 100, 127, 118], bbox_raw_img=[103, 100, 127, 118])
        _, report = refine_pins([comp], {'led': [(70, 100), (147, 100)]}, image)
        self.assertEqual(report['led']['sources'], ['body_attached_red_green_transition'] * 2)

    def test_earlier_red_green_boundary_has_priority_over_a_later_node(self):
        image, comp, old = raster.RasterAndJunctionTests().resistor()
        cv2.circle(image, (100, 155), 3, (0, 0, 200), -1)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'][-1], (100, 125.5))
        self.assertEqual(report['r']['sources'][-1], 'body_attached_red_green_transition')

    def test_an_earlier_green_pixel_on_a_thick_red_stem_still_precedes_the_node(self):
        image, comp, old = raster.RasterAndJunctionTests().resistor()
        image[110:141, 99:102] = (0, 0, 200)
        image[120, 100] = (0, 140, 0)
        cv2.circle(image, (100, 140), 3, (0, 0, 200), -1)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['r'][-1], (100, 119.5))
        self.assertNotIn('first_red_junction_center', report['r']['sources'])
        rotate = lambda point: (point[1], image.shape[1] - 1 - point[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp = dict(comp, bbox_img=[*corners.min(axis=0), *corners.max(axis=0)])
        comp['bbox_raw_img'] = comp['bbox_img']
        rotated, rotated_report = refine_pins([comp], {'r': list(map(rotate, old['r']))}, np.rot90(image))
        self.assertIn(rotate((100, 119.5)), rotated['r'])
        self.assertNotIn('first_red_junction_center', rotated_report['r']['sources'])

    def ground_pair(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        cv2.rectangle(image, (96, 85), (104, 105), (0, 0, 200), 1)
        cv2.line(image, (100, 65), (100, 85), (0, 0, 200), 1)
        cv2.line(image, (100, 20), (100, 64), (0, 140, 0), 1)
        cv2.line(image, (100, 105), (100, 150), (0, 0, 200), 1)
        cv2.line(image, (20, 135), (180, 135), (0, 140, 0), 1)
        cv2.circle(image, (100, 135), 3, (0, 0, 200), -1)
        for y, left, right in ((150, 90, 110), (155, 94, 106), (160, 97, 103)):
            cv2.line(image, (left, y), (right, y), (0, 0, 200), 1)
        comps = [dict(id='r', type='r', bbox_img=[96, 85, 104, 105], bbox_raw_img=[96, 85, 104, 105]),
                 dict(id='g', type='gnd', bbox_img=[88, 130, 112, 163], bbox_raw_img=[88, 130, 112, 163])]
        return image, comps, {'r': [(100, 70), (100, 120)], 'g': [(100, 129), (100, 170)]}

    def test_ground_loose_box_and_resistor_share_the_actual_node_centre(self):
        image, comps, old = self.ground_pair()
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['g'], [(100, 135)])
        self.assertEqual(result['r'][-1], (100, 135))
        self.assertEqual(report['g']['sources'], ['first_red_junction_center'])

    def test_unrelated_green_wire_below_ground_bars_does_not_replace_node(self):
        image, comps, old = self.ground_pair()
        cv2.line(image, (20, 168), (180, 168), (0, 140, 0), 1)
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['g'], [(100, 135)])

    def test_white_break_on_red_stem_prevents_borrowing_a_node(self):
        image, comps, old = self.ground_pair()
        image[143, 99:102] = 255
        result, report = refine_pins(comps, old, image)
        self.assertNotIn('first_red_junction_center', report.get('g', {}).get('sources', []))
        self.assertNotEqual(result['g'], [(100, 135)])

    def test_new_coordinate_sources_are_preserved_by_final_assembly(self):
        points = [(50, 45), (50, 95)]
        stage = {'passive_pin_report': {'p': dict(type='r', after=points,
                 coordinate_rule='image_color_terminal_rules',
                 sources=['first_red_junction_center', 'shared_single_green_pixel'])}}
        self.assertEqual(assemble._measured_passive_points(stage, 'p', 'r', points), dict(enumerate(points)))
        self.assertEqual(assemble._measured_passive_points(stage, 'p', 'box', points), {})
        self.assertEqual(assemble._measured_passive_points(stage, 'p', 'r', [(50, 46), (50, 95)]), {})

    def test_monochrome_does_not_gain_colour_coordinate_records(self):
        image, comps, old = self.ground_pair()
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})


if __name__ == '__main__':
    unittest.main()
