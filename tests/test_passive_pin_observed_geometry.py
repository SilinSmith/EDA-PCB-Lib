"""Image-evidence regressions for inset bodies, offset axes and turning wires."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import test_passive_pins as fixtures
from stage1.passive_pins import refine_pins
from stage2 import assemble


class ObservedColourGeometryTests(unittest.TestCase):
    def resistor(self):
        return fixtures.PassivePinTests().resistor()

    def test_opposed_observed_contacts_resolve_a_square_detector_box(self):
        image, comp, old = self.resistor()
        comp['bbox_raw_img'] = [90, 90, 110, 110]
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 74.5), (100, 125.5)])
        self.assertIs(report['r']['axis_y'], True)

    def test_one_contact_does_not_resolve_an_uncertain_axis(self):
        image, comp, old = self.resistor()
        comp['bbox_raw_img'] = [90, 90, 110, 110]
        image[110:181, 99:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_two_observed_axes_remain_ambiguous(self):
        image, comp, old = self.resistor()
        comp.update(bbox_img=[90, 90, 110, 110], bbox_raw_img=[90, 90, 110, 110])
        cv2.line(image, (75, 100), (96, 100), (0, 0, 200), 1)
        cv2.line(image, (104, 100), (125, 100), (0, 0, 200), 1)
        cv2.line(image, (20, 100), (74, 100), (0, 140, 0), 1)
        cv2.line(image, (126, 100), (180, 100), (0, 140, 0), 1)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def inset_outline(self):
        image, comp, old = self.resistor()
        image[:] = 255
        cv2.rectangle(image, (90, 75), (110, 125), (0, 0, 200), 1)
        cv2.line(image, (100, 55), (100, 75), (0, 0, 200), 1)
        cv2.line(image, (100, 125), (100, 145), (0, 0, 200), 1)
        cv2.line(image, (100, 20), (100, 54), (0, 140, 0), 1)
        cv2.line(image, (100, 146), (100, 180), (0, 140, 0), 1)
        return image, comp, old

    def test_enclosing_red_outline_anchors_an_inset_detector_box(self):
        image, comp, old = self.inset_outline()
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 54.5), (100, 145.5)])

    def test_open_outline_does_not_invent_body_containment(self):
        image, comp, old = self.inset_outline()
        image[75:126, 110] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_observed_green_turn_supplies_a_terminal(self):
        image, comp, old = self.resistor()
        image[20:75, 99:102] = 255
        cv2.line(image, (100, 74), (100, 72), (0, 140, 0), 1)
        cv2.line(image, (65, 72), (100, 72), (0, 140, 0), 1)
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 74.5), (100, 125.5)])
        self.assertEqual(report['r']['connection_evidence'][0]['kind'], 'turning_green_wire')

    def test_short_straight_wire_is_not_confused_with_an_nc_cross(self):
        image, comp, old = self.resistor()
        image[20:75, 99:102] = 255
        cv2.line(image, (100, 69), (100, 74), (0, 140, 0), 1)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(len(result['r']), 2)
        self.assertEqual(report['r']['connection_evidence'][0]['kind'], 'straight_green_wire')

    def test_thick_diagonal_nc_cross_has_no_wire_continuation(self):
        image, comp, old = self.resistor()
        image[20:75, 96:105] = 255
        cv2.line(image, (94, 64), (106, 76), (0, 140, 0), 3)
        cv2.line(image, (94, 76), (106, 64), (0, 140, 0), 3)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 125.5)])

    def test_detached_transverse_wire_does_not_supply_a_terminal(self):
        image, comp, old = self.resistor()
        image[20:75, 99:102] = 255
        cv2.line(image, (65, 70), (135, 70), (0, 140, 0), 1)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 125.5)])

    def test_short_green_stub_cannot_jump_a_white_row_to_a_parallel_wire(self):
        image, comp, old = self.resistor()
        image[20:75, 99:102] = 255
        cv2.line(image, (100, 73), (100, 74), (0, 140, 0), 1)
        cv2.line(image, (65, 71), (135, 71), (0, 140, 0), 1)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result['r'], [(100, 125.5)])

    def offset_led(self):
        image, _, old = self.resistor()
        image[:] = 255
        cv2.polylines(image, [np.array([(90, 90), (110, 90), (100, 110)], np.int32)],
                      True, (0, 0, 200), 1)
        cv2.line(image, (100, 75), (100, 90), (0, 0, 200), 1)
        cv2.line(image, (100, 110), (100, 125), (0, 0, 200), 1)
        cv2.line(image, (100, 20), (100, 74), (0, 140, 0), 1)
        cv2.line(image, (100, 126), (100, 180), (0, 140, 0), 1)
        comp = dict(id='r', type='led', bbox_img=[83, 90, 97, 110], bbox_raw_img=[83, 90, 97, 110])
        return image, comp, old

    def test_two_aligned_led_contacts_calibrate_an_offset_box(self):
        image, comp, old = self.offset_led()
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 74.5), (100, 125.5)])

    def test_offset_contact_needs_an_independent_opposed_contact(self):
        image, comp, old = self.offset_led()
        image[110:181, 99:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def ground(self, widths=(40, 28, 16, 6)):
        image, _, old = self.resistor()
        image[:] = 255
        for y, width in zip((100, 106, 112, 118), widths):
            cv2.line(image, (100 - width // 2, y), (100 + width // 2, y), (0, 0, 200), 1)
        cv2.line(image, (100, 75), (100, 100), (0, 0, 200), 1)
        cv2.line(image, (100, 20), (100, 74), (0, 140, 0), 1)
        comp = dict(id='r', type='gnd', bbox_img=[90, 106, 110, 121], bbox_raw_img=[90, 106, 110, 121])
        return image, comp, old

    def test_tapering_ground_bars_recover_the_stem_outside_an_inset_box(self):
        image, comp, old = self.ground()
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['r'], [(100, 74.5)])
        rotate = lambda p: (p[1], 219 - p[0])
        box = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        turned = dict(comp, bbox_img=[*box.min(axis=0), *box.max(axis=0)])
        turned['bbox_raw_img'] = turned['bbox_img']
        result, _ = refine_pins([turned], {'r': [rotate(p) for p in old['r']]}, np.rot90(image))
        np.testing.assert_allclose(result['r'], [rotate((100, 74.5))])

    def test_unrelated_parallel_strokes_do_not_calibrate_a_ground_stem(self):
        image, comp, old = self.ground(widths=(40, 40, 40, 40))
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_measured_axis_preserves_two_output_pins_when_box_axis_is_wrong(self):
        stage, assigned = fixtures.PassivePinPublicationTests().fixture()
        comp = stage['components'][0]
        comp.update(type='c', bbox_raw_img=[40, 65, 60.5, 85])
        stage['passive_pin_report']['p']['type'] = 'c'
        with patch.object(assemble, 'DROP_UNKEYED', False), \
             patch.object(assemble, 'NET_DROP_UNKEYED', False):
            doc = assemble.build_doc(stage, assigned, 'square-capacitor')
        key = next(iter(doc['components']))
        self.assertEqual(len(doc['pins'][key]), 2)
        self.assertEqual(sorted((p['point']['x'], p['point']['y']) for p in doc['pins'][key].values()),
                         [(50, 5), (50, 55)])
        self.assertNotEqual(doc['__pin_index']['p']['0'], doc['__pin_index']['p']['1'])


if __name__ == '__main__':
    unittest.main()
