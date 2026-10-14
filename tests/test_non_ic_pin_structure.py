"""Count recovery is supported by ink, and never claims precise gray positions."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage1.non_ic_pin_structure import reconcile


class NonIcPinStructureTests(unittest.TestCase):
    def ground(self):
        image = np.full((220, 220, 3), 255, np.uint8)
        for y, width in [(95, 48), (104, 34), (113, 20), (122, 8)]:
            image[y:y + 3, 100 - width // 2:100 + width // 2] = (0, 0, 0)
        image[50:95, 99:102] = (0, 0, 0)
        comp = dict(id='g', type='gnd', bbox_img=[89, 103, 111, 124], bbox_raw_img=[89, 103, 111, 124])
        old = {'g': [(89, 94), (111, 94), (100, 134), (120, 110)]}
        return image, comp, old

    def test_ground_bar_and_label_candidates_form_one_port(self):
        image, comp, old = self.ground()
        frozen = copy.deepcopy(old)
        result, report = reconcile([comp], old, image)
        self.assertEqual(len(result['g']), 1)
        self.assertIn(result['g'][0], old['g'])
        self.assertEqual(report['g']['kind'], 'single_port_count_reconciliation')
        self.assertEqual(report['g']['observed_outward_side'], -1)
        self.assertIn('not_precision_verified', report['g']['coordinate_status'])
        self.assertEqual(old, frozen)

    def test_ground_cardinality_survives_a_quarter_turn(self):
        image, comp, old = self.ground()
        rotate = lambda p: (p[1], 219 - p[0])
        corners = np.array([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp['bbox_img'] = comp['bbox_raw_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        old['g'] = [rotate(p) for p in old['g']]
        result, report = reconcile([comp], old, np.rot90(image))
        self.assertEqual(len(result['g']), 1)
        self.assertIn(result['g'][0], old['g'])
        self.assertFalse(report['g']['observed_axis_y'])

    def test_ground_without_a_visible_stem_keeps_uncertain_candidates(self):
        image, comp, old = self.ground()
        image[50:95, 97:104] = 255
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_candidates_only_on_the_label_side_are_not_promoted(self):
        image, comp, _ = self.ground()
        old = {'g': [(90, 134), (110, 136)]}
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_multiple_observed_ground_stems_are_ambiguous(self):
        image, comp, old = self.ground()
        for y, width in [(135, 44), (144, 30), (153, 18)]:
            image[y:y + 2, 100 - width // 2:100 + width // 2] = 0
        image[127:135, 99:102] = 0
        comp['bbox_img'] = comp['bbox_raw_img'] = [89, 92, 111, 156]
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def power(self):
        image = np.full((220, 220, 3), 255, np.uint8)
        image[95:99, 85:117] = 0
        image[99:150, 99:103] = 0
        cv2.putText(image, '3.3V', (81, 89), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        comp = dict(id='v', type='v', bbox_img=[95, 92, 106, 102], bbox_raw_img=[95, 92, 106, 102])
        return image, comp, {'v': [(100, 82), (116, 97), (100.5, 112), (85, 97)]}

    def test_gray_power_label_and_cap_do_not_create_extra_ports(self):
        image, comp, old = self.power()
        result, report = reconcile([comp], old, image)
        self.assertEqual(result['v'], [(100.5, 112)])
        self.assertEqual(report['v']['retained_raw_indices'], [2])

    def test_an_unconnected_power_cap_does_not_confirm_an_axis(self):
        image, comp, old = self.power()
        image[99:151, 97:105] = 255
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_a_ground_is_not_reinterpreted_as_a_power_cap(self):
        image, comp, old = self.ground()
        comp['type'] = 'v'
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def resistor(self):
        image = np.full((220, 220, 3), 255, np.uint8)
        path = np.array([(40, 100), (80, 100), (85, 95), (90, 105), (95, 95), (100, 105),
                         (105, 95), (110, 105), (115, 100), (170, 100)], np.int32)
        cv2.polylines(image, [path], False, (0, 0, 0), 3)
        comp = dict(id='r', type='r', bbox_img=[84, 94, 116, 106], bbox_raw_img=[84, 94, 116, 106])
        return image, comp, {'r': [(74, 100.5)]}

    def test_missing_resistor_port_needs_a_connected_opposite_arm(self):
        image, comp, old = self.resistor()
        result, report = reconcile([comp], old, image)
        self.assertEqual(len(result['r']), 2)
        self.assertEqual(result['r'][0], old['r'][0])
        self.assertEqual(report['r']['kind'], 'observed_opposed_stub_count_recovery')
        self.assertTrue(report['r']['recovered_stroke_pixels_img'])

    def test_a_white_gap_blocks_an_apparently_aligned_resistor_arm(self):
        image, comp, old = self.resistor()
        image[94:107, 121:123] = 255
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_resistor_with_no_opposite_arm_is_not_filled_by_type_alone(self):
        image, comp, old = self.resistor()
        image[94:107, 118:] = 255
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_inset_resistor_box_can_follow_a_zigzag_to_the_straight_arm(self):
        image, comp, old = self.resistor()
        comp['bbox_img'] = comp['bbox_raw_img'] = [91, 95, 106, 105]
        old['r'] = [(81, 97.5)]
        result, _ = reconcile([comp], old, image)
        self.assertEqual(len(result['r']), 2)

    def test_a_square_detector_does_not_guess_the_resistor_axis(self):
        image, comp, old = self.resistor()
        comp['bbox_img'] = comp['bbox_raw_img'] = [90, 90, 110, 110]
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_colour_terminal_evidence_is_authoritative(self):
        image, comp, old = self.resistor()
        result, report = reconcile([comp], old, image, {'r': {'after': old['r']}})
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_coloured_ink_is_outside_gray_count_recovery(self):
        image, comp, old = self.ground()
        image[88:91, 99:102] = (0, 150, 0)
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_ic_and_already_single_candidates_are_unchanged(self):
        image, comp, old = self.power()
        comp['type'] = 'box'
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})
        comp['type'] = 'v'
        old['v'] = [(100, 112)]
        result, report = reconcile([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_missing_image_keeps_all_original_candidates(self):
        _, comp, old = self.power()
        result, report = reconcile([comp], old, None)
        self.assertEqual(result, old)
        self.assertEqual(report, {})


class CountRecoveryPipelineTests(unittest.TestCase):
    def run_graph(self, wire_y):
        from test_pipeline_pin_network import PipelinePinNetworkTests
        from stage1 import non_ic_pin_structure
        component = dict(id='r', type='r', bbox_img=[100, 100, 132, 112],
                         center_img=[116, 106], pending=False)
        before = {'r': [(90, 106)]}
        recovered = {'r': [(90, 106), (142, 106)]}
        evidence = {'r': {'kind': 'observed_opposed_stub_count_recovery', 'new_raw_indices': [1]}}
        segments = [[[60, 106], [90, 106]], [[140, wire_y], [180, wire_y]]]
        with patch.object(non_ic_pin_structure, 'reconcile', return_value=(recovered, evidence)):
            return PipelinePinNetworkTests().run_case([component], segments, supplied_pins=before)

    def test_recovered_port_is_counted_without_borrowing_a_nearby_parallel_wire(self):
        doc, _, graph_call = self.run_graph(110)
        self.assertEqual(graph_call.kwargs['pin_synth']['r'], [False, True])
        self.assertEqual(len(doc['pins']['r']), 2)
        self.assertEqual([net['pins'] for net in doc['nets']], [[['r', 0]]])
        self.assertEqual(doc['pin_join']['orphans'][0]['pin'], 1)

    def test_a_wire_reaching_the_recovered_estimate_still_connects(self):
        doc, _, graph_call = self.run_graph(106)
        self.assertEqual(graph_call.kwargs['pin_synth']['r'], [False, True])
        self.assertEqual(len(doc['pins']['r']), 2)
        self.assertEqual([net['pins'] for net in doc['nets']], [[['r', 0]], [['r', 1]]])


if __name__ == '__main__':
    unittest.main()
