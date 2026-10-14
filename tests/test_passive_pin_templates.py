"""Unmarked elbows, measured lengths, and weak-span fallback regressions."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage1.passive_pins import refine_pins
from stage2 import assemble

RED, GREEN, BLUE = (0, 0, 190), (0, 140, 0), (180, 0, 0)


def component(cid, typ, box):
    return dict(id=cid, type=typ, pending=False, bbox_img=list(box), bbox_raw_img=list(box),
                center_img=[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def corner_scene():
    image = np.full((260, 420, 3), 255, np.uint8)
    for x in (196, 204):
        cv2.line(image, (x, 90), (x, 110), RED, 1)
    cv2.line(image, (170, 100), (195, 100), RED, 1)
    cv2.line(image, (170, 100), (170, 135), RED, 1)
    cv2.rectangle(image, (150, 135), (190, 180), RED, 1)
    cv2.line(image, (205, 100), (245, 100), RED, 1)
    for x, half in ((245, 10), (249, 6), (253, 2)):
        cv2.line(image, (x, 100 - half), (x, 100 + half), RED, 1)
    # A separate observed red/green symbol establishes this sheet's roles.
    for x in (296, 304):
        cv2.line(image, (x, 210), (x, 230), RED, 1)
    cv2.line(image, (280, 220), (295, 220), RED, 1)
    cv2.line(image, (305, 220), (320, 220), RED, 1)
    cv2.line(image, (260, 220), (279, 220), GREEN, 1)
    cv2.line(image, (321, 220), (350, 220), GREEN, 1)
    comps = [component('c', 'c', [193, 86, 207, 114]),
             component('ic', 'box', [150, 135, 190, 180]),
             component('g', 'gnd', [243, 87, 256, 113]),
             component('ref', 'c', [293, 206, 307, 234])]
    pins = {'c': [(180, 100), (220, 100)], 'ic': [(170, 135)],
            'g': [(225, 100)], 'ref': [(280, 220), (320, 220)]}
    return image, comps, pins


def weak_scene():
    image = np.full((200, 320, 3), 255, np.uint8)
    comps, pins = [], {}
    for name, x in [('target', 100), ('ref', 200)]:
        for y in (96, 104):
            cv2.line(image, (x - 10, y), (x + 10, y), RED, 1)
        cv2.line(image, (x, 81), (x, 95), RED, 1)
        cv2.line(image, (x, 105), (x, 140), RED, 1)
        cv2.line(image, (x, 45), (x, 80), GREEN, 1)
        for y, half in ((140, 10), (144, 6), (148, 2)):
            cv2.line(image, (x - half, y), (x + half, y), RED, 1)
        comps.extend([component(name, 'c', [x - 13, 91, x + 13, 109]),
                      component(name + '-g', 'gnd', [x - 13, 137, x + 13, 151])])
        pins[name], pins[name + '-g'] = [(x, 82), (x, 120)], [(x, 121)]
    image[120:122, 200] = GREEN
    image[120:122, 101] = (207, 219, 220)
    return image, comps, pins


class PassiveTemplateTests(unittest.TestCase):
    def test_two_straight_pins_at_elbow_locate_both_capacitor_ends(self):
        image, comps, pins = corner_scene()
        result, report = refine_pins(comps, pins, image)
        self.assertEqual(result['c'], [(170.0, 100.0), (230.0, 100.0)])
        self.assertEqual(report['c']['sources'], ['direct_pin_corner', 'symmetric_opposed_terminal'])
        self.assertEqual(result['g'], [(230.0, 100.0)])
        self.assertEqual(result['ic'], pins['ic'])

    def test_corner_survives_quarter_turn_and_does_not_move_ic(self):
        image, comps, pins = corner_scene()
        width = image.shape[1]
        transform = lambda pt: (pt[1], width - 1 - pt[0])
        for comp in comps:
            a, b, c, d = comp['bbox_img']
            comp.update(bbox_img=[b, width - 1 - c, d, width - 1 - a],
                        bbox_raw_img=[b, width - 1 - c, d, width - 1 - a])
        rotated = {cid: [transform(pt) for pt in pts] for cid, pts in pins.items()}
        result, _ = refine_pins(comps, rotated, np.rot90(image))
        self.assertEqual(sorted(result['c']), sorted(map(transform, [(170, 100), (230, 100)])))
        self.assertEqual(result['ic'], rotated['ic'])

    def test_red_T_is_not_a_two_pin_elbow(self):
        image, comps, pins = corner_scene()
        cv2.line(image, (145, 100), (169, 100), RED, 1)
        _, report = refine_pins(comps, pins, image)
        self.assertNotIn('direct_pin_corner', report.get('c', {}).get('sources', []))

    def test_a_blank_in_the_peer_pin_does_not_supply_an_elbow(self):
        image, comps, pins = corner_scene()
        image[118:121, 169:172] = 255
        _, report = refine_pins(comps, pins, image)
        self.assertNotIn('direct_pin_corner', report.get('c', {}).get('sources', []))

    def test_monochrome_elbow_with_coloured_label_stays_deferred(self):
        image, comps, pins = corner_scene()
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cv2.putText(image, 'C1', (190, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, BLUE, 1)
        self.assertEqual(refine_pins(comps, pins, image), (pins, {}))

    def test_weak_span_uses_real_same_symbol_lengths_with_separate_ends(self):
        image, comps, pins = weak_scene()
        result, report = refine_pins(comps, pins, image)
        self.assertEqual(result['target'][-1], (100.0, 119.5))
        self.assertEqual(result['target-g'], [(100.0, 121.5)])
        self.assertEqual(report['target']['sources'][-1], 'same_symbol_terminal_length')
        proof = report['target']['connection_evidence'][-1]
        self.assertTrue(proof['weak_colour_fallback'])
        self.assertTrue(proof['reference_terminals'])

    def test_template_cannot_cross_a_white_gap(self):
        image, comps, pins = weak_scene()
        image[112:115, 99:102] = 255
        _, report = refine_pins(comps, pins, image)
        self.assertNotIn('same_symbol_terminal_length', report.get('target', {}).get('sources', []))

    def test_a_single_white_cell_next_to_a_mixed_mark_is_not_invented_ink(self):
        image, comps, pins = weak_scene()
        image[119, 100] = 255
        _, report = refine_pins(comps, pins, image)
        self.assertNotIn('same_symbol_terminal_length', report.get('target', {}).get('sources', []))

    def test_template_does_not_override_an_observed_boundary(self):
        image, comps, pins = weak_scene()
        image[113:125, 100] = GREEN
        result, report = refine_pins(comps, pins, image)
        self.assertEqual(result['target'][-1], (100.0, 112.5))
        self.assertNotEqual(report['target']['sources'][-1], 'same_symbol_terminal_length')

    def test_final_assembly_delivers_corner_and_template_coordinates(self):
        for scene in (corner_scene, weak_scene):
            image, comps, pins = scene()
            result, report = refine_pins(comps, pins, image)
            stage = dict(components=comps, pins=result, passive_pin_report=report,
                         image=dict(w=image.shape[1], h=image.shape[0]), nets=[])
            with patch.object(assemble, 'DROP_UNKEYED', False):
                doc = assemble.build_doc(stage, {}, 'synthetic')
            for n, comp in enumerate(comps):
                if comp['id'] not in report:
                    continue
                for i, pt in enumerate(result[comp['id']]):
                    pk = doc['__pin_index'][comp['id']][str(i)]
                    self.assertEqual(doc['pins'][f'p{n:03d}'][pk]['point'], dict(x=pt[0], y=image.shape[0] - pt[1]))

    def test_inputs_remain_unchanged(self):
        image, comps, pins = weak_scene()
        before = copy.deepcopy((image, comps, pins))
        refine_pins(comps, pins, image)
        np.testing.assert_array_equal(image, before[0])
        self.assertEqual(comps, before[1]); self.assertEqual(pins, before[2])

    def test_filled_resistor_preserves_its_visible_two_cell_span_boundary(self):
        image=np.full((220,260,3),255,np.uint8)
        cv2.rectangle(image,(100,96),(120,104),RED,-1)
        cv2.line(image,(80,100),(99,100),RED,1)
        cv2.line(image,(30,100),(79,100),GREEN,1)
        cv2.line(image,(121,100),(190,100),RED,1)
        image[100,140:142]=(90,130,138)
        comp=component('r','r',[99,95,121,105])
        result,report=refine_pins([comp],{'r':[(80,100),(140,100)]},image)
        self.assertEqual(result['r'],[(79.5,100.0),(139.5,100.0)])
        self.assertEqual(report['r']['sources'][-1],'mixed_colour_span_boundary')


if __name__ == '__main__':
    unittest.main()
