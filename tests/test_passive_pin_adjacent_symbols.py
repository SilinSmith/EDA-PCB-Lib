"""A coil can join a dot; adjacent ground bars belong to their own symbol."""
import unittest

import cv2
import numpy as np

from stage1.passive_pins import refine_pins


class AdjacentSymbolTests(unittest.TestCase):
    def coil_nodes(self):
        image = np.full((160, 160, 3), 255, np.uint8)
        red = (0, 0, 200)
        for y in (49, 50, 99, 100):
            cv2.line(image, (20, y), (140, y), (0, 140, 0), 1)
        curve = [(99, 52), (98, 53), (97, 54), (96, 55), (95, 56),
                 (96, 58), (101, 61), (96, 65), (95, 69), (101, 73),
                 (96, 77), (95, 81), (101, 85), (96, 89), (97, 93), (99, 97)]
        cv2.polylines(image, [np.asarray(curve)], False, red, 2)
        for top in (47, 97):
            for y, width in enumerate((2, 4, 6, 6, 4, 2)):
                start = 97 + (6 - width) // 2
                image[top + y, start:start + width] = red
        comp = dict(id='l', type='l', bbox_img=[94, 56, 102, 93], bbox_raw_img=[94, 56, 102, 93])
        return image, comp, {'l': [(98, 40), (98, 110)]}

    def test_curved_coil_end_does_not_move_the_dot_centre_into_its_appendage(self):
        image, comp, old = self.coil_nodes()
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result['l'], [(99.5, 49.5), (99.5, 99.5)])
        self.assertEqual(report['l']['sources'], ['first_red_junction_center'] * 2)

    def test_white_row_between_coil_and_dot_is_not_crossed(self):
        image, comp, old = self.coil_nodes()
        image[54, 90:110] = 255
        result, _ = refine_pins([comp], old, image)
        self.assertNotIn((99.5, 49.5), result['l'])

    def test_coil_dot_centres_survive_rotation(self):
        image, comp, old = self.coil_nodes()
        rotate = lambda point: (point[1], 159 - point[0])
        corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
        comp['bbox_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        comp['bbox_raw_img'] = comp['bbox_img']
        result, _ = refine_pins([comp], {'l': list(map(rotate, old['l']))}, np.rot90(image))
        self.assertEqual(result['l'], sorted([rotate((99.5, 49.5)), rotate((99.5, 99.5))]))

    def test_monochrome_coil_nodes_keep_previous_candidates(self):
        image, comp, old = self.coil_nodes()
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        result, report = refine_pins([comp], old, image)
        self.assertEqual((result, report), (old, {}))

    def power_and_ground(self):
        image = np.full((180, 180, 3), 255, np.uint8)
        red = (0, 0, 200)
        cv2.line(image, (110, 65), (110, 95), red, 1)
        cv2.line(image, (90, 75), (110, 75), red, 1)
        cv2.line(image, (20, 75), (89, 75), (0, 140, 0), 1)
        for x, top, bottom in ((114, 79, 91), (118, 82, 88)):
            cv2.line(image, (x, top), (x, bottom), red, 1)
        cv2.line(image, (110, 96), (110, 150), (0, 140, 0), 1)
        comps = [dict(id='v', type='v', bbox_img=[108, 68, 112, 76], bbox_raw_img=[108, 68, 112, 76]),
                 dict(id='g', type='gnd', bbox_img=[113, 79, 121, 92], bbox_raw_img=[113, 79, 121, 92])]
        return image, comps, {'v': [(110, 95), (90, 75)], 'g': [(110, 85)]}

    def test_power_cannot_publish_a_boundary_reached_through_another_ground_bar(self):
        image, comps, old = self.power_and_ground()
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['v'], [(89.5, 75)])
        self.assertEqual(len(report['v']['after']), 1)

    def test_power_without_ground_shape_evidence_keeps_the_ambiguity(self):
        image, comps, old = self.power_and_ground()
        result, report = refine_pins(comps[:1], old, image)
        self.assertEqual(result['v'], old['v'])
        self.assertNotIn('v', report)

    def test_a_ground_bar_at_the_final_red_cell_is_not_an_intervening_bar(self):
        image, comps, old = self.power_and_ground()
        image[:, 114:122] = 255
        image[96:151, 110] = 255
        red = (0, 0, 200)
        cv2.line(image, (90, 65), (90, 95), red, 1)
        for x, top, bottom in ((94, 79, 91), (98, 82, 88)):
            cv2.line(image, (x, top), (x, bottom), red, 1)
        comps[1]['bbox_img'] = comps[1]['bbox_raw_img'] = [93, 79, 101, 92]
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['v'], [(89.5, 75)])
        self.assertEqual(len(report['v']['after']), 1)


if __name__ == '__main__':
    unittest.main()
