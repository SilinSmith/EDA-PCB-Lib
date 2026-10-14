"""One-port bodies and large junctions need actual connected image evidence."""
import unittest

import cv2
import numpy as np

from stage1.passive_pins import refine_pins
from stage1 import passive_terminal_geometry as geometry


class SinglePortStructureTests(unittest.TestCase):
    def power_elbow(self, span=2):
        image = np.full((180, 180, 3), 255, np.uint8)
        image[90:92, 75:87] = (127, 127, 207)
        image[92:101, 80:82] = (127, 127, 207)
        image[101:101 + span, 82:101] = (127, 127, 207)
        image[101:101 + span, 80:82] = (30, 110, 110)
        cv2.rectangle(image, (100, 95), (120, 112), (127, 127, 207), 1)
        power = dict(id='v', type='v', bbox_img=[73, 84, 84, 94], bbox_raw_img=[73, 84, 84, 94])
        peer = dict(id='u', type='box', bbox_img=[100, 95, 120, 112], bbox_raw_img=[100, 95, 120, 112])
        return image, [power, peer], {'v': [(78, 116)], 'u': [(100, 101)]}

    def test_short_mixed_elbow_uses_own_red_colour_interface(self):
        image, components, old = self.power_elbow()
        result, report = refine_pins(components, old, image)
        np.testing.assert_allclose(result['v'], [(80.5, 100.5)])
        self.assertEqual(report['v']['sources'], ['mixed_colour_span_boundary'])
        proof = report['v']['connection_evidence'][0]
        self.assertEqual(proof['kind'], 'observed_turning_short_colour_span')
        self.assertEqual(proof['peer_id'], 'u')
        self.assertEqual(result['u'], old['u'])
        self.assertEqual(old['v'], [(78, 116)])

    def test_single_pixel_elbow_uses_shared_pixel_centre(self):
        image, components, old = self.power_elbow(span=1)
        result, report = refine_pins(components, old, image)
        np.testing.assert_allclose(result['v'], [(80.5, 101)])
        self.assertEqual(report['v']['sources'], ['shared_single_green_pixel'])

    def test_elbow_requires_a_detected_peer(self):
        image, components, old = self.power_elbow()
        result, report = refine_pins(components[:1], {'v': old['v']}, image)
        self.assertEqual(result['v'], old['v'])
        self.assertNotIn('v', report)

    def test_elbow_cannot_cross_a_white_gap(self):
        image, components, old = self.power_elbow()
        image[101:103, 88] = 255
        result, report = refine_pins(components, old, image)
        self.assertEqual(result['v'], old['v'])
        self.assertNotIn('v', report)

    def test_elbow_with_two_peer_candidates_remains_ambiguous(self):
        image, components, old = self.power_elbow()
        components.append(dict(components[1], id='u2'))
        old['u2'] = [(100, 102)]
        result, report = refine_pins(components, old, image)
        self.assertEqual(result['v'], old['v'])
        self.assertNotIn('v', report)

    def test_plain_red_elbow_is_two_straight_pins(self):
        image, components, old = self.power_elbow()
        image[101:103, 80:82] = (127, 127, 207)
        # An independent measured pin establishes the image's red-pin,
        # green-wire convention before the unmarked elbow is inferred.
        cv2.line(image, (140, 140), (140, 150), (0, 0, 200), 1)
        cv2.line(image, (140, 151), (140, 170), (0, 140, 0), 1)
        cv2.line(image, (134, 140), (146, 140), (0, 0, 200), 1)
        components.append(dict(id='ref', type='v', bbox_img=[134, 137, 146, 142], bbox_raw_img=[134, 137, 146, 142]))
        old['ref'] = [(140, 155)]
        result, report = refine_pins(components, old, image)
        np.testing.assert_allclose(result['v'], [(80.5, 101.5)])
        self.assertEqual(report['v']['sources'], ['direct_pin_corner'])

    def test_mixed_elbow_supports_quarter_turns(self):
        image, components, old = self.power_elbow()
        rotate = lambda p: (p[1], image.shape[1] - 1 - p[0])
        for component in components:
            points = np.array([rotate(component['bbox_img'][:2]), rotate(component['bbox_img'][2:])])
            component['bbox_raw_img'] = component['bbox_img'] = [*points.min(axis=0), *points.max(axis=0)]
        result, _ = refine_pins(components, {cid: [rotate(p) for p in ps] for cid, ps in old.items()}, np.rot90(image))
        np.testing.assert_allclose(result['v'], [rotate((80.5, 100.5))])

    def test_a_disconnected_text_T_cannot_supply_a_terminal(self):
        image, components, old = self.power_elbow()
        image[99:104, 78:103] = 255
        result, report = refine_pins(components, old, image)
        self.assertEqual(result['v'], old['v'])
        self.assertNotIn('v', report)

    def compact_ground(self):
        image = np.full((180, 180, 3), 255, np.uint8)
        for first, last, width in [(100, 102, 14), (102, 104, 8), (104, 106, 4)]:
            image[first:last, 100 - width // 2:100 + width // 2] = (127, 127, 207)
        image[80:100, 99:101] = (127, 127, 207)
        image[30:80, 99:101] = (80, 180, 80)
        comp = dict(id='g', type='gnd', bbox_img=[79, 100, 102, 120], bbox_raw_img=[79, 100, 102, 120])
        return image, comp, {'g': [(90, 80)]}

    def test_compact_ground_recovers_a_detector_axis_shifted_by_its_label(self):
        image, comp, old = self.compact_ground()
        result, _ = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['g'], [(99.5, 79.5)])

    def test_compact_ground_is_not_a_single_power_bar(self):
        image, comp, old = self.compact_ground()
        comp['type'] = 'v'
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_unattached_compact_ground_cannot_borrow_a_parallel_wire(self):
        image, comp, old = self.compact_ground()
        image[80, 98:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_grayscale_compact_ground_is_deferred(self):
        image, comp, old = self.compact_ground()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result, report = refine_pins([comp], old, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        self.assertEqual(result, old)
        self.assertEqual(report, {})


class LargeBlueJunctionTests(unittest.TestCase):
    def capacitor(self):
        image = np.full((220, 220, 3), 255, np.uint8)
        cv2.line(image, (86, 100), (114, 100), (255, 0, 0), 1)
        cv2.line(image, (86, 110), (114, 110), (255, 0, 0), 1)
        image[94:100, 99:102] = (255, 0, 0)
        image[68:94, 99:102] = (12, 12, 12)
        image[111:130, 99:102] = (12, 12, 12)
        image[130:175, 99:102] = (0, 0, 200)
        image[59:62, 25:176] = (0, 150, 0)
        cv2.circle(image, (100, 60), 7, (128, 0, 0), -1)
        comp = dict(id='c', type='c', bbox_img=[91, 96, 109, 115], bbox_raw_img=[91, 96, 109, 115])
        return image, comp, {'c': [(100, 85), (100, 140)]}

    def test_large_direct_dot_owns_its_centre_after_internal_body_colour_change(self):
        image, comp, old = self.capacitor()
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['c'], [(100, 60), (100, 129.5)])
        self.assertEqual(report['c']['sources'][0], 'first_filled_junction_center')
        self.assertEqual(report['c']['connection_evidence'][0]['node_pigment'], 'blue')

    def test_large_dot_does_not_jump_a_white_stem_gap(self):
        image, comp, old = self.capacitor()
        image[73, 98:103] = 255
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['c'], [(100, 129.5)])
        self.assertNotIn('first_filled_junction_center', report['c']['sources'])

    def test_earlier_wire_interface_has_priority_over_a_large_dot(self):
        image, comp, old = self.capacitor()
        image[68:81, 99:102] = (0, 150, 0)
        result, report = refine_pins([comp], old, image)
        np.testing.assert_allclose(result['c'], [(100, 80.5), (100, 129.5)])
        self.assertEqual(report['c']['sources'][0], 'lead_wire_colour_boundary')

    def test_large_square_glyph_is_not_a_junction(self):
        image, _, _ = self.capacitor()
        image[45:76, 85:116] = 255
        image[54:67, 94:107] = (128, 0, 0)
        nodes = geometry._filled_nodes(geometry._pigments(image))
        self.assertFalse(any(abs(n['point'][0] - 100) < 3 and abs(n['point'][1] - 60) < 3 for n in nodes))


if __name__ == '__main__':
    unittest.main()
