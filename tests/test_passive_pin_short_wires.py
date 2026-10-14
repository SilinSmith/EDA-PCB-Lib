"""Short spans are valid contacts; peer attribution still needs actual evidence."""
import unittest

import cv2
import numpy as np

from stage1.passive_pins import refine_pins


class ShortWireTests(unittest.TestCase):
    def pair(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        components = []
        for cid, top, bottom, first, last in (('a', 80, 100, 65, 115), ('b', 140, 160, 120, 180)):
            cv2.rectangle(image, (96, top), (104, bottom), (0, 0, 200), 1)
            cv2.line(image, (100, first), (100, top), (0, 0, 200), 1)
            cv2.line(image, (100, bottom), (100, last), (0, 0, 200), 1)
            components.append(dict(id=cid, type='r', bbox_img=[96, top, 104, bottom],
                                   bbox_raw_img=[96, top, 104, bottom]))
        cv2.line(image, (100, 20), (100, 64), (0, 140, 0), 1)
        cv2.line(image, (100, 116), (100, 119), (0, 140, 0), 1)
        cv2.line(image, (100, 181), (100, 220), (0, 140, 0), 1)
        return image, components, {'a': [(100, 70), (100, 110)], 'b': [(100, 130), (100, 170)]}

    def test_both_boundaries_of_short_wire_are_measured_with_peer_evidence(self):
        image, comps, old = self.pair()
        result, report = refine_pins(comps, old, image)
        np.testing.assert_allclose(result['a'], [(100, 64.5), (100, 115.5)])
        np.testing.assert_allclose(result['b'], [(100, 119.5), (100, 180.5)])
        for cid, index, peer in (('a', 1, 'b'), ('b', 0, 'a')):
            proof = report[cid]['connection_evidence'][index]
            self.assertEqual(proof['kind'], 'short_green_wire_to_detected_body')
            self.assertEqual(proof['peer_id'], peer)
            self.assertEqual(len(proof['green_pixels_img']), 4)

    def test_two_pixel_green_wire_does_not_lose_its_first_raster_cell(self):
        image, comps, old = self.pair()
        image[118:120, 100] = (0, 0, 200)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])
        self.assertEqual(result['b'], [(100, 117.5), (100, 180.5)])
        self.assertEqual(len(report['a']['connection_evidence'][1]['green_pixels_img']), 2)

    def test_missing_peer_detection_still_allows_the_observed_short_contact(self):
        image, comps, old = self.pair()
        result, report = refine_pins(comps[:1], old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])
        self.assertNotIn('peer_id', report['a']['connection_evidence'][1])

    def test_white_break_inside_green_wire_is_not_bridged(self):
        image, comps, old = self.pair()
        image[118, 100] = 255
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5)])
        self.assertEqual(result['b'], [(100, 180.5)])

    def test_detached_facing_red_stem_is_not_peer_body_evidence(self):
        image, comps, old = self.pair()
        image[132:138, 99:102] = 255
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])
        self.assertNotIn('peer_id', report['a']['connection_evidence'][1])

    def test_red_detour_around_a_white_break_does_not_confirm_a_direct_peer_lead(self):
        image, comps, old = self.pair()
        image[132:138, 99:102] = 255
        cv2.polylines(image, [np.array([(100, 130), (108, 130), (108, 140), (100, 140)])],
                      False, (0, 0, 200), 1)
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])
        self.assertNotIn('peer_id', report['a']['connection_evidence'][1])

    def test_two_peer_boxes_sharing_red_body_remain_ambiguous(self):
        image, comps, old = self.pair()
        comps.append(dict(comps[1], id='duplicate'))
        old['duplicate'] = old['b']
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])
        self.assertNotIn('peer_id', report['a']['connection_evidence'][1])

    def test_nc_cross_is_rejected_even_with_a_facing_peer(self):
        image, comps, old = self.pair()
        image[116:120, 96:105] = 255
        cv2.line(image, (96, 113), (104, 121), (0, 140, 0), 1)
        cv2.line(image, (96, 121), (104, 113), (0, 140, 0), 1)
        result, _ = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5)])

    def test_peer_through_an_intervening_detection_cannot_confirm_short_wire(self):
        image, comps, old = self.pair()
        comps.append(dict(id='barrier', type='box', bbox_img=[95, 128, 105, 133]))
        old['barrier'] = [(95, 130)]
        result, report = refine_pins(comps, old, image)
        self.assertEqual(result['a'], [(100, 64.5), (100, 115.5)])
        self.assertNotIn('peer_id', report['a']['connection_evidence'][1])
        self.assertEqual(result['barrier'], old['barrier'])

    def test_quarter_turn_preserves_both_short_wire_boundaries(self):
        image, comps, old = self.pair()
        rotate = lambda point: (point[1], 239 - point[0])
        for comp in comps:
            corners = np.asarray([rotate(comp['bbox_img'][:2]), rotate(comp['bbox_img'][2:])])
            comp['bbox_img'] = comp['bbox_raw_img'] = [*corners.min(axis=0), *corners.max(axis=0)]
        result, _ = refine_pins(comps, {cid:[rotate(v) for v in points] for cid,points in old.items()}, np.rot90(image))
        np.testing.assert_allclose(sorted(result['a']), sorted(map(rotate, [(100, 64.5), (100, 115.5)])))
        np.testing.assert_allclose(sorted(result['b']), sorted(map(rotate, [(100, 119.5), (100, 180.5)])))


if __name__ == '__main__':
    unittest.main()
