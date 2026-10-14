"""Incomplete borders reject a contour only with contradictory IC context."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from stage1 import bigic


class BigICBorderTests(unittest.TestCase):
    def rectangle(self):
        ink = np.zeros((400, 600), np.uint8)
        cv2.rectangle(ink, (100, 100), (300, 200), 255, 1)
        return ink

    def test_solid_border_is_supported_with_contour_rounding(self):
        ink = self.rectangle()
        self.assertEqual(bigic._border_support(ink, (100, 100, 301, 201)), [1.0] * 4)
        self.assertEqual(bigic._border_support(ink, (99, 99, 302, 202)), [1.0] * 4)

    def test_missing_straight_side_is_not_rescued_by_interior_ink(self):
        ink = self.rectangle()
        ink[198:204, 170:240] = 0
        cv2.line(ink, (170, 180), (240, 180), 255, 1)
        support = bigic._border_support(ink, (100, 100, 301, 201))
        self.assertLess(support[2], bigic.BORDER_MIN)
        self.assertEqual(support[0], 1.0)

    def test_full_detector_rejects_step_shaped_wire_contour(self):
        ink = np.zeros((400, 600), np.uint8)
        contour = np.array([(100, 100), (300, 100), (300, 200),
                            (240, 200), (240, 180), (170, 180),
                            (170, 200), (100, 200)], np.int32)
        cv2.polylines(ink, [contour], True, 255, 1)
        bgr = cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)
        with patch.object(bigic, "_edge_stubs", return_value=4):
            legacy = bigic.detect_rects(bgr, border_min=0)
            guarded = bigic.detect_rects(bgr, border_min=bigic.BORDER_MIN)
        self.assertTrue(legacy)
        self.assertEqual(guarded, [])

    def test_full_detector_preserves_solid_rectangle(self):
        bgr = cv2.cvtColor(255 - self.rectangle(), cv2.COLOR_GRAY2BGR)
        with patch.object(bigic, "_edge_stubs", return_value=4):
            found = bigic.detect_rects(bgr)
        self.assertTrue(found)
        self.assertTrue(all(min(record["border"]) == 1 for record in found))

    def test_partial_border_alone_does_not_drop_standalone_ic(self):
        record = dict(x1=100., y1=100., x2=300., y2=200., w=200., h=100.,
                      fill=.95, stub=4, border=[1., .75, .85, .65])
        components = []
        with patch.object(bigic, "detect_rects", return_value=[record]):
            added = bigic.add_components(np.full((400, 600, 3), 255, np.uint8),
                                         components, SimpleNamespace(add=lambda *args, **kw: None))
        self.assertEqual(added, ["b0000"])

    def test_partial_border_with_body_cut_rejects_wire_loop(self):
        record = dict(x1=100., y1=100., x2=300., y2=200., w=200., h=100.,
                      fill=.9, stub=4, border=[1., 1., .73, 1.])
        components = [dict(id="existing", type="box", bbox_img=[180., 180., 260., 240.])]
        with patch.object(bigic, "detect_rects", return_value=[record]):
            added = bigic.add_components(np.full((400, 600, 3), 255, np.uint8),
                                         components, SimpleNamespace(add=lambda *args, **kw: None))
        self.assertEqual(added, [])
        self.assertEqual(len(components), 1)

    def test_boundary_contact_and_inside_boxes_do_not_count_as_body_cut(self):
        self.assertFalse(bigic._straddles_body_boundary([100, 100, 300, 200], [180, 200, 260, 240]))
        self.assertFalse(bigic._straddles_body_boundary([100, 100, 300, 200], [180, 130, 260, 170]))
        self.assertTrue(bigic._straddles_body_boundary([100, 100, 300, 200], [180, 180, 260, 240]))

    def completion(self, *, closed_half=False, neighbor=False, source="yolo", colored=False, shifted=False):
        bgr=np.full((500,600,3),255,np.uint8)
        cv2.rectangle(bgr,(100,50),(300,450),(0,0,255) if colored else (0,0,0),1)
        for row in (100,170,260,340,410):
            cv2.line(bgr,(80,row),(100,row),(0,0,0),1)
        old=[108.,47.,308.,449.] if shifted else [100.,50.,301.,251.]
        if closed_half:cv2.line(bgr,(100,250),(300,250),(0,0,0),1)
        component=dict(id="ic",type="box",source=source,bbox_img=old,bbox_raw_img=list(old))
        comps=[component]
        if neighbor:comps.append(dict(id="other",type="r",source="yolo",bbox_img=[180,330,200,350]))
        rect=dict(x1=100.,y1=50.,x2=301.,y2=451.,border=[1.]*4,stub=5)
        return bigic._complete_detected_body(bgr,comps,rect),component,old

    def test_truncated_semantic_body_is_completed_without_changing_identity_or_raw_detector_box(self):
        corrected,c,old=self.completion()
        self.assertTrue(corrected)
        self.assertEqual(c["bbox_img"],[100.,50.,301.,451.])
        self.assertEqual(c["bbox_raw_img"],old)
        self.assertEqual((c["id"],c["source"]),("ic","yolo"))
        self.assertEqual(c["bbox_json"],[100.,49.,301.,450.])
        self.assertEqual(c["bbox_completion"]["expanded_side"],"bottom")

    def test_closed_real_half_neighbor_or_shape_only_and_color_context_prevent_expansion(self):
        for kwargs in (dict(closed_half=True),dict(neighbor=True),dict(source="cv:bigic"),dict(colored=True)):
            with self.subTest(kwargs=kwargs):
                corrected,c,old=self.completion(**kwargs)
                self.assertFalse(corrected)
                self.assertEqual(c["bbox_img"],old)

    def test_nearby_shifted_semantic_frame_can_follow_all_four_measured_sides(self):
        corrected,c,old=self.completion(shifted=True)
        self.assertTrue(corrected)
        self.assertEqual(c["bbox_raw_img"],old)
        self.assertEqual(c["bbox_completion"]["method"],"four_nearby_measured_sides")


if __name__ == "__main__":
    unittest.main()
