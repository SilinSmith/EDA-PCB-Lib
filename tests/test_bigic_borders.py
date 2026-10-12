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


if __name__ == "__main__":
    unittest.main()


# ── §5.28 门：本模块测的是队友分支未并入 main 的边框证据（border）链 ──
# 前置符号缺失 ⇒ setUpModule 抛 SkipTest，整模块标为 skipped（一条断言都不改）；
# 队友那条绑定路径并进来后本门自动打开，原测试原样开跑。
def setUpModule():
    import _unmerged_gate
    _unmerged_gate.gate([
            ('stage1.bigic', ('BORDER_MIN', '_border_support',
                              '_straddles_body_boundary'))])
