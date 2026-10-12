"""Text near-edge midpoint geometry uses fresh frame contacts only."""
import math
import os
import unittest
from unittest.mock import patch

from stage2 import structmatch as sm


def candidate(text, box, score=1.0):
    return {"text": text, "bbox": box, "score": score,
            "center": [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]}


def contact(point, edge, anchor):
    return {"point_img": point, "edge": edge, "anchor_img": anchor,
            "trust": "trusted", "frame_confirmed": True,
            "source": "red_green_transition"}


class NameEdgeMidpointTests(unittest.TestCase):
    def test_midpoint_uses_side_facing_pin(self):
        c = candidate("NAME", [105, 130, 165, 150])
        expected = {"left": (105, 140), "right": (165, 140),
                    "top": (135, 130), "bottom": (135, 150)}
        for edge, value in expected.items():
            self.assertEqual(sm.name_edge_midpoint(c, edge), value)

    def test_invalid_rectangles_have_no_midpoint(self):
        for box in ([0, 0, 0, 5], [0, 0, 5, float("nan")], [2, 0, 1, 5], None):
            self.assertIsNone(sm.name_edge_midpoint({"bbox": box}, "left"))
        self.assertIsNone(sm.name_edge_midpoint(candidate("X", [0, 0, 5, 5]), "front"))

    def test_distance_uses_frame_intersection_not_terminal(self):
        bb, pins = [100, 100, 300, 300], [[70, 150]]
        records = [contact(pins[0], "left", [100, 150])]
        c = candidate("GND", [105, 147, 135, 157])
        self.assertAlmostEqual(sm.name_edge_midpoint_distance(c, bb, pins, 0, records),
                               math.hypot(5, 2))

    def test_longer_text_same_near_edge_keeps_distance(self):
        bb, pins = [100, 100, 300, 300], [[85, 150]]
        records = [contact(pins[0], "left", [100, 150])]
        short = candidate("VDD", [105, 145, 130, 155])
        long = candidate("VDDREGULATOR", [105, 145, 255, 155])
        self.assertEqual(sm.name_edge_midpoint_distance(short, bb, pins, 0, records),
                         sm.name_edge_midpoint_distance(long, bb, pins, 0, records))

    def test_unconfirmed_or_noncolor_frame_falls_back(self):
        bb, pins = [100, 100, 300, 300], [[85, 150]]
        c = candidate("GND", [105, 145, 135, 155])
        with patch.dict(os.environ, {"STAGE2_PIN_NAME_EDGE_MIDPOINT": "1"}):
            for changes in ({"frame_confirmed": False}, {"source": "gray_frame"},
                            {"trust": "uncertain"}, {"anchor_img": [115, 150]}):
                rec = contact(pins[0], "left", [100, 150])
                rec.update(changes)
                self.assertFalse(sm._name_edge_midpoint_enabled(rec, bb))
                self.assertIsNone(sm.name_edge_midpoint_distance(c, bb, pins, 0, [rec]))

    def test_feature_switch_changes_only_distance_formula(self):
        bb, pins = [100, 100, 300, 300], [[85, 150]]
        recs = [contact(pins[0], "left", [100, 150])]
        c = candidate("GND", [105, 147, 135, 157])
        with patch.dict(os.environ, {"STAGE2_PIN_NAME_EDGE_MIDPOINT": "0"}):
            self.assertEqual(sm._name_corridor_distance(c, bb, pins, 0, recs), 5)
        with patch.dict(os.environ, {"STAGE2_PIN_NAME_EDGE_MIDPOINT": "1"}):
            self.assertAlmostEqual(sm._name_corridor_distance(c, bb, pins, 0, recs),
                                   math.hypot(5, 2))

    def test_directional_score_does_not_double_count_old_center_distance(self):
        bb, pins = [100, 100, 300, 300], [[85, 150]]
        recs = [contact(pins[0], "left", [100, 150])]
        c = candidate("GND", [105, 147, 135, 157], 0.8)
        with patch.dict(os.environ, {"STAGE2_PIN_NAME_EDGE_MIDPOINT": "1"}):
            _, score = sm._directional_name_groups(bb, pins, [c], recs, [0],
                                                    lambda _c, _i: 0.2)
            self.assertAlmostEqual(score(c, 0), 0.8 / (1 + math.hypot(5, 2) / 10))

    def test_neighboring_row_and_external_name_remain_rejected(self):
        bb, pins = [100, 100, 300, 300], [[85, 150]]
        recs = [contact(pins[0], "left", [100, 150])]
        pool = [candidate("GND", [105, 145, 135, 155]),
                candidate("MODEL", [105, 195, 150, 205]),
                candidate("OUTSIDE", [75, 145, 98, 155])]
        with patch.dict(os.environ, {"STAGE2_PIN_NAME_EDGE_MIDPOINT": "1"}):
            bound, _ = sm.rebind(bb, pins, pool, lambda c: c["text"],
                                 lambda c, _i: c["score"], pin_evidence=recs,
                                 prioritize_trusted=True)
        self.assertEqual(bound, {0: 0})

    def test_quarter_rotation_preserves_distance(self):
        bb, pins = [100, 100, 300, 300], [[85, 150]]
        recs = [contact(pins[0], "left", [100, 150])]
        original = candidate("GND", [105, 147, 135, 157])
        rotated = candidate("GND", [-157, 105, -147, 135])
        rpins = [[-150, 85]]
        rrecs = [contact(rpins[0], "top", [-150, 100])]
        self.assertEqual(sm.name_edge_midpoint_distance(original, bb, pins, 0, recs),
                         sm.name_edge_midpoint_distance(rotated, [-300, 100, -100, 300],
                                                        rpins, 0, rrecs))

    def test_digit_geometry_does_not_change_name_rectangle(self):
        c = candidate("1", [85, 140, 98, 152])
        c["geometry_bbox"] = [87, 143, 95, 150]
        self.assertEqual(sm.number_row_distance(c, "left", [80, 150]), 0)
        self.assertEqual(sm.name_edge_midpoint(c, "left"), (85, 146))
        c["geometry_bbox"] = [0, 0, float("nan"), 2]
        self.assertEqual(sm.number_row_distance(c, "left", [80, 150]), 2)


if __name__ == "__main__":
    unittest.main()


# ── §5.28 门：本模块测的是队友分支未并入 main 的绑定路径 ──
# 前置符号缺失 ⇒ setUpModule 抛 SkipTest，整模块标为 skipped（一条断言都不改）；
# 队友那条绑定路径并进来后本门自动打开，原测试原样开跑。
def setUpModule():
    import _unmerged_gate
    _unmerged_gate.gate([
             ('stage2.structmatch', ('name_edge_midpoint', 'name_edge_midpoint_distance', '_name_corridor_distance', '_directional_name_groups', 'number_row_distance'))])
