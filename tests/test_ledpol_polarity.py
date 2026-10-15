"""`stage2.ledpol`：LED/二极管极性判据（合成符号图上的方向、退化情形、掩膜）。

判据要回答的是「**横杠（阴极 K）**画在哪一端」，输入是元件符号的暗红笔画掩膜与
两个候选端点（json 坐标，`H` 是图高）。符号用合成图元钉住：三角形（底边在阳极侧、
两条斜边收拢到阴极侧的顶点）+ 横杠 + 贯穿引线；横杠与底边的长度比按语料常态
（220 池实测 28~35px vs 28~35px，比值 ≥0.82）取值，使两条横笔都能被剖面找到。
"""
import unittest

import numpy as np

from stage2 import ledpol

H, W = 60, 41


def glyph(bar_len=12, base_len=15):
    """竖直 LED：底边在上（阳极）、斜边收拢到顶点、横杠在下（阴极 K）、引线贯穿。"""
    m = np.zeros((H, W), dtype=bool)
    cx = W // 2
    m[8, cx - base_len // 2: cx + base_len // 2 + 1] = True          # 底边（阳极侧）
    for i in range(11):                                              # 斜边 → 顶点
        x = max(0, (base_len // 2) * (10 - i) // 10)
        m[9 + i, cx - x: cx + x + 1] = True
    m[26, cx - bar_len // 2: cx + bar_len // 2 + 1] = True           # 横杠（阴极侧）
    m[:, cx] = True                                                  # 引线
    return m


def to_json(img_y):
    """图像 y → json y（判据的入参是 json 坐标）。"""
    return float(H - img_y)


class LedPolarityTests(unittest.TestCase):
    def test_bar_side_is_reported_as_k(self):
        # 候选 K 端指在横杠那侧 ⇒ 判据认可（'K'）
        got, prof = ledpol.k_end(glyph(), (20, to_json(26)), (20, to_json(8)), H)
        self.assertEqual(got, "K")
        self.assertTrue(prof)

    def test_flipped_input_flips_verdict(self):
        # 同一个符号，把「候选 K 端」指到底边（阳极）侧 ⇒ 判据说该换向（'A'）
        got, _ = ledpol.k_end(glyph(), (20, to_json(8)), (20, to_json(26)), H)
        self.assertEqual(got, "A")

    def test_bar_longer_than_base_still_decided(self):
        # 横杠比底边长也不影响：判据只看两条横笔之间的斜边跨度，不看游程长短
        got, _ = ledpol.k_end(glyph(bar_len=20), (20, to_json(26)), (20, to_json(8)), H)
        self.assertEqual(got, "K")

    def test_blunt_symbol_is_undecided(self):
        # 两条横笔之间的跨度差 < 2px（这里用一条没有斜边的光杆 + 两个横笔）⇒ 不判
        m = np.zeros((H, W), dtype=bool)
        m[10, 13:28] = True
        m[26, 13:28] = True
        m[:, 20] = True
        got, _ = ledpol.k_end(m, (20, to_json(26)), (20, to_json(10)), H)
        self.assertIsNone(got)

    def test_blank_mask_is_undecided(self):
        got, _ = ledpol.k_end(np.zeros((H, W), dtype=bool),
                              (20, to_json(26)), (20, to_json(8)), H)
        self.assertIsNone(got)

    def test_red_mask_keeps_component_strokes_only(self):
        bgr = np.full((H, W, 3), 255, dtype=np.uint8)
        bgr[5:15, 5:15] = (60, 61, 151)      # 元件体：暗红（BGR）
        bgr[20:30, 5:15] = (120, 150, 40)    # 导线：绿
        bgr[35:45, 5:15] = (157, 108, 100)   # 文字：蓝
        m = ledpol.red_mask(bgr)
        self.assertTrue(m[10, 10])
        self.assertFalse(m[25, 10])
        self.assertFalse(m[40, 10])


if __name__ == "__main__":
    unittest.main()
