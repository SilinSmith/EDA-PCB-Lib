# -*- coding: utf-8 -*-
"""调试 axis 全局解码：打印每条轴的候选窗口与分。"""
import sys

sys.path.insert(0, "Y:/workspace/EDA/eda-pcb")
from fusion import axis as A  # noqa: E402

items = [
    A.Item(key="a1", box=[104.0, 147.0, 157.0, 165.0], text="C151", score=0.885, reads=[("C151", 0.885)]),
    A.Item(key="a2", box=[222.0, 147.0, 278.0, 165.0], text="C152", score=0.913, reads=[("C152", 0.913)]),
    A.Item(key="a3", box=[340.0, 147.0, 395.0, 165.0], text="C153", score=0.868, reads=[("C153", 0.868)]),
    A.Item(key="a4", box=[457.0, 147.0, 514.0, 165.0], text="C154", score=0.767, reads=[("C154", 0.767)]),
    A.Item(key="a5", box=[605.0, 147.0, 660.0, 165.0], text="C15", score=0.925, reads=[("C15", 0.925)]),
    A.Item(key="a6", box=[753.0, 145.0, 813.0, 166.0], text="C156", score=0.776, reads=[("C156", 0.776)]),
    A.Item(key="c1", box=[164.0, 587.0, 225.0, 608.0], text="C157", score=0.833, reads=[("C157", 0.833)]),
    A.Item(key="c2", box=[311.0, 588.0, 366.0, 606.0], text="C5", score=0.965, reads=[("C5", 0.965)]),
    A.Item(key="b1", box=[488.0, 646.0, 548.0, 667.0], text="C15", score=0.955, reads=[("C15", 0.955)]),
    A.Item(key="b2", box=[605.0, 648.0, 661.0, 666.0], text="C10", score=0.840, reads=[("C10", 0.840)]),
    A.Item(key="b3", box=[753.0, 646.0, 809.0, 667.0], text="C181", score=0.733, reads=[("C181", 0.733)]),
]
axes = A.build_axes(items, min_members=2)
print("轴数 %d，长度 %s" % (len(axes), [len(a) for a in axes]))
for ax in axes:
    print("--- 轴 len=%d  cy 带 %.1f  读数 %s" % (len(ax), min(i.cy for i in ax.items),
                                                [i.text for i in ax.items]))
    for c in A.decode(ax, topk=8):
        print("    lo=%-5s hi=%-5s value=%.3f  cands=%s"
              % (c.lo, c.hi, c.value, [d.get("cand") for d in c.detail]))
plan = A.decode_prefix(axes, topk=4)
print("全局 plan:", plan.as_dict() if plan else None)
