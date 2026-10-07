# -*- coding: utf-8 -*-
"""诊断 axis 两层解码在 case 1 三条真实轴上的行为（读数逐条抄自 _assigned/1.json）。"""
import sys
from fusion.axis import (Item, build_axes, decode, decode_prefix, score_at,
                         keep_value, MIN_GAIN, KEEP_BASE, RUN_BONUS)

items = [
    Item(key="a1", box=[104.0, 147.0, 157.0, 165.0], text="C151", score=0.885, reads=[("C151", 0.885)]),
    Item(key="a2", box=[222.0, 147.0, 278.0, 165.0], text="C152", score=0.913, reads=[("C152", 0.913)]),
    Item(key="a3", box=[340.0, 147.0, 395.0, 165.0], text="C153", score=0.868, reads=[("C153", 0.868)]),
    Item(key="a4", box=[457.0, 147.0, 514.0, 165.0], text="C154", score=0.767, reads=[("C154", 0.767)]),
    Item(key="a5", box=[605.0, 147.0, 660.0, 165.0], text="C15", score=0.925, reads=[("C15", 0.925)]),
    Item(key="a6", box=[753.0, 145.0, 813.0, 166.0], text="C156", score=0.776, reads=[("C156", 0.776)]),
    Item(key="c1", box=[164.0, 587.0, 225.0, 608.0], text="C157", score=0.833, reads=[("C157", 0.833)]),
    Item(key="c2", box=[311.0, 588.0, 366.0, 606.0], text="C5", score=0.965, reads=[("C5", 0.965)]),
    Item(key="b1", box=[488.0, 646.0, 548.0, 667.0], text="C15", score=0.955, reads=[("C15", 0.955)]),
    Item(key="b2", box=[605.0, 648.0, 661.0, 666.0], text="C10", score=0.840, reads=[("C10", 0.840)]),
    Item(key="b3", box=[753.0, 646.0, 809.0, 667.0], text="C181", score=0.733, reads=[("C181", 0.733)]),
]

axes = build_axes(items, min_members=2)
print("聚出 %d 条轴" % len(axes))
for ax in axes:
    need = max(2, (len(ax) + 1) // 2)
    print("== n=%d cy=%.1f keep=%.3f need=%d reads=%s"
          % (len(ax), min(it.cy for it in ax.items), keep_value(ax), need,
             [it.text for it in ax.items]))
    for c in decode(ax, topk=6):
        nh = sum(1 for p in c.per if p is not None)
        ok = "OK" if (nh >= need and c.value > keep_value(ax) + MIN_GAIN) else "--"
        print("   %s lo=%-5d val=%.3f hits=%d cands=%s"
              % (ok, c.lo, c.value, nh, [d.get("cand") for d in c.detail]))

plan = decode_prefix(axes, topk=4)
print("\nPLAN =", plan.as_dict() if plan else None)
