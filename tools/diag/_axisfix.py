# -*- coding: utf-8 -*-
"""把 `fusion/axis.py::_selftest` 里的真实夹具拿出来单独看：新长度律下
局部解码与全局铺号各是什么，以及每条 item 的长度似然。

夹具来自 `work/_submit_out220r36/runs/stage2/_assigned/1.json` 的 `desig_slots`
（GT：A 轴 `C151..C156`、C 轴 `C157/C158`、B 轴 `C159/C160/C161`）。
"""
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from fusion import axis as A          # noqa: E402

items = [
    A.Item(key="a1", box=[104.0, 147.0, 157.0, 165.0], text="C151", score=0.885,
           reads=[("C151", 0.885)]),
    A.Item(key="a2", box=[222.0, 147.0, 278.0, 165.0], text="C152", score=0.913,
           reads=[("C152", 0.913)]),
    A.Item(key="a3", box=[340.0, 147.0, 395.0, 165.0], text="C153", score=0.868,
           reads=[("C153", 0.868)]),
    A.Item(key="a4", box=[457.0, 147.0, 514.0, 165.0], text="C154", score=0.767,
           reads=[("C154", 0.767)]),
    A.Item(key="a5", box=[605.0, 147.0, 660.0, 165.0], text="C15", score=0.925,
           reads=[("C15", 0.925)]),
    A.Item(key="a6", box=[753.0, 145.0, 813.0, 166.0], text="C156", score=0.776,
           reads=[("C156", 0.776)]),
    A.Item(key="c1", box=[164.0, 587.0, 225.0, 608.0], text="C157", score=0.833,
           reads=[("C157", 0.833)]),
    A.Item(key="c2", box=[311.0, 588.0, 366.0, 606.0], text="C5", score=0.965,
           reads=[("C5", 0.965)]),
    A.Item(key="b1", box=[488.0, 646.0, 548.0, 667.0], text="C15", score=0.955,
           reads=[("C15", 0.955)]),
    A.Item(key="b2", box=[605.0, 648.0, 661.0, 666.0], text="C10", score=0.840,
           reads=[("C10", 0.840)]),
    A.Item(key="b3", box=[753.0, 646.0, 809.0, 667.0], text="C181", score=0.733,
           reads=[("C181", 0.733)]),
]

print("PAD=%.3f FALLBACK=%.3f SIG=%.3f LEN_CONFLICT=%.2f"
      % (A._PAD, A._ADV_FB, A.CHAR_SIG, A.LEN_CONFLICT))
print("%-4s %-7s %-6s %-6s %-7s %-7s %s"
      % ("key", "text", "w", "h", "w/h", "pred", "len_prob"))
for it in items:
    print("%-4s %-7s %-6.0f %-6.0f %-7.3f %-7.3f %.4f"
          % (it.key, it.text, it.w, it.h, it.w / it.h,
             A.pred_wh_text(it.text), A.len_prob_text(it.w, it.h, it.text)))

axes = A.build_axes(items, min_members=2)
print("\naxes:", [(a.prefix, len(a.items), [x.text for x in a.items],
                   a.note.get("w_med"), a.note.get("trunc")) for a in axes])
for a in axes:
    print("  keep_value =", round(A.keep_value(a), 3))
    top = A.decode(a, lam=1.0, topk=4)
    print("   local top4:", [(t.lo, round(t.value, 3), t.per) for t in top])
plan = A.decode_prefix(axes, topk=4)
if plan is None:
    print("\nglobal plan = None")
else:
    print("\nglobal plan: value=%.3f gaps=%s" % (plan.value, plan.gaps))
    for a, asg, lo in plan.per_axis:
        print("   %-2s lo=%-4s per=%s" % (a.prefix, lo, asg.per))
        print("      cands=%s" % [d.get("cand") for d in asg.detail])
