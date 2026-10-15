# -*- coding: utf-8 -*-
"""§5.126：导线通道 tp/fn/fp 全账（判定剩余空间在 fn 侧还是 fp 侧）。"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import discover, load_json, DEFAULT_DATA_ROOT   # noqa: E402
from schematic_parser.metrics import EvalConfig                            # noqa: E402
from tools.diag import _wirecut                                            # noqa: E402


def main():
    a_dir = sys.argv[1] if len(sys.argv) > 1 else "work/_fast_out_deterministic_default"
    index = {}
    for c in discover(DEFAULT_DATA_ROOT):
        index.setdefault(c.case_id, c)
    cfg = EvalConfig()
    T = [0, 0, 0, 0, 0]
    print("%-5s %6s %6s %6s %6s %8s %8s %8s" %
          ("case", "nGT", "nPred", "tp", "fn", "fp", "prec", "rec"))
    for cid in [str(i) for i in range(1, 11)]:
        case = index.get(cid)
        fp = os.path.join(a_dir, "final_results", "%s_Pred.json" % cid)
        if case is None or not os.path.exists(fp):
            continue
        gt, pred = load_json(case.json_path), load_json(fp)
        r = _wirecut.score(gt, pred, cfg)[0]
        t = getattr(r, "tp", None)
        f = getattr(r, "fn", None)
        p = getattr(r, "fp", None)
        if t is None:
            print("  (metric 无 tp/fn/fp 属性：%s)" % [k for k in dir(r) if not k.startswith("_")])
            return
        T[0] += t
        T[1] += f
        T[2] += p
        pr = t / (t + p) if t + p else 0.0
        rc = t / (t + f) if t + f else 0.0
        print("%-5s %6d %6d %6d %6d %6d %8.4f %8.4f"
              % (cid, t + f, t + p, t, f, p, pr, rc))
    t, f, p = T[0], T[1], T[2]
    f1 = 2.0 * t / (2.0 * t + f + p) if t else 0.0
    print("\n合计 tp=%d fn=%d fp=%d  →  prec %.4f / rec %.4f / F1 %.4f"
          % (t, f, p, t / (t + p) if t + p else 0, t / (t + f) if t + f else 0, f1))
    print("把 fn 全消掉（fp 不变）的 micro F1 上界 = %.4f"
          % ((2.0 * t + 2.0 * f) / (2.0 * (t + f) + p)))
    print("把 fp 全消掉（fn 不变）的 micro F1 上界 = %.4f"
          % ((2.0 * t) / (2.0 * t + f + 0.0)))
    print("两者全消 = 1.0000；当前 micro F1 = %.4f，报告口径 F1 = %.4f"
          % (f1, getattr(r, "f1", float("nan"))))


if __name__ == "__main__":
    main()
