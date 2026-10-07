# -*- coding: utf-8 -*-
"""按 IoU 配对（不看键）比较两版预测的 component.value 正确率。

    python work\\tools\\diag\\_valiou.py --a work\\stage2_v17 --b work\\stage2_v18
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import discover, load_json  # noqa: E402

VAL_TYPES = ("r", "c", "l", "fuse", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin")


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    it = (x1 - x0) * (y1 - y0)
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1])); bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return it / (aa + bb - it)


def norm(s):
    return "" if s is None else str(s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--b", default=str(ROOT / "work" / "stage2_v18"))
    a = ap.parse_args()
    A, B = Path(a.a), Path(a.b)
    st = Counter()
    for c in discover():
        if not c.json_path:
            continue
        fa, fb = A / ("%s.json" % c.case_id), B / ("%s.json" % c.case_id)
        if not (fa.exists() and fb.exists()):
            continue
        gt = load_json(c.json_path)
        gcomp = {k: v for k, v in (gt.get("components") or {}).items()
                 if v.get("bbox") and str(v.get("type") or "").lower() in VAL_TYPES}
        pa = json.loads(fa.read_text(encoding="utf-8")).get("components") or {}
        pb = json.loads(fb.read_text(encoding="utf-8")).get("components") or {}
        for gk, gv in gcomp.items():
            gval = norm(gv.get("value"))
            for tag, pp in (("a", pa), ("b", pb)):
                best, bk = 0.0, None
                for k, v in pp.items():
                    if not v.get("bbox") or str(v.get("type") or "").lower() not in VAL_TYPES:
                        continue
                    s = iou(v["bbox"], gv["bbox"])
                    if s > best:
                        best, bk = s, k
                if best < 0.5 or bk is None:
                    st[tag + "_nomatch"] += 1
                    continue
                st[tag + "_gtempty" if not gval else tag + "_gtfull"] += 1
                if not gval:
                    st[tag + "_wrote_on_empty"] += (1 if norm(pp[bk].get("value")) else 0)
                else:
                    st[tag + "_ok"] += (1 if norm(pp[bk].get("value")) == gval else 0)
    print("按 IoU 配对（GT 分立元件，不看键）")
    for tag, who in (("a", Path(a.a).name), ("b", Path(a.b).name)):
        full = st[tag + "_gtfull"]; ok = st[tag + "_ok"]
        emp = st[tag + "_gtempty"]; wr = st[tag + "_wrote_on_empty"]
        print("  %-12s GT非空 %4d 值对 %4d = %.4f | GT空 %4d 多写 %4d | 没配上 %d"
              % (who, full, ok, ok / max(1, full), emp, wr, st[tag + "_nomatch"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
