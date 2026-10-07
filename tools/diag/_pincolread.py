# -*- coding: utf-8 -*-
"""`stage1/pincol.py` 的号码窗直接送进 CRNN，读出来到底对不对？

    python work\\tools\\diag\\_pincolread.py --case 0143 --key U1

对每个枚举出的脚，用 `pincol.win_boxes` 切出 `out_win`（框外号码窗、已收紧），
跑 `TextEngine.read`，再把读法与「最近 GT 引脚号」对账。
用来判定 `STAGE2_PINCOL_WIN` 到底有没有读到号 —— 读了没读对，还是根本没读出来。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EDA = Path(__file__).resolve().parents[3] / "eda-pcb"
sys.path[:0] = [str(EDA)]

from schematic_parser.data import DEFAULT_DATA_ROOT, IC_TYPES, discover, load_json  # noqa: E402
from stage1 import pincol as PC  # noqa: E402
from stage2.text import TextEngine, imread_unicode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--stage1", default=r"Y:\workspace\EDA\work\_fast_out\runs\stage1")
    ap.add_argument("--roles", default="")
    ap.add_argument("--win", default="out_win",
                    help="out_win=框外引脚号 / in_win=框内引脚名")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    p = os.path.join(a.stage1, "%s.stage1.json" % a.case)
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    bgr = imread_unicode((d.get("image") or {}).get("path"))
    cols = d.get("pincol") or {}
    if a.key:
        cols = {k: v for k, v in cols.items() if k == a.key}
    wb = PC.win_boxes(bgr, cols, keys=tuple(a.win.split(",")))
    print("窗数", len(wb))
    if not wb:
        return 1
    roles = tuple(a.roles.split(",")) if a.roles else None
    eng = TextEngine()
    reads = eng.read(bgr, [b for b, _c, _k in wb]) if roles is None \
        else eng.read(bgr, [b for b, _c, _k in wb], roles)

    # GT 引脚点（图像坐标）
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    H = ((d.get("image") or {}).get("h")) or 0
    gnum = []
    if a.case in cases:
        gt = load_json(cases[a.case].json_path)
        gp = gt.get("pins") or {}
        for k, v in (gt.get("components") or {}).items():
            if str((v or {}).get("type", "")).lower() not in IC_TYPES:
                continue
            for pk, info in (gp.get(k) or {}).items():
                pt = (info or {}).get("point") or {}
                if "x" in pt and "y" in pt:
                    gnum.append((float(pt["x"]), float(H) - float(pt["y"]),
                                 pk.replace("pin_", ""), k))
    ok = bad = miss = 0
    for i, (b, cid, key) in enumerate(wb):
        rec = reads[i] if i < len(reads) else {"reads": {}}
        rd = rec.get("reads") or {}
        got = {r: v for r, v in rd.items() if v and v[0]}
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        g = min(gnum, key=lambda t: (t[0] - cx) ** 2 + (t[1] - cy) ** 2) if gnum else None
        near = g and abs(g[0] - cx) <= 6 and abs(g[1] - cy) <= 6
        num = (rd.get("pin_number") or ("", 0))[0]
        if not near:
            miss += 1
            flag = "（无GT）"
        elif num and num == g[2]:
            ok += 1
            flag = "✓"
        elif num:
            bad += 1
            flag = "✗GT=%s" % g[2]
        else:
            miss += 1
            flag = "空GT=%s" % g[2]
        if a.limit and i >= a.limit:
            break
        print("  %-4s %-28s 近GT=%-5s %-12s %s"
              % (cid, "[%d,%d,%d,%d]" % tuple(b), g[2] if near else "-", flag,
                 " ".join("%s=%s" % (k, v[0]) for k, v in sorted(got.items()))))
    print("读对 %d / 读错 %d / 空或无GT %d（共 %d）" % (ok, bad, miss, len(wb)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
