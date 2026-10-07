# -*- coding: utf-8 -*-
"""同号引脚（`box` IC）的位置误差普查：我们的引脚点 vs GT 的引脚点。

问题：GT 的引脚点贴在**符号框边上**，我们的 stage1 引脚点是从框沿引脚方向追出去
碰到导线才停（可能 10~45px 外）⇒ `metrics.pin_f1` 的 `position_mismatch`。

    python work\\tools\\diag\\_pinpos.py --pred work\\_gt10_vlmh
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M          # noqa: E402
from schematic_parser.data import discover          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", type=int, default=12, help="打印前 N 个样本")
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    cases = {c.case_id: c for c in discover()}
    want = [x.strip() for x in a.cases.split(",") if x.strip()] or None
    cfg = M.EvalConfig()
    dists, inside, outs = [], 0, 0
    samples = []
    for cid, c in sorted(cases.items()):
        if want and cid not in want:
            continue
        pj = pred_dir / ("%s.json" % cid)
        if not (pj.is_file() and c.json_path):
            continue
        gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pr = json.loads(pj.read_text(encoding="utf-8"))
        _f1, keymap = M.assign_components(gt, pr, cfg)
        g2p = {gk: pk for pk, gk in keymap.items()}
        for gk, pk in g2p.items():
            gv = (gt.get("components") or {}).get(gk) or {}
            if str(gv.get("type") or "").lower() not in M.IC_TYPES:
                continue
            gbox = gv.get("bbox") or gv.get("box")
            gl = (gt.get("pins") or {}).get(gk) or {}
            pl = (pr.get("pins") or {}).get(pk) or {}
            pfl = {M.pin_key_to_number(k): v for k, v in pl.items()}
            for gkey, ginfo in gl.items():
                num = M.pin_key_to_number(gkey)
                p = pfl.get(num)
                if p is None:
                    continue
                try:
                    gx = float((ginfo.get("point") or {})["x"])
                    gy = float((ginfo.get("point") or {})["y"])
                    px = float((p.get("point") or {})["x"])
                    py = float((p.get("point") or {})["y"])
                except Exception:  # noqa: BLE001
                    continue
                d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                dists.append(d)
                if gbox:
                    x0, y0, x1, y1 = [float(v) for v in gbox]
                    if x0 <= gx <= x1 and y0 <= gy <= y1:
                        inside += 1
                    else:
                        outs += 1
                if len(samples) < a.show:
                    samples.append((cid, gk, num, (round(gx), round(gy)),
                                    (round(px), round(py)), round(d, 1)))
    print("同号引脚样本 n=%d" % len(dists))
    if dists:
        q = statistics.quantiles(dists, n=20)
        print("  距离 d: min=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f max=%.1f"
              % (min(dists), q[4], statistics.median(dists), q[14], q[17], max(dists)))
        print("  <= pin_tol(%.1f): %d (%.3f)" % (cfg.pin_tol,
              sum(1 for d in dists if d <= cfg.pin_tol),
              sum(1 for d in dists if d <= cfg.pin_tol) / len(dists)))
    print("  GT 引脚点落在 GT 符号框内: %d / 框外: %d" % (inside, outs))
    print()
    for s in samples:
        print("   case %-6s %-5s num=%-5s GT=%-12s PR=%-12s d=%s" % s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
