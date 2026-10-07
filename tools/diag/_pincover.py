# -*- coding: utf-8 -*-
"""量「引脚几何」的可用性：GT box 元件的引脚、我们预测里的引脚、stage1 检出的引脚点。

问题：`PinF1` 只拿 0.0474。已知「引脚号从来没从图上读」。但如果**几何点**
（引脚在图上的位置）本来就检出来了，那 VLM 读出「号->名」之后按序贴回去就行；
如果几何点本来就少，就得先补检测。这个脚本回答的就是这个。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_pincover.py --pred work\\stage2_v17 --stage1 work\\stage1_v6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

import schematic_parser.data as D  # noqa: E402
from schematic_parser.data import discover, load_json  # noqa: E402

IC = {"box", "ic", "boxic"}


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--stage1", default=str(ROOT / "work" / "stage1_v6"))
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--samples", type=int, default=12)
    a = ap.parse_args(argv)

    pred_dir = Path(a.pred); s1_dir = Path(a.stage1)
    n_gt_box = n_gt_pins = n_pred_pins = n_s1_pts = 0
    n_pair_pt = 0          # GT 引脚点在我们检出的点里有对应（<=tol）
    n_comp = n_matched = 0
    samples = []

    for c in discover():
        if not c.json_path:
            continue
        pf = pred_dir / ("%s.json" % c.case_id)
        sf = s1_dir / ("%s.stage1.json" % c.case_id)
        if not pf.exists() or not sf.exists():
            continue
        gt = load_json(c.json_path)
        pred = json.loads(pf.read_text(encoding="utf-8"))
        s1 = json.loads(sf.read_text(encoding="utf-8"))
        H = float(s1.get("image", {}).get("h") or 0)
        gcomp = gt.get("components") or {}
        gpins = gt.get("pins") or {}
        pcomp = pred.get("components") or {}
        ppins = pred.get("pins") or {}
        # stage1 元件的引脚点（图像坐标，y 向下）-> 预测坐标（y = H - y）
        s1pts = []
        for cid, pts in (s1.get("pins") or {}).items():
            for p in pts or []:
                s1pts.append((float(p[0]), H - float(p[1])))
        for k, v in gcomp.items():
            if str(v.get("type") or "").lower() not in IC:
                continue
            n_comp += 1
            n_gt_box += 1
            gp = gpins.get(k) or {}
            n_gt_pins += len(gp)
            gb = v.get("bbox")
            if not gb:
                continue
            best, bk = 0.0, None
            for pk, pv in pcomp.items():
                bb = pv.get("bbox")
                if not bb:
                    continue
                s = iou(gb, bb)
                if s > best:
                    best, bk = s, pk
            if best >= 0.5:
                n_matched += 1
                n_pred_pins += len(ppins.get(bk) or {})
                for _pk, info in (gp.items()):
                    pt = (info or {}).get("point") or {}
                    try:
                        q = (float(pt.get("x")), float(pt.get("y")))
                    except Exception:  # noqa: BLE001
                        continue
                    if any((q[0] - s[0]) ** 2 + (q[1] - s[1]) ** 2 <= a.tol ** 2
                           for s in s1pts):
                        n_pair_pt += 1
                if len(samples) < a.samples:
                    samples.append((c.case_id, k, len(gp), len(ppins.get(bk) or {}),
                                    best, str(v.get("Name"))[:18]))

    print("GT box 元件 %d 个，其中与预测框 IoU>=0.5 的 %d 个" % (n_gt_box, n_matched))
    print("GT box 引脚总数            %d" % n_gt_pins)
    print("  配对元件里我们的引脚数    %d  (%.4f)" % (n_pred_pins, n_pred_pins / max(1, n_gt_pins)))
    print("  GT 引脚点被 stage1 覆盖   %d  (%.4f)  [tol=%g px]"
          % (n_pair_pt, n_pair_pt / max(1, n_gt_pins), a.tol))
    print()
    print("%-8s %-6s %6s %6s %6s  %s" % ("case", "key", "GT", "ours", "IoU", "Name"))
    for s in samples:
        print("%-8s %-6s %6d %6d %6.2f  %s" % s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
