# -*- coding: utf-8 -*-
"""把「宽框引脚号」的裁片摊开看（含 35px 邻域），用来判「框里到底有几个字形」。

用法：
    python work/tools/diag/_widebox.py --dir work/_fast_out_off9_numd --minw 22 \
        --out work/preview/_widebox.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

CASES = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out_off9_numd"))
    ap.add_argument("--minw", type=float, default=22.0)
    ap.add_argument("--pad", type=float, default=35.0)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--out", default=str(ROOT / "work" / "preview" / "_widebox.png"))
    ap.add_argument("--cases", default="")
    a = ap.parse_args()

    D = Path(a.dir)
    only = [c for c in a.cases.split(",") if c] or CASES
    tiles = []
    for case in only:
        ap_path = D / "runs" / "stage2" / "_assigned" / ("%s.json" % case)
        s1_path = D / "runs" / "stage1" / ("%s.stage1.json" % case)
        if not ap_path.is_file() or not s1_path.is_file():
            continue
        rec = load(ap_path)
        s1 = load(s1_path)
        png = None
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _boxread import find_png  # 原始图（不是 runs/stage1 里的调试叠图）
        png = find_png(case)
        if png is None:
            continue
        img = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_COLOR)
        H = img.shape[0]
        for cid, c in (rec.get("comps") or {}).items():
            pins = c.get("pins") or {}
            for k, pv in pins.items():
                num = (pv or {}).get("number") or {}
                t = str(num.get("text") or "")
                b = num.get("box")
                if not t or not b:
                    continue
                w = float(b[2]) - float(b[0])
                h = float(b[3]) - float(b[1])
                if w < a.minw:
                    continue
                x1, y1, x2, y2 = b
                print("%-3s %-6s pin=%-3s w=%4.1f h=%4.1f text=%-6r conf=%s"
                      % (case, cid, k, w, h, t, round(float(num.get("conf") or 0), 2)))
                ix1 = max(0, int(x1 - a.pad)); ix2 = min(img.shape[1], int(x2 + a.pad))
                iy1 = max(0, int(y1 - a.pad)); iy2 = min(H, int(y2 + a.pad))
                crop = img[iy1:iy2, ix1:ix2].copy()
                cv2.rectangle(crop, (int(x1) - ix1, int(y1) - iy1),
                              (int(x2) - ix1, int(y2) - iy1), (0, 0, 255), 1)
                sc = cv2.resize(crop, None, fx=a.scale, fy=a.scale,
                                interpolation=cv2.INTER_NEAREST)
                head = np.zeros((22, sc.shape[1], 3), np.uint8)
                cv2.putText(head, "%s/%s p%s  read=%s  w=%.0f" % (case, cid, k, t, w),
                            (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                tiles.append(np.vstack([head, sc]))

    if tiles:
        W = max(t.shape[1] for t in tiles)
        out = np.vstack([np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles])
        op = Path(a.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", out)[1].tofile(str(op))
        print("->", op, out.shape)
    print("tiles", len(tiles))


if __name__ == "__main__":
    main()
