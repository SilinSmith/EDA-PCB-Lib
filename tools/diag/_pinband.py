# -*- coding: utf-8 -*-
"""量「某颗 IC 框外的引脚号文字框」到底有没有被检出、又有没有被读成号。

`_numprobe.py` 用的是 `pincol` 的 26px 判据；而 `stage2/assign.py` 给引脚号的实际
窗口宽得多（`STAGE2_PINNUM_NEAR=60`，从**引脚点**量）。于是「池子里少了号」有两种
完全不同的原因，必须分开：
  (a) **文字框没检出** —— 图上那排号在 `stage1.texts` 里就找不到；
  (b) **检出但没读成号** —— 框在，OCR 结果过不了 `assign.PINNUM_RE`
      （`^[A-Za-z]{0,2}\\d{1,3}[A-Za-z]?$`）或 `fit_pinnum`。
本工具把框外面的小文字框按「离框边的距离」列出来，并尽量带上 OCR 读到的文本。

⚠ 文本内容不在 `stage1.texts` 里（那里只有 `id` / `bbox_img`），所以
「读到了什么」只能从 `runs/stage2/_assigned/<case>.json` 的 `pins[*].number`
反查；本工具默认只量 (a)。

用法：
    python work\\tools\\diag\\_pinband.py --dir work\\_probe_dbg --case 1 --cid b0000 --reach 70
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "work", "_probe_dbg"))
    ap.add_argument("--case", default="1")
    ap.add_argument("--cid", default="")
    ap.add_argument("--reach", type=float, default=70.0)
    ap.add_argument("--maxw", type=float, default=60.0)
    ap.add_argument("--maxh", type=float, default=40.0)
    a = ap.parse_args()

    p = os.path.join(a.dir, "runs", "stage1", a.case + ".stage1.json")
    s = json.load(open(p, encoding="utf-8"))
    ts = s.get("texts") or []
    print("case %s · texts 总数 %d" % (a.case, len(ts)))
    for c in (s.get("components") or []):
        if c.get("type") not in ("box", "ic", "boxic", "bigic"):
            continue
        if a.cid and c.get("id") != a.cid:
            continue
        bb = c.get("bbox_img") or c.get("bbox") or []
        if len(bb) < 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bb[:4]]
        rows = []
        for t in ts:
            b = t.get("bbox_img")
            if not b:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in b]
            w, h = bx2 - bx1, by2 - by1
            if w <= 0 or h <= 0 or w > a.maxw or h > a.maxh:
                continue
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            if not (y0 - 20 <= cy <= y1 + 20):
                continue
            dl, dr = x0 - bx2, bx1 - x1
            if dl >= dr:
                d, side = dl, "L"
            else:
                d, side = dr, "R"
            if 0 <= d <= a.reach:
                rows.append((d, side, cx, cy, w, h, t.get("id")))
        rows.sort()
        print("\n== %s  box=[%.0f,%.0f,%.0f,%.0f] （%.0fx%.0f）  框外小文字框 %d 个（<=%.0fpx）"
              % (c.get("id"), x0, y0, x1, y1, x1 - x0, y1 - y0, len(rows), a.reach))
        for d, side, cx, cy, w, h, tid in rows:
            print("   %s d=%5.1f c=(%4.0f,%4.0f) %2.0fx%-3.0f %s"
                  % (side, d, cx, cy, w, h, tid))
        left = sum(1 for r in rows if r[1] == "L")
        print("   ⇒ 左 %d / 右 %d" % (left, len(rows) - left))
    return 0


if __name__ == "__main__":
    sys.exit(main())
