# -*- coding: utf-8 -*-
"""位号的可达性上限：GT box 元件的位号，在它周围 R 像素内的 stage1 文字框里，
有没有一个（CRNN 已经读出来的）就等于 GT 键？

    python work\\tools\\diag\\_desigring.py --stage1 work\\stage1_v6 --limit 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser.data import discover, load_json  # noqa: E402

IC = ("box", "ic", "boxic")
DES_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")
RADII = (40, 70, 100, 150, 220, 320, 500)


def clean(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s or ""))


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "work" / "stage1_v6"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=10)
    a = ap.parse_args()
    S1 = Path(a.stage1)
    hit = Counter()
    tot = Counter()
    shape = Counter()
    ex = []
    for c in discover():
        if not c.json_path:
            continue
        f = S1 / ("%s.json" % c.case_id)
        if not f.exists():
            continue
        s1 = json.loads(f.read_text(encoding="utf-8"))
        txt = s1.get("texts") or []
        boxes = []
        for t in txt:
            bb = t.get("bbox") or t.get("box")
            if not bb:
                continue
            boxes.append((bb, clean(t.get("text"))))
        gt = load_json(c.json_path)
        for k, g in (gt.get("components") or {}).items():
            if str(g.get("type") or "").lower() not in IC or not g.get("bbox"):
                continue
            gc = center(g["bbox"])
            near = sorted(((dist(gc, center(bb)), s) for bb, s in boxes))[:400]
            for R in RADII:
                tot[R] += 1
                cands = [s for d, s in near if d <= R]
                if k in cands:
                    hit[R] += 1
                    break
            # 记录最近的 5 个框的读法，用来人看
            if len(ex) < a.examples:
                top = sorted(((dist(gc, center(bb)), s) for bb, s in boxes))[:5]
                ex.append((c.case_id, k, [s for _d, s in top]))
        if a.limit and tot[RADII[0]] >= a.limit:
            break
    print("GT box 元件 %d 个（有 stage1 文字的图）" % tot[RADII[0]])
    for R in RADII:
        print("  R<=%4d  位号落在 stage1 文字框里  %5d / %5d = %.4f"
              % (R, hit[R], tot[R], hit[R] / max(1, tot[R])))
    print("\n最近 5 个文字框的 CRNN 读法样例（case key | 读法）：")
    for cid, k, reads in ex:
        print("   %s %s | %s" % (cid, k, reads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
