# -*- coding: utf-8 -*-
"""**裁片口径核对**：同一条语料裁片，分别用两个权重读一遍。

用来判定「域内微调权重换进 stage2 反而掉分」是**模型漂移**还是**裁片口径不同**：
  1. 从 `work\\official_text\\textrec_manifest.jsonl` 挑若干行（`--role/--text/--case`）；
  2. 用 `--ckpt-a`（默认交付权重）与 `--ckpt-b`（默认域内微调权重）各读一遍；
  3. 同时把裁片拼成 `--sheet` 方便肉眼比。

用法：
    python -X utf8 work\\tools\\dev\\_cropchk.py --role designator --text C152,C153,C159,R164
    python -X utf8 work\\tools\\dev\\_cropchk.py --role value --text 10nF,1uF --limit 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

WORK = pathlib.Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(WORK / "official_text" / "textrec_manifest.jsonl"))
    ap.add_argument("--role", default="designator")
    ap.add_argument("--text", default="", help="逗号分隔的 GT 文本过滤（空 = 不过滤）")
    ap.add_argument("--case", default="", help="逗号分隔的 case 过滤")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--ckpt-a", default=str(WORK / "ocr_mix" / "runs" / "best.pt"))
    ap.add_argument("--ckpt-b", default=str(WORK / "ocr_official_v1" / "best.pt"))
    ap.add_argument("--sheet", default=str(WORK / "_cropchk.png"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    want_t = set(t.strip() for t in a.text.split(",") if t.strip())
    want_c = set(t.strip() for t in a.case.split(",") if t.strip())
    rows = []
    with open(a.manifest, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if r.get("role") != a.role:
                continue
            if want_t and str(r.get("text")) not in want_t:
                continue
            if want_c and str(r.get("case")) not in want_c:
                continue
            rows.append(r)
            if len(rows) >= a.limit:
                break
    print("[行数] %d  role=%s" % (len(rows), a.role))

    from PIL import Image, ImageDraw
    from textrec.infer import Recognizer
    A = Recognizer.load(a.ckpt_a, device=a.device)
    B = Recognizer.load(a.ckpt_b, device=a.device)

    tiles = []
    for r in rows:
        p = pathlib.Path(r["image"])
        if not p.is_absolute():
            p = WORK / "official_text" / p
        if not p.is_file():
            print("  缺图", p)
            continue
        im = Image.open(p).convert("L")
        bb = tuple(r["bbox"])
        ta = A.read_many(im, [(bb, r["role"])])[0]
        tb = B.read_many(im, [(bb, r["role"])])[0]
        mark = "  " if ta[0] == tb[0] else "≠ "
        print("%scase=%s GT=%-8s src=%-16s base=%-10r(%.2f) ft=%-10r(%.2f)"
              % (mark, r.get("case"), r.get("text"), str(r.get("src"))[:16], ta[0], ta[1], tb[0], tb[1]))
        crop = im.crop((int(bb[0]) - 2, int(bb[1]) - 2, int(bb[2]) + 2, int(bb[3]) + 2))
        tiles.append(crop)
    if tiles and a.sheet:
        W = max(t.width for t in tiles) + 4
        H = sum(t.height + 4 for t in tiles)
        out = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(out)
        y = 0
        for t in tiles:
            out.paste(t, (2, y))
            y += t.height + 4
        out.save(a.sheet)
        print("[sheet]", a.sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
