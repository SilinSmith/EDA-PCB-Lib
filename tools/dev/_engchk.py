# -*- coding: utf-8 -*-
"""**口径对照**：同一条框，走「语料口径（裸框裁片）」与「stage2 运行口径（`pad_box` 外扩）」
两条路各读一遍，权重可用 `--ckpt-a/--ckpt-b` 分别指定。

用途：判定域内微调权重在管线里掉分（见 `eda-pcb\\docs\\DIAGNOSTICS.md`「域内微调权重不能直接
换进 stage2」）到底是**模型漂移**还是**裁片口径不同** —— 语料裁片是**未外扩**的原始框
（`image.crop(bbox)`），而 `stage2\\text.py:384 TextEngine.read` 会先 `pad_box`
（水平 `PAD_MIN=2.0`、垂直 `PAD_FRAC_Y=0.20/PAD_MIN_Y=3.0`）再喂模型。

用法：
    python -X utf8 work\\tools\\dev\\_engchk.py --role designator --text C152,C155 --case 1
    python -X utf8 work\\tools\\dev\\_engchk.py --role value --text 10nF --limit 8 --sheet work\\_engchk.png
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
    ap.add_argument("--text", default="")
    ap.add_argument("--case", default="")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--ckpt-a", default=str(WORK / "ocr_mix" / "runs" / "best.pt"))
    ap.add_argument("--ckpt-b", default=str(WORK / "ocr_official_v1" / "best.pt"))
    ap.add_argument("--sheet", default=str(WORK / "_engchk.png"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    want_t = set(t.strip() for t in a.text.split(",") if t.strip())
    want_c = set(t.strip() for t in a.case.split(",") if t.strip())
    rows = []
    with open(a.manifest, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("role") != a.role:
                continue
            if want_t and str(r.get("text")) not in want_t:
                continue
            if want_c and str(r.get("case")) not in want_c:
                continue
            rows.append(r)
            if len(rows) >= a.limit:
                break
    print("[行数] %d" % len(rows))

    import numpy as np
    from PIL import Image, ImageDraw
    from textrec.infer import Recognizer
    from stage2.text import TextEngine, imread_unicode, pad_box

    R = {"a": Recognizer.load(a.ckpt_a, device=a.device), "b": Recognizer.load(a.ckpt_b, device=a.device)}
    E = {"a": TextEngine(ckpt=a.ckpt_a, device=a.device), "b": TextEngine(ckpt=a.ckpt_b, device=a.device)}

    tiles = []
    for r in rows:
        p = pathlib.Path(r["image"])
        img = Image.open(p).convert("RGB")
        g = img.convert("L")
        bgr = np.asarray(img)[:, :, ::-1].copy()
        bb = tuple(float(v) for v in r["bbox"])
        out = {}
        for k in ("a", "b"):
            raw = R[k].read_many(g, [(bb, r["role"])])[0][0]
            eng = E[k].read(bgr, [bb], roles=(r["role"],))[0]["reads"][r["role"]][0]
            out[k] = (raw, eng)
        flag = "" if out["a"] == out["b"] else "≠"
        print("%-2s case=%-5s GT=%-8s | 语料口径 a=%-10r b=%-10r | stage2口径 a=%-10r b=%-10r"
              % (flag, r.get("case"), r.get("text"), out["a"][0], out["b"][0], out["a"][1], out["b"][1]))
        cb = pad_box(list(bb))
        crop = g.crop((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))
        padc = g.crop((int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3])))
        tiles.append((crop, padc))

    if tiles and a.sheet:
        H = sum(max(t[0].height, t[1].height) + 6 for t in tiles)
        W = max(t[0].width + t[1].width + 12 for t in tiles)
        out_img = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(out_img)
        y = 0
        for crop, padc in tiles:
            out_img.paste(crop, (2, y))
            out_img.paste(padc, (crop.width + 10, y))
            d.line([(crop.width + 4, y), (crop.width + 4, y + max(crop.height, padc.height))], fill=0)
            y += max(crop.height, padc.height) + 6
        out_img.save(a.sheet)
        print("[sheet]", a.sheet, "（左=语料口径裸框，右=stage2 口径外扩框）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
