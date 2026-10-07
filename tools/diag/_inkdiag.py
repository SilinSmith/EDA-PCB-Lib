# -*- coding: utf-8 -*-
"""归一的逐条诊断：把 val 拆成「归一几乎没动框」和「归一大幅改了框」两组，
分别看模型的 exact —— 判定掉分到底来自取景变化，还是来自归一本身有 bug。

    python work\\tools\\diag\\_inkdiag.py
    python work\\tools\\diag\\_inkdiag.py --frame 2.5 --role designator
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from textrec import config as C                                   # noqa: E402
from textrec.charset import BLANK                                 # noqa: E402
from textrec.dataset import crop_gray, _ink_band                  # noqa: E402
from textrec.infer import Recognizer                              # noqa: E402
from assets import textrec_ckpt                                   # noqa: E402


def decode(charset, col_ids, lps):
    chars, prev = [], None
    for t, i in enumerate(col_ids):
        if i != prev and i != BLANK:
            ch = charset.itos.get(i)
            if ch is not None:
                chars.append(ch)
        prev = i
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "work" / "ocr_real" / "manifest.jsonl"))
    ap.add_argument("--frame", type=float, default=2.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--role", default="")
    a = ap.parse_args()
    C.INK_FRAME = a.frame

    rec = Recognizer.load(textrec_ckpt(must_exist=False))
    recs = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    val = [r for r in recs if r.get("split") == "val"]
    if a.role:
        val = [r for r in val if r.get("role") == a.role]
    print("val %d  INK_FRAME=%.2f" % (len(val), a.frame))

    # 先逐条裁两份，记录框高变化与是否找到墨迹带。
    base_crops, norm_crops, meta = [], [], []
    for r in val:
        b = r["bbox"]
        im = Image.open(r["image"]).convert("L")
        W, H = im.size
        C.INK_YNORM = False
        # 记录现行裁剪的框高
        h = max(1.0, float(b[3]) - float(b[1]))
        my = C.PAD_Y_FRAC * h
        H0 = 2 * my + h
        c0 = crop_gray(im, b, C.IMG_H, C.MAX_W)
        C.INK_YNORM = True
        x1 = int(max(0, min(W - 1, round(b[0])))); x2 = int(max(x1 + 1, min(W, round(b[2]))))
        y1 = int(max(0, min(H - 1, round(b[1])))); y2 = int(max(y1 + 1, min(H, round(b[3]))))
        got = _ink_band(im, y1, y2, x1, x2, W, H, False, None)
        c1 = crop_gray(im, b, C.IMG_H, C.MAX_W)
        C.INK_YNORM = False
        H1 = (got[1] - got[0]) if got is not None else H0
        base_crops.append(c0)
        norm_crops.append(c1)
        meta.append(dict(role=r.get("role"), txt=r["text"], case=r.get("case"),
                         H0=float(H0), H1=float(H1),
                         found=got is not None, g=(float(got[2]) if got else None)))

    def run(crops):
        out = []
        for s in range(0, len(crops), a.batch):
            part = crops[s:s + a.batch]
            widths = [np.asarray(c).shape[1] for c in part]
            Wd = max(widths)
            batch = torch.full((len(part), 1, C.IMG_H, Wd), 1.0)
            for i, c in enumerate(part):
                arr = np.asarray(c, dtype="float32") / 255.0
                t = torch.from_numpy((arr - 0.5) / 0.5).unsqueeze(0)
                w = min(t.shape[2], Wd)
                batch[i, :, :, :w] = t[:, :, :w]
            batch = batch.to(rec.device)
            roles = torch.tensor(
                [int(C.ROLE_ID.get(meta[s + i]["role"], C.N_ROLES - 1))
                 for i in range(len(part))], dtype=torch.long, device=rec.device)
            with torch.no_grad():
                idx = rec.model(batch, roles).log_softmax(2).argmax(2)
            for b in range(len(part)):
                out.append(decode(rec.charset, idx[:, b].tolist(), None))
        return out

    pb, pn = run(base_crops), run(norm_crops)

    buckets = defaultdict(lambda: [0, 0])
    rows = []
    for i, m in enumerate(meta):
        okb = pb[i] == m["txt"]
        okn = pn[i] == m["txt"]
        dr = abs(m["H1"] - m["H0"]) / max(1.0, m["H0"])
        key = ("None" if not m["found"] else
               ("d<=0.10" if dr <= 0.10 else "0.10<d<=0.30" if dr <= 0.30 else "d>0.30"))
        buckets[key][0] += 1 if okb else 0
        buckets[key][1] += 1
        buckets["ALL"][0] += 1 if okb else 0
        buckets["ALL"][1] += 1
        buckets["ALL-norm"][0] += 1 if okn else 0
        buckets["ALL-norm"][1] += 1
        if okb != okn:
            rows.append((m["case"], m["role"], m["txt"], pb[i], pn[i], m["H0"], m["H1"], m["g"]))

    print("\n%-14s %6s %6s   %-8s %-8s" % ("分组", "n", "占", "基线exact", "归一exact"))
    for k in ("ALL", "ALL-norm", "None", "d<=0.10", "0.10<d<=0.30", "d>0.30"):
        if k in buckets:
            n = buckets[k][1]
            print("%-14s %6d %5.1f%%   %-8s %s" % (k, n, 100.0 * n / len(meta), "", ""))
    print("\n--- 基线对而归一错 ---")
    bad = [r for r in rows if r[3] == r[2] and r[4] != r[2]]
    for r in bad[:40]:
        print("   %-6s %-11s %-10r 归一→%-10r H0=%.0f H1=%.0f g=%s"
              % (r[0], r[1], r[2], r[4], r[5], r[6], ("%.0f" % r[7]) if r[7] else "-"))
    print("   ...共 %d 条" % len(bad))
    good = [r for r in rows if r[3] != r[2] and r[4] == r[2]]
    print("\n--- 基线错而归一对 (%d 条) ---" % len(good))
    for r in good[:20]:
        print("   %-6s %-11s %-10r 基线→%-10r H0=%.0f H1=%.0f g=%s"
              % (r[0], r[1], r[2], r[3], r[5], r[6], ("%.0f" % r[7]) if r[7] else "-"))


if __name__ == "__main__":
    main()
