# -*- coding: utf-8 -*-
"""CRNN 验证集快速评测——训练用的那份 manifest（work\\ocr_real\\manifest.jsonl，
3829 条：train 3033 / val 796）+ 训练用的 `collate`，秒级出 CER / exact。

用途：任何改 `textrec` 预处理或解码的实验，先用它筛一轮，再去跑 220 张端到端。

    python work\\tools\\diag\\_crnneval.py                          # 现行行为
    python work\\tools\\diag\\_crnneval.py --pad-w 384               # 整批补白到固定宽度
    python work\\tools\\diag\\_crnneval.py --trunc                   # 按 input_len 截断解码
    TEXTREC_BUCKET=1 python work\\tools\\diag\\_crnneval.py          # 分桶（应与逐条一致）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from textrec import config as C                      # noqa: E402
from textrec.charset import BLANK                    # noqa: E402
from textrec.dataset import CropDataset, collate     # noqa: E402
from textrec.infer import Recognizer                 # noqa: E402
from assets import textrec_ckpt                      # noqa: E402


def decode(charset, col_ids, lps):
    chars, keep, prev = [], [], None
    for t, i in enumerate(col_ids):
        if i != prev and i != BLANK:
            ch = charset.itos.get(i)
            if ch is not None:
                chars.append(ch)
                keep.append(lps[t][i])
        prev = i
    return "".join(chars), (float(sum(keep) / len(keep)) if keep else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "work" / "ocr_real" / "manifest.jsonl"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pad-w", type=int, default=0,
                    help=">0 时把整批补白到这个固定宽度（0=补到本批最大宽度）")
    ap.add_argument("--trunc", action="store_true", help="按 w//WIDTH_REDUCTION 截断解码")
    ap.add_argument("--only", default="", help="只看某个 role")
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条错的")
    ap.add_argument("--shrink", default="",
                    help="先把真值框按 'w,h' 比例向中心收缩，模拟我们检测出的紧框"
                         "（实测 det/gt 宽 p50=0.77、高 p50=0.54）")
    ap.add_argument("--pad-frac", type=float, default=None,
                    help="模拟 stage2 `pad_box` 的外扩比例（默认不套 pad_box）")
    ap.add_argument("--pad-min", type=float, default=2.0)
    ap.add_argument("--pad-max", type=float, default=99.0)
    a = ap.parse_args()

    rec = Recognizer.load(textrec_ckpt(must_exist=False))
    recs = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    val = [r for r in recs if r.get("split") == "val"]
    if a.only:
        val = [r for r in val if r.get("role") == a.only]
    if a.shrink:
        fw, fh = [float(v) for v in a.shrink.replace("x", ",").split(",")]
        for r in val:
            x1, y1, x2, y2 = r["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            hw, hh = (x2 - x1) * fw / 2.0, (y2 - y1) * fh / 2.0
            r["bbox"] = [cx - hw, cy - hh, cx + hw, cy + hh]
    if a.pad_frac is not None:
        for r in val:
            b = r["bbox"]
            p = min(a.pad_max, max(a.pad_min, a.pad_frac * max(1.0, b[3] - b[1])))
            r["bbox"] = [b[0] - p, b[1] - p, b[2] + p, b[3] + p]
    ds = CropDataset(val, rec.charset, augment=False)
    print("[data] val %d（丢弃未知字符 %d）手动补白=%s 截断=%s 分桶=%s shrink=%s pad_frac=%s"
          % (len(ds), ds.n_dropped_unk, a.pad_w or "批内最大", a.trunc,
             os.environ.get("TEXTREC_BUCKET", "0") != "0", a.shrink or "-", a.pad_frac))

    tot = [0, 0, 0, 0]           # ed, len, exact, n
    per = {}
    shown = 0
    for s in range(0, len(ds), a.batch):
        part = ds.records[s:s + a.batch]
        imgs, roles, widths = [], [], []
        from textrec.dataset import crop_gray
        from PIL import Image
        for r in part:
            im = Image.open(r["image"]).convert("L")
            crop = crop_gray(im, r["bbox"], ds.img_h, ds.max_w)
            import numpy as np
            arr = np.asarray(crop, dtype="float32") / 255.0
            imgs.append(torch.from_numpy((arr - 0.5) / 0.5).unsqueeze(0))
            widths.append(arr.shape[1])
            roles.append(int(r.get("role_id", C.ROLE_ID.get(r.get("role"), C.N_ROLES - 1))))
        W = a.pad_w or max(widths)
        h = imgs[0].shape[1]
        batch = torch.full((len(imgs), 1, h, W), 1.0)
        for i, t in enumerate(imgs):
            w = min(t.shape[2], W)
            batch[i, :, :, :w] = t[:, :, :w]
        batch = batch.to(rec.device)
        rt = torch.tensor(roles, dtype=torch.long, device=rec.device)
        with torch.no_grad():
            logp = rec.model(batch, rt).log_softmax(2)
            idx = logp.argmax(2)
        T = idx.shape[0]
        for b, r in enumerate(part):
            tmax = T if not a.trunc else max(1, min(T, min(widths[b], W) // C.WIDTH_REDUCTION))
            pred, _c = decode(rec.charset, idx[:tmax, b].tolist(), logp[:tmax, b].tolist())
            txt = r["text"]
            ed = sum(1 for x, y in zip(pred, txt) if x != y) + abs(len(pred) - len(txt))
            tot[0] += ed
            tot[1] += len(txt)
            tot[2] += 1 if pred == txt else 0
            tot[3] += 1
            d = per.setdefault(r.get("role"), [0, 0, 0, 0])
            d[0] += ed
            d[1] += len(txt)
            d[2] += 1 if pred == txt else 0
            d[3] += 1
            if a.show and pred != txt and shown < a.show:
                shown += 1
                print("   %-28s %-14r -> %-14r" % (r.get("case"), txt, pred))
    n = len(ds)
    print("CER %.4f   exact %.4f (%d/%d)" % (tot[0] / max(1, tot[1]), tot[2] / max(1, n), tot[2], n))
    for k, (e, l, ok, dn) in sorted(per.items()):
        print("   %-12s n=%-4d CER %.4f  exact %.4f" % (k, dn, e / max(1, l), ok / max(1, dn)))


if __name__ == "__main__":
    main()
