# -*- coding: utf-8 -*-
"""渲染鲁棒性阶梯：拿**真实裁片**做单变量改造（γ 曲线 / 模糊 / 重采样 / 二值化 / 整体压黑），
看文字模型 exact 怎么随「笔画核心深浅、灰阶级数、笔画粗细」变化。

用途（§5.169/§5.170）：给合成数据增广定验收判据——控制量是 **dark/ink 剖面**，不是灰阶级数；
二值化（−28.5pp）与模糊到发虚（dark<0.05 时 −23pp 起）是失败模式。

用法（仓根）：
    python -X utf8 tools/diag/_aliasladder.py --manifest <manifest.jsonl> [--limit N] [--variants a,b,c]
每个框**单独**当一张图喂 `read_many`（无版面上下文），所以绝对值低于有上下文的评测口径；
本表只在同协议内做配对比较。`--variants` 留空 = 全部。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from textrec.infer import Recognizer
from textrec.train import levenshtein

DEFAULT_MANIFEST = r"Y:\workspace\EDA\work\fullpop_r8\ruler.jsonl"
DEFAULT_CKPT = r"Y:\workspace\EDA\eda-pcb\weights\textrec_best.pt"


def stats(a):
    g = a.astype(np.uint8)
    return len(np.unique(g)), float((g < 200).mean()), float((g < 128).mean())


VARIANTS = ["orig", "bin", "blur0.5", "blur0.8", "blur1.2", "updown4", "blur1.2_thr",
            "blur1.2_dark", "orig_dark", "orig_light", "gam0.5", "gam1.5", "gam2.0", "gam2.5"]


def make(a, name):
    im = Image.fromarray(a.astype(np.uint8))
    if name == "orig":
        return a
    if name.startswith("gam"):                   # 保背景 255、只改墨迹核心深浅（γ>1 更深）
        g_ = float(name[3:])
        return np.clip(255.0 * (a.astype(np.float32) / 255.0) ** g_, 0, 255).astype(np.uint8)
    if name == "orig_dark":                      # 整体压黑（连背景）——设计失败的对照
        return np.clip(a.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if name == "orig_light":                     # 整体抬亮（降低暗像素占比）
        return np.clip(255.0 - (255.0 - a.astype(np.float32)) * 0.45, 0, 255).astype(np.uint8)
    if name == "bin":
        return np.where(a < 128, 0, 255).astype(np.uint8)
    if name.startswith("blur"):
        r = float(name[4:].split("_")[0])
        arr = np.asarray(im.filter(ImageFilter.GaussianBlur(r))).astype(np.uint8)
        if name.endswith("_thr"):
            return np.where(arr < 128, 0, 255).astype(np.uint8)
        if name.endswith("_dark"):
            return np.clip(arr.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
        return arr
    if name == "updown4":
        b = im.resize((im.width * 4, im.height * 4), Image.LANCZOS).resize(im.size, Image.LANCZOS)
        return np.asarray(b).astype(np.uint8)
    raise ValueError(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    ap.add_argument("--variants", default="", help="逗号分隔；留空 = 全部")
    a = ap.parse_args(argv)
    global VARIANTS
    if a.variants:
        VARIANTS = [v for v in a.variants.split(",") if v in VARIANTS]

    rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    rec = Recognizer.load(Path(a.ckpt), device=a.device)

    agg = {v: {"n": 0, "exact": 0, "lev": 0, "gt": 0, "lv": [], "ik": []} for v in VARIANTS}
    for p, rs in by_img.items():
        im = Image.open(p).convert("L")
        for r in rs:
            x1, y1, x2, y2 = [int(round(v)) for v in r["bbox"]]
            a0 = np.asarray(im.crop((x1, y1, x2, y2)), dtype=np.uint8)
            if a0.size == 0 or not r["text"]:
                continue
            for v in VARIANTS:
                arr = make(a0, v)
                g = Image.fromarray(arr)
                pred, _conf = rec.read_many(g, [((0, 0, g.width, g.height), r["role"])])[0]
                agg[v]["n"] += 1
                agg[v]["exact"] += int(pred == r["text"])
                agg[v]["lev"] += levenshtein(pred, r["text"])
                agg[v]["gt"] += len(r["text"])
                if len(agg[v]["lv"]) < 400:
                    agg[v]["lv"].append(levenshtein(pred, r["text"]))
                    agg[v]["ik"].append(stats(arr))
    print("%-12s %5s %8s %8s %10s  灰阶中位 ink dark" % ("variant", "n", "exact", "CER", "lev/sample"))
    for v in VARIANTS:
        d = agg[v]
        lv = np.array(d["lv"]) if d["lv"] else np.array([0])
        ik = np.array(d["ik"]) if d["ik"] else np.zeros((1, 3))
        print("%-12s %5d %8.4f %8.4f %10.3f  %5.0f %.3f %.3f"
              % (v, d["n"], d["exact"] / max(1, d["n"]), d["lev"] / max(1, d["gt"]), lv.mean(),
                 np.median(ik[:, 0]), ik[:, 1].mean(), ik[:, 2].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
