# -*- coding: utf-8 -*-
"""在**真实**语料上评一次——这是唯一诚实的泛化检查。

合成语料的验证集只能说明「没训坏」，说明不了「在真图上能用」。真实标注只有
Dataset300 那 32 个 case 的 `CVJsonStd`（文字在顶层 `texts` 里，带 content 与
location），所以这份评测集很小但**来源是真的**，必须看。

    python -m textrec.eval_real --ckpt textrec/runs/best.pt

输出三个东西：
  1. 按角色的 CER / 完全正确率；
  2. **位号（designator）完全正确率**——这是真正决定分数的指标，因为位号读错
     一个字符会连带丢掉该组件、它的引脚、以及挂在它身上的所有网络；
  3. **置信度阈值扫描**：拒识 vs 误读的取舍曲线。管线应该卡一个阈值，
     宁可拒识也不要吞下错字符串。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from . import config as C
from .infer import Recognizer
from .manifest import load as load_manifest
from .train import levenshtein

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="在真实语料上评 textrec")
    ap.add_argument("--ckpt", default=str(C.RUNS_DIR / "best.pt"))
    ap.add_argument("--manifest", default=str(C.MANIFEST))
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个 case")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dump", default="", help="把读错的样本写成 jsonl")
    ap.add_argument("--role", default="", help="只看某个角色")
    args = ap.parse_args(argv)

    rec = Recognizer.load(Path(args.ckpt), device=args.device)
    print("[recognizer] " + rec.debug())

    recs = [r for r in load_manifest(Path(args.manifest)) if r["split"] == "real"]
    if args.role:
        recs = [r for r in recs if r["role"] == args.role]
    by_image: Dict[str, List[dict]] = defaultdict(list)
    for r in recs:
        by_image[r["image"]].append(r)
    keys = sorted(by_image)
    if args.limit:
        keys = keys[: args.limit]
    if not keys:
        print("!! 真实评测集为空，先跑 python -m textrec.manifest")
        return 2
    n_total = sum(len(by_image[k]) for k in keys)
    print("[data] 真实 %d 个 case / %d 条文字" % (len(keys), n_total))

    # 逐图跑，一次把该图所有文字行批量喂进去。
    stats = defaultdict(lambda: [0, 0, 0, 0])       # ed, len, n, exact
    errors: List[dict] = []
    per_item: List[tuple] = []                      # (role, pred, gt, conf)
    for k in keys:
        items = by_image[k]
        with Image.open(k) as im:
            im = im.convert("L")
            preds = rec.read_many(im, [(r["bbox"], r["role"]) for r in items])
        for r, (pred, conf) in zip(items, preds):
            gt = r["text"]
            s = stats[r["role"]]
            s[0] += levenshtein(pred, gt)
            s[1] += max(1, len(gt))
            s[2] += 1
            s[3] += int(pred == gt)
            per_item.append((r["role"], pred, gt, conf))
            if pred != gt and args.dump:
                errors.append({"image": k, "case": r["case"], "role": r["role"],
                               "gt": gt, "pred": pred, "conf": round(conf, 4),
                               "bbox": r["bbox"]})

    print("\n%-12s %7s %8s %9s %9s" % ("role", "n", "CER", "exact", "完全正确数"))
    for role in sorted(stats):
        ed, ln, n, ex = stats[role]
        print("%-12s %7d %8.4f %9.4f %9d"
              % (role, n, ed / max(1, ln), ex / max(1, n), ex))

    tot_ed = sum(v[0] for v in stats.values())
    tot_ln = sum(v[1] for v in stats.values())
    tot_n = sum(v[2] for v in stats.values())
    tot_ex = sum(v[3] for v in stats.values())
    print("%-12s %7d %8.4f %9.4f %9d"
          % ("ALL", tot_n, tot_ed / max(1, tot_ln), tot_ex / max(1, tot_n), tot_ex))

    # ---- 位号：真正决定分数的那个数字 ----
    d = stats.get("designator")
    if d and d[2]:
        print("\n★ 位号 designator: n=%d  CER=%.4f  完全正确率=%.4f"
              % (d[2], d[0] / max(1, d[1]), d[3] / max(1, d[2])))
        print("  （这条线的上界见 docs/算法路线设计.md：现有模板匹配端到端是 0.0000）")

    # ---- 置信度阈值扫描 ----
    print("\n置信度阈值扫描（拒识 vs 误读）")
    print("%8s %9s %10s %10s %10s" % ("thr", "保留数", "保留占比", "保留内CER", "保留内exact"))
    for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = [(p, g) for (_r, p, g, c) in per_item if c >= thr]
        if not keep:
            print("%8.2f %9d %10s %10s %10s" % (thr, 0, "-", "-", "-"))
            continue
        ed = sum(levenshtein(p, g) for p, g in keep)
        ln = sum(max(1, len(g)) for _p, g in keep)
        ex = sum(int(p == g) for p, g in keep)
        print("%8.2f %9d %9.2f%% %10.4f %10.4f"
              % (thr, len(keep), 100.0 * len(keep) / max(1, len(per_item)),
                 ed / max(1, ln), ex / len(keep)))

    if args.dump and errors:
        p = Path(args.dump)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print("\n%d 条读错的样本 -> %s" % (len(errors), p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
