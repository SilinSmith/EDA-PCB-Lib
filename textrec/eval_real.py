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


def _inter(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _lenbucket(t: str) -> str:
    """GT 长度分桶（与 §5.156 / §5.161 的验收判据同一口径）。"""
    n = len(t)
    return ("1-2" if n <= 2 else "3-4" if n <= 4 else
            "5-7" if n <= 7 else "8-11" if n <= 11 else "12+")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="在真实语料上评 textrec")
    ap.add_argument("--ckpt", default=str(C.RUNS_DIR / "best.pt"))
    ap.add_argument("--manifest", default=str(C.MANIFEST))
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个 case")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dump", default="", help="把读错的样本写成 jsonl")
    ap.add_argument("--role", default="", help="只看某个角色")
    ap.add_argument("--ov-thr", type=float, default=0.10,
                    help="重叠桶阈值：与同图其它文本框的最大包含度 >= 该值记为「有重叠」")
    ap.add_argument("--ov-strong", type=float, default=0.30,
                    help="强重叠桶阈值（同一个包含度口径）")
    ap.add_argument("--no-ov", action="store_true", help="不打印重叠分桶")
    ap.add_argument("--by-len", action="store_true", dest="by_len",
                    help="额外打印 role × GT 长度分桶表（验收判据口径，见 §5.161）")
    args = ap.parse_args(argv)

    rec = Recognizer.load(Path(args.ckpt), device=args.device)
    print("[recognizer] " + rec.debug())

    recs = [r for r in load_manifest(Path(args.manifest)) if r["split"] == "real"]
    # ---- 重叠桶：**只用 GT 自己的框**算（框与框的交 / 较小框面积），不涉及模型的任何输出 ----
    # 真实语料（Dataset300 CVJsonStd）里密集排布的相邻行会让「一个框的裁片含到别行的字」，
    # 这就是本线要对付的「重叠文本」。桶定义必须在评任何权重的**同一个口径**下固定，见 §5.156。
    if not args.no_ov:
        _by_img: Dict[str, List[dict]] = defaultdict(list)
        for r in recs:
            _by_img[r["image"]].append(r)
        for _k, _rs in _by_img.items():
            _a = [_area(r["bbox"]) for r in _rs]
            for _i, _r in enumerate(_rs):
                _best = 0.0
                for _j, _s in enumerate(_rs):
                    if _i == _j or _a[_i] <= 0 or _a[_j] <= 0:
                        continue
                    _iv = _inter(_r["bbox"], _s["bbox"])
                    if _iv > 0:
                        _best = max(_best, _iv / min(_a[_i], _a[_j]))
                _r["ov"] = _best
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
    stats_len = defaultdict(lambda: [0, 0, 0, 0])   # (role, 长度桶) -> ed, len, n, exact
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
            if args.by_len:
                t = stats_len[(r["role"], _lenbucket(gt))]
                t[0] += levenshtein(pred, gt)
                t[1] += max(1, len(gt))
                t[2] += 1
                t[3] += int(pred == gt)
            per_item.append((r["role"], pred, gt, conf, r.get("ov", -1.0)))
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

    # ---- role × GT 长度：验收判据（§5.156 人类裁定 / §5.161 预注册）看的表 ----
    if args.by_len and stats_len:
        buckets = ("1-2", "3-4", "5-7", "8-11", "12+")
        print("\nrole × GT 长度（格内 = exact，判据只看 >=5 那一列）")
        print("%-12s %8s %8s %8s %8s %8s | %8s %7s"
              % ("role", *buckets, ">=5", "n>=5"))
        for role in sorted({k[0] for k in stats_len}):
            cells, ge5n, ge5ex = [], 0, 0
            for b in buckets:
                v = stats_len.get((role, b))
                cells.append("%8.4f" % (v[3] / v[2]) if v and v[2] else "       -")
                if b not in ("1-2", "3-4") and v:
                    ge5n += v[2]
                    ge5ex += v[3]
            print("%-12s %s | %8s %7d"
                  % (role, " ".join(cells),
                     ("%.4f" % (ge5ex / ge5n)) if ge5n else "       -", ge5n))
        cells, ge5n, ge5ex = [], 0, 0
        for b in buckets:
            n = ex = 0
            for k2, v in stats_len.items():
                if k2[1] == b:
                    n += v[2]
                    ex += v[3]
            cells.append("%8.4f" % (ex / n) if n else "       -")
            if b not in ("1-2", "3-4"):
                ge5n += n
                ge5ex += ex
        print("%-12s %s | %8s %7d"
              % ("ALL", " ".join(cells),
                 ("%.4f" % (ge5ex / ge5n)) if ge5n else "       -", ge5n))

    # ---- 位号：真正决定分数的那个数字 ----
    d = stats.get("designator")
    if d and d[2]:
        print("\n★ 位号 designator: n=%d  CER=%.4f  完全正确率=%.4f"
              % (d[2], d[0] / max(1, d[1]), d[3] / max(1, d[2])))
        print("  （这条线的上界见 docs/算法路线设计.md：现有模板匹配端到端是 0.0000）")

    # ---- 重叠分桶：验收线「重叠框 exact 提升且对照组不降」的基线（见 §5.156）----
    if not args.no_ov and any(it[4] >= 0 for it in per_item):
        def _bucket(ov: float):
            if ov < 0:
                return None
            if ov < args.ov_thr:
                return "clean"
            return "mild" if ov < args.ov_strong else "strong"

        bt = defaultdict(lambda: [0, 0, 0, 0])          # ed, len, n, exact
        bt_d = defaultdict(lambda: [0, 0, 0, 0])        # 位号单独一份
        for role, p, g, _c, ov in per_item:
            b = _bucket(ov)
            if b is None:
                continue
            for tbl in ((bt, bt_d) if role == "designator" else (bt,)):
                s = tbl[b]
                s[0] += levenshtein(p, g)
                s[1] += max(1, len(g))
                s[2] += 1
                s[3] += int(p == g)
        print("\n重叠分桶（桶只由 GT 框自身的包含度定：clean <%.2f ≤ mild <%.2f ≤ strong；"
              "模型输出不参与分桶）" % (args.ov_thr, args.ov_strong))
        print("%-8s %7s %8s %9s" % ("bucket", "n", "CER", "exact"))
        for b in ("clean", "mild", "strong"):
            s = bt[b]
            print("%-8s %7d %8.4f %9.4f"
                  % (b, s[2], s[0] / max(1, s[1]), s[3] / max(1, s[2])))
        if bt_d["clean"][2] or bt_d["mild"][2] or bt_d["strong"][2]:
            print("  ↳ 位号 designator：" + "  ".join(
                "%s n=%d exact=%.4f" % (b, bt_d[b][2], bt_d[b][3] / max(1, bt_d[b][2]))
                for b in ("clean", "mild", "strong")))
        # 朴素分桶会被 role/长度组成混淆（clean 桶里长 value 串占多数）⇒ 必须按 role 与长度控制住
        rb = defaultdict(lambda: [0, 0])                # (role, bucket) -> [n, exact]
        bl = defaultdict(lambda: [0, 0])                # (len 桶, bucket) -> [n, exact]
        lg = defaultdict(list)                          # bucket -> [gt 长度]
        for role, p, g, _c, ov in per_item:
            b = _bucket(ov)
            if b is None:
                continue
            rb[(role, b)][0] += 1
            rb[(role, b)][1] += int(p == g)
            L = len(g)
            lb = "1-2" if L <= 2 else ("3-4" if L <= 4 else "5+")
            bl[(lb, b)][0] += 1
            bl[(lb, b)][1] += int(p == g)
            lg[b].append(L)
        print("\n  按 role 控制（每格 n / exact）：")
        print("  %-12s %14s %14s %14s" % ("role", "clean", "mild", "strong"))
        for role in sorted(set(k[0] for k in rb)):
            cells = []
            for b in ("clean", "mild", "strong"):
                n, ex = rb[(role, b)]
                cells.append("%5d / %.3f" % (n, ex / n) if n else "      -      ")
            print("  %-12s %14s %14s %14s" % (role, cells[0], cells[1], cells[2]))
        print("  按 GT 长度控制：")
        print("  %-12s %14s %14s %14s" % ("len", "clean", "mild", "strong"))
        for lb in ("1-2", "3-4", "5+"):
            cells = []
            for b in ("clean", "mild", "strong"):
                n, ex = bl[(lb, b)]
                cells.append("%5d / %.3f" % (n, ex / n) if n else "      -      ")
            print("  %-12s %14s %14s %14s" % (lb, cells[0], cells[1], cells[2]))
        for b in ("clean", "mild", "strong"):
            v = sorted(lg[b])
            if v:
                print("  GT 长度中位 %-6s %.1f（n=%d）" % (b, v[len(v) // 2], len(v)))

    # ---- 置信度阈值扫描 ----
    print("\n置信度阈值扫描（拒识 vs 误读）")
    print("%8s %9s %10s %10s %10s" % ("thr", "保留数", "保留占比", "保留内CER", "保留内exact"))
    for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = [(p, g) for (_r, p, g, c, _ov) in per_item if c >= thr]
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
