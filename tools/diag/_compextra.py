# -*- coding: utf-8 -*-
"""把 `component.extra`（我们多出来的元件 = 纯 fp）按**类别 / 尺寸 / 与已匹配框的关系**分桶。

`component` 权重 0.30，是最大的一块。220 张当前 tp 5320 / fp 4414 / fn 4289。

用法：
    python work\\tools\\diag\\_compextra.py --pred work\\_submit_out220r26\\runs\\stage2 [--top 20] [--show 0001]

输出：
  ① 全库汇总：fp 的 `canon_type` 分布 vs tp 的分布（看是不是某几类在造 fp）
  ② fp 的 bbox 面积分位 vs tp 的
  ③ fp 与「已匹配 tp 框」的 IoU 关系：完全重叠（重复检出）/ 部分重叠（切碎）/ 无重叠（凭空）
  ④ `--show <case>` 打印单张的 fp/tp/fn 逐条
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eda-pcb"))
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser import data as D                                    # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components         # noqa: E402


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = x1 - x0, y1 - y0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b):
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))


def pct(vals, q):
    if not vals:
        return 0.0
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--root", default=D.DEFAULT_DATA_ROOT)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--show", default="")
    ap.add_argument("--iou", type=float, default=0.30, help="判定「切碎」的 IoU 门槛")
    a = ap.parse_args()

    cases = {c.case_id: c for c in D.discover(a.root)}
    files = sorted(Path(a.pred).glob("*.json"))
    print("预测目录 %s  文件 %d 份" % (a.pred, len(files)))

    by_type_fp, by_type_tp = Counter(), Counter()
    by_type_fn = Counter()
    afp, atp = [], []
    rel = Counter()
    overlap_with = Counter()
    tot_fp = tot_tp = tot_fn = 0
    per_case = []

    for f in files:
        case = cases.get(f.stem)
        if case is None:
            continue
        gt = D.load_json(case.json_path)
        pr = D.load_json(str(f))
        cfg = EvalConfig()
        _f1, km = assign_components(gt, pr, cfg)
        gc = gt.get("components") or {}
        pc = pr.get("components") or {}
        matched_p = set(km.keys())
        matched_g = set(km.values())
        fps = [(k, v) for k, v in pc.items() if k not in matched_p]
        tps = [(k, v) for k, v in pc.items() if k in matched_p]

        # 已匹配的预测框（用来判「切碎」）
        tpb = {}
        for k, v in tps:
            b = v.get("bbox")
            if b:
                tpb[k] = b
        n_cut = n_dup = n_free = 0
        for k, v in fps:
            et = str(v.get("type") or "")
            by_type_fp[et] += 1
            b = v.get("bbox")
            if not b:
                rel["no_bbox"] += 1
                continue
            afp.append(area(b))
            best_k, best_i = None, 0.0
            for k2, b2 in tpb.items():
                if k2 == k:
                    continue
                j = iou(b, b2)
                if j > best_i:
                    best_i, best_k = j, k2
            if best_i >= 0.60:
                rel["dup(>=0.60)"] += 1
                n_dup += 1
                overlap_with[best_k] += 1
            elif best_i >= a.iou:
                rel["cut(%.2f~0.60)" % a.iou] += 1
                n_cut += 1
            elif best_i > 0.0:
                rel["touch(0~%.2f)" % a.iou] += 1
                n_free += 1
            else:
                rel["free(0)"] += 1
                n_free += 1
        for k, v in tps:
            by_type_tp[str(v.get("type") or "")] += 1
            b = v.get("bbox")
            if b:
                atp.append(area(b))
        for k, v in gc.items():
            if k not in matched_g:
                by_type_fn[str(v.get("type") or "")] += 1
        tot_fp += len(fps)
        tot_tp += len(tps)
        tot_fn += sum(1 for k in gc if k not in matched_g)
        per_case.append((len(fps), f.stem))

    print("\n总计  tp %d  fp %d  fn %d" % (tot_tp, tot_fp, tot_fn))
    print("\n① fp 的类别分布（fp 数 / 该类在 tp 中的数 / fn 数）")
    rows = by_type_fp.most_common()
    for et, n in rows[:a.top]:
        print("   %-14s fp %5d   tp %6d   fn %5d   fp 占该类预测 %.1f%%"
              % (et or "(空)", n, by_type_tp.get(et, 0), by_type_fn.get(et, 0),
                 100.0 * n / max(1, n + by_type_tp.get(et, 0))))
    print("   --- 未列出的 fp 合计 %d" % sum(n for _, n in rows[a.top:]))

    print("\n② bbox 面积分位（px²）")
    for name, arr in (("tp", atp), ("fp", afp)):
        print("   %-3s n=%6d  p10 %9.0f  p25 %9.0f  p50 %9.0f  p75 %9.0f  p90 %9.0f"
              % (name, len(arr), pct(arr, .10), pct(arr, .25), pct(arr, .50),
                 pct(arr, .75), pct(arr, .90)))

    print("\n③ fp 与已匹配 tp 框的 IoU 关系")
    tot = sum(rel.values()) or 1
    for k, n in rel.most_common():
        print("   %-16s %5d  (%.1f%%)" % (k, n, 100.0 * n / tot))
    if overlap_with:
        print("   被重复检出的 tp 框 top: %s" % overlap_with.most_common(8))

    print("\n④ fp 最多的 case top%d" % a.top)
    for n, cid in sorted(per_case, reverse=True)[:a.top]:
        print("   %-16s fp %4d" % (cid, n))

    if a.show:
        case = cases.get(a.show)
        if case:
            gt = D.load_json(case.json_path)
            pr = D.load_json(str(Path(a.pred) / ("%s.json" % a.show)))
            cfg = EvalConfig()
            _f1, km = assign_components(gt, pr, cfg)
            gc, pc = gt.get("components") or {}, pr.get("components") or {}
            mg = set(km.values())
            print("\n=== case %s  GT %d 颗 / 我们 %d 颗 / 配上 %d" % (a.show, len(gc), len(pc), len(km)))
            for k, v in pc.items():
                if k in km:
                    continue
                print("   FP  %-8s %-12s bbox=%s" % (k, v.get("type"), v.get("bbox")))
            for k, v in gc.items():
                if k not in mg:
                    print("   FN  %-8s %-12s bbox=%s Name=%s" % (k, v.get("type"), v.get("bbox"), v.get("Name")))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
