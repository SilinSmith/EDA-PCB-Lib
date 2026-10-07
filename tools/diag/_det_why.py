# -*- coding: utf-8 -*-
"""元件检测（第一阶段）的漏检 / 多检归因（用户 m07122）。

    & 'K:\python\311\python.exe' Y:\workspace\EDA\work\tools\diag\_det_why.py --stage1 Y:\workspace\EDA\work\stage1_all14

回答三个问题：
1. 漏检的 GT 元件里，有多少是**导线端点就在它身上**（用户说的「导线延伸下来、
   检测末端就能检出」）？—— 这是可以用几何补回来的那一部分。
2. 多检里有多少是**我们自己重复检出同一个符号**（框重叠）？—— 不是数据集问题。
3. 按 **GT 框面积**分档看召回，验证「小规模检测抽风」。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "viz"))

import comp_cases as CC                                            # noqa: E402

BUCKETS = [(0, 100), (100, 200), (200, 400), (400, 900), (900, 2500),
           (2500, 10000), (10000, 10 ** 9)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def pt_rect(d, r):
    return ((max(r[0] - d[0], 0.0, d[0] - r[2]) ** 2
             + max(r[1] - d[1], 0.0, d[1] - r[3]) ** 2) ** 0.5)


def bucket(a):
    for lo, hi in BUCKETS:
        if lo <= a < hi:
            return "%d-%d" % (lo, hi if hi < 10 ** 9 else 999999)
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "work" / "stage1_all14"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.1)
    ap.add_argument("--ep", type=float, default=12.0)
    args = ap.parse_args()

    idx = CC.gt_index(CC.DEFAULT_DATA_ROOT)
    s1dir = Path(args.stage1)
    cases = sorted(p.name.split(".")[0] for p in s1dir.glob("*.stage1.json"))
    if args.limit:
        cases = cases[:args.limit]

    gt_by_type = Counter()
    hit_by_type = Counter()
    gt_by_bucket = Counter()
    hit_by_bucket = Counter()
    fn_ep = Counter()          # 漏检里有导线端点贴着它的
    fn_notype = Counter()
    fp_dup = 0
    fp_all = 0
    n_gt_tot = n_hit_tot = 0
    worst = []
    fn_examples = []

    for case in cases:
        rec = idx.get(case)
        s1 = s1dir / ("%s.stage1.json" % case)
        if rec is None or not rec.json_path or not s1.exists():
            continue
        doc = json.loads(s1.read_text(encoding="utf-8"))
        gdoc = json.loads(Path(rec.json_path).read_text(encoding="utf-8"))
        H = float(doc["image"]["h"])
        comps = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
        boxes = [[float(v) for v in c["bbox_img"]] for c in comps]
        eps = (doc.get("wire_detail") or {}).get("endpoints") or []

        gt = []
        for k, v in (gdoc.get("components") or {}).items():
            b = v.get("bbox")
            if not b or len(b) != 4:
                continue
            gt.append((k, v.get("type") or "?",
                       [float(b[0]), H - float(b[3]), float(b[2]), H - float(b[1])]))

        # 贪心一对一配对
        cands = []
        for gi, (_k, _t, gb) in enumerate(gt):
            for pi, pb in enumerate(boxes):
                v = iou(gb, pb)
                if v >= args.iou:
                    cands.append((v, gi, pi))
        cands.sort(key=lambda x: -x[0])
        ug, up = set(), set()
        for _v, gi, pi in cands:
            if gi in ug or pi in up:
                continue
            ug.add(gi)
            up.add(pi)

        n_gt_tot += len(gt)
        n_hit_tot += len(ug)
        nmiss = 0
        for gi, (_k, gt_type, gb) in enumerate(gt):
            a = (gb[2] - gb[0]) * (gb[3] - gb[1])
            gt_by_type[gt_type] += 1
            gt_by_bucket[bucket(a)] += 1
            if gi in ug:
                hit_by_type[gt_type] += 1
                hit_by_bucket[bucket(a)] += 1
                continue
            nmiss += 1
            # 最近的导线端点
            dmin = 1e9
            for e in eps:
                d = pt_rect(e, gb)
                if d < dmin:
                    dmin = d
            if dmin <= args.ep:
                fn_ep[gt_type] += 1
                fn_notype["端点贴着"] += 1
                if len(fn_examples) < 30 and gt_type in ("gnd", "v", "c", "r"):
                    fn_examples.append({"case": case, "gt": _k, "type": gt_type,
                                        "ep_d": round(dmin, 1), "box": [round(x, 1) for x in gb]})
            else:
                fn_notype["端点不贴"] += 1
        # 多检
        for pi, pb in enumerate(boxes):
            if pi in up:
                continue
            fp_all += 1
            dup = max((iou(pb, boxes[j]) for j in range(len(boxes)) if j != pi),
                      default=0.0)
            if dup >= 0.5:
                fp_dup += 1
        worst.append((nmiss, case, len(gt)))

    print("=" * 74)
    print("GT 元件 %d，检出 %d，召回 %.4f" % (n_gt_tot, n_hit_tot,
                                              n_hit_tot / max(1, n_gt_tot)))
    print("\n---- 按 GT 类型 ----")
    rows = []
    for t, n in gt_by_type.most_common():
        h = hit_by_type.get(t, 0)
        rows.append((n - h, t, n, h, h / max(1, n)))
    for miss, t, n, h, r in sorted(rows, reverse=True)[:26]:
        print("  %-16s GT %5d  检出 %5d  召回 %.3f  漏 %5d  其中端点贴着 %4d"
              % (t, n, h, r, miss, fn_ep.get(t, 0)))
    print("\n---- 按 GT 框面积分档（「小规模抽风」）----")
    for lo, hi in BUCKETS:
        key = "%d-%d" % (lo, hi if hi < 10 ** 9 else 999999)
        n = gt_by_bucket.get(key, 0)
        if not n:
            continue
        h = hit_by_bucket.get(key, 0)
        print("  %-12s GT %5d  检出 %5d  召回 %.3f" % (key, n, h, h / n))
    print("\n---- 多检 ----")
    print("  多检总数 %d，其中「与另一个我们自己的框 IoU>=0.5」= 重复检出 %d (%.1f%%)"
          % (fp_all, fp_dup, 100.0 * fp_dup / max(1, fp_all)))
    print("\n---- 漏检与导线端点 ----")
    tot_miss = fn_notype["端点贴着"] + fn_notype["端点不贴"]
    print("  漏检 %d；其中 %.1f%% 的 GT 框 <=%.0fpx 内就有我们**已经检出**的导线端点"
          % (tot_miss, 100.0 * fn_notype["端点贴着"] / max(1, tot_miss), args.ep))
    print("\n---- 漏检最多的 12 张 ----")
    for m, c, n in sorted(worst, reverse=True)[:12]:
        print("  %-14s 漏 %3d / GT %3d" % (c, m, n))
    print("\n---- 端点贴着的漏检样例 ----")
    for e in fn_examples[:20]:
        print("  %-12s GT=%-12s %-6s 端点距离 %5.1f box=%s"
              % (e["case"], e["gt"], e["type"], e["ep_d"], e["box"]))


if __name__ == "__main__":
    main()
