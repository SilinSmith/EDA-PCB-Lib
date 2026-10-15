# -*- coding: utf-8 -*-
"""真实文字语料的**重叠文本框体检**：语料里到底有没有重叠，是哪种重叠，文字是否不同。

为什么要它：本线（文字感知训练侧 + 重叠文本分解）的验收判据是「重叠框上 exact 恢复率
显著提升且对照组不降」。要判这条线有没有戏，先得量**语料本身**：重叠框有多少、是同一段
文字被标两次还是两段不同文字真的贴在一起、交叠是「整宽横带」还是「字形压字形」。这些量
**只读 GT 框**（框与框的交 / 较小框面积），不涉及模型任何输出，因此对新 GT 同样适用——
新 GT 落地后用同一把尺子量一遍，就知道它是不是同一种重叠。

两种模式：
    census  按 role / case 统计重叠规模，并给每框最大包含度分布
    pairs   按 (role, role) 统计重叠**对**，区分同文/异文，给 IoU 分布与最大 IoU 样例

用法（CWD 任意）：
    python -X utf8 eda-pcb\tools\diag\_textov.py --mode census
    python -X utf8 eda-pcb\tools\diag\_textov.py --mode pairs --thr 0.30 --top 10
    python -X utf8 eda-pcb\tools\diag\_textov.py --manifest <新 GT 清单> --mode census

口径（写死，勿随手改）：
    包含度 = 交面积 / **较小框**面积   —— 「小框有多少比例被别的框盖住」
    IoU    = 交面积 / 并面积         —— 真叠印才会大；相邻行只相切时很小
    census 的 --thr 默认 0.10（轻微相交即算），--strong 默认 0.30
    pairs  的 --thr 默认 0.30（只配对「明显相交」的框，避免打印污染）
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MANIFEST = r"Y:\workspace\EDA\eda-pcb\textrec\data\manifest.jsonl"


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0, None
    return (x1 - x0) * (y1 - y0), (x0, y0, x1, y1)


def _area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _load(path, split=""):
    by_case = collections.OrderedDict()
    n = 0
    with io.open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if split and r.get("split") != split:
                continue
            key = r.get("case") or r.get("image")
            by_case.setdefault(key, {"image": r.get("image"), "rows": []})["rows"].append(r)
            n += 1
    return by_case, n


def census(a):
    by_case, n = _load(a.manifest, a.split)
    role_cnt = collections.Counter()
    role_ov = collections.Counter()
    role_st = collections.Counter()
    n_ov = n_st = n_tiny = tiny_ov = n_dup = 0
    cases_ov = []
    ratios = []
    for case, blk in by_case.items():
        rows = blk["rows"]
        areas = [_area(r["bbox"]) for r in rows]
        ov_here = 0
        for i, r in enumerate(rows):
            role_cnt[r["role"]] += 1
            if r["bbox"][3] - r["bbox"][1] <= a.tiny:
                n_tiny += 1
            best = 0.0
            for j, s in enumerate(rows):
                if i == j or areas[i] <= 0 or areas[j] <= 0:
                    continue
                iv, _ = _inter(r["bbox"], s["bbox"])
                if iv <= 0:
                    continue
                if iv >= 0.95 * min(areas[i], areas[j]):
                    n_dup += 1
                best = max(best, iv / min(areas[i], areas[j]))
            ratios.append(best)
            if best >= a.thr:
                n_ov += 1
                role_ov[r["role"]] += 1
                ov_here += 1
                if r["bbox"][3] - r["bbox"][1] <= a.tiny:
                    tiny_ov += 1
            if best >= a.strong:
                n_st += 1
                role_st[r["role"]] += 1
        if ov_here:
            cases_ov.append((case, ov_here, len(rows)))

    print("清单：%s%s" % (a.manifest, ("  split=%s" % a.split) if a.split else ""))
    print("case 数 %d，文本框 %d" % (len(by_case), n))
    print("按 role：%s" % dict(role_cnt))
    print()
    print("包含度 >= %.2f：%d / %d = %.1f%%"
          % (a.thr, n_ov, n, 100.0 * n_ov / max(1, n)))
    print("  按 role：%s" % {k: "%d/%d" % (role_ov[k], role_cnt[k]) for k in role_cnt})
    print("包含度 >= %.2f（强）：%d / %d = %.1f%%"
          % (a.strong, n_st, n, 100.0 * n_st / max(1, n)))
    print("  按 role：%s" % {k: "%d/%d" % (role_st[k], role_cnt[k]) for k in role_cnt})
    print("  近似完全重复（包含度 >= 0.95）：%d" % n_dup)
    print("  含至少一个重叠框的 case：%d / %d" % (len(cases_ov), len(by_case)))
    print("  框高 <= %dpx 的 %d 个，其中重叠 %d 个" % (a.tiny, n_tiny, tiny_ov))
    if cases_ov:
        print("  重叠最多的 8 个 case：" + "，".join(
            "%s %d/%d" % t for t in sorted(cases_ov, key=lambda x: -x[1])[:8]))
    nz = sorted(x for x in ratios if x > 0)
    print()
    print("每框最大包含度：=0 的 %d 个；>0 的 %d 个中位 %.3f p90 %.3f 最大 %.3f"
          % (sum(1 for x in ratios if x == 0), len(nz),
             nz[len(nz) // 2] if nz else -1, nz[int(len(nz) * 0.9)] if nz else -1,
             nz[-1] if nz else -1))


def pairs(a):
    by_case, n = _load(a.manifest, a.split)
    rp = collections.Counter()
    same = collections.Counter()
    ious = []
    cand = []
    for case, blk in by_case.items():
        rows = blk["rows"]
        areas = [_area(r["bbox"]) for r in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if areas[i] <= 0 or areas[j] <= 0:
                    continue
                iv, ib = _inter(rows[i]["bbox"], rows[j]["bbox"])
                if iv <= 0:
                    continue
                c = iv / min(areas[i], areas[j])
                if c < a.thr:
                    continue
                iou = iv / (areas[i] + areas[j] - iv)
                key = tuple(sorted([rows[i]["role"], rows[j]["role"]]))
                rp[key] += 1
                ious.append(iou)
                if rows[i]["text"] == rows[j]["text"]:
                    same[key] += 1
                else:
                    cand.append((iou, c, case, rows[i], rows[j], ib))

    print("清单：%s（%d 个文本框）" % (a.manifest, n))
    print("重叠对（包含度 >= %.2f）：共 %d 对" % (a.thr, sum(rp.values())))
    print("%-30s %7s %7s %7s" % ("role 对", "对数", "同文", "异文"))
    for k in sorted(rp, key=lambda x: -rp[x]):
        print("%-30s %7d %7d %7d" % ("+".join(k), rp[k], same[k], rp[k] - same[k]))
    if ious:
        ious.sort()
        m = len(ious)
        print()
        print("IoU：p10 %.3f 中位 %.3f p90 %.3f 最大 %.3f；IoU>=0.2 的 %d 对 (%.1f%%)"
              % (ious[int(m * 0.1)], ious[m // 2], ious[int(m * 0.9)], ious[-1],
                 sum(1 for x in ious if x >= 0.2),
                 100.0 * sum(1 for x in ious if x >= 0.2) / m))
        print("（IoU 中位远小于包含度 ⇒ 交叠是「整宽横带」的相邻行，不是字形压字形）")
    print()
    print("=== 文字不同且 IoU 最大的 %d 对 ===" % a.top)
    for iou, c, case, ri, rj, ib in sorted(cand, key=lambda x: -x[0])[:a.top]:
        print("case %-6s IoU %.3f 包含 %.3f  [%s]%-12s @%s  ×  [%s]%-12s @%s  交=%s"
              % (case, iou, c, ri["role"], repr(ri["text"]), ri["bbox"],
                 rj["role"], repr(rj["text"]), rj["bbox"], ib))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="真实文字语料的重叠框体检")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--split", default="real",
                    help="只取该 split（默认 real = eval_real 打分的那一份；空串 = 全部）")
    ap.add_argument("--mode", choices=("census", "pairs"), default="census")
    ap.add_argument("--thr", type=float, default=None, help="census 默认 0.10，pairs 默认 0.30")
    ap.add_argument("--strong", type=float, default=0.30)
    ap.add_argument("--tiny", type=int, default=6, help="census：小字框的高阈值（px）")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)
    if a.thr is None:
        a.thr = 0.10 if a.mode == "census" else 0.30
    if a.mode == "census":
        census(a)
    else:
        pairs(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
