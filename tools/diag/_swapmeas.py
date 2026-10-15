"""一对一归属的**天花板**测量：把「元件 ↔ 位号实例」从独立最近邻改成邻域内双射，
在给定的 `_assigned` 目录上有多少 GT 位号能从错变对。

动机（`docs/RESULTS.md` §2.9(b)）：639 个 GT 位号里 **78 个（12.2%）**「正确读法就在
**这颗元件自己的**候选清单里、且 ≤60px，却被**邻座的号**顶掉」，例子全是相邻换位
（`1 C151→C152 @54.6px`、`2 R1→R2 @22.8px`）。⇒ 这是**归属**问题（缺一对一约束），
不是读数问题。`fusion/solve.py` 现有的「唯一性」只在**同一个值撞车**时生效，管不了
「两颗元件各自拿了对方的号」这种**互换**（两个值都只出现一次，唯一性判据看不见）。

本探针只做**离线重排**，不碰读数、不碰 stage2：

  · 元件 = `comps` 里的 `cid`；不需要元件自己的框 —— `desig_slots[cid][k]["d"]`
    已经是「该元件框 → 该文字框」的距离（`stage2/assign.py` 构造候选时算的）；
  · 实例 = `box_i`（stage1 文字框下标）⇒ **同一个物理文字框在不同 cid 的清单里共享
    同一个 `box_i`**，这正是「偷邻居的号」的形式化；
  · 约束 = 每个实例最多归一个元件、每个元件最多一个实例；元件可以**落空**（罚 `--miss`）；
  · 目标 = 最小化 Σ `d`。当前分配一定在可行域里（候选来自各自的 `desig_slots`）
    ⇒ 重排**不会比现状差**（这是「只增不减」的形式保证，端到端仍要实测，
    因为 28 张上 ±0.3 落在流水线混沌带内，见 `docs/RESULTS.md` §5.5）。

冲突图按 `d <= --radius` 连边分连通块，块内跑 Hungarian（`scipy.optimize`）。
每个块很小（相邻几颗元件），所以顺带打印块规模分布 —— 这是实装成本的上界。

输出：① 与 `adjmeas.py --poolscan` 同尺子的五桶（对表用）；② 归属正确数 before/after
与 修好 / 弄坏 / 净增；③ 改动面（多少颗元件的号被换掉）；④ 块规模分布；⑤ 例子。

用法：
  python -X utf8 eda-pcb\\tools\\diag\\_swapmeas.py --assigned work\\_s2_220mix\\_assigned ^
      --cases-file work\\_cases_w28.txt --radius 60 --miss 40 --show 12
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjmeas as AM                                            # noqa: E402

try:                                                            # noqa: E402
    import numpy as _np
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:                                               # noqa: BLE001
    _np = None
    _lsa = None

BIG = 1e7


# ------------------------------------------------------------------ 读入
def read_cases(path) -> list:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace(",", "\n")
    out = []
    for ln in txt.splitlines():
        t = ln.strip()
        if t and not t.startswith("#"):
            out.append(t)
    return out


def _inst_key(box_i, box):
    if box_i is not None:
        return "i:%s" % box_i
    return "b:%s" % ",".join("%.0f" % float(v) for v in (box or [])[:4])


def pool_of(A: dict):
    """`desig_slots` -> {cid: {instance: (canon, d, box)}}，外加 {cid: (canon, 当前实例)}。"""
    pools: dict = {}
    for cid, lst in (A.get("desig_slots") or {}).items():
        m: dict = {}
        for cd in lst or []:
            if not cd.get("box"):
                continue
            can = str(cd.get("canon") or cd.get("text") or "")
            if not can:
                continue
            k = _inst_key(cd.get("box_i"), cd.get("box"))
            d = float(cd.get("d") or 0.0)
            old = m.get(k)
            if old is None or d < old[1]:
                m[k] = (can, d, [float(v) for v in cd["box"][:4]])
        pools[cid] = m
    own: dict = {}
    for cid, rec in (A.get("comps") or {}).items():
        dg = (rec or {}).get("designator") or {}
        can = str(dg.get("canon") or dg.get("text") or "")
        if not can:
            own[cid] = (None, None)
            continue
        k = _inst_key(dg.get("box_i"), dg.get("box"))
        m = pools.get(cid) or {}
        if k not in m:                                          # box_i 缺失时按值回找
            cand = [(kk, v) for kk, v in m.items() if v[0] == can]
            k = min(cand, key=lambda kv: kv[1][1])[0] if cand else None
        own[cid] = (can, k)
    return pools, own


# ------------------------------------------------------------------ 分块 + 解
def clusters(pools: dict, radius: float):
    edges: dict = collections.defaultdict(list)
    for cid, m in pools.items():
        for k, (_c, d, _b) in m.items():
            if d <= radius:
                edges[cid].append(k)
                edges[k].append(cid)
    seen, out = set(), []
    for n in list(edges):
        if n in seen:
            continue
        seen.add(n)
        stack, comp = [n], set()
        while stack:
            x = stack.pop()
            comp.add(x)
            for y in edges[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        cids = sorted(x for x in comp if not x.startswith("i:") and not x.startswith("b:"))
        inst = sorted(x for x in comp if x.startswith("i:") or x.startswith("b:"))
        if cids and inst:
            out.append((cids, inst))
    return out


def solve_cluster(cids, insts, pools, radius, miss):
    """块内最小 Σd 的「元件→实例」指派；返回 {cid: inst|None}。"""
    R, C = len(cids), len(insts)
    cost = _np.full((R, C + R), BIG)
    for i, cid in enumerate(cids):
        m = pools[cid]
        for j, k in enumerate(insts):
            if k in m and m[k][1] <= radius:
                cost[i, j] = m[k][1]
        cost[i, C + i] = miss                                   # 落空
    ri, ci = _lsa(cost)
    res = {}
    for i, j in zip(ri, ci):
        res[cids[i]] = insts[j] if j < C else None
    return res


# ------------------------------------------------------------------ 主
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assigned", required=True)
    ap.add_argument("--cases-file", required=True)
    ap.add_argument("--radius", type=float, default=60.0)
    ap.add_argument("--miss", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--explain", type=int, default=0,
                    help="打印 N 个 near_not_chosen 的现场（谁拿了正确框、距离各多少）")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                           # noqa: BLE001
        pass
    if _lsa is None:
        print("需要 scipy.optimize.linear_sum_assignment")
        return 2

    adir = Path(a.assigned)
    cases = read_cases(a.cases_file)
    if a.limit:
        cases = cases[:a.limit]

    buckets = collections.Counter()
    n_ok_b = n_ok_a = 0
    n_tot = 0
    fixes, breaks, churn_rows = [], [], []
    size_hist = collections.Counter()
    n_cl = 0
    gain_hist = collections.Counter()
    mismatch = 0
    rows = []
    explain: list = []

    for case in cases:
        p = adir / ("%s.json" % case)
        if not p.exists():
            continue
        sp, gt = AM.find_gt(case)
        png = AM.find_png(case)
        if not gt or not png:
            continue
        _w, h_img = AM.png_hw(png)
        A = json.loads(p.read_text(encoding="utf-8"))
        pools, own = pool_of(A)
        if not pools:
            continue

        # 同一个 box_i 在不同 cid 下的 canon 不一致 = 归属问题之外还有读数分歧
        seen_can: dict = {}
        for cid, m in pools.items():
            for k, (can, _d, _b) in m.items():
                if k in seen_can and seen_can[k] != can:
                    mismatch += 1
                seen_can.setdefault(k, can)

        # GT 项目 -> 我们的 cid（与 adjmeas --poolscan 同尺子：最近的候选框，≤250px）
        gt_map = []
        for key, _pref, _num, cx, cy, _w2, _h2 in AM.gt_items(gt, h_img):
            bd, bsid = 1e18, None
            for cid, m in pools.items():
                for _k, (_c, _d, b) in m.items():
                    bx, by = AM.box_center(b)
                    dd = math.hypot(bx - cx, by - cy)
                    if dd < bd:
                        bd, bsid = dd, cid
            if bsid is None or bd > 250.0:
                buckets["no_slot_unknown"] += 1
                continue
            n_tot += 1
            if own.get(bsid, (None, None))[0] == key:
                buckets["chosen_ok"] += 1
            else:
                dmin = 1e18
                m = pools.get(bsid) or {}
                for _k, (can, d, _b) in m.items():
                    if can == key:
                        dmin = min(dmin, d)
                if dmin > 1e17:
                    buckets["not_in_pool"] += 1
                elif dmin <= a.radius:
                    buckets["near_not_chosen"] += 1
                    if len(explain) < a.explain:
                        ob, obi = own.get(bsid, (None, None))
                        obx = (pools.get(bsid) or {}).get(obi) or (None, None, None)
                        corr = []
                        for k, (can, d, b) in (pools.get(bsid) or {}).items():
                            if can != key:
                                continue
                            rivals = []
                            for cid2, m2 in pools.items():
                                if cid2 == bsid or k not in m2:
                                    continue
                                o2, oi2 = own.get(cid2, (None, None))
                                o2x = (m2.get(oi2) or (None, None, None))[1]
                                rivals.append((cid2, round(m2[k][1], 1), o2, o2x))
                            corr.append((k, round(d, 1), rivals))
                        explain.append(dict(case=case, key=key, sid=bsid, own=ob,
                                            own_d=None if obx[1] is None else round(obx[1], 1),
                                            own_i=obi, corr=corr))
                else:
                    buckets["far_not_chosen"] += 1
            gt_map.append((key, bsid))

        ok_b = sum(1 for key, sid in gt_map if own.get(sid, (None, None))[0] == key)
        n_ok_b += ok_b
        n_tot_here = len(gt_map)

        # ---- 重排
        choice = {cid: (own.get(cid, (None, None))[0], own.get(cid, (None, None))[1])
                  for cid in pools}
        changed = 0
        for cids, insts in clusters(pools, a.radius):
            n_cl += 1
            size_hist[len(cids)] += 1
            cur = 0.0
            for cid in cids:
                k = choice[cid][1]
                if k in pools[cid]:
                    cur += pools[cid][k][1]
            res = solve_cluster(cids, insts, pools, a.radius, a.miss)
            new = 0.0
            for cid in cids:
                k = res.get(cid)
                if k is not None and k in pools[cid]:
                    new += pools[cid][k][1]
            gain_hist[round(cur - new, 1)] += 1
            for cid in cids:
                k = res.get(cid)
                if k is None or k == choice[cid][1]:
                    continue
                can = pools[cid][k][0]
                if can != choice[cid][0]:
                    changed += 1
                    choice[cid] = (can, k)

        own_a = {cid: choice[cid][0] for cid in pools}
        ok_a = sum(1 for key, sid in gt_map if own_a.get(sid) == key)
        n_ok_a += ok_a
        for key, sid in gt_map:
            b, a2 = own.get(sid, (None, None))[0], own_a.get(sid)
            if b != key and a2 == key and len(fixes) < 400:
                fixes.append((case, key, b or "-", a2))
            if b == key and a2 != key and len(breaks) < 400:
                breaks.append((case, key, b, a2 or "-"))
        if changed:
            churn_rows.append((case, changed, ok_b, ok_a, n_tot_here))
        rows.append((case, ok_b, ok_a, n_tot_here, changed))

    print("=" * 78)
    print("一对一归属天花板（离线重排，改归属不改读数）")
    print("  料 %s  半径 %.0fpx  落空罚 %.0f  case 数 %d" % (adir, a.radius, a.miss, len(rows)))
    print("-" * 78)
    tot0 = sum(buckets.values())
    for k in ("chosen_ok", "near_not_chosen", "far_not_chosen", "not_in_pool", "no_slot_unknown"):
        v = buckets.get(k, 0)
        print("  %-18s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, tot0)))
    print("-" * 78)
    print("  归属正确（GT 位号 %d 个）：before %d (%.1f%%) -> after %d (%.1f%%)  Δ=%+d"
          % (n_tot, n_ok_b, 100.0 * n_ok_b / max(1, n_tot),
             n_ok_a, 100.0 * n_ok_a / max(1, n_tot), n_ok_a - n_ok_b))
    print("  修好 %d ｜ 弄坏 %d ｜ 净 %+d ｜ 被换号的元件 %d 颗（%d 张图有改动）"
          % (len(fixes), len(breaks), len(fixes) - len(breaks),
             sum(r[4] for r in rows), len(churn_rows)))
    print("  块规模分布（块数 %d）%s" % (n_cl, dict(sorted(size_hist.items()))))
    print("  同实例读数分歧 %d 处" % mismatch)
    if gain_hist:
        g = sorted(gain_hist.items())
        print("  每块 Σd 下降分布（前 12）：%s" % g[:12])
    if rows:
        print("-" * 78)
        print("  %-16s %6s %6s %6s %6s" % ("case", "before", "after", "GT", "换号"))
        for case, ob, oa, nt, ch in rows[:40]:
            print("  %-16s %6d %6d %6d %6d%s" % (case, ob, oa, nt, ch,
                                                 "" if ob == oa else "   Δ%+d" % (oa - ob)))
    if fixes and a.show:
        print("-" * 78)
        print("  ★ 修好的例子（case, GT, 原选中, 重排后）")
        for e in fixes[:a.show]:
            print("    %-16s %-8s %-8s -> %-8s" % e)
    if breaks and a.show:
        print("  弄坏的例子")
        for e in breaks[:a.show]:
            print("    %-16s %-8s %-8s -> %-8s" % e)
    if explain:
        print("-" * 78)
        print("  ★ near_not_chosen 现场（我方元件 sid：选中的值@距离 ← 正确框@距离；正确框的**其他**候选者）")
        for e in explain:
            print("    %-16s GT %-8s sid %-8s 选中 %-8s@%s 正确 %s"
                  % (e["case"], e["key"], e["sid"], e["own"], e["own_d"],
                     ", ".join("%s@%s" % (k, d) for k, d, _r in e["corr"])))
            for k, d, rivals in e["corr"]:
                if rivals:
                    print("        %s 也被 %s" % (k, " ｜ ".join(
                        "%s(自己拿 %s@%s，到这框 %s)" % (c, o, od, dd)
                        for c, dd, o, od in rivals)))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
