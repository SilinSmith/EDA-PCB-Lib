# -*- coding: utf-8 -*-
"""**相邻信息度量**：位号候选的「行 / 列 run」结构里，邻座的号能不能救回读不出来的号。

背景（User ref m40034 第三项：「高维空间里分布元器件」的构想能不能用来衡量相邻信息）：
`fusion/axis.py` 的 `build_axes()` 只做了最窄的一种对齐（**同前缀 + 同一 y 带 + 水平**），
`docs/reports/工程哲学.md:260` 自认「只做了最窄的一种」。本探针**先量信号**，不先改代码。

口径
  · 真值：GT `components` 的位号键（镜像帧）⇒ 翻成 PNG 帧 `(x, H - y)` 与候选框同帧。
  · 候选：`_assigned/<case>.json` 的 `desig_slots[sid]` = [{text, canon, conf, box, d, fit, ...}]。
  · 对每个 GT 位号，取半径 R 内最近的候选框当它的「槽」。

每个 GT 位号分桶
  `read_ok`            选中的就是这个号
  `pool_ok`            池子里有这个号，但没选中（★ 选择问题，`fusion/axis.py` 正在管）
  `missing_implied`    ★ 池子里没有这个号，但**同一 run 的邻座号**把号推得出来（相邻信息的收益上限）
  `missing_orphan`     池子里没有、邻座也推不出来
  `no_slot`            R 内没有任何候选框

run 的定义（两种朝向都算）
  同前缀、按主轴排序，成员两两之间「副轴差 <= `--band`(默认 8px)」且
  「主轴间距 ∈ [0.25, 4.0] × 中位框宽」且号**严格单调**、成员数 >= `--min-run`(默认 3)。
  水平 run 用 y 当副轴、x 当主轴；竖直 run 用 x 当副轴、y 当主轴。
  `missing_implied` 的判据：用同 run 的**其他成员的真值号**对主轴位置做线性插值/端点外推，
  取整后等于本槽真值号 ⇒ 邻座信息足够。

用法
  python -X utf8 work\\tools\\dev\\_adjmeas.py --assigned work\\_s2_220mix\\_assigned \\
      --cases-file work\\_cases_w28.txt --radius 60 --show 12
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
DR = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
SPLITS = ("200_train_cases", "10_GTcase", "10GT")
DES_RE = re.compile(r"^([A-Za-z]{1,3})(\d{1,4})$")


# ---------------------------------------------------------------- GT / 图
def find_gt(case: str):
    for sp in SPLITS:
        d = DR / sp / case
        if not d.is_dir():
            continue
        js = sorted(d.glob("%s*.json" % case))
        js = [p for p in js if "target_named" in p.name] or js
        for p in js:
            try:
                g = json.loads(p.read_text(encoding="utf-8"))
            except Exception:                                       # noqa: BLE001
                continue
            if g.get("components"):
                return sp, g
    return None, None


def find_png(case: str):
    for sp in SPLITS:
        d = DR / sp / case
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.png")):
            if "__full" not in f.name:
                return f
    return None


def png_hw(p: Path):
    """只读 PNG 头，不解码整张图。"""
    try:
        b = p.open("rb").read(33)
        if b[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
    except Exception:                                               # noqa: BLE001
        pass
    import cv2
    im = cv2.imread(str(p))
    return (im.shape[1], im.shape[0]) if im is not None else (0, 0)


def box_center(b):
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_wh(b):
    return abs(float(b[2]) - float(b[0])), abs(float(b[3]) - float(b[1]))


def gt_items(gt, h_img):
    """GT 位号 -> [(key, prefix, num, cx, cy_png, w, h)]"""
    out = []
    for key, val in (gt.get("components") or {}).items():
        m = DES_RE.match(str(key))
        b = None
        if isinstance(val, dict):
            b = val.get("bbox") or val.get("box")
        if not b or len(b) < 4:
            continue
        cx, cy = box_center(b)
        w, h = box_wh(b)
        cy = float(h_img) - cy                                     # 镜像帧 -> PNG 帧
        if m:
            out.append((key, m.group(1), int(m.group(2)), cx, cy, w, h))
    return out


# ---------------------------------------------------------------- run
def build_runs(items, band=8.0, pitch_lo=0.25, pitch_hi=4.0, min_run=3):
    """items: [(key, prefix, num, cx, cy, w, h)] -> 水平/竖直 run 列表。"""
    runs = []
    by_prefix = collections.defaultdict(list)
    for it in items:
        by_prefix[it[1]].append(it)
    for pref, group in by_prefix.items():
        for horiz in (True, False):
            main_i, side_i, size_i = (3, 4, 5) if horiz else (4, 3, 6)
            seq = sorted(group, key=lambda t: t[main_i])
            med = sorted(t[size_i] for t in seq)[len(seq) // 2] if seq else 0.0
            if med <= 0:
                continue
            cur = []
            for t in seq:
                if cur:
                    p = cur[-1]
                    if (abs(t[side_i] - p[side_i]) <= band
                            and pitch_lo * med <= (t[main_i] - p[main_i]) <= pitch_hi * med):
                        cur.append(t)
                        continue
                    if len(cur) >= min_run:
                        runs.append((pref, horiz, list(cur)))
                    cur = []
                cur.append(t)
            if len(cur) >= min_run:
                runs.append((pref, horiz, list(cur)))
    return runs


def run_numbers(nums):
    """严格单调（任一方向）才算一条号 run。"""
    if len(nums) < 3:
        return False
    inc = all(b > a for a, b in zip(nums, nums[1:]))
    dec = all(b < a for a, b in zip(nums, nums[1:]))
    return inc or dec


def implied_number(run, idx, main_i):
    """用同 run 其他成员的真值号对主轴位置线性拟合，取整。"""
    xs = [run[j][main_i] for j in range(len(run)) if j != idx]
    ys = [run[j][2] for j in range(len(run)) if j != idx]
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-9:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    pred = my + slope * (run[idx][main_i] - mx)
    return int(round(pred)), slope


# ---------------------------------------------------------------- 主
def main():
    ap = argparse.ArgumentParser(description="相邻信息度量（位号 run）")
    ap.add_argument("--assigned", default=r"work\_s2_220mix\_assigned")
    ap.add_argument("--cases-file", default=r"work\_cases_w28.txt")
    ap.add_argument("--radius", type=float, default=60.0)
    ap.add_argument("--band", type=float, default=8.0, help="副轴容差 px")
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--json", default="")
    ap.add_argument("--poolscan", action="store_true",
                    help="只做「正确读法去哪了」的分类：选中/近处池里/远处池里/池里没有")
    a = ap.parse_args()

    adir = (WORK.parent / a.assigned) if not Path(a.assigned).is_absolute() else Path(a.assigned)
    cf = (WORK.parent / a.cases_file) if not Path(a.cases_file).is_absolute() else Path(a.cases_file)
    cases = [c.strip() for c in cf.read_text(encoding="utf-8-sig").replace("\n", ",").split(",") if c.strip()]
    if a.limit:
        cases = cases[:a.limit]

    pool_only = a.poolscan
    far_hist = collections.Counter()
    if pool_only:
        buckets = collections.Counter()
        examples = []
        for case in cases:
            ap_ = adir / ("%s.json" % case)
            if not ap_.exists():
                continue
            sp, gt = find_gt(case)
            png = find_png(case)
            if not gt or not png:
                continue
            _w, h_img = png_hw(png)
            A = json.loads(ap_.read_text(encoding="utf-8"))
            own, pools = {}, {}
            for sid, lst in (A.get("desig_slots") or {}).items():
                pools[sid] = [str(c.get("text") or "") for c in (lst or []) if c.get("box")]
                d = (A.get("comps", {}).get(sid) or {}).get("designator")
                own[sid] = str((d or {}).get("text") or "")
            for key, pref, num, cx, cy, w, h in gt_items(gt, h_img):
                best, bd, bsid = None, 1e18, None
                for sid, lst in (A.get("desig_slots") or {}).items():
                    for c in lst or []:
                        b = c.get("box")
                        if not b:
                            continue
                        bx, by = box_center(b)
                        dd = math.hypot(bx - cx, by - cy)
                        if dd < bd:
                            best, bd, bsid = c, dd, sid
                if bsid is None or bd > 250.0:
                    buckets["no_slot_unknown"] += 1
                    continue
                if own.get(bsid) == key:
                    buckets["chosen_ok"] += 1
                    continue
                # 正确读法在不在**这颗元件自己的**候选清单里？距离多远？
                dmin = 1e18
                for c in (A.get("desig_slots") or {}).get(bsid) or []:
                    if str(c.get("text") or "") == key and c.get("box"):
                        bx, by = box_center(c["box"])
                        dmin = min(dmin, math.hypot(bx - cx, by - cy))
                if dmin > 1e17:
                    buckets["not_in_pool"] += 1
                elif dmin <= 60.0:
                    buckets["near_not_chosen"] += 1
                    if len(examples) < 20:
                        examples.append((case, key, own.get(bsid) or "-", dmin))
                else:
                    buckets["far_not_chosen"] += 1
                    for lo, hi in ((60, 80), (80, 100), (100, 150), (150, 250),
                                   (250, 10 ** 6)):
                        if lo <= dmin < hi:
                            far_hist[(lo, hi)] += 1
                            break
                    if len(examples) < 20:
                        examples.append((case, key, own.get(bsid) or "-", dmin))
        tot0 = sum(buckets.values())
        print("=" * 78)
        print("位号读法「去哪了」：GT 位号 %d 个（只按元件自己的候选清单 `desig_slots` 判）" % tot0)
        print("目录 %s" % adir)
        print("-" * 78)
        for k in ("chosen_ok", "near_not_chosen", "far_not_chosen", "not_in_pool",
                  "no_slot_unknown"):
            v = buckets.get(k, 0)
            print("  %-18s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, tot0)))
        print("-" * 78)
        print("  远距（>60px）正确读法的距离分布")
        for lo, hi in ((60, 80), (80, 100), (100, 150), (150, 250), (250, 10 ** 6)):
            v = far_hist.get((lo, hi), 0)
            if v:
                print("    %4d~%-8s %4d" % (lo, ("%d" % hi) if hi < 10 ** 6 else "+", v))
        if examples:
            print("-" * 78)
            print("  ★ 池子里有正确读法、但没选中的例子（case, GT, 我方选中, 正确框到 GT 的距离）")
            for e in examples:
                print("    %-12s %-8s -> %-8s | %.1fpx" % e)
        print("-" * 78)
        return

    buckets = collections.Counter()
    by_axis = collections.Counter()
    prefix_stat = collections.defaultdict(collections.Counter)
    examples, per_case = [], []
    n_cand_total = 0

    for case in cases:
        ap_ = adir / ("%s.json" % case)
        if not ap_.exists():
            print("  [skip] %s (no _assigned)" % case)
            continue
        sp, gt = find_gt(case)
        png = find_png(case)
        if not gt or not png:
            print("  [skip] %s (gt=%s png=%s)" % (case, bool(gt), bool(png)))
            continue
        _w, h_img = png_hw(png)
        A = json.loads(ap_.read_text(encoding="utf-8"))
        slots = A.get("desig_slots") or {}
        chosen = {}
        cands = []
        seen = set()
        for sid, lst in slots.items():
            d = (A.get("comps", {}).get(sid) or {}).get("designator")
            ctext = str((d or {}).get("text") or "")
            chosen[sid] = ctext
            for c in lst or []:
                b = c.get("box")
                if not b:
                    continue
                key = (str(c.get("text") or ""), round(float(b[0]), 1), round(float(b[1]), 1))
                if key in seen:
                    continue
                seen.add(key)
                cands.append({"text": str(c.get("text") or ""), "box": b})
        n_cand_total += len(cands)
        items = gt_items(gt, h_img)
        # GT 位号 -> 最近候选槽
        slots_gt = []
        for key, pref, num, cx, cy, w, h in items:
            best, bd = None, 1e18
            for c in cands:
                bx, by = box_center(c["box"])
                dd = math.hypot(bx - cx, by - cy)
                if dd < bd:
                    best, bd = c, dd
            if best is None or bd > a.radius:
                buckets["no_slot"] += 1
                prefix_stat[pref]["no_slot"] += 1
                continue
            in_pool = any(DES_RE.match(c["text"] or "") and
                          str(c["text"]) == key for c in cands
                          if math.hypot(box_center(c["box"])[0] - cx,
                                        box_center(c["box"])[1] - cy) <= a.radius)
            ctexts = {str(c["text"]) for c in cands
                      if math.hypot(box_center(c["box"])[0] - cx,
                                    box_center(c["box"])[1] - cy) <= a.radius}
            read = str(best["text"])
            slots_gt.append({"key": key, "pref": pref, "num": num, "cx": cx, "cy": cy,
                             "read": read, "in_pool": in_pool, "ctexts": ctexts})
        # run 归属
        runs = build_runs(items, band=a.band, min_run=a.min_run)
        run_of = {}
        for ri, (pref, horiz, members) in enumerate(runs):
            if not run_numbers([m[2] for m in members]):
                continue
            for j, m in enumerate(members):
                run_of[m[0]] = (ri, pref, horiz, members, j)
        for s in slots_gt:
            if s["read"] == s["key"]:
                b = "read_ok"
            elif s["in_pool"]:
                b = "pool_ok"
            else:
                b = "missing_orphan"
            axis = "-"
            imp = None
            if s["key"] in run_of:
                ri, pref, horiz, members, j = run_of[s["key"]]
                axis = "h" if horiz else "v"
                if b == "missing_orphan":
                    r = implied_number(members, j, 3 if horiz else 4)
                    if r is not None:
                        imp = r[0]
                        if imp == s["num"]:
                            b = "missing_implied"
                            if len(examples) < a.show:
                                examples.append((case, s["key"], s["read"], imp, axis,
                                                 "%.1f" % r[1], len(members)))
            buckets[b] += 1
            by_axis["%s:%s" % (axis, b)] += 1
            prefix_stat[s["pref"]][b] += 1
        per_case.append((case, sp, len(items), len(cands), len(runs), len(run_of),
                         sum(1 for s in slots_gt if s["key"] in run_of)))

    tot = sum(buckets.values())
    if pool_only:
        print("=" * 78)
        print("位号读法「去哪了」：GT 位号 %d 个（只按元件自己的候选清单 `desig_slots` 判）" % tot)
        print("目录 %s" % adir)
        print("-" * 78)
        for k in ("chosen_ok", "near_not_chosen", "far_not_chosen", "not_in_pool",
                  "no_slot_unknown"):
            v = buckets.get(k, 0)
            print("  %-18s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, tot)))
        print("-" * 78)
        print("  远距（60px~）正确读法的距离分布")
        for lo, hi in ((60, 80), (80, 100), (100, 150), (150, 250), (250, 10 ** 6)):
            v = far_hist.get((lo, hi), 0)
            if v:
                print("    %4d~%-6s %4d" % (lo, ("%d" % hi) if hi < 10 ** 6 else "+", v))
        if examples:
            print("-" * 78)
            print("  ★ 池子里有正确读法、但没选中的例子（case, GT, 我方选中, 正确框到 GT 的距离）")
            for e in examples:
                print("    %-12s %-8s -> %-8s | %.1fpx" % e)
        print("-" * 78)
        return

    print("=" * 78)
    print("相邻信息度量：%d 张 case，GT 位号 %d 个，位号候选框合计 %d" % (len(per_case), tot, n_cand_total))
    print("目录 %s  半径 %.0fpx  副轴带 %.1fpx  min-run %d" % (adir, a.radius, a.band, a.min_run))
    print("-" * 78)
    for k in ("read_ok", "pool_ok", "missing_implied", "missing_orphan", "no_slot"):
        v = buckets.get(k, 0)
        print("  %-16s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, tot)))
    print("-" * 78)
    print("  按朝向（run 内）")
    for k in sorted(by_axis):
        print("    %-20s %5d" % (k, by_axis[k]))
    print("  按前缀（前 12）")
    for pref, cnt in sorted(prefix_stat.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        s = sum(cnt.values())
        print("    %-6s n=%-5d ok=%-4d pool_ok=%-4d implied=%-3d orphan=%-4d no_slot=%d"
              % (pref, s, cnt["read_ok"], cnt["pool_ok"], cnt["missing_implied"],
                 cnt["missing_orphan"], cnt["no_slot"]))
    if examples:
        print("-" * 78)
        print("  ★ 邻座信息足够、但池子里没读出来的例子（case, GT, 我方选中, 邻座推出, 轴向, 斜率, run 长度）")
        for e in examples:
            print("    %-12s %-8s -> %-8s | implied %-6s %s slope=%s n=%d" % e)
    print("-" * 78)
    print("  逐 case（case, split, GT位号, 候选框, run数, run内位号, 判读）")
    for c, sp, ni, nc, nr, nin, _ in per_case:
        print("    %-14s %-18s gt=%-4d cand=%-4d runs=%-3d in_run=%-4d"
              % (c, sp, ni, nc, nr, nin))
    if a.json:
        out = {"cases": len(per_case), "gt_total": tot, "buckets": dict(buckets),
               "by_axis": dict(by_axis),
               "prefix": {k: dict(v) for k, v in prefix_stat.items()},
               "examples": examples}
        (WORK.parent / a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        print("[json] -> %s" % a.json)


if __name__ == "__main__":
    main()
