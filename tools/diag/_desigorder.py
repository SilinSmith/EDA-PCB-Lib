"""GT 的位号是不是**按画的顺序编的**？—— 决定「猜位号」能不能打赢「读位号」。

动机（三个已实证的事实）：
  1. `_desigsrc.py` 在 220 张上：994 个未对上的 GT `box` 里 **294 个（49.5%）我们的键是
     占位符 `pNNN`**（**没读到任何位号**），且它们**全部**类型 ∈ IC ⇒ 位号读取本来就去读了。
  2. 但把裁片放大到几乎整张图，模型仍然回 `NONE` —— 因为**图里根本没有位号**：
     * `0048-Datasheet.png`：IC 框旁只有引脚名（VCC/VSEN/…），**没有任何 `U1`**；
     * `0057-Datasheet.png`：只有器件型号 `S-8225B`，**没有 `U1`**；
     * `0102-jlc.png`：符号名画的是 **`Flash`**（不是 `U3`），型号 `W25Q16JVUXIQ`。
     * 而 `0143-jlc.png`（真竞赛图）位号 `U1` **画得清清楚楚**。
  3. `10GT(?)`（竞赛原始 GT）**21/21 = 100%**，派生拆分集只有 21%~47%。
  ⇒ **读不出来不是我们弱，是那些图上没印。**

那么唯一的赢法就变成**按约定去猜**：如果这批 GT 是被「按某种几何/拓扑顺序重新编号」
成 `U1,U2,U3…` 的，猜顺序就能白拿这些键（而键同时是 `pin_f1` 与 `netH_f1` 的闸门）。

本脚本只做一件事：**量若干种排序约定与 GT 编号的一致率**，给出「猜」的准确率上界。
  * `reading`  自上而下、自左而右（按框中心，行容差 0.5×中位框高）
  * `reading_x` 自左而右、自上而下
  * `area`     框面积降序
  * `pins`     引脚数降序、再 reading
  * `reverse_reading` 与 `reading` 相反
判据：把该 case 该前缀组（≥3 个）按约定排成一列，看数字序列是否恰好是 `1..n`（严格升序且从 1 开始）。

用法：
  python work\\tools\\diag\\_desigorder.py
  python work\\tools\\diag\\_desigorder.py --source Datasheet --show 8
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import data as D                                    # noqa: E402
from schematic_parser.metrics import canon_type, normalise_doc            # noqa: E402

DESIG = re.compile(r"^([A-Za-z]{1,4})(\d{1,4})$")
ORDERS = ("reading", "reading_x", "area", "area_asc", "pins", "json",
          "perim", "w_desc", "h_desc", "y_desc", "x_desc")


def _bb(v):
    b = (v or {}).get("bbox") or (v or {}).get("box")
    return [float(x) for x in b[:4]] if b and len(b) >= 4 else None


def source_of(stem: str) -> str:
    return stem.split("-", 1)[1].rsplit("_target", 1)[0] if "-" in stem else "10GT(?)"


def _reading(items):
    """自上而下、自左而右；行容差 = 0.5 × 中位框高。"""
    hs = sorted(b[3] - b[1] for _k, b, _n in items)
    tol = 0.5 * hs[len(hs) // 2] if hs else 10.0
    out, row, cy = [], [], None
    for it in sorted(items, key=lambda t: -((t[1][1] + t[1][3]) / 2.0)):
        c = (it[1][1] + it[1][3]) / 2.0
        if cy is None or abs(c - cy) <= tol:
            row.append(it)
            cy = c if cy is None else (cy + c) / 2.0
        else:
            out += sorted(row, key=lambda t: (t[1][0] + t[1][2]) / 2.0)
            row, cy = [it], c
    out += sorted(row, key=lambda t: (t[1][0] + t[1][2]) / 2.0)
    return out


def _orders(items, pin_n):
    cx = lambda t: (t[1][0] + t[1][2]) / 2.0                              # noqa: E731
    cy = lambda t: (t[1][1] + t[1][3]) / 2.0                              # noqa: E731
    area = lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1])            # noqa: E731
    perim = lambda t: (t[1][2] - t[1][0]) + (t[1][3] - t[1][1])           # noqa: E731
    return {
        "reading": [k for k, _b, _n in _reading(items)],
        "reading_x": [k for k, _b, _n in sorted(items, key=lambda t: (cx(t), -cy(t)))],
        "area": [k for k, b, _n in sorted(items, key=lambda t: -area(t))],
        "area_asc": [k for k, b, _n in sorted(items, key=lambda t: area(t))],
        "pins": [k for k, _b, n in sorted(items, key=lambda t: (-t[2], cx(t)))],
        # GT JSON 的**字典插入顺序** —— 标注工具常常就是按这个顺序编的
        "json": [k for k, _b, _n in items],
        "perim": [k for k, b, _n in sorted(items, key=lambda t: -perim(t))],
        "w_desc": [k for k, b, _n in sorted(items, key=lambda t: -(t[1][2] - t[1][0]))],
        "h_desc": [k for k, b, _n in sorted(items, key=lambda t: -(t[1][3] - t[1][1]))],
        "y_desc": [k for k, _b, _n in sorted(items, key=lambda t: -cy(t))],
        "x_desc": [k for k, _b, _n in sorted(items, key=lambda t: -cx(t))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    want = {x.strip() for x in a.cases.split(",") if x.strip()} or None

    hit = Counter()
    grp = Counter()
    bysrc = defaultdict(Counter)
    tot_el = Counter()
    shown = 0

    for c in D.discover(D.DEFAULT_DATA_ROOT):
        if not c.json_path or (want and c.case_id not in want):
            continue
        src = source_of(Path(c.json_path).stem)
        if a.source and a.source.lower() not in src.lower():
            continue
        gt = normalise_doc(D.load_json(c.json_path))
        pin_n = Counter()
        for p in (gt.get("pins") or {}).values():
            pin_n[(p or {}).get("component")] += 1
        groups = defaultdict(list)
        for k, v in (gt.get("components") or {}).items():
            if canon_type((v or {}).get("type")) not in D.IC_TYPES:
                continue
            m = DESIG.match(k or "")
            b = _bb(v)
            if m and b:
                groups[m.group(1).upper()].append((k, b, pin_n.get(k, 0)))
        for pre, items in groups.items():
            if len(items) < 3:
                continue
            grp[(src, "组")] += 1
            g = sorted(items, key=lambda t: int(DESIG.match(t[0]).group(2)))
            truth = [k for k, _b, _n in g]
            nums = [int(DESIG.match(k).group(2)) for k in truth]
            strict = nums == list(range(1, len(nums) + 1))
            if set(nums) == set(range(1, len(nums) + 1)):
                grp[(src, "是1..n")] += 1
            grp[(src, "元素")] += len(truth)
            ords = _orders(items, pin_n)
            for name, seq in ords.items():
                if seq == truth:
                    hit[(src, name)] += 1
                    bysrc[src][name] += 1
                # 逐元素：按该顺序把 pre1..pren 摊下去，算对了几个
                guess = ["%s%d" % (pre, i + 1) for i in range(len(seq))]
                ne = sum(1 for k, gk in zip(seq, guess) if k == gk)
                bysrc[src][name + ".el"] += ne
                tot_el[(src, name)] += ne
            # 逐元素上界：如果知道组内正确顺序，最多能对几个（= 数字集合的交集大小）
            tot_el[(src, "oracle")] += len(set(nums) & set(range(1, len(nums) + 1)))
            if a.show and shown < a.show and not strict:
                n_ok = len(set(nums) & set(range(1, len(nums) + 1)))
                shown += 1
                print("  %-6s %-12s GT=%-30s 数字集是1..n=%s  oracle逐元素=%d/%d"
                      % (c.case_id, pre, truth[:10], set(nums) == set(range(1, len(nums) + 1)),
                         n_ok, len(nums)))

    print("=== GT `box` 位号是否「按顺序编」：各排序约定与 GT 编号序列完全一致的比例 ===")
    print("（只统计同一前缀、≥3 个成员的组）")
    print("%-18s %6s | %s" % ("来源", "组数", "  ".join("%-15s" % o for o in ORDERS)))
    tot = Counter()
    for src in sorted({k[0] for k in grp}, key=lambda s: -grp[(s, "组")]):
        n = grp[(src, "组")]
        tot["组"] += n
        for o in ORDERS:
            tot[o] += bysrc[src][o]
        print("%-18s %6d | %s" % (src, n, "  ".join(
            "%-6s %7.1f%%" % ("", 100.0 * bysrc[src][o] / n) if n else "-" for o in ORDERS)))
    if tot["组"]:
        print("%-18s %6d | %s" % ("合计", tot["组"], "  ".join(
            "%-6s %7.1f%%" % ("", 100.0 * tot[o] / tot["组"]) for o in ORDERS)))
    print()
    print("=== 同一个问题，但判据换成**逐元素**准确率（组内按约定顺序摊 pre1..pren）===")
    print("%-18s %6s %8s | %s" % (
        "来源", "组数", "是1..n", "  ".join("%-13s" % o for o in ORDERS) + "   oracle"))
    T = Counter()
    for src in sorted({k[0] for k in grp}, key=lambda s: -grp[(s, "组")]):
        n, ne = grp[(src, "组")], grp[(src, "元素")]
        for o in ORDERS:
            T[o] += bysrc[src][o + ".el"]
        T["元素"] += ne
        T["oracle"] += tot_el[(src, "oracle")]
        T["是1..n"] += grp[(src, "是1..n")]
        print("%-18s %6d %7.1f%% | %s   %6.1f%%" % (
            src, n, 100.0 * grp[(src, "是1..n")] / n if n else 0.0,
            "  ".join("%-13s" % ("%.1f%%" % (100.0 * bysrc[src][o + ".el"] / ne if ne else 0.0))
                      for o in ORDERS),
            100.0 * tot_el[(src, "oracle")] / ne if ne else 0.0))
    if T["元素"]:
        print("%-18s %6d %7.1f%% | %s   %6.1f%%" % (
            "合计", T["组"], 100.0 * T["是1..n"] / T["组"] if T["组"] else 0.0,
            "  ".join("%-13s" % ("%.1f%%" % (100.0 * T[o] / T["元素"])) for o in ORDERS),
            100.0 * T["oracle"] / T["元素"]))
        print()
        print("⇒ `oracle` = 如果连「组内谁是第几个」都猜对，逐元素命中率的上界；")
        print("  某来源 `reading`/`area` 与 oracle 的差距就是「顺序约定能捞回多少」。")


if __name__ == "__main__":
    main()
