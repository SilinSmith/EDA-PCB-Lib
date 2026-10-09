"""位号**归属**可修性测量（GT 锚定版）：`_adjmeas --poolscan` 的 78 个
`near_not_chosen` 里，到底有多少是「**能靠一对一/互最近邻修好的**」，有多少是
「**距离判据本身就和真值反着来**」（那种必须靠顺序/轴先验，距离没法修）。

口径修正（与 `_adjmeas.py --poolscan` 的差别，很重要）：
`_adjmeas` 用「离 GT 元件中心最近的**候选文本框**」反推我们哪颗元件 —— 当这颗元件
恰好抢了邻居的号时，反推出来的 cid 就是错的那颗，桶会串味。本探针改成
`_desattr.py` 的口径：`prod`（stage2 产物，键=我方位号）与 `_assigned`（键=cid）
**按插入序 zip** 得到 `键→cid`，再用**元件框中心**（JSON 帧，不翻 y）找 GT 的近邻
（≤`--gate`）。GT bbox 与 prod bbox 同帧（`_desacc.py` 也是直接比）。

判据：
  · `key_ok`        我们的预测键 == GT 键；
  · `act`（可行动集）= 键错、但**正确读法就在这颗元件自己的 `desig_slots` 里**
    且 `d <= --radius`；这正是 §2.9(b) 的 `near_not_chosen`，只是换成了 GT 锚定；
  · 对 `act` 再分两种**可否用几何修**：
      - `mutual`  正确实例的「最近元件」**就是**我们 → 互最近邻规则就能修（可实现）；
      - `rival`   正确实例的最近元件是别人 → **距离判据与真值反着来**，只能靠
                  顺序/轴先验（`fusion/axis.py` 那一路）；
  · 顺带量：正确实例现在被谁拿着、那个人自己的 GT 键是不是我们**当前选中的那个值**
    （= 真·互换；`fusion/solve.py` 的唯一性判据看不见这种，因为两个值都只出现一次）。

用法：
  python -X utf8 eda-pcb\\tools\\diag\\_swapceil.py --prod work\\_s2_220mix \\
      --assigned work\\_s2_220mix\\_assigned --cases-file work\\_cases_w28.txt --show 10
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

# GT 里的伪位号（∅12 / p0045）不是真位号，它们**不该**进这套统计 ——
# 第一版忘了滤，`ordinary` 被灌到 1119、`no_read` 虚高到 68.5%
# （与 `_desacc.py:20` 同一条正则）。
PSEUDO = re.compile(r"^(?:∅\d+|p\d{3,4})$")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjmeas as AM                                            # noqa: E402


def centre(b):
    if not b or len(b) < 4:
        return None
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def read_cases(path) -> list:
    txt = Path(path).read_text(encoding="utf-8-sig", errors="ignore").replace(",", "\n")
    return [t.strip() for t in txt.splitlines() if t.strip() and not t.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default=r"Y:\workspace\EDA\work\_s2_220mix")
    ap.add_argument("--assigned", default="")
    ap.add_argument("--cases-file", default=r"Y:\workspace\EDA\work\_cases_w28.txt")
    ap.add_argument("--gate", type=float, default=40.0)
    ap.add_argument("--radius", type=float, default=60.0)
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                           # noqa: BLE001
        pass
    prod_dir = Path(a.prod)
    asg_dir = Path(a.assigned) if a.assigned else prod_dir / "_assigned"

    T = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    ex = []
    dgap = collections.Counter()
    rival_kind = collections.Counter()

    for nm in read_cases(a.cases_file):
        pp, pa = prod_dir / ("%s.json" % nm), asg_dir / ("%s.json" % nm)
        if not pp.exists() or not pa.exists():
            T["缺产物"] += 1
            continue
        prod = json.loads(pp.read_text(encoding="utf-8"))
        asg = json.loads(pa.read_text(encoding="utf-8"))
        gp, gt = AM.find_gt(nm)
        if gt is None:
            T["无GT"] += 1
            continue
        pk_list = list((prod.get("components") or {}).keys())
        cid_list = list((asg.get("comps") or {}).keys())
        if len(pk_list) != len(cid_list):
            T["元件数不一致"] += 1
            continue
        cid_of = dict(zip(pk_list, cid_list))
        pctr = {pk: centre((prod["components"][pk] or {}).get("bbox")) for pk in pk_list}
        pools = {}
        for cid, m in (asg.get("comps") or {}).items():
            pass
        # 每个 cid 的候选：instance(=box_i) -> (canon, d, box)
        for cid, lst in (asg.get("desig_slots") or {}).items():
            mm = {}
            for cd in lst or []:
                if not cd.get("box"):
                    continue
                can = str(cd.get("canon") or cd.get("text") or "")
                if not can:
                    continue
                k = cd.get("box_i")
                k = ("i:%s" % k) if k is not None else \
                    ("b:%s" % ",".join("%.0f" % float(v) for v in cd["box"][:4]))
                d = float(cd.get("d") or 0.0)
                if k not in mm or d < mm[k][1]:
                    mm[k] = (can, d, [float(v) for v in cd["box"][:4]])
            pools[cid] = mm
        # 每个 cid 的当前位置 = 其**选中候选**的文本框中心（我们拿不到元件本体框）
        #   —— 这正是本探针的短板，所以在两类判据里都只用它做「同一把尺子」的比较。
        chosen = {}
        for cid, rec in (asg.get("comps") or {}).items():
            dg = (rec or {}).get("designator") or {}
            can = str(dg.get("canon") or dg.get("text") or "")
            k = dg.get("box_i")
            k = ("i:%s" % k) if k is not None else (
                ("b:%s" % ",".join("%.0f" % float(v) for v in dg["box"][:4])) if dg.get("box") else None)
            chosen[cid] = (can, k, (pools.get(cid) or {}).get(k, (None, None, None))[1])

        # GT 元件 -> 我们的 pk/cid
        g_of_cid = {}
        for gk, gv in (gt.get("components") or {}).items():
            if PSEUDO.match(str(gk)):
                continue
            gc = centre((gv or {}).get("bbox"))
            if gc is None:
                continue
            near, bd = None, 1e18
            for pk in pk_list:
                c0 = pctr.get(pk)
                if c0 is None:
                    continue
                dd = math.hypot(c0[0] - gc[0], c0[1] - gc[1])
                if dd < bd:
                    near, bd = pk, dd
            if near is None or bd > a.gate:
                T["no_pred"] += 1
                continue
            cid = cid_of.get(near)
            if cid is None:
                T["no_pred"] += 1
                continue
            g_of_cid[cid] = gk

        # 每个实例的「最近元件」（用元件框中心；没有元件框 ⇒ 用各自候选的 d 代替）
        for gk, gv in (gt.get("components") or {}).items():
            if PSEUDO.match(str(gk)):
                continue
            gc = centre((gv or {}).get("bbox"))
            if gc is None:
                continue
            near, bd = None, 1e18
            for pk in pk_list:
                c0 = pctr.get(pk)
                if c0 is None:
                    continue
                dd = math.hypot(c0[0] - gc[0], c0[1] - gc[1])
                if dd < bd:
                    near, bd = pk, dd
            if near is None or bd > a.gate:
                continue
            cid = cid_of.get(near)
            if cid is None:
                continue
            T["ordinary"] += 1
            can, ck, cd = chosen.get(cid, (None, None, None))
            if can and can.lower() == gk.lower():
                T["key_ok"] += 1
                continue
            # 正确读法在这颗元件自己的池子里吗、多远？
            mi = [(k, v) for k, v in (pools.get(cid) or {}).items()
                  if v[0].lower() == gk.lower()]
            if not mi:
                T["no_read"] += 1
                continue
            k_correct, (can_c, d_c, box_c) = min(mi, key=lambda kv: kv[1][1])
            if d_c > a.radius:
                T["far"] += 1
                continue
            T["act"] += 1
            if cd is not None and d_c is not None:
                dgap["正确更近" if d_c <= cd else "选中更近"] += 1
            # 这个正确实例现在被谁拿着（谁把它选成了自己的 designator）？
            holder = None
            for cid2, (can2, k2, _d2) in chosen.items():
                if k2 == k_correct:
                    holder = cid2
            if holder is None:
                rival_kind["无人持有（自由）"] += 1
            else:
                gh = g_of_cid.get(holder)
                if gh is not None and can and gh.lower() == can.lower():
                    rival_kind["持有者真值 == 我们选中的值（真·互换）"] += 1
                else:
                    rival_kind["持有者真值 %s（非互换）" % (gh or "?")] += 1
            # 正确实例的「最近元件」是不是我们？（用我们元件框中心的代理：
            #   这里退一步用「谁到该文本框中点最近」，即候选框几何）
            bx, by = AM.box_center(box_c)
            best_cid, best_d = None, 1e18
            for pk in pk_list:
                c0 = pctr.get(pk)
                if c0 is None:
                    continue
                dd = math.hypot(c0[0] - bx, c0[1] - by)
                if dd < best_d:
                    best_d, best_cid = dd, cid_of.get(pk)
            T["act_inst_nearest_is_us" if best_cid == cid else "act_inst_nearest_is_rival"] += 1
            if len(ex) < a.show:
                ex.append((nm, gk, can or "-", None if cd is None else round(cd, 1),
                           round(d_c, 1), holder, g_of_cid.get(holder) or "-"))

    o = max(1, T["ordinary"])
    print("=" * 78)
    print("位号归属可修性（GT 锚定；gate %.0fpx / radius %.0fpx）" % (a.gate, a.radius))
    print("-" * 78)
    for k in ("ordinary", "no_pred", "key_ok", "no_read", "far", "act"):
        print("  %-28s %5d  %5.1f%%" % (k, T[k], 100.0 * T[k] / o))
    print("     其中 正确实例最近元件是我们  %d   ← 互最近邻规则可修" % T["act_inst_nearest_is_us"])
    print("     其中 正确实例最近元件是别人  %d   ← 距离与真值反着来，只能靠顺序/轴先验"
          % T["act_inst_nearest_is_rival"])
    print("  可行动集里：%s" % dict(dgap))
    print("  正确实例当前持有者：")
    for k, v in rival_kind.most_common():
        print("     %-34s %4d" % (k, v))
    if ex:
        print("-" * 78)
        print("  %-14s %-8s %-8s %-7s %-7s %-8s %s" % ("case", "GT", "我方选中", "选中d", "正确d", "持有者", "持有者真值"))
        for e in ex:
            print("  %-14s %-8s %-8s %-7s %-7s %-8s %s" % e)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
