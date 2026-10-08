"""量「一块文字框被两颗元件同时当成位号」的规模，并离线试 `own.assign` 全局一对一的收益。

背景（`own` 节点的「尚未实现」那一半）：`eda-pcb/cvparts/own.py` 的 `assign()` 是
**整体最小代价的一对一**（直接复用官方 `schematic_parser.metrics._assign`），
但全仓库**从未被调用**。`stage2/assign.py:755-765` 现在的位号裁决是：

  1. 每颗元件**各自**从自己的候选池里挑最优 ⇒ 两块元件可以同时选中**同一块文字框**；
  2. 再按**位号字符串** `canon` 贪心去重 ⇒ 撞号时低分者**整颗丢掉**位号（不是改拿次优框）。

`own.assign` 的说法是：把「元件 × 文字框」整体解一次一对一，一块框只能是**一颗**元件的主人，
抢输的元件在同一次求解里拿到自己的**次优框**。

本工具不改管线，只读 `runs/stage2/_assigned/<case>.json`（里面有 `comps[*]["designator"]`
与 `desig_slots`＝每颗元件的全部候选，带 `box_i`/`box`/`canon`/`score`），做三件事：

  · 数现状：`dup_box`（同一块框被 >=2 颗元件选中）、`dup_canon`（同一读法被 >=2 颗选中）、
    `n_desig`（拿到位号的元件数）；
  · 离线跑 `own.assign` 那一路，给出同样三个数 + 「改判了几个元件」；
  · 打印两臂**不一致**的元件明细（`--show`）。

用法：
  python work/tools/diag/_ownasgn.py --dir work/_fast_out_novlm9
  python work/tools/diag/_ownasgn.py --dir work/_fast_out_novlm9 --show --cases 1,2
  python work/tools/diag/_ownasgn.py --dir work/_fast_out_bs1_9 --only 8
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PKG = os.path.join(ROOT, "eda-pcb")
if PKG not in sys.path:
    sys.path.insert(0, PKG)


def _load(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _cases_in(d: str) -> List[str]:
    sub = os.path.join(d, "runs", "stage2", "_assigned")
    if not os.path.isdir(sub):
        return []
    names = [f[:-5] for f in os.listdir(sub) if f.endswith(".json")]
    return sorted(names, key=lambda s: (len(s), s))


def _key(info: Optional[dict]):
    if not info:
        return None
    bi = int(info.get("box_i", -1))
    if bi >= 0:
        return ("i", bi)
    b = info.get("box")
    return ("b", tuple(round(float(v), 2) for v in b)) if b else None


def _cur_pick(doc: dict) -> Dict[str, dict]:
    """现状那一臂：直接用文档里已经落盘的 `designator`。"""
    out = {}
    for cid, rec in (doc.get("comps") or {}).items():
        d = rec.get("designator")
        if d and d.get("canon"):
            out[cid] = d
    return out


def _asgn_pick(doc: dict) -> Dict[str, dict]:
    """`own.assign` 那一臂：(元件 x 文字框) 整体一对一，再按 canon 去重。

    与现状的差别只在第一步 —— 第二步（canon 唯一）保持不变，因为真图上位号确实唯一。
    """
    from cvparts import own as own_mod

    slots: Dict[str, List[dict]] = doc.get("desig_slots") or {}
    comps = doc.get("comps") or {}
    comp_bb = {}
    for cid, rec in comps.items():
        b = rec.get("bbox") or rec.get("bbox_img")
        comp_bb[cid] = tuple(float(v) for v in b) if b and len(b) == 4 else (0.0, 0.0, 0.0, 0.0)

    # 标的 = 一块具体的文字框（用 `_key` 归一，跨元件共享）
    items, seen = [], {}
    score: Dict[tuple, float] = {}
    info_of: Dict[tuple, dict] = {}
    for cid, alts in slots.items():
        for a in alts or []:
            k = _key(a)
            if k is None:
                continue
            score[(cid, k)] = float(a.get("score") or 0.0)
            if k not in seen:
                seen[k] = True
                items.append(own_mod.Party("tb|%s" % (k,), tuple(a.get("box") or (0,) * 4), "text"))
                info_of[k] = a
    claimants = [own_mod.Party(cid, comp_bb[cid], "comp")
                 for cid in slots
                 if any(_key(x) in seen for x in (slots.get(cid) or []))]

    got, _led = own_mod.assign(
        claimants, items,
        gate=lambda c, i: (c.id, _key_of_item(i.id)) in score,
        cost=lambda c, i: -score[(c.id, _key_of_item(i.id))])

    first: List[tuple] = []
    for cid, iid in got.items():
        k = _key_of_item(iid)
        a = info_of.get(k)
        if a:
            first.append((float(a.get("score") or 0.0), cid, a))
    first.sort(key=lambda x: -x[0])
    picked, seen_canon = {}, set()
    for _sc, cid, a in first:
        can = a.get("canon")
        if not can or can in seen_canon:
            continue
        seen_canon.add(can)
        picked[cid] = a
    return picked


def _key_of_item(iid: str):
    """把标的 id（`tb|('i', 12)`）还原成 `_key`。"""
    s = iid[3:]
    try:
        return eval(s)  # 只吃我们自己写出来的 ('i', N) / ('b', (..))
    except Exception:
        return None


def _gt_index() -> Dict[str, str]:
    from schematic_parser.data import DEFAULT_DATA_ROOT, discover
    idx: Dict[str, str] = {}
    for c in discover(DEFAULT_DATA_ROOT):
        idx[str(c.case_id)] = c.json_path
        idx[os.path.splitext(os.path.basename(c.png_path))[0]] = c.json_path
    return idx


def _cid_bbox(d: str, case: str) -> Dict[str, tuple]:
    """stage1 的 `components[*]` 里取 `id -> bbox_json`（与 `_assigned` 同一坐标系）。"""
    p = os.path.join(d, "runs", "stage1", "%s.stage1.json" % case)
    doc = _load(p) or {}
    out = {}
    for c in doc.get("components") or []:
        b = c.get("bbox_json") or c.get("bbox_img")
        if b and len(b) == 4:
            out[str(c.get("id"))] = tuple(float(v) for v in b)
    return out


def _nearest_gt_key(bb: Optional[tuple], gtc: Dict[str, dict], tol: float = 20.0):
    """按**角点距离**找最近的 GT 元件（官方口径 `bbox_mode='corners'`）。"""
    if not bb:
        return None, None
    best, bd = None, 1e9
    for k, v in gtc.items():
        b = v.get("bbox") or v.get("bbox_json")
        if not b or len(b) != 4:
            continue
        d = max(abs(bb[i] - float(b[i])) for i in range(4))
        if d < bd:
            best, bd = k, d
    return (best, bd) if bd <= tol else (None, bd)


def _stats(pick: Dict[str, dict], doc: dict) -> dict:
    by_box: Dict[tuple, List[str]] = {}
    by_canon: Dict[str, List[str]] = {}
    for cid, d in pick.items():
        k = _key(d)
        if k is not None:
            by_box.setdefault(k, []).append(cid)
        by_canon.setdefault(str(d.get("canon")), []).append(cid)
    n_comp = len(doc.get("comps") or {})
    return {
        "n_comp": n_comp,
        "n_desig": len(pick),
        "dup_box": sum(1 for v in by_box.values() if len(v) > 1),
        "dup_box_comp": sum(len(v) for v in by_box.values() if len(v) > 1),
        "dup_canon": sum(1 for v in by_canon.values() if len(v) > 1),
    }


def _headroom(a, names: List[str]) -> int:
    """位号通道的**天花板**：本来手里就有正确读法、却没被选中的元件有多少。

    对每张图：先按官方 `corners` 口径把我们的元件与 GT 元件配成一对（一颗 GT 只认最近的
    一颗，重复检出取 dev 最小的），再看这一颗的 `designator` 对不对，以及
    `desig_slots[cid]` 里**有没有**那条正确的读法。四种落点：

      ok           读对了
      none         什么都没选，但候选池里有正确读法 ⇒ **可救**
      none_blind   什么都没选，候选池里也没有 ⇒ 只能靠识别器
      wrong        选错了；候选池里有正确读法 ⇒ **可救**（tie-break 问题）
      wrong_blind  选错了，候选池里也没有 ⇒ 识别器问题

    用法：`--headroom --gt --dir work/_fast_out_novlm9`
    """
    import re

    gti = _gt_index()
    key_re = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")
    T = {"n_comp": 0, "n_gtkey": 0, "ok": 0, "none": 0, "none_blind": 0,
         "wrong": 0, "wrong_blind": 0, "extra": 0}
    print("%-14s %6s %5s %5s %5s %5s %5s" %
          ("case", "gtKey", "ok", "none", "n_bl", "wrong", "w_bl"))
    for n in names:
        doc = _load(os.path.join(a.dir, "runs", "stage2", "_assigned", "%s.json" % n))
        gtc = (_load(gti.get(n)) or {}).get("components") or {} if gti.get(n) else {}
        if not doc or not gtc:
            continue
        bbs = _cid_bbox(a.dir, n)
        slots = doc.get("desig_slots") or {}
        cur = _cur_pick(doc)
        # 一颗 GT 只认最近的一颗（重复检出取 dev 最小）
        pair: Dict[str, tuple] = {}
        for cid, bb in bbs.items():
            gk, dev = _nearest_gt_key(bb, gtc)
            if not gk or not key_re.match(str(gk)) or dev is None:
                continue
            if gk not in pair or dev < pair[gk][1]:
                pair[gk] = (cid, dev)
        c = {"n_gtkey": 0, "ok": 0, "none": 0, "none_blind": 0, "wrong": 0, "wrong_blind": 0}
        for gk, (cid, _dev) in pair.items():
            c["n_gtkey"] += 1
            cands = {str(x.get("canon")) for x in (slots.get(cid) or []) if x.get("canon")}
            got = str((cur.get(cid) or {}).get("canon") or "")
            if got == gk:
                c["ok"] += 1
            elif not got:
                if gk in cands:
                    c["none"] += 1
                else:
                    c["none_blind"] += 1
            else:
                if gk in cands:
                    c["wrong"] += 1
                else:
                    c["wrong_blind"] += 1
        T["n_comp"] += len(bbs)
        for k in c:
            T[k] = T.get(k, 0) + c[k]
        print("%-14s %6d %5d %5d %5d %5d %5d" %
              (n, c["n_gtkey"], c["ok"], c["none"], c["none_blind"],
               c["wrong"], c["wrong_blind"]))
    rec = T["none"] + T["wrong"]
    print("-" * 72)
    print("合计：我们的元件 %d ｜ 按 corners 口径认到 GT 位号元件 **%d**"
          % (T["n_comp"], T["n_gtkey"]))
    print("  读对 **%d** ｜ 没读 %d ｜ 读错 %d"
          % (T["ok"], T["none"] + T["none_blind"], T["wrong"] + T["wrong_blind"]))
    print("  其中**候选池里本来就有正确读法**（归属/tie-break 可救）= **%d**"
          "（没读 %d + 读错 %d）" % (rec, T["none"], T["wrong"]))
    print("  天花板（识别器给不出正确的）= %d（没读 %d + 读错 %d）"
          % (T["none_blind"] + T["wrong_blind"], T["none_blind"], T["wrong_blind"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="快测输出目录（里面有 runs/stage2/_assigned）")
    ap.add_argument("--cases", default="", help="逗号分隔的 case 子串过滤")
    ap.add_argument("--only", default="")
    ap.add_argument("--show", action="store_true", help="打印两臂不一致的元件明细")
    ap.add_argument("--gt", action="store_true",
                    help="对改判的元件去问 GT：哪一臂的读法才是这颗元件的真位号")
    ap.add_argument("--headroom", action="store_true",
                    help="（需 --gt）量位号通道的**天花板**：本来手里就有正确读法却没被选中的元件数")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    want = [s.strip() for s in (a.cases or a.only).split(",") if s.strip()]
    names = [n for n in _cases_in(a.dir)
             if (not want or any(n == w or n.endswith(w) for w in want))]
    if not names:
        print("没找到 _assigned，检查 --dir：%s" % a.dir)
        return 1

    if a.headroom:
        return _headroom(a, names)

    tot_cur = {"n_comp": 0, "n_desig": 0, "dup_box": 0, "dup_box_comp": 0, "dup_canon": 0}
    tot_as = dict(tot_cur)
    n_diff = 0
    n_cur_ok = n_as_ok = n_cur_bad = n_as_bad = 0
    gti = _gt_index() if a.gt else {}
    print("%-14s %5s %5s %5s %5s   %5s %5s %5s %5s" %
          ("case", "comp", "cur", "dupB", "dupC", "asgn", "dupB", "dupC", "diff"))
    for n in names:
        doc = _load(os.path.join(a.dir, "runs", "stage2", "_assigned", "%s.json" % n))
        if not doc:
            continue
        cur = _cur_pick(doc)
        ab = _stats(cur, doc)
        try:
            asg = _asgn_pick(doc)
        except Exception as exc:
            print("%-14s asgn 崩了：%s" % (n, exc))
            continue
        bb = _stats(asg, doc)
        diff = []
        for cid in sorted(set(cur) | set(asg)):
            k1 = _key(cur.get(cid))
            k2 = _key(asg.get(cid))
            if k1 != k2:
                diff.append((cid, (cur.get(cid) or {}).get("canon"), k1,
                             (asg.get(cid) or {}).get("canon"), k2))
        n_diff += len(diff)
        for k in tot_cur:
            tot_cur[k] += ab[k]
            tot_as[k] += bb[k]
        print("%-14s %5d %5d %5d %5d   %5d %5d %5d %5d" %
              (n, ab["n_comp"], ab["n_desig"], ab["dup_box_comp"], ab["dup_canon"],
               bb["n_desig"], bb["dup_box_comp"], bb["dup_canon"], len(diff)))
        if a.show or a.gt:
            gtc = {}
            if a.gt:
                gp = gti.get(n)
                gtc = (_load(gp) or {}).get("components") or {} if gp else {}
                bbs = _cid_bbox(a.dir, n)
            for row in diff[:a.top]:
                cid = row[0]
                if a.gt:
                    gt_key, dev = _nearest_gt_key(bbs.get(cid), gtc)
                    c_ok = row[1] is not None and row[1] == gt_key
                    a_ok = row[3] is not None and row[3] == gt_key
                    n_cur_ok += 1 if c_ok else 0
                    n_as_ok += 1 if a_ok else 0
                    if row[1] and not c_ok:
                        n_cur_bad += 1
                    if row[3] and not a_ok:
                        n_as_bad += 1
                    print("      %-8s GT=%-6s dev=%-6s cur=%-6s%s   asgn=%-6s%s"
                          % (cid, gt_key, ("%.1f" % dev) if dev is not None else "-",
                             row[1], "  ✓" if c_ok else ("  ✗" if row[1] else "  -"),
                             row[3], "  ✓" if a_ok else ("  ✗" if row[3] else "  -")))
                else:
                    print("      %-8s cur=%s%s  ->  asgn=%s%s"
                          % (cid, row[1], row[2], row[3], row[4]))

    print("-" * 78)
    print("合计  元件 %d ｜ 现状拿到位号 %d / dup框 %d / dup读法 %d"
          % (tot_cur["n_comp"], tot_cur["n_desig"], tot_cur["dup_box_comp"], tot_cur["dup_canon"]))
    print("      own.assign 拿到位号 %d / dup框 %d / dup读法 %d ｜ 改判 %d 个元件"
          % (tot_as["n_desig"], tot_as["dup_box_comp"], tot_as["dup_canon"], n_diff))
    if a.gt and n_diff:
        print("      GT 裁决：现状改判点上答对 %d / 答错 %d ｜ own.assign 答对 %d / 答错 %d"
              % (n_cur_ok, n_cur_bad, n_as_ok, n_as_bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
