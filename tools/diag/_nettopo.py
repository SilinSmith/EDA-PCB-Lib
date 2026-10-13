# -*- coding: utf-8 -*-
"""netH ③「拓扑丢」的**逐成员归因** —— 我们到底把这颗脚放进了哪条网？

`_netloss.py` 把 netH 的 recall 损失拆成 ①缺键 / ②a缺号 / ②b∅缺号 / ③拓扑，其中：
  * ①（92 条）= GT 的 `∅NNN` 电源符号，度量把我们的 `pNNN` 重写成 `∅NNN`（坐标匹配）后
    仍有 92 条对不上 ⇒ 这些元件的键压根没进 `assign_components` 的 keymap（**没有对应元件**）；
  * ③（18 条）= 键和号**都在**我们的 `pins` 里，却没并进正确的那条网。

本探针只回答 ③（以及 ②a）的**下一步动作**：逐条打印「这颗脚在我们这边出现在哪些网里」，
从而区分三种完全不同的病因：
  a) **没有出现在任何网里** ⇒ 它被漏掉（比如元件是被「内涵键回收」新挂上键的，脚没跟着建）；
  b) **单独成一条网 / 与少数同伴成网** ⇒ 「网络过碎」，是分组/串接问题；
  c) **并进了另一条（通常是包含 GT 那条的子集/超集）网** ⇒ 合并错误。

用法：
    python -X utf8 work/_netwhy.py                       # 官方 10 张，当前交付态
    python -X utf8 work/_netwhy.py --dir work/_fast_out_cd10on --cases 3,7,10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
DSROOT = EDA / "yolo_part" / "Dataset(waitforpreprocess)"
from schematic_parser import metrics as M                       # noqa: E402

CASES = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")


def gt_of(case):
    for p in DSROOT.rglob(case + ".json"):
        if p.parent.name == case:
            return json.load(open(p, encoding="utf-8"))
    return None


def member_owner(doc):
    """{成员元组: [(网名, 成员数, 成员串列表), ...]}。"""
    own = defaultdict(list)
    for name, s in M._net_sets(doc.get("nets") or {}):
        mems = sorted("%s.%s" % (a, b) for a, b in s)
        for mem in s:
            own[mem].append((name, len(s), mems))
    return own


def member_detail(doc, key):
    """该元件在我们产物里的脚（号 + 名），以及它出现在哪些网里。"""
    pins = (doc.get("pins") or {}).get(key) or {}
    got = sorted(str(k).replace("pin_", "") for k in pins)
    nets = []
    for name, s in M._net_sets(doc.get("nets") or {}):
        if any(str(m[0]) == str(key) for m in s if isinstance(m, tuple) and len(m) == 2):
            nets.append((name, len(s), sorted("%s.%s" % (a, b) for a, b in s)))
    return got, nets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out_cd10on"))
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    cases = [c.strip() for c in a.cases.split(",") if c.strip()] or list(CASES)
    cfg = M.EvalConfig()
    agg = defaultdict(int)
    for case in cases:
        g = gt_of(case)
        pf = Path(a.dir) / "final_results" / (case + "_Pred.json")
        if g is None or not pf.exists():
            print("%-5s 跳过（缺 GT 或产物）" % case)
            continue
        pred = json.load(open(pf, encoding="utf-8"))
        _comp, keymap = M.assign_components(g, pred, cfg, None)
        p2 = M.rewrite_keys(pred, keymap)
        gs, ps = M._net_sets(g.get("nets") or {}), M._net_sets(p2.get("nets") or {})
        gs, ps = [(n, s) for n, s in gs if s], [(n, s) for n, s in ps if s]
        nums = defaultdict(set)
        for k, v in (p2.get("pins") or {}).items():
            for pn in (v or {}):
                nums[k].add(str(pn).replace("pin_", ""))
        pairs = M.match_nets(gs, ps)
        where = {}
        for pi, gi in pairs:
            for mem in (gs[gi][1] & ps[pi][1]):
                where[mem] = gi
        own = member_owner(p2)
        topo, num = [], []
        for gi, (gname, gset) in enumerate(gs):
            for mem in gset:
                if not isinstance(mem, tuple) or len(mem) != 2:
                    continue
                k, n = str(mem[0]), str(mem[1])
                if mem in where:
                    continue
                if k not in nums:
                    continue                                  # ①缺键，本轮不看
                if n not in nums[k]:
                    num.append((k, n, gname, gset))
                else:
                    topo.append((k, n, gname, gset, own.get((k, n)) or []))
        if not topo and not num:
            print("%-5s ✓ 无 ②a/③" % case)
            continue
        print("=" * 100)
        print("case %s：②a %d 条、③ %d 条" % (case, len(num), len(topo)))
        for k, n, gname, gset in num:
            got, nets = member_detail(p2, k)
            print("  ②a %s.%s  应在网 %s%s" % (k, n, gname, sorted("%s.%s" % (x, y) for x, y in gset)))
            print("       我们这颗元件的脚号 = %s；出现在 %d 条网里" % (got, len(nets)))
            agg["num-hit" if got else "num-nopin"] += 1
        for k, n, gname, gset, mine in topo:
            nsize = mine[0][1] if mine else 0
            kind = "a-没有任何网" if not mine else ("b-过碎(≤2 成员)" if nsize <= 2 else "c-并进了别的网")
            agg[kind] += 1
            print("  ③ %-16s [%s] 应在 %s（%d 成员：%s）"
                  % ("%s.%s" % (k, n), kind, gname, len(gset),
                     ",".join(sorted("%s.%s" % (x, y) for x, y in gset))[:90]))
            if mine:
                for name, size, mem in mine:
                    print("       我们放在 %s（%d 成员）：%s" % (name, size, ",".join(mem)[:90]))
            else:
                print("       我们的产物里**任何网都没包含这颗脚**")
    print()
    print("归因合计：", dict(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
