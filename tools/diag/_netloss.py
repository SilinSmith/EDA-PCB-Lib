"""网络（hypergraph）失分的**三层分解** —— 把 netH 的损失拆成可行动的三块。

`net_hypergraph_f1` 是 micro 口径：`tp = Σ|预测集 ∩ GT集|`（匈牙利配对后），
`fp = 预测成员总数 − tp`，`fn = GT成员总数 − tp`。
recall 的上界 = 「我们交出的成员数 / GT成员数」，而这个上界被三件事卡住：

  ①**键**：该成员的元件键，在**度量自己的键重写之后**还在不在？
     （必须复用 `metrics.assign_components` + `rewrite_keys`：GT 的 `∅NNN`
      是**坐标匹配**的，度量会把我们的 `pNNN` 改写成 GT 的 `∅NNN`。
      自己在外部按字符串比对会把这批算成「缺键」，从而严重高估设计器的锅。）
  ②**号**：键对了，但这个引脚**有没有号**？（`_Pred.json` 的 `pins[key]` 只收有号的脚，
     没号的脚**永远不可能**成为网络成员 —— 官方 Q6 说 ∅ 类的号是随机的，另说）
  ③**拓扑**：键和号都在，但有没有被并进**正确的**那条网？（「网络过碎」）

用法：
    python work\\tools\\diag\\_netloss.py --dir work\\_fast_out_off9_nopincol
    python work\\tools\\diag\\_netloss.py --dir work\\_fast_out_off9_nopincol --cases 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
DSROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
GT9 = ("1", "2", "3", "4", "5", "7", "8", "9", "10")
MEM = re.compile(r"^(.+)\.([^.]+)$")

from schematic_parser import metrics as M                      # noqa: E402


def gt_of(case):
    for p in DSROOT.rglob(case + ".json"):
        if p.parent.name == case:
            return json.load(open(p, encoding="utf-8"))
    return None


def net_sets(doc):
    """用度量自己的解析器 -> [(名字, frozenset('key.number'))]，丢掉空网。"""
    return [(n, s) for n, s in M._net_sets(doc.get("nets") or {}) if s]


def pred_members(doc):
    """我们**能**参与网络的全部成员：{key: set(number)}。"""
    nums = defaultdict(set)
    for k, v in (doc.get("pins") or {}).items():
        for pn in (v or {}):
            m = re.match(r"^pin_(.+)$", pn)
            if m:
                nums[k].add(m.group(1))
            elif pn:
                nums[k].add(str(pn))
    return nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out_off9_nopincol"))
    ap.add_argument("--cases", default="")
    args = ap.parse_args()
    dirp = Path(args.dir)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or list(GT9)

    cfg = M.EvalConfig()
    tot = dict(m=0, tp=0, key=0, num=0, num_u=0, topo=0)
    print("%-5s %6s %5s %5s %5s %5s %5s  %5s   %s" %
          ("case", "GT成员", "tp", "缺键", "缺号", "∅没号", "拓扑", "成员",
           "样例（缺键 / 缺号 / ∅没号 / 拓扑丢）"))
    for case in cases:
        g = gt_of(case)
        pf = dirp / "final_results" / (case + "_Pred.json")
        if g is None or not pf.exists():
            print("%-5s 跳过（缺 GT 或产物）" % case)
            continue
        pred = json.load(open(pf, encoding="utf-8"))
        comp, keymap = M.assign_components(g, pred, cfg, None)
        p2 = M.rewrite_keys(pred, keymap)
        gs, ps = net_sets(g), net_sets(p2)
        nums = pred_members(p2)
        pairs = M.match_nets(gs, ps)                      # [(pred_idx, gt_idx)]
        where = {}
        for pi, gi in pairs:
            for mem in (gs[gi][1] & ps[pi][1]):
                where[mem] = gi
        n_tp = n_key = n_num = n_num_u = n_topo = 0
        ex_key, ex_num, ex_num_u, ex_topo = [], [], [], []
        ms = [x for s in gs for x in s[1]]
        for mem in ms:
            if not isinstance(mem, tuple) or len(mem) != 2:
                continue
            k, num = str(mem[0]), str(mem[1])
            if mem in where:
                n_tp += 1
                continue
            if k not in nums:
                n_key += 1
                ex_key.append("%s.%s" % (k, num))
            elif num not in nums[k]:
                if M.is_unkeyed(k):
                    n_num_u += 1
                    ex_num_u.append((k, num))
                else:
                    n_num += 1
                    ex_num.append((k, num))
            else:
                n_topo += 1
                ex_topo.append("%s.%s" % (k, num))
        for k, v in dict(m=len(ms), tp=n_tp, key=n_key, num=n_num,
                         num_u=n_num_u, topo=n_topo).items():
            tot[k] += v
        print("%-5s %6d %5d %5d %5d %5d %5d  %5d   %s" %
              (case, len(ms), n_tp, n_key, n_num, n_num_u, n_topo, len(nums),
               " | ".join([",".join(ex_key[:3]),
                           ",".join("%s.%s" % t for t in ex_num[:3]),
                           ",".join("%s.%s" % t for t in ex_num_u[:2]),
                           ",".join(ex_topo[:3])])))
    print()
    m = tot["m"]
    print("合计 GT 网络成员 %d" % m)
    for lbl, k in (("① 缺键（位号没匹配上）", "key"),
                   ("②a 有名元件的脚没号（可真修）", "num"),
                   ("②b ∅ 电源/地的脚没号（号随机，官方 Q6）", "num_u"),
                   ("③ 键号都在但没并进正确的网（拓扑）", "topo"),
                   ("④ 已经拿对（tp）", "tp")):
        print("  %-40s %4d  %5.1f%%" % (lbl, tot[k], 100.0 * tot[k] / m))
    print()
    print("⇒ 当前 recall = %.3f（tp/成员）" % (tot["tp"] / m))
    print("   修①后上限            %.3f" % ((tot["tp"] + tot["num"] + tot["num_u"] + tot["topo"]) / m))
    print("   修①+②a后上限         %.3f" % ((tot["tp"] + tot["num_u"] + tot["topo"]) / m))
    print("   修①+②a+③后上限       %.3f（②b 的 ∅ 号码随机，视为不可赢）"
          % ((tot["tp"] + tot["num_u"]) / m))


if __name__ == "__main__":
    main()
