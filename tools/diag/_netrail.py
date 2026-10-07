# -*- coding: utf-8 -*-
"""按「电源/地轨引脚名」合并网络的离线实验（`netH` 权重 0.35）。

动机来自用户 m12359 的常识：**引脚名叫 GND，那它必然接在接地网络上**。
`_netwhy.py` 实测（10GT 9 张，GT 网络引脚 419 个）：

| 类别 | 个数 | 占比 | 含义 |
|---|---:|---:|---|
| `A 元件键就没有` | 106 | 0.253 | 要靠位号（见 `_ickey.py`） |
| `B 有元件没这个脚` | 5 | 0.060 | 引脚检测漏了 |
| **`C 有脚但不在网里`** | **50** | **0.119** | 引脚我们**有**，但没有任何网络引用它 |
| **`D 在网里但配错`** | **96** | **0.229** | 引脚在网里，只是那个网没配到这条 GT 网 |
| `E 已命中` | 142 | 0.339 | 就是 tp |

⇒ 一半以上的 `fn` 是 **C + D**（`50 + 96 = 146`），两者共同的特征是
**「同一个电源/地轨被切成了好几段」**。这个脚本就测「把共享同名轨引脚的网并起来」。

⚠️ 注意 `_netmerge.py --rail` 早先报过 `+0.00`，但那个脚本的「网络数」列在多数 case 上是 0
（读网络的路径有问题），所以那个负结果**不可信**，这里用官方 `evaluate_case` 重做。

用法：`python work\\tools\\diag\\_netrail.py [--pred 目录] [--cases 1,2,3]`
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402

TEN = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
FAST = TEN + ["0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]

RAIL = {"gnd", "gnda", "gndd", "agnd", "dgnd", "pgnd", "vss", "vssa", "vcc", "vdd",
        "vddd", "vdda", "vddio", "3v3", "5v", "12v", "vbat", "vbus", "vin", "vout"}


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def refs_of(hg: str):
    out = []
    for tok in str(hg or "").strip().strip("()").split(","):
        tok = tok.strip()
        if "." in tok:
            c, _, p = tok.rpartition(".")
            out.append((c.strip(), p.strip()))
    return out


def rail_merge(doc: dict, names: set) -> tuple:
    """把「共享同名轨引脚」的网络并起来。返回 (新文档, 并掉了几个网)。"""
    d = copy.deepcopy(doc)
    nets = d.get("nets") or {}
    if len(nets) < 2:
        return d, 0
    pins = d.get("pins") or {}
    names_list = list(nets)
    parent = {k: k for k in names_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for n in names_list:
        for comp, num in refs_of(nets[n].get("hyperGraph", "")):
            nm = str(((pins.get(comp) or {}).get("pin_%s" % num) or {}).get("pinname") or "")
            if nm.strip().lower() not in names:
                continue
            key = nm.strip().lower()        # ← 按**名字**并，不是按 (元件,引脚)
            if key in owner:
                ra, rb = find(owner[key]), find(n)
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[key] = n

    groups = {}
    for n in names_list:
        groups.setdefault(find(n), []).append(n)
    if len(groups) == len(names_list):
        return d, 0

    out = {}
    for members in groups.values():
        rec = dict(nets[members[0]])
        edges, refs, seen = {}, [], set()
        for n in members:
            for ref in refs_of(nets[n].get("hyperGraph", "")):
                if ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
            for ev in (nets[n].get("edges") or {}).values():
                edges["edge_%d" % (len(edges) + 1)] = ev
        rec["edges"] = edges
        rec["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"
        out["net_%d" % (len(out) + 1)] = rec
    d["nets"] = out
    return d, len(names_list) - len(out)


def _net_names(nets, pins):
    """每个网络里出现过的「轨名」集合。"""
    out = {}
    for n, rec in nets.items():
        s = set()
        for comp, num in refs_of(rec.get("hyperGraph", "")):
            nm = str(((pins.get(comp) or {}).get("pin_%s" % num) or {}).get("pinname") or "")
            if nm.strip().lower() in RAIL:
                s.add(nm.strip().lower())
        out[n] = s
    return out


def orphan_rail(doc: dict, names: set = RAIL) -> tuple:
    """把「有引脚、但没有任何网络引用它」的轨引脚，挂到同名的最大网络上去。

    这是 `_netwhy.py` 里 `C_有脚但不在网里` 那一类（10GT 上 50 个 / 0.119）的对策，
    比「全局按名字合并」保守得多 —— 只动本来就悬空的引脚。
    """
    d = copy.deepcopy(doc)
    nets = d.get("nets") or {}
    pins = d.get("pins") or {}
    if not nets or not pins:
        return d, 0
    in_nets = set()
    for rec in nets.values():
        in_nets |= set(refs_of(rec.get("hyperGraph", "")))
    # 每个轨名 -> 最大的那个网络
    per_name = {}
    for n, s in _net_names(nets, pins).items():
        size = len(refs_of(nets[n].get("hyperGraph", "")))
        for nm in s:
            if nm not in per_name or size > per_name[nm][1]:
                per_name[nm] = (n, size)

    added = 0
    for comp, pk in pins.items():
        num = str(pk if isinstance(pk, str) else "").replace("pin_", "")
        for pnum, rec in (pins.get(comp) or {}).items():
            num = str(pnum).replace("pin_", "")
            nm = str((rec or {}).get("pinname") or "").strip().lower()
            if nm not in names or (comp, num) in in_nets:
                continue
            hit = per_name.get(nm)
            if not hit:
                continue
            rec_n = nets[hit[0]]
            refs = refs_of(rec_n.get("hyperGraph", ""))
            if (comp, num) in refs:
                continue
            refs.append((comp, num))
            rec_n["hyperGraph"] = "(" + ",".join("%s.%s" % t for t in refs) + ")"
            in_nets.add((comp, num))
            added += 1
    return d, added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arm", default="rail", choices=["rail", "orphan"])
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or (FAST if a.all else TEN)
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()

    agg = collections.Counter()
    print("%-7s %8s %8s %8s | %-14s %s" %
          ("case", "原总分", "并后", "Δ", "netH 原->新", "网数 原->新"))
    for cid in cases:
        c, _how = resolve(cid, exact, digits)
        gt = load_json(c.json_path)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        s0 = evaluate_case(gt, pr, case_id=cid, cfg=cfg)
        d, merged = (orphan_rail(pr) if a.arm == "orphan" else rail_merge(pr, RAIL))
        s1 = evaluate_case(gt, d, case_id=cid, cfg=cfg)
        agg["s0"] += s0.score
        agg["s1"] += s1.score
        agg["n0"] += s0.net_hypergraph.f1
        agg["n1"] += s1.net_hypergraph.f1
        agg["l0"] += s0.net_line.f1
        agg["l1"] += s1.net_line.f1
        agg["tp0"] += s0.net_hypergraph.tp
        agg["tp1"] += s1.net_hypergraph.tp
        agg["fp0"] += s0.net_hypergraph.fp
        agg["fp1"] += s1.net_hypergraph.fp
        agg["fn0"] += s0.net_hypergraph.fn
        agg["fn1"] += s1.net_hypergraph.fn
        agg["c"] += 1
        n0 = len(pr.get("nets") or {})
        n1 = len(d.get("nets") or {})
        print("%-7s %8.2f %8.2f %+8.2f | %.4f -> %.4f   %d -> %d%s" %
              (cid, s0.score, s1.score, s1.score - s0.score,
               s0.net_hypergraph.f1, s1.net_hypergraph.f1, n0, n1,
               "" if merged == 0 else "  (并掉 %d)" % merged))

    n = max(1, agg["c"])
    print()
    print("合计 %d 张：总分 %.2f -> %.2f (%+.2f) | netH %.4f -> %.4f (%+.4f) | line %.4f -> %.4f"
          % (n, agg["s0"] / n, agg["s1"] / n, (agg["s1"] - agg["s0"]) / n,
             agg["n0"] / n, agg["n1"] / n, (agg["n1"] - agg["n0"]) / n,
             agg["l0"] / n, agg["l1"] / n))
    print("netH 明细  tp %d->%d   fp %d->%d   fn %d->%d"
          % (agg["tp0"], agg["tp1"], agg["fp0"], agg["fp1"], agg["fn0"], agg["fn1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
