# -*- coding: utf-8 -*-
"""**自短接体检**：某颗两脚元件（r/c/l/d/led/fuse/crystal_2pin）的**两个端脚出现在同
一条网里**——这在原理图上几乎一定是错的（元件被自己短接）。

这是 `stamp_pins` 槽位 bug 的**外部症状计数器**：`num_of[(cid,0)] == num_of[(cid,2)]`
会让两条本该分开的 stage1 网在 `merge_shared_pins` 里被合并，于是「电容两端短路」。
它也用作修复的**侧证**（side signal）：修好之后自短接数应显著下降。

只读 `final_results/<case>_Pred.json`（我方）与官方 GT json，不跑流水线：

    python -X utf8 eda-pcb/tools/diag/_shortchk.py --run work/_fast_out_fin --split 10GT
    python -X utf8 eda-pcb/tools/diag/_shortchk.py --run work/_fast_out_sfix0 --diff work/_fast_out_sfix1
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover                     # noqa: E402

NET_RE = re.compile(r"\(([^()]*)\)")
TWO_PIN = frozenset({"r", "c", "l", "d", "led", "fuse", "crystal_2pin", "crystal"})


def refs_of(hg: str):
    """`"(C157.2,C158.2,U1.8,∅200.198)"` -> `[('C157','2'), ..., ('∅200','198')]`"""
    out = []
    for grp in NET_RE.findall(hg or ""):
        for tok in grp.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "." in tok:
                comp, pin = tok.rsplit(".", 1)
                out.append((comp, pin))
            else:
                out.append((tok, ""))
    return out


def load_nets(path: Path):
    """返回 `{网名: set((元件, 脚))}`。`final_results/<case>_Pred.json` 与 GT 同构。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    nets = d.get("nets") or {}
    out = {}
    for name, v in nets.items():
        hg = v.get("hyperGraph") if isinstance(v, dict) else v
        out[name] = set(refs_of(hg if isinstance(hg, str) else ""))
    return d, out


def comp_types(doc) -> dict:
    cs = doc.get("components") or {}
    out = {}
    if isinstance(cs, dict):
        for k, v in cs.items():
            if isinstance(v, dict):
                out[k] = str(v.get("type") or "").lower()
    elif isinstance(cs, list):
        for v in cs:
            if isinstance(v, dict):
                out[str(v.get("id") or v.get("Name") or "")] = str(v.get("type") or "").lower()
    return out


def scan(run: Path, cid: str, verbose=False):
    pred_p = run / "final_results" / ("%s_Pred.json" % cid)
    if not pred_p.exists():
        return None
    pdoc, pnets = load_nets(pred_p)
    types = comp_types(pdoc)
    # 每个元件在 GT 里出现在哪几条网
    gt_p = None
    for c in discover(DEFAULT_DATA_ROOT):
        if c.case_id == cid:
            gt_p = Path(c.json_path)
            break
    gt_of = {}
    if gt_p and gt_p.exists():
        gdoc, gnets = load_nets(gt_p)
        for gname, refs in gnets.items():
            for comp, _pin in refs:
                gt_of.setdefault(comp, set()).add(gname)
    bad = []
    for name, refs in pnets.items():
        by_comp = collections.defaultdict(set)
        for comp, pin in refs:
            by_comp[comp].add(pin)
        for comp, pins in by_comp.items():
            if comp.startswith("∅") or len(pins) < 2:
                continue
            if types.get(comp, "") not in TWO_PIN:
                continue
            g = gt_of.get(comp) or set()
            false = len(g) > 1          # GT 把两端分在不同网里 ⇒ 我们这边是真错
            bad.append((name, comp, sorted(pins), false, sorted(g)))
    n_false = sum(1 for b in bad if b[3])
    print("  case %-6s 我方网 %2d ｜ 自短接元件 %2d（其中与 GT 冲突 %2d）"
          % (cid, len(pnets), len(bad), n_false))
    if verbose:
        for name, comp, pins, false, g in bad[:10]:
            print("     %s: %s 脚=%s %s GT网=%s"
                  % (name, comp, pins, "**假合并**" if false else "(GT 也同网)",
                     ",".join(g) if g else "?"))
    return {"nets": len(pnets), "short": len(bad), "false": n_false}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_fin")
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--case", default="")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diff", default="", help="第二套 run 目录：只列两次都跑过的 case 的对比")
    a = ap.parse_args()
    runs = [Path(a.run)]
    if a.diff:
        runs.append(Path(a.diff))
    for r in runs:
        if not r.is_absolute():
            r = (ROOT / r).resolve()
    cases = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split]
    if a.case:
        cases = [c for c in cases if c.case_id == a.case]
    res = []
    for c in cases[:a.n]:
        row = [scan(r, c.case_id, a.verbose) for r in runs]
        res.append((c.case_id, row))
    print("\n== 汇总 ==")
    for i, r in enumerate(runs):
        tot = collections.Counter()
        for _cid, row in res:
            if row[i]:
                tot.update(row[i])
        print("  %-46s 网 %3d ｜ 自短接 %3d ｜ **与 GT 冲突 %3d**"
              % (str(r), tot["nets"], tot["short"], tot["false"]))
    if len(runs) == 2:
        print("\n  逐案对比（旧 -> 新）：")
        for cid, row in res:
            if row[0] and row[1]:
                d = row[1]["false"] - row[0]["false"]
                flag = "  <== 改善" if d < 0 else ("  !! 变差" if d > 0 else "")
                print("     case %-6s 假合并 %2d -> %2d (%+d)%s"
                      % (cid, row[0]["false"], row[1]["false"], d, flag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
