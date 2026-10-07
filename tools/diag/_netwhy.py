# -*- coding: utf-8 -*-
"""网络（`netH`，权重 0.35）的丢分逐条拆开：GT 的网络引脚里，我们缺的那些**为什么缺**。

背景：`netH` 是 **micro** 口径 ——
    `tp = Σ|pred_net ∩ gt_net|`、`fp = pred_pins - tp`、`fn = gt_pins - tp`。
所以 `fn`（GT 有、我们没有的网络引脚）是唯一的敌人；多出来的错引脚只是 `fp`，
在分子里损失小得多。**结论：要涨就得让「GT 的网络引脚」更多地出现在我们的网络里。**

分类（对 GT 的每一个 (元件, 引脚号) 引用，互斥、从上往下第一个命中）：
  `A 元件键就没有`  —— 这个元件名在我们的 `components` 里都不存在（键没配上）。
  `B 有元件没这个脚` —— 元件在，但这个引脚号在我们的 `pins[comp]` 里没有。
  `C 有脚但不在网里` —— 引脚我们有，但它没被任何网络引用（`assemble.py` 的 `NET_DROP_UNKEYED`
                        把它踢了，或者 stage1 就没把它连上）。
  `D 在网里但配错`  —— 引脚在网里，只是那个网没配到这条 GT 网（匈牙利配对问题）。
  `E 已命中`        —— 就是 `tp`。

用法：`python work\\tools\\diag\\_netwhy.py [--pred 目录] [--cases 1,2,3] [--show 40]`
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402
from schematic_parser.metrics import match_nets, net_pin_set  # noqa: E402

TEN = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
FAST = TEN + ["0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]
HG = re.compile(r"\(([^)]*)\)")


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def refs_of(hg: str):
    """`"(U1.45,C2.1)"` -> `[("U1","45"),("C2","1")]`（按最后一个点切）。"""
    out = []
    for tok in str(hg or "").strip().strip("()").split(","):
        tok = tok.strip()
        if "." in tok:
            c, _, p = tok.rpartition(".")
            out.append((c.strip(), p.strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or TEN
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    tot = collections.Counter()
    for cid in cases:
        c, _how = resolve(cid, exact, digits)
        gt = load_json(c.json_path)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        g_nets = [(k, frozenset(net_pin_set(v))) for k, v in (gt.get("nets") or {}).items()]
        p_nets = [(k, frozenset(net_pin_set(v))) for k, v in (pr.get("nets") or {}).items()]
        pairs = match_nets(g_nets, p_nets)
        matched_partner = {}
        for pi, gi in pairs:                        # 官方返回的是 (预测下标, GT 下标)
            matched_partner[g_nets[gi][0]] = p_nets[pi][0]
        # 我们这边所有网络引用的脚
        in_any_net = set()
        for _k, s in p_nets:
            in_any_net |= s
        p_pins = pr.get("pins") or {}
        r = collections.Counter()
        for gk, gv in g_nets:
            partner = matched_partner.get(gk)
            pset = dict(p_nets).get(partner, frozenset()) if partner else frozenset()
            for (comp, num) in gv:
                r["gt_pins"] += 1
                if comp not in (pr.get("components") or {}):
                    r["A_元件键就没有"] += 1
                elif (comp, num) not in {(comp, str(k).replace("pin_", ""))
                                         for k in (p_pins.get(comp) or {})}:
                    r["B_有元件没这个脚"] += 1
                elif (comp, num) not in in_any_net:
                    r["C_有脚但不在网里"] += 1
                elif (comp, num) in pset:
                    r["E_已命中"] += 1
                else:
                    r["D_在网里但配错"] += 1
        tot.update(r)
        print("case %-6s GT网 %2d / 我们网 %3d ｜ GT网络引脚 %3d：A %3d  B %3d  C %3d  D %3d  E %3d"
              % (cid, len(g_nets), len(p_nets), r["gt_pins"], r["A_元件键就没有"],
                 r["B_有元件没这个脚"], r["C_有脚但不在网里"], r["D_在网里但配错"], r["E_已命中"]))

    n = max(1, tot["gt_pins"])
    print("\n== 合计 GT 网络引脚 %d" % tot["gt_pins"])
    for k in ("A_元件键就没有", "B_有元件没这个脚", "C_有脚但不在网里",
              "D_在网里但配错", "E_已命中"):
        print("   %-14s %5d  (%.3f)" % (k, tot[k] / 1, tot[k] / n))
    print("\n   ⇒ 真·可修的是 B（元件在、引脚检测漏了）与 C（引脚在、没进网络）；"
          "A 要靠位号（见 _ickey.py）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
