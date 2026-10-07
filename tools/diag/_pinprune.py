# -*- coding: utf-8 -*-
"""引脚剪枝的离线实验：不动管线，只删预测里的引脚，再按官方口径重打分。

动机（`_pinjunk.py` 实测，快速回归集 19 张）：
    只统计 GT 认定为 IC 的元件（官方 `pin_f1` 的 `box_comps` 就是这一批）——
    **GT 引脚 587 个，我们输出 225 个，其中 104 个（46.2%）的 `pin_N` 键在 GT 里根本不存在**。
    多出来的键在官方口径下**全是 fp**，删掉只涨不跌（对应的 GT 键本来就是 fn，删了 fn 不变）。

这里测几条**只看自己、不看 GT** 的剪枝规则：
  A `重复号`：同一个元件里同一个引脚号出现多次 ⇒ 只留第一个（一个引脚号只有一个引脚）。
  B `长名字`：`pinname` 长度 > 8（真引脚名极少超过 8 个字符，比如 `/NRESET` 是 7）。
  C `名字像型号`：`pinname` 去掉 `/` 后长度 ≥4 且是元件 `Name` 的子串。
  D `数字名`：`pinname` 以数字开头且长度 ≥3（`16`/`114` 这种，是引脚号漏进了名字位）。

用法：`python work\\tools\\diag\\_pinprune.py [--pred 目录] [--cases 1,2,3]`
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402

FAST = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
        "0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]

NUM = re.compile(r"^(\d+)")


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def prune(doc: dict, rules: str) -> tuple:
    """按 `rules`（子集 "ABCD"）删引脚，返回 (新文档, 删掉的个数)。"""
    d = copy.deepcopy(doc)
    gone = 0
    for comp, pins in list((d.get("pins") or {}).items()):
        if not isinstance(pins, dict):
            continue
        nm = str(((d.get("components") or {}).get(comp) or {}).get("Name") or "")
        seen = set()
        keep = {}
        for pk, pv in pins.items():
            pv = pv or {}
            name = str(pv.get("pinname") or "")
            drop = False
            m = NUM.match(str(pk).replace("pin_", ""))
            num = m.group(1) if m else pk
            if "A" in rules:
                if num in seen:
                    drop = True
                else:
                    seen.add(num)
            if not drop and "B" in rules and len(name) > 8:
                drop = True
            if not drop and "C" in rules and len(name) >= 4 and nm \
                    and name.replace("/", "").upper() in nm.upper():
                drop = True
            if not drop and "D" in rules and name[:1].isdigit() and len(name) >= 3:
                drop = True
            if drop:
                gone += 1
            else:
                keep[pk] = pv
        if keep:
            d["pins"][comp] = keep
        else:
            d["pins"].pop(comp, None)
    return d, gone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or FAST
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()

    ARMS = ["", "A", "B", "D", "AB", "ABD", "ABCD"]
    agg = {k: collections.Counter() for k in ARMS}
    agg["_n"] = 0
    per = []
    for cid in cases:
        c, _how = resolve(cid, exact, digits)
        gt = load_json(c.json_path)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        row = {}
        for arm in ARMS:
            d, _gone = prune(pr, arm) if arm else (pr, 0)
            s = evaluate_case(gt, d, case_id=cid, cfg=cfg)
            row[arm] = s
            agg[arm]["pin"] += s.pin.f1
            agg[arm]["comp"] += s.component.f1
            agg[arm]["net"] += s.net_hypergraph.f1
            agg[arm]["line"] += s.net_line.f1
            agg[arm]["tot"] += s.score
            agg[arm]["tp"] += s.pin.tp
            agg[arm]["fp"] += s.pin.fp
            agg[arm]["fn"] += s.pin.fn
        agg["_n"] += 1
        per.append((cid, row))

    n = max(1, agg["_n"])
    print("%-6s %8s %8s %8s %8s %8s %6s %6s %6s" %
          ("剪枝", "总分", "元件", "引脚", "网络", "导线", "引脚tp", "引脚fp", "引脚fn"))
    for arm in ARMS:
        t = agg[arm]
        print("%-6s %8.2f %8.4f %8.4f %8.4f %8.4f %6d %6d %6d" %
              (arm or "无", t["tot"] / n, t["comp"] / n, t["pin"] / n,
               t["net"] / n, t["line"] / n, t["tp"], t["fp"], t["fn"]))
    print()
    base = agg[""]["tot"] / n
    for arm in ARMS[1:]:
        print("  %-5s 总分 %+.2f   引脚 F1 %+.4f" %
              (arm, agg[arm]["tot"] / n - base, (agg[arm]["pin"] - agg[""]["pin"]) / n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
