# -*- coding: utf-8 -*-
"""把两个「220 张融合臂」目录按官方口径打分，并按 suite 拆分（官方 20 / 派生 200）。

为什么要它：`_rescore.py` 只给**总均分**，而 220 张里 200 张是派生集、20 张是官方集，
平均分会把方向相反的档抵消掉。本工具输出 `stage2_eval` 兼容的 per-case JSON（可直接喂
`work\\tools\\diag\\_split220.py`），并自己打一张按 suite 的表。

用法：
    python -X utf8 work\\tools\\diag\\_ro220score.py --a work\\_ro220b --b work\\_ro220o
    # --a = 新默认（不设 FUSION_RUN_* 变量），--b = 旧默认（ALTWINDOW=1 ORDER=0）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA\eda-pcb")
WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(WORK / "tools" / "eval"))

from schematic_parser.data import load_json                       # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case    # noqa: E402
import stage2_eval as SE                                          # noqa: E402

W = {"component_f1": 0.30, "pin_f1": 0.25, "net_hypergraph_f1": 0.35, "net_line_f1": 0.10}


def suite(n: str) -> str:
    if re.match(r"^\d{1,2}$", n):
        return "10GT"
    if re.match(r"^\d{4}$", n):
        return "train200"
    return "10_GTcase"


def score_dir(d: Path, cfg: EvalConfig):
    idx, digits = SE.build_index(SE.DEFAULT_DATA_ROOT)
    out, miss = {}, []
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_") or "." in f.stem:
            continue
        case, _how = SE.resolve(f.stem, idx, digits)
        if case is None:
            miss.append(f.stem)
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        s = evaluate_case(gt, pred, f.stem, cfg)
        out[str(f.stem)] = s.as_dict()
    return out, miss


def wsum(c: dict) -> float:
    return 100 * sum(W[k] * float(c.get(k) or 0) for k in W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="臂 A 目录（新默认）")
    ap.add_argument("--b", required=True, help="臂 B 目录（旧默认）")
    ap.add_argument("--la", default="new")
    ap.add_argument("--lb", default="old")
    ap.add_argument("--json-a", default=None)
    ap.add_argument("--json-b", default=None)
    a = ap.parse_args()

    cfg = EvalConfig(net_pin_collapse_nonbox=True)      # 新口径（当前默认）
    A, ma = score_dir(Path(a.a), cfg)
    B, mb = score_dir(Path(a.b), cfg)
    for tag, m in ((a.la, ma), (a.lb, mb)):
        if m:
            print("[warn] %s 配不上的 %d 个: %s" % (tag, len(m), m[:6]))

    for path, arm in ((a.json_a, A), (a.json_b, B)):
        if path:
            Path(path).write_text(json.dumps(
                {"config": vars(cfg), "cases": list(arm.values())},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print("写到 %s" % path)

    common = [k for k in A if k in B]
    print("\n两臂交集 %d 张（A=%d B=%d）" % (len(common), len(A), len(B)))
    groups: dict[str, list[str]] = {}
    for n in common:
        groups.setdefault(suite(n), []).append(n)

    keys = ("component_f1", "pin_f1", "net_hypergraph_f1", "net_line_f1")
    print("\n%-11s %4s %8s %8s %8s | %8s %8s | %8s %8s | %8s %8s | %8s %8s"
          % ("suite", "n", a.la, a.lb, "Δ(A-B)",
             "comp A", "comp B", "pin A", "pin B", "netH A", "netH B", "netL A", "netL B"))
    print("-" * 132)
    for g in ("10GT", "10_GTcase", "train200"):
        ns = groups.get(g, [])
        if not ns:
            continue
        row = [sum(wsum(A[n]) for n in ns) / len(ns), sum(wsum(B[n]) for n in ns) / len(ns)]
        cells = []
        for k in keys:
            cells.append(sum(float(A[n].get(k) or 0) for n in ns) / len(ns))
            cells.append(sum(float(B[n].get(k) or 0) for n in ns) / len(ns))
        print("%-11s %4d %8.4f %8.4f %+8.4f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f"
              % (g, len(ns), row[0], row[1], row[0] - row[1], *cells))
    for g, ns in (("官方 20", groups.get("10GT", []) + groups.get("10_GTcase", [])),
                  ("全部 220", common)):
        if not ns:
            continue
        va = sum(wsum(A[n]) for n in ns) / len(ns)
        vb = sum(wsum(B[n]) for n in ns) / len(ns)
        print("%-11s %4d %8.4f %8.4f %+8.4f" % (g, len(ns), va, vb, va - vb))

    # Δ = A − B（正 = A 更好）；两臂是「新默认 vs 旧默认」时，正 = 翻转有收益
    d = sorted(((wsum(A[n]) - wsum(B[n]), n) for n in common), reverse=True)
    up = [x for x in d if x[0] > 1e-9]
    dn = [x for x in d if x[0] < -1e-9]
    print("\n逐张：A 更好 %d / A 更差 %d / 不变 %d" % (len(up), len(dn), len(common) - len(up) - len(dn)))
    print("  A 更好最多: %s" % ", ".join("%s %+.2f" % (n, v) for v, n in d[:6]))
    print("  A 更差最多: %s" % ", ".join("%s %+.2f" % (n, v) for v, n in d[-6:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
