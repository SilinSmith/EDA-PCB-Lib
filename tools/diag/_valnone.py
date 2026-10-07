# -*- coding: utf-8 -*-
"""「GT 有值、我们写 None」的 351 条：到底是**没读到**还是**读到了被丢**？

这是当前最大的一桶失配（`_textmiss.py` on `stage2_v15`：V 我们没写值(None) **351**）。
分两问：
  A. `_assigned/<case>.json` 里这颗元件的 `value` 有没有 `text`？
     - 有 text 但最终输出是 null ⇒ **是 `assemble.py` 的 `VALUE_LIKE` 门控丢的**（后处理可修）
     - 没 text ⇒ 值这个角色压根没被选中（`fit_value` 全 0 或框没进候选）
  B. 被丢的那些 text 长什么样。

用法：python work\tools\diag\_valnone.py --pred work\stage2_v15
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, VALUE_TYPES, canon_type  # noqa: E402
from stage2_eval import build_index, resolve  # noqa: E402

CFG = EvalConfig()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "work" / "stage2_v15"))
    ap.add_argument("--samples", type=int, default=25)
    a = ap.parse_args()
    pred_dir = Path(a.pred)
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    cat = Counter()
    texts = Counter()
    samples = []
    for f in sorted(pred_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        case, _ = resolve(f.stem, exact, digits)
        if case is None:
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gc, pc = gt.get("components") or {}, pred.get("components") or {}
        # cid -> canon 位号，用来回查 _assigned
        ap_path = pred_dir / "_assigned" / f.name
        assigned = (json.loads(ap_path.read_text(encoding="utf-8"))
                    if ap_path.exists() else {})
        by_canon = {}
        for cid, rec in (assigned.get("comps") or {}).items():
            d = rec.get("designator") or {}
            c = d.get("canon")
            if c:
                by_canon.setdefault(c, cid)

        for k in gc:
            if k not in pc:
                continue
            g, p = gc[k], pc[k]
            if canon_type(g.get("type")) not in VALUE_TYPES:
                continue
            gv = g.get("value")
            if not gv:
                continue                       # GT 本来就空，不是这桶
            if p.get("value"):
                continue                       # 我们写了东西，也不是这桶
            cat["n"] += 1
            cid = by_canon.get(k)
            rec = (assigned.get("comps") or {}).get(cid, {}) if cid else {}
            val = rec.get("value") or {}
            t = val.get("text")
            if t:
                cat["A_读到了但被丢"] += 1
                texts[t] += 1
            elif cid:
                cat["B_这个角色没选中"] += 1
            else:
                cat["C_连元件都没对上"] += 1
            if len(samples) < a.samples:
                samples.append((f.stem, k, rec.get("type") or g.get("type"),
                                gv, t, val.get("canon"), rec.get("designator", {}).get("canon")))

    print("「GT 有值 / 我们 None」共 %d 条" % cat["n"])
    for k, v in cat.most_common():
        if k != "n":
            print("   %-18s %d  (%.1f%%)" % (k, v, 100.0 * v / max(1, cat["n"])))
    print("\n被丢掉的 text Top20（`VALUE_LIKE` 要带字母/Ω/μ，纯数字会被丢成 null）：")
    for t, n in texts.most_common(20):
        print("   %-14s x%d" % (t, n))
    print("\n样例  case / 键 / 类型 / GT值 / 我们读到的text / canon / 我们输出的键")
    for s in samples:
        print("   %-12s %-6s %-10s %-14s %-14s %-10s %s" % s)


if __name__ == "__main__":
    main()
