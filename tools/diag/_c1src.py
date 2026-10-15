"""`_assigned/<case>.json` 的**出处归因**：每个字段到底是哪一层定的。

用途：位号/值/引脚号出错时，**先看 `_by`，再看分数**。
融合层的 `solve` 会给每个选中的值写下 `_by`（如 `solve:prior:run_altwindow`、
`solve:vlm:enrich`、`solve:ocr:designator`），而 `run_altwindow` 这种名字只在
`fusion/priors.py` 里出现 ⇒ 一眼就能定位到「哪一层编出来的」。

    python -X utf8 tools\diag\_c1src.py                       # 默认 work\_f9_B_pin, case 1, designator
    python -X utf8 tools\diag\_c1src.py --dir work\_full220\runs\fusion_nollm --cases 1,2
    python -X utf8 tools\diag\_c1src.py --field value --show all
    python -X utf8 tools\diag\_c1src.py --field pin --case 1

`--field designator|value|pin`：`pin` 展开每根引脚的 number/name（含 `pinrun` 这类先验补号）。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _load(d: Path, case: str):
    for name in (f"{case}.json", f"{case}_assigned.json"):
        p = d / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _rows(doc, field: str, comp_id: str, comp: dict):
    if field == "designator":
        v = comp.get("designator") or {}
        yield comp_id, comp.get("type") or comp.get("cls") or "", str(v.get("text")), str(v.get("canon")), v.get("_by"), v.get("conf")
    elif field == "value":
        v = comp.get("value") or {}
        yield comp_id, comp.get("type") or comp.get("cls") or "", str(v.get("text")), "", v.get("_by"), v.get("conf")
    else:
        for slot, pin in (comp.get("pins") or {}).items():
            for sub in ("number", "name"):
                v = (pin or {}).get(sub) or {}
                if not v:
                    continue
                yield (f"{comp_id}.{slot}", comp.get("type") or "", str(v.get("text")),
                       sub, v.get("_by"), v.get("conf"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"work\_f9_B_pin", help="含 _assigned 的产物目录（交付产物根目录）")
    ap.add_argument("--assigned", default="", help="直接指定 _assigned 目录（默认 <dir>\\_assigned）")
    ap.add_argument("--cases", default="1")
    ap.add_argument("--field", choices=("designator", "value", "pin"), default="designator")
    ap.add_argument("--show", choices=("changed", "all"), default="all",
                    help="changed = 只打印 text 与 canon 不同的（先验改过读数的）")
    args = ap.parse_args(argv)

    root = Path(args.assigned) if args.assigned else Path(args.dir) / "_assigned"
    if not root.is_dir():
        print(f"没有 _assigned 目录：{root}")
        return 2
    tally = collections.Counter()
    for case in [c.strip() for c in args.cases.split(",") if c.strip()]:
        doc = _load(root, case)
        if doc is None:
            print(f"[case {case}] 找不到 {root}\\{case}.json")
            continue
        print(f"\n=== case {case}  field={args.field}  {root} ===")
        n = 0
        for cid in sorted(doc.get("comps") or {}):
            for row in _rows(doc, args.field, cid, doc["comps"][cid]):
                cid2, ctype, text, canon, by, conf = row
                if args.show == "changed" and text == canon:
                    continue
                arrow = f"{text or '∅'} -> {canon}" if canon else f"{text or '∅'}"
                print(f"  {cid2:>8s} {ctype:<12s} {arrow:<22s} by={by or '?':<26s} conf={conf if conf is None else round(float(conf), 3)}")
                tally[by or "?"] += 1
                n += 1
        print(f"  （{n} 条）")
    if tally:
        print("\n=== 出处统计 ===")
        for k, v in tally.most_common():
            print(f"  {v:6d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
