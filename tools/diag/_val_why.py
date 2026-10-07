# -*- coding: utf-8 -*-
"""官方的 `component.text_mismatch` 到底是什么？—— 按 GT 类型拆开「文字错」。

    & 'K:\\python\\311\\python.exe' Y:\\workspace\\EDA\\work\\tools\\diag\\_val_why.py --pred work\\stage2_v7

`metrics.py:149-157` 的 `component_text_ok` 只对两类做逐字符比较：

    IC_TYPES    = {box, ic, boxic}          -> 比 `Name`（器件型号）
    VALUE_TYPES = {r, resistor, c, capacitor} -> 比 `value`（阻容值）
    其余类型一律算对

所以 `text_mismatch` 只可能来自 `box` 的**器件名**和 `r`/`c` 的**值**。本脚本把
「键已经对上了、只是文字不等」的那些配对数按类型拆开，并给出：

* 每类的（GT 期望, 我们写的）Top 对；
* 我们的值属于哪一档：`empty`（null/空）/ `norm_eq`（只差大小写与分隔符）/
  `subst`（只差 1~2 个字符，明显是同一个串读错）/ `other`（完全不同 —— 很可能
  是**选错了文本框**，比如把 `0603` 当成阻值）；
* 与 GT 期望串的编辑距离分布。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "viz"))

import comp_cases as CC                                            # noqa: E402
from PIL import Image                                              # noqa: E402
from schematic_parser.data import IC_TYPES, VALUE_TYPES             # noqa: E402

NORM = re.compile(r"[^0-9A-Za-z]")


def norm(s) -> str:
    return NORM.sub("", str(s or "")).upper()


def edit(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def kind(exp, got) -> str:
    e, g = norm(exp), norm(got)
    if g == "":
        return "empty"
    if e == g:
        return "norm_eq"
    d = edit(e, g)
    if d <= max(1, len(e) // 3):
        return "subst"
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v7"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args(argv)

    pred_dir = Path(args.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / args.pred
    idx = CC.gt_index(CC.DEFAULT_DATA_ROOT)

    files = sorted(pred_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if args.limit:
        files = files[:args.limit]

    n_pair = 0
    n_key_ok = 0
    by_type = Counter()
    by_kind = Counter()
    pairs = Counter()
    dist = Counter()
    done = 0

    for f in files:
        case = f.stem
        c = idx.get(case) or idx.get(re.sub(r"\D", "", case))
        if c is None or not c.json_path:
            continue
        gdoc = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pdoc = json.loads(f.read_text(encoding="utf-8"))
        try:
            W, H = Image.open(str(c.png_path)).size
        except Exception:
            continue
        g = CC.gt_components(gdoc, H)
        p = CC.pred_components(pdoc, H)
        prs, _ug, _up = CC.pair(g, p, 0.1)
        done += 1
        for i, j, _v in prs:
            gi, pj = g[i], p[j]
            n_pair += 1
            if str(gi["key"]) != str(pj["id"]):
                continue
            n_key_ok += 1
            ct = gi["type"]
            if ct in IC_TYPES:
                exp, got, role = gi["name"], pj["name"], "Name"
            elif ct in VALUE_TYPES:
                exp, got, role = gi["value"], pj["value"], "value"
            else:
                continue
            if (exp or "") == (got or ""):
                continue
            k = kind(exp, got)
            by_type["%s/%s" % (ct, role)] += 1
            by_kind[k] += 1
            pairs[(ct, role, str(exp), str(got))] += 1
            if k in ("subst",):
                dist[min(edit(norm(exp), norm(got)), 8)] += 1

    print("扫描 %d 张，配对上 %d 个元件，键相同 %d 个" % (done, n_pair, n_key_ok))
    print("\n== 文字不等，按 (GT类型/字段) ==")
    for k, v in by_type.most_common():
        print("  %-22s %5d" % (k, v))
    tot = sum(by_type.values())
    print("  合计 %d" % tot)
    print("\n== 我们的值属于哪一档 ==")
    for k, v in by_kind.most_common():
        print("  %-10s %5d  (%.1f%%)" % (k, v, 100.0 * v / max(1, tot)))
    print("\n== 编辑距离（subst 档，归一化后）==")
    for k in sorted(dist):
        print("  d=%d  %5d" % (k, dist[k]))
    print("\n== (GT期望 -> 我们写的) Top %d ==" % args.top)
    for (ct, role, exp, got), v in pairs.most_common(args.top):
        print("  %-14s %-6s %-22s -> %-22s  x%d  [%s]"
              % (ct, role, exp[:22], got[:22], v, kind(exp, got)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
