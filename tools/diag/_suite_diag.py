# -*- coding: utf-8 -*-
"""按套看某一臂的 per-case 诊断分布（pin/component 的 missing/extra/mismatch 计数）。

用法：
  python -X utf8 work\tools\diag\_suite_diag.py --json work\_ro220b_metrics.json
  python -X utf8 work\tools\diag\_suite_diag.py --json work\_ro220b_metrics.json --metric pin
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def suite(n: str) -> str:
    if re.match(r"^\d{1,2}$", n):
        return "10GT"
    if re.match(r"^\d{4}$", n):
        return "train200"
    return "10_GTcase"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--metric", default="", help="只看带这个前缀的诊断键（pin/component/…）")
    ap.add_argument("--show-f1", action="store_true")
    a = ap.parse_args()
    d = json.loads(Path(a.json).read_text(encoding="utf-8"))
    cs = d["cases"]
    agg = collections.defaultdict(collections.Counter)
    f1 = collections.defaultdict(list)
    ncase = collections.Counter()
    for c in cs:
        s = suite(str(c.get("case_id")))
        ncase[s] += 1
        for k, v in (c.get("diagnostic") or {}).items():
            if a.metric and not k.startswith(a.metric):
                continue
            if isinstance(v, int):
                agg[s][k] += v
        for key in ("component_f1", "pin_f1", "net_hypergraph_f1", "net_line_f1"):
            if c.get(key) is not None:
                f1[(s, key)].append(c[key])
    print("case 数:", dict(ncase))
    for s in ("10GT", "10_GTcase", "train200"):
        if not agg.get(s):
            continue
        print("[%s] %s" % (s, dict(agg[s].most_common())))
    if a.show_f1:
        for (s, k), vs in sorted(f1.items()):
            print("  %-10s %-18s mean=%.4f n=%d" % (s, k, sum(vs) / len(vs), len(vs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
