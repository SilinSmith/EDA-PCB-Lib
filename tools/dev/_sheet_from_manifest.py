# -*- coding: utf-8 -*-
"""从已落盘的文字语料 manifest 里挑一批出来拼对照图（核标签真假 / 看还没修好的）。

用法：
  # 看「我们原来读得跟 GT 差很远」的那批（最有可能是错配，也最该修）
  python -X utf8 work\tools\dev\_sheet_from_manifest.py --manifest work\official_text\manifest.jsonl ^
      --min-dist 3 --n 24 --out work\_sheet_far

  # 只看 value
  python -X utf8 work\tools\dev\_sheet_from_manifest.py --role value --n 24 --out work\_sheet_val
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(WORK / "tools" / "dev"))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import mk_official_textset as M       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(WORK / "official_text" / "manifest.jsonl"))
    ap.add_argument("--role", default="")
    ap.add_argument("--min-dist", type=int, default=0)
    ap.add_argument("--max-dist", type=int, default=99)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=str(WORK / "_sheet_probe"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rows = []
    for ln in Path(a.manifest).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        d = int(r.get("dist", M.edit_dist(r.get("was") or "", r["text"], 9)))
        if d < a.min_dist or d > a.max_dist:
            continue
        if a.role and r.get("role") != a.role:
            continue
        r["dist"] = d
        rows.append(r)
    print("候选 %d 条；按 role %s" % (len(rows), dict(Counter(r["role"] for r in rows))))
    random.seed(a.seed)
    random.shuffle(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    M.make_sheet(rows[:a.n], out, a.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
