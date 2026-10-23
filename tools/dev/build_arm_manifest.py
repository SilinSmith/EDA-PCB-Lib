# -*- coding: utf-8 -*-
"""Assemble a fine-tune manifest from three sources: synthetic, real, new-domain.

The delivery weight was produced by `--synth 120000` + the real 300-case corpus
+ the harvested new-dataset crops (see docs/RESULTS.md §5.357). Keeping the
synthetic share large is what stops catastrophic forgetting of the old domain
(§5.356.2); the new-domain share is what buys the gain.

  cd eda-pcb
  python -X utf8 tools\\dev\\build_arm_manifest.py --synth 120000 \
      --real work\\_real_only.jsonl --new work\\_man_new_ocr.jsonl \
      --out work\\_man_mix.jsonl

All inputs are JSONL with {"image","case","src","split","role","role_id","text","bbox"}.
Sampling is deterministic (seeded) so the same command gives the same manifest.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path: Path, drop_val: bool = True) -> list:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if drop_val and r.get("split") == "val":
                continue
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=str(Path("work/textrec_merged.jsonl")),
                    help="合成语料 jsonl（取其中 src==synth 的行）")
    ap.add_argument("--synth-n", type=int, default=120000, help="合成行上限（按类均匀抽）")
    ap.add_argument("--real", default="work/_real_only.jsonl", help="真实 300 案语料")
    ap.add_argument("--new", default="work/_man_new_ocr.jsonl", help="新集收字")
    ap.add_argument("--new-repeat", type=int, default=7,
                    help="新集过采样倍数（交付配方是 7：8665 条 → 6.2 万条）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20250)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    parts = []
    synth = [r for r in load(Path(a.synth)) if r.get("src") == "synth"]
    if a.synth_n and len(synth) > a.synth_n:
        by_role = {}
        for r in synth:
            by_role.setdefault(r.get("role"), []).append(r)
        quota = max(1, a.synth_n // max(1, len(by_role)))
        picked = []
        for role, rows in by_role.items():
            rnd.shuffle(rows)
            picked.extend(rows[:min(quota, len(rows))])
        rnd.shuffle(synth)
        seen = {id(r) for r in picked}
        if len(picked) < a.synth_n:
            picked.extend([r for r in synth if id(r) not in seen][:a.synth_n - len(picked)])
        synth = picked
    parts.extend(synth)
    for src in (a.real, a.new):
        p = Path(src)
        if p.exists():
            rows = load(p)
            if src == a.new and a.new_repeat > 1:
                rows = rows * a.new_repeat
            parts.extend(rows)
        else:
            print("[warn] missing", p)
    rnd.shuffle(parts)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in parts:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in parts:
        counts[r.get("src", "?")] = counts.get(r.get("src", "?"), 0) + 1
    print("[out] %s  n=%d  %s" % (out, len(parts), counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
