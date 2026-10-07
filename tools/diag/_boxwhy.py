# -*- coding: utf-8 -*-
"""box(IC) 位号为什么挂不上 —— 官方口径下唯一的 pinF1 来源。

只回答一个问题：GT 里 type=box 的元件，我们的预测里有没有同名的键？
没有的话，是我们压根没检出这个框、检出了但没读出字、还是读出了但读错了？

用法：
    python work\tools\diag\_boxwhy.py [--pred work\stage2_v12] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from stage2_eval import build_index, resolve  # noqa: E402


def canon(t):
    return {"ic": "box", "boxic": "box"}.get((t or "").strip(), (t or "").strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "work" / "stage2_v12"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=25)
    a = ap.parse_args()

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    files = [f for f in sorted(Path(a.pred).glob("*.json")) if not f.name.startswith("_")]
    if a.limit:
        files = files[:a.limit]

    cnt = Counter()
    keys_missing = Counter()      # GT 键没出现在预测里
    near = []                     # (gt_key, 最近的预测键, 相似度, case)
    dims = []                     # box 的 GT 框尺寸
    for f in files:
        case, _ = resolve(f.stem, exact, digits)
        if case is None:
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gc, pc = gt.get("components") or {}, pred.get("components") or {}
        gbox = {k: v for k, v in gc.items() if canon((v or {}).get("type")) == "box"}
        pkeys = list(pc)
        for k, v in gbox.items():
            cnt["gt_box"] += 1
            b = (v or {}).get("bbox") or [0, 0, 0, 0]
            dims.append((abs(b[2] - b[0]), abs(b[3] - b[1])))
            if k in pc:
                cnt["key_hit"] += 1
                if canon((pc[k] or {}).get("type")) == "box":
                    cnt["type_ok"] += 1
                continue
            cnt["key_miss"] += 1
            keys_missing[k] += 1
            best, bs = "", 0.0
            for pk in pkeys:
                s = SequenceMatcher(None, k, pk).ratio()
                if s > bs:
                    best, bs = pk, s
            if bs >= 0.5:
                near.append((f.stem, k, best, round(bs, 2)))
            else:
                near.append((f.stem, k, best, round(bs, 2)))
    print("配对 %d 张" % len(files))
    print("GT box 元件 %d；键命中 %d (%.3f)；其中 type 也对 %d (%.3f)"
          % (cnt["gt_box"], cnt["key_hit"], cnt["key_hit"] / max(1, cnt["gt_box"]),
             cnt["type_ok"], cnt["type_ok"] / max(1, cnt["gt_box"])))
    print("键没配上 %d (%.3f)"
          % (cnt["key_miss"], cnt["key_miss"] / max(1, cnt["gt_box"])))
    if dims:
        ws = sorted(d[0] for d in dims)
        hs = sorted(d[1] for d in dims)
        print("GT box 框尺寸：w 中位 %.0f  p10 %.0f p90 %.0f ；h 中位 %.0f p10 %.0f p90 %.0f"
              % (ws[len(ws) // 2], ws[len(ws) // 10], ws[len(ws) * 9 // 10],
                 hs[len(hs) // 2], hs[len(hs) // 10], hs[len(hs) * 9 // 10]))
    print("最常没配上的 GT 键:", keys_missing.most_common(20))
    print("\n键没配上的样例（GT键 -> 预测里最像的键, 相似度）：")
    for row in near[:a.samples]:
        print("   %-8s %-24s -> %-24s %s" % row)


if __name__ == "__main__":
    main()
