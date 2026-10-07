# -*- coding: utf-8 -*-
"""GT 的 box(IC) 元件为什么还是配不上键 —— 把 636 个 miss 逐条归因。

回答的问题：VLM 位号补强只救回了 54 个（0.361 -> 0.416），剩下的卡在哪？
分类（按与 GT 框 IoU>=0.5 的那个预测元件的状态）：

  A 没检出        —— 预测里没有任何 IoU>=0.5 的元件
  B 类型不在 IC   —— 找到了，但我们的 type 不是 box/ic/boxic，VLM 压根没被问
  C 仍是兜底键    —— type 对，但键还是 pNNN（VLM 没读出来 / 被冲突消解挡掉）
  D 读成了别的键  —— 键是别的合法位号（VLM 读错，或读到邻居）
  E 键被别人占了  —— 我们的预测里出现了这个 GT 键，但落在另一个框上

用法：
    python work\tools\diag\_desigwhy.py --pred work\_submit_out220v\runs\stage3_vlm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402
from stage2_eval import build_index, resolve  # noqa: E402

FKEY = re.compile(r"^p\d{3,4}$")
DESIG = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "work" / "_submit_out220v" / "runs" / "stage3_vlm"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--min-iou", type=float, default=0.5)
    a = ap.parse_args()

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    files = [f for f in sorted(Path(a.pred).glob("*.json")) if not f.name.startswith("_")]
    if a.limit:
        files = files[:a.limit]

    cnt = Counter()
    types_seen = Counter()
    samples = []
    for f in files:
        out = resolve(f.stem, exact, digits)
        if not out:
            cnt["0_图配不上"] += 1
            continue
        case, _how = out if isinstance(out, tuple) else (out, "?")
        gt = load_json(case.json_path)
        pred = json.loads(Path(f).read_text(encoding="utf-8"))
        gcomp = {k: v for k, v in (gt.get("components") or {}).items()
                 if str(v.get("type") or "").lower() in ("box", "ic", "boxic")}
        pcomp = pred.get("components") or {}
        pkeys = set(pcomp)
        pboxes = [(k, v.get("bbox")) for k, v in pcomp.items() if v.get("bbox")]
        for gk, gv in gcomp.items():
            if gk in pkeys:
                cnt["ok"] += 1
                continue
            if gk in pboxes and False:
                pass
            gb = gv.get("bbox")
            if not gb:
                cnt["A_没检出"] += 1
                continue
            best, bi = None, 0.0
            for pk, pb in pboxes:
                v = iou(gb, pb)
                if v > bi:
                    best, bi = pk, v
            if best is None or bi < a.min_iou:
                cnt["A_没检出"] += 1
                if len(samples) < a.samples:
                    samples.append("%s GT %s -> 无配对 (best %.2f)" % (f.stem, gk, bi))
                continue
            pc = pcomp[best]
            ty = str(pc.get("type") or "").lower()
            types_seen[ty] += 1
            if ty not in ("box", "ic", "boxic"):
                cnt["B_类型不在IC"] += 1
                if len(samples) < a.samples:
                    samples.append("%s GT %s -> %s type=%s (VLM 没被问)" % (f.stem, gk, best, ty))
                continue
            if FKEY.match(best):
                cnt["C_仍是兜底键"] += 1
                if len(samples) < a.samples:
                    samples.append("%s GT %s -> %s 兜底键未改名" % (f.stem, gk, best))
                continue
            if DESIG.match(best):
                cnt["D_读成了别的键"] += 1
                if len(samples) < a.samples:
                    samples.append("%s GT %s -> %s (读错/读到邻居)" % (f.stem, gk, best))
                continue
            cnt["F_其它键"] += 1
            if len(samples) < a.samples:
                samples.append("%s GT %s -> %s (type=%s)" % (f.stem, gk, best, ty))

    tot = sum(v for k, v in cnt.items() if k != "0_图配不上")
    print("配不上的 20 张跳过：%d" % cnt.get("0_图配不上", 0))
    print("GT box 元件 %d" % tot)
    for k in ("ok", "A_没检出", "B_类型不在IC", "C_仍是兜底键", "D_读成了别的键", "F_其它键"):
        v = cnt.get(k, 0)
        print("  %-14s %5d  %.3f" % (k, v, v / max(1, tot)))
    print("\n配上了 IoU>=%.2f 的预测元件的 type 分布：" % a.min_iou)
    for t, v in types_seen.most_common(14):
        print("  %-12s %d" % (t, v))
    if samples:
        print("\n样例：")
        for s in samples[:a.samples]:
            print("  " + s)


if __name__ == "__main__":
    main()
