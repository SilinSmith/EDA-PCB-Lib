# -*- coding: utf-8 -*-
"""**位号挂载体检**：把 GT 元件与产物元件按 JSON 坐标 IoU 配对，统计
  ① 键对（含大小写无关）② 我们是伪键 `pNNN`（=没挂上位号）③ 挂了别的位号 ④ 位置没配上。

用法：
    python -X utf8 work\tools\dev\_desacc.py --dir work\_ro220b --limit 60
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
DR = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
SPLITS = ("200_train_cases", "10_GTcase", "10GT")
PSEUDO = re.compile(r"^(?:∅\d+|p\d{3,4})$")


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def find_gt(case: str):
    for sp in SPLITS:
        d = DR / sp / case
        if not d.is_dir():
            continue
        js = sorted(d.glob("%s*.json" % case))
        js = [p for p in js if "target_named" in p.name] or js
        for p in js:
            try:
                g = json.loads(p.read_text(encoding="utf-8"))
            except Exception:                                        # noqa: BLE001
                continue
            if g.get("components"):
                return p, g
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(WORK / "_ro220b"))
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条键错的例子")
    a = ap.parse_args()

    raw = Path(a.cases_file).read_text(encoding="utf-8")
    cases, seen = [], set()
    for tok in re.split(r"[,\s]+", raw):
        if not re.match(r"^[0-9A-Za-z_]+$", tok or "") or tok in seen:
            continue
        seen.add(tok)
        cases.append(tok)
    cases = cases[: a.limit]

    cnt = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    wrong_pairs = []
    for c in cases:
        p, gt = find_gt(c)
        fp = Path(a.dir) / ("%s.json" % c)
        if p is None:
            cnt["无 GT 文件"] += 1
            continue
        if not fp.is_file():
            cnt["无产物"] += 1
            continue
        pr = json.loads(fp.read_text(encoding="utf-8")).get("components") or {}
        used = set()
        for gk, gv in (gt.get("components") or {}).items():
            gb = gv.get("bbox")
            if not gb:
                continue
            best, bi = 0.0, None
            for pk, pv in pr.items():
                if pk in used:
                    continue
                v = iou(gb, pv.get("bbox") or [0, 0, 0, 0])
                if v > best:
                    best, bi = v, pk
            ty = str(gv.get("type") or "?")
            if bi is None or best < a.min_iou:
                cnt["位置没配上"] += 1
                by_type[ty]["位置没配上"] += 1
                continue
            used.add(bi)
            if gk.upper() == str(bi).upper():
                cnt["键对"] += 1
                by_type[ty]["键对"] += 1
            elif PSEUDO.match(str(bi)):
                cnt["伪键（没挂位号）"] += 1
                by_type[ty]["伪键"] += 1
            else:
                cnt["挂了别的位号"] += 1
                by_type[ty]["挂错"] += 1
                if len(wrong_pairs) < a.show:
                    wrong_pairs.append((c, gk, bi, [round(float(x)) for x in gb]))

    tot = sum(cnt.values())
    print("== %s  (%d 张 case，配对 %d 个 GT 元件) ==" % (a.dir, len(cases), tot))
    for k, v in cnt.most_common():
        print("  %-18s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, tot)))
    print("-- 按类型 --")
    for ty, cc in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        s = sum(cc.values())
        print("  %-10s n=%-5d 键对 %.0f%%  伪键 %.0f%%  挂错 %.0f%%  没配上 %.0f%%"
              % (ty, s, 100.0 * cc["键对"] / s, 100.0 * cc["伪键"] / s,
                 100.0 * cc["挂错"] / s, 100.0 * cc["位置没配上"] / s))
    for w in wrong_pairs:
        print("   例: case=%s GT=%-7s ours=%-7s bbox_json=%s" % w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
