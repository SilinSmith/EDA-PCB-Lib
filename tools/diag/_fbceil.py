# -*- coding: utf-8 -*-
"""给「兜底键」那 2601 个元件量一个上限：它们的 GT 位号，在图上**附近**读得出来吗？

对每个「检测对/类型对/位置对、但键是 pNNN」的元件，把它 bbox 外扩 `--near` 像素内的
stage1 文字框全部过一遍 CRNN（designator 角色），看 GT 的键有没有出现在这些读法里。

* 出现  -> 「文字路线」走得通：VLM 补读（或更好的归属）有可能捞回来
* 不出现 -> 位号压根没印在符号旁边（或印成了别的字），这部分是数据本身的天花板

    python work\\tools\\diag\\_fbceil.py --pred work\\_crnntrunc\\final_results --limit 220
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import schematic_parser.metrics as M                      # noqa: E402
import stage2_eval as SE                                  # noqa: E402
from stage2.text import TextEngine, imread_unicode        # noqa: E402

DES = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")


def rdist(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--near", type=float, default=80.0)
    ap.add_argument("--by-type", action="store_true")
    a = ap.parse_args()

    root = Path(a.pred)
    s1dir = root.parent / "runs" / "stage1"
    ex, di = SE.build_index(SE.DEFAULT_DATA_ROOT)
    eng = TextEngine()

    files = sorted(root.glob("*_Pred.json"))
    if a.limit:
        files = files[: a.limit]

    bucket = Counter()
    per_type = {}
    n = 0
    for f in files:
        cid = f.stem.replace("_Pred", "")
        s1f = s1dir / (cid + ".stage1.json")
        if not s1f.is_file():
            continue
        try:
            c, _ = SE.resolve(cid, ex, di)
        except Exception:
            continue
        g = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        p = json.loads(f.read_text(encoding="utf-8"))
        gc, pc = g.get("components") or {}, p.get("components") or {}
        if not gc or not pc:
            continue
        _f1, km = M.assign_components(g, p, M.EvalConfig(), {})
        used_g = set(km.values())
        ug = [k for k in gc if k not in used_g and not str(k).startswith("\u2205")]
        if not ug:
            continue
        s1 = json.loads(s1f.read_text(encoding="utf-8"))
        texts = s1.get("texts") or []
        if not texts:
            continue
        bgr = imread_unicode(c.png_path)
        H = float(bgr.shape[0])
        # GT 的 bbox 是左下原点；stage1 文字框的 bbox_img 是图像坐标
        for gk in ug:
            gv = gc[gk]
            gb = gv.get("bbox")
            if not gb:
                continue
            gib = [gb[0], H - gb[3], gb[2], H - gb[1]]
            near = [(rdist(gib, t["bbox_img"]), t) for t in texts]
            near = [x for x in near if x[0] <= a.near]
            if not near:
                bucket["C_附近根本没有文字框"] += 1
                per_type.setdefault(M.canon_type(gv.get("type")), Counter())[
                    "C_附近根本没有文字框"] += 1
                n += 1
                continue
            out = eng.read(bgr, [tuple(float(v) for v in t["bbox_img"]) for _d, t in near],
                           ["designator"])
            got = set()
            for rr in out:
                txt = (rr.get("reads") or {}).get("designator", ("", 0.0))[0]
                if txt and DES.match(txt):
                    got.add(txt.upper())
            n += 1
            key = str(gk).upper()
            if key in got:
                b = "A_附近读得出这个键"
            elif got:
                b = "B_附近有位号形状的读法，但没有这个键"
            else:
                b = "D_附近读不出任何位号形状"
            bucket[b] += 1
            per_type.setdefault(M.canon_type(gv.get("type")), Counter())[b] += 1

    print("扫了 %d 张，带真名但没配上的 GT 元件 %d 个（附近半径 %gpx）" % (len(files), n, a.near))
    for k, v in bucket.most_common():
        print("  %-34s %6d  %5.1f%%" % (k, v, 100.0 * v / max(1, n)))
    if a.by_type:
        print()
        for t, cnt in sorted(per_type.items(), key=lambda kv: -sum(kv[1].values()))[:14]:
            print("  %-12s n=%-5d %s" % (t, sum(cnt.values()),
                                         "  ".join("%s=%d" % (k.split("_")[0], v) for k, v in cnt.most_common())))


if __name__ == "__main__":
    main()
