# -*- coding: utf-8 -*-
"""把「官方渲染图上的域内文字语料」（`mk_official_textset.py` 的产物）转成 `textrec` 能直接
训练的 manifest，并按需掺入合成语料、按 case 切 train/val。

为什么需要它
------------
`textrec` 的 manifest 每条要 `image/case/src/split/role/role_id/text/bbox`（见
`textrec/manifest.py` 顶部注释），而域内语料只有 `image/case/role/text/bbox/dist`。

切分纪律
--------
**按 case 切**，不按条切：同一个 case 的文字分布是相关的，按条切会让同 case 的裁片同时出现在
train 和 val 里，valCER 会虚低。默认把官方公开 case `1,5` 与每 10 个派生 case 留一个进 val。

用法
----
  python -X utf8 work\tools\dev\mk_textrec_manifest.py ^
      --official work\official_text\manifest.jsonl ^
      --out work\official_text\textrec_manifest.jsonl --repeat 6 --synth-train 40000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from textrec import config as C          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default=str(WORK / "official_text" / "manifest.jsonl"))
    ap.add_argument("--out", default=str(WORK / "official_text" / "textrec_manifest.jsonl"))
    ap.add_argument("--synth", default=str(C.MANIFEST))
    ap.add_argument("--synth-train", type=int, default=40000)
    ap.add_argument("--synth-val", type=int, default=2000)
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--font", default=str(WORK / "fonttext" / "manifest.jsonl"),
                    help="官方字体合成语料（`mk_fonttext.py` 产物）；缺省目录不存在则跳过")
    ap.add_argument("--font-train", type=int, default=180000)
    ap.add_argument("--font-val", type=int, default=4000)
    ap.add_argument("--repeat", type=int, default=6,
                    help="域内训练条重复几遍（域内只有几千条，合成有 36 万条）")
    ap.add_argument("--val-cases", default="1,5")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--max-dist", type=int, default=2,
                    help="丢掉「原读数与 GT 编辑距离 > 此值」的行——实测 dist≥3 那批大多是"
                         "**错配**（`GT='10uF' was='5V'` 这种），留下来等于灌噪声")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    val_fixed = {s.strip() for s in a.val_cases.split(",") if s.strip()}
    rows = []
    for ln in Path(a.official).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        role = r.get("role")
        if role not in C.ROLE_ID:
            continue
        d = int(r.get("dist", 99))
        if d > a.max_dist:
            continue
        rows.append({"image": r["image"], "case": r["case"], "src": "official",
                     "role": role, "role_id": C.ROLE_ID[role],
                     "text": r["text"], "bbox": r["bbox"], "dist": d,
                     "was": r.get("was")})
    # 按 case 切
    cases = sorted({r["case"] for r in rows})
    val_cases = set(val_fixed & set(cases))
    if a.val_every > 0:
        for i, c in enumerate(sorted(cases)):
            if re.match(r"^\d{4}$", c) and i % a.val_every == 0:
                val_cases.add(c)
    for r in rows:
        r["split"] = "val" if r["case"] in val_cases else "train"
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]

    if a.repeat > 1:
        tr = [dict(r) for r in tr for _ in range(a.repeat)]

    synth_tr, synth_va = [], []
    if not a.no_synth:
        rng = random.Random(a.seed)
        for ln in Path(a.synth).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") == "train":
                synth_tr.append(r)
            elif r.get("split") == "val":
                synth_va.append(r)
        rng.shuffle(synth_tr)
        rng.shuffle(synth_va)
        synth_tr = synth_tr[:a.synth_train]
        synth_va = synth_va[:a.synth_val]
        for r in synth_tr:
            r["split"] = "train"
        for r in synth_va:
            r["split"] = "val"

    font_tr, font_va = [], []
    fpath = Path(a.font)
    if fpath.is_file() and (a.font_train or a.font_val):
        for ln in fpath.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            r.pop("dist", None)
            if r.get("split") == "train":
                font_tr.append(r)
            else:
                font_va.append(r)
        rng2 = random.Random(a.seed + 17)
        rng2.shuffle(font_tr)
        rng2.shuffle(font_va)
        font_tr = font_tr[:a.font_train]
        font_va = font_va[:a.font_val]
        for r in font_tr:
            r["split"] = "train"
        for r in font_va:
            r["split"] = "val"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allr = tr + va + synth_tr + synth_va + font_tr + font_va
    with out.open("w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rep(tag, rs):
        c = Counter(x["role"] for x in rs)
        print("  %-16s n=%-7d %s" % (tag, len(rs), dict(c.most_common())))
    print("-> %s" % out)
    rep("official train", tr)
    rep("official val", va)
    rep("synth train", synth_tr)
    rep("synth val", synth_va)
    print("val cases:", sorted(val_cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
