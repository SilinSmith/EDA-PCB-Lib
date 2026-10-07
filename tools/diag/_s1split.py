# -*- coding: utf-8 -*-
"""在 stage1 产物上「就地」重做 T 形分叉切段，并同步重建 pins / nets。

目的：让 junction 切段的 A/B 变成**干净的** —— 检测结果、文字框、元件表全部沿用同一份
stage1 产物，只有线段与由它派生的 pins/nets 变。直接重跑 stage1 是不行的：YOLO/CV 有
逐次抖动（同一张 `0001` 两次跑出 91 vs 92 个元件、前两个元件的顺序还会互换），
做出来的差值是运行噪声而不是切段的贡献。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\tools\\diag\\_s1split.py --in work\\stage1_v6 --out work\\_s1js
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from stage1 import wires as wires_mod          # noqa: E402
from stage1 import nets as nets_mod            # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--tol", type=float, default=1.5)
    a = ap.parse_args(argv)

    src = Path(a.indir)
    if not src.is_absolute():
        src = ROOT / src
    dst = Path(a.outdir)
    if not dst.is_absolute():
        dst = ROOT / dst
    dst.mkdir(parents=True, exist_ok=True)

    n_case = n_seg0 = n_seg1 = n_split_cases = 0
    for f in sorted(src.glob("*.stage1.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        wd = d.get("wire_detail") or {}
        segs0 = wd.get("segments") or []
        segs1, nsplit = wires_mod._split_at_nodes(
            [[list(map(float, s[0])), list(map(float, s[1]))] for s in segs0], tol=a.tol)
        endpoints, junctions = wires_mod._rebuild_nodes(segs1)

        comps = d.get("components") or []
        texts = [tuple(float(v) for v in t["bbox_img"]) for t in (d.get("texts") or [])]
        comp_pins = nets_mod.build_pins(comps, segs1, texts=texts)
        nets, seg_groups = nets_mod.build_nets(comp_pins, segs1)

        d["pins"] = {cid: [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pl]
                     for cid, pl in comp_pins.items()}
        d["nets"] = nets
        wd["segments"] = segs1
        wd["endpoints"] = endpoints
        wd["junctions"] = junctions
        wd["groups"] = seg_groups
        d["wire_detail"] = wd
        w = d.get("wires") or {}
        w["n_segments"] = len(segs1)
        w["n_segments_split"] = nsplit
        w["n_endpoints"] = len(endpoints)
        w["n_junctions"] = len(junctions)
        d["wires"] = w
        st = d.get("stats") or {}
        st["n_pins"] = sum(len(v) for v in comp_pins.values())
        st["n_nets"] = len(nets)
        st["n_pins_in_nets"] = sum(len(n["pins"]) for n in nets)
        d["stats"] = st

        n_case += 1
        n_seg0 += len(segs0)
        n_seg1 += len(segs1)
        if nsplit:
            n_split_cases += 1
        (dst / f.name).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    print(f"cases {n_case}  segments {n_seg0} -> {n_seg1}  切过的图 {n_split_cases}")
    print(f"产物 -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
