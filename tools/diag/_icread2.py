# -*- coding: utf-8 -*-
"""`_icread.py` 的 220 张版：把「GT 那颗 IC 的位号，图上到底读不读得出来」
在**默认链**（`work\\_submit_out220r36`）上量一遍。

`_icread.py` 原来跑的是 `work\\_fast_out`（fast.py 的 19 张回归集），结论
「A_读得出这个键 2/38 = 0.053」不能直接搬到 220 张。本脚本只改三处：
  ① `--pred`   = `...\\runs\\stage2`（每张一个 `<case>.json`，不是 `_Pred.json`）
  ② `--stage1` = `...\\runs\\stage1`
  ③ 不传 `--cases` 就遍历 stage1 目录里的**全部** case（含裸名 `1`..`10`）

判据（与 `_icread.py` 逐字一致）：把 GT 元件框转到**图像坐标**（`H - y`），
取 `bbox_img` 空间里与它**间隙距离 ≤ pad** 的 stage1 文字框，交给
`TextEngine.read(bgr, boxes, ["designator"])`，看有没有哪个读出 GT 键。

    A 读得出这个键  -> 只是没被归属/没进池子（可救）
    B 读了别的位号  -> 归属错或识别错（部分可救）
    D 啥也没读出来  -> **检测/分割问题**（下游无解）

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_icread2.py
    python work\\tools\\diag\\_icread2.py --pad 60 --cases 0001,0003 --limit 40
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json      # noqa: E402
from schematic_parser.metrics import (IC_TYPES, EvalConfig, assign_components,
                                      canon_type, normalise_doc)    # noqa: E402
from stage2_eval import build_index, resolve as _resolve            # noqa: E402
from stage2.text import TextEngine, imread_unicode                  # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"


def _gap_dist(a, b):
    dx = max(a[0] - b[2], 0, b[0] - a[2])
    dy = max(a[1] - b[3], 0, b[1] - a[3])
    return (dx * dx + dy * dy) ** 0.5


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--stage1", default=str(RUN / "stage1"))
    ap.add_argument("--pad", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=16)
    a = ap.parse_args(argv)

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    if a.cases.strip():
        ids = [s.strip() for s in a.cases.split(",") if s.strip()]
        cases = []
        for cid in ids:
            c, _ = _resolve(cid, exact, digits)
            if c is None:
                c, _ = _resolve(cid.lstrip("0"), exact, digits)
            if c is not None and c.json_path:
                cases.append(c)
    else:
        cases = []
        for p in sorted(Path(a.stage1).glob("*.stage1.json")):
            stem = p.name.split(".")[0]
            c, _ = _resolve(stem, exact, digits)
            if c is not None and c.json_path:
                cases.append(c)
    if a.limit:
        cases = cases[:a.limit]
    print("[i] 解析到 case %d 张；pred=%s stage1=%s"
          % (len(cases), a.pred, a.stage1), flush=True)
    for c in cases[:3]:
        print("    例 %s json=%s" % (c.case_id, c.json_path), flush=True)

    engine = TextEngine(device="auto")
    tot = hit_key = read_other = read_none = 0
    rows = []
    per_case = collections.Counter()

    for ci, c in enumerate(cases, 1):
        cid = c.case_id
        # 文件名要用 stage1 的 stem：数据集里 `1` 与 `0001` 是**两张不同的图**
        # （`1` 有 22 个元件 / 21 条网络，`0001` 有 83 个元件 / 64 条网络），
        # 所以必须从 GT 的目录名取名，不能 `zfill(4)`。
        stem = Path(c.json_path).parent.name
        pf = Path(a.pred) / ("%s.json" % stem)
        sf = Path(a.stage1) / ("%s.stage1.json" % stem)
        if not pf.exists() or not sf.exists():
            continue
        gt = load_json(Path(c.json_path))
        pr = json.loads(pf.read_text(encoding="utf-8"))
        g, p = normalise_doc(gt), normalise_doc(pr)
        try:
            _f1, km = assign_components(g, p, EvalConfig(), {})
        except Exception:                                    # noqa: BLE001
            km = {}
        matched = set(km.values())
        st = json.loads(sf.read_text(encoding="utf-8"))
        boxes = [(t["bbox_img"][0], t["bbox_img"][1], t["bbox_img"][2], t["bbox_img"][3])
                 for t in (st.get("texts") or []) if t.get("bbox_img")]
        try:
            bgr = imread_unicode(c.png_path)
        except Exception:                                    # noqa: BLE001
            continue
        H = bgr.shape[0]

        todo = []
        n_ic = 0
        for k, v in (gt.get("components") or {}).items():
            if canon_type(v.get("type")) not in IC_TYPES or not v.get("bbox"):
                continue
            n_ic += 1
            if k in matched:
                continue
            b = v["bbox"]
            img = (b[0], H - b[3], b[2], H - b[1])
            near = [bb for bb in boxes if _gap_dist(img, bb) <= a.pad]
            tot += 1
            todo.append((k, img, near, v.get("Name")))
        if ci <= 3:
            print("     %s: comps=%d IC=%d matched=%d todo=%d texts=%d"
                  % (stem, len(gt.get("components") or {}), n_ic, len(matched),
                     len(todo), len(boxes)), flush=True)

        allb = []
        for _k, _im, near, _n in todo:
            allb.extend(near)
        uniq = sorted(set(allb))
        reads = engine.read(bgr, uniq, ["designator"]) if uniq else []
        rmap = {}
        for r in reads:
            rmap[tuple(r["bbox"])] = (r.get("reads") or {}).get("designator")

        for k, img, near, name in todo:
            got = []
            for bb in near:
                r = rmap.get(tuple(bb))
                if r and r[0]:
                    got.append(str(r[0]))
            up = k.upper()
            if any(s.strip().upper() == up for s in got):
                hit_key += 1
                verdict = "A_读得出这个键"
            elif got:
                read_other += 1
                verdict = "B_读了别的"
            else:
                read_none += 1
                verdict = "D_附近啥也没读出来"
            per_case[verdict] += 1
            if len(rows) < a.show:
                rows.append((stem, k, name, len(near), verdict, got[:4]))
        if ci % 40 == 0:
            print("  ... %d/%d 张，累计 IC %d" % (ci, len(cases), tot), flush=True)

    print("=" * 78)
    print("GT IC 元件（键没配上的，默认链 220 张）%d 个" % tot)
    print("  A_附近读得出这个键   %4d  %.3f" % (hit_key, hit_key / max(1, tot)))
    print("  B_读了别的位号       %4d  %.3f" % (read_other, read_other / max(1, tot)))
    print("  D_附近啥也没读出来   %4d  %.3f" % (read_none, read_none / max(1, tot)))
    print("=" * 78)
    print("按判决：", dict(per_case))
    print("样例：")
    for r in rows:
        print("   %-8s GT=%-6s Name=%-16s 最近框%d  %-20s %s"
              % (r[0], r[1], str(r[2])[:16], r[3], r[4], r[5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
