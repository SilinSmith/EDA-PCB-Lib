# -*- coding: utf-8 -*-
"""池子 A/B 的产量体检：读一个输出目录的 stage1 产物，打印每个 case 的 `pin_mono` 产量。

用途：判断 `STAGE1_MONO_FRAME_PINS=1` 在池子 10 张上到底有没有「输入」
（官方 10 张上四边 `confirmed=False`、`added/calibrated` 全 0，所以分数一动不动）。

用法：
    python -X utf8 ..\\work\\_poolmono.py --run poolA1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="输出目录后缀，如 poolA1（读 work/_fast_out_<run>）")
    ap.add_argument("--cmp", default="", help="另一个输出目录后缀；逐 case 比较 stage1 的 pins 是否逐位相同")
    a = ap.parse_args()
    d = ROOT / "work" / ("_fast_out_%s" % a.run) / "runs" / "stage1"
    files = sorted(d.glob("*.stage1.json"))
    if not files:
        print("no stage1 docs:", d)
        return 1
    print("run=%s  docs=%d" % (a.run, len(files)))
    tot = dict(recs=0, added=0, calibrated=0, held=0, stroke=0, marker=0, allskip=0)
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print("  ! %s: %s" % (f.name, exc))
            continue
        pm = doc.get("pin_mono")
        if pm is None:
            print("  %-28s pin_mono key ABSENT" % f.name.split(".")[0])
            continue
        if not pm:
            print("  %-28s pin_mono EMPTY" % f.name.split(".")[0])
            continue
        rows = []
        for cid, rec in sorted(pm.items()):
            if not rec:
                rows.append("%s:<empty>" % cid)
                continue
            n_add = len(rec.get("added") or [])
            n_cal = len(rec.get("calibrated") or [])
            n_held = len(rec.get("held") or [])
            n_sk = len(rec.get("skipped_sides") or [])
            n_row = len(rec.get("stroke_rows") or [])
            n_mk = len(rec.get("marker_rows") or [])
            tot["recs"] += 1
            tot["added"] += n_add
            tot["calibrated"] += n_cal
            tot["held"] += n_held
            tot["stroke"] += n_row
            tot["marker"] += n_mk
            if n_sk >= 4:
                tot["allskip"] += 1
            rows.append("%s:+%d/~%d/h%d/rows%d/mk%d/skip%d(old%s->new%s)" % (
                cid, n_add, n_cal, n_held, n_row, n_mk, n_sk,
                rec.get("old_count"), rec.get("new_count")))
        print("  %-28s %s" % (f.name.split(".")[0], " | ".join(rows)))
    print("TOTAL recs=%d added=%d calibrated=%d held=%d stroke_rows=%d marker_rows=%d all4skip=%d" % (
        tot["recs"], tot["added"], tot["calibrated"], tot["held"],
        tot["stroke"], tot["marker"], tot["allskip"]))
    if a.cmp:
        d2 = ROOT / "work" / ("_fast_out_%s" % a.cmp) / "runs" / "stage1"
        print("--- 对照 %s：stage1 pins 是否逐位相同 ---" % a.cmp)
        for f in files:
            g = d2 / f.name
            if not g.exists():
                print("  %-28s (对照组缺此 case)" % f.name.split(".")[0])
                continue
            x = json.loads(f.read_text(encoding="utf-8")).get("pins") or {}
            y = json.loads(g.read_text(encoding="utf-8")).get("pins") or {}
            nx = {k: len(v) for k, v in x.items()}
            ny = {k: len(v) for k, v in y.items()}
            same = "same" if x == y else ("counts-same" if nx == ny else "DIFF")
            print("  %-28s %s  (%s vs %s)" % (f.name.split(".")[0], same,
                                              sum(nx.values()), sum(ny.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
