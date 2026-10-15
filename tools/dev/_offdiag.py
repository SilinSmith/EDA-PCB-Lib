"""官方口径失分归因：把 component / pin 的失败原因按类别汇总（不改任何东西）。

    python -X utf8 work\\tools\\dev\\_offdiag.py --run work\\_fast_out_maj2
    python -X utf8 work\\tools\\dev\\_offdiag.py --run work\\_fast_out_maj2 --cases 1,8

输出：
  * component：extra / missing / 属性不符（type/Name/value/bbox 各自计数）+ 属性不符样例
  * pin：extra / missing / pin_name / tip_position 计数 + 错名样例（GT 名 → 我方名）
  * connectivity / wire：软 tp 与分母
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import official as OFF  # noqa: E402

GT_ROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work\\_fast_out_maj2", help="fast.py 输出目录")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    run = ROOT / args.run
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    comp = Counter()
    pin = Counter()
    names = Counter()
    val_mismatch = []
    name_mismatch = []
    bbox_big = []
    rows = []
    for cid in cases:
        gp = GT_ROOT / cid / ("%s.json" % cid)
        pp = run / "final_results" / ("%s_Pred.json" % cid)
        if not (gp.exists() and pp.exists()):
            print("skip %s" % cid)
            continue
        gt = json.loads(gp.read_text(encoding="utf-8"))
        pred = json.loads(pp.read_text(encoding="utf-8"))
        sc = OFF.evaluate_case(gt, pred, cid)
        rows.append((cid, sc))
        for ex in sc.component.detail.get("examples", []):
            comp[ex.get("error")] += 1
            if ex.get("error") == "attribute_or_position_mismatch":
                for f in ex.get("failed_checks") or []:
                    comp["chk:" + f] += 1
                if "bbox" in (ex.get("failed_checks") or []):
                    bbox_big.append((cid, ex["component_id"],
                                     ex.get("bbox_max_coordinate_error_px")))
                if "value" in (ex.get("failed_checks") or []):
                    k = ex["component_id"]
                    pv = (pred.get("components", {}).get(k) or {}).get("value")
                    gv = (gt.get("components", {}).get(k) or {}).get("value")
                    val_mismatch.append((cid, k, gv, pv))
                if "Name" in (ex.get("failed_checks") or []):
                    k = ex["component_id"]
                    pv = (pred.get("components", {}).get(k) or {}).get("Name")
                    gv = (gt.get("components", {}).get(k) or {}).get("Name")
                    name_mismatch.append((cid, k, gv, pv))
        for ex in sc.pin.detail.get("examples", []):
            if ex.get("error"):
                pin[ex["error"]] += 1
                continue
            for f in ex.get("failed_checks") or []:
                pin["chk:" + f] += 1
                if f == "pin_name":
                    k, pid = ex["pin_id"]
                    gpn = ((gt.get("pins", {}).get(k) or {}).get(pid) or {}).get("pinname")
                    ppn = ((pred.get("pins", {}).get(k) or {}).get(pid) or {}).get("pinname")
                    names["%s -> %s" % (gpn, ppn)] += 1

    print("== component 失败计数 ==")
    for k, v in comp.most_common():
        print("   %-32s %d" % (k, v))
    print("== pin 失败计数 ==")
    for k, v in pin.most_common():
        print("   %-32s %d" % (k, v))
    print("== pin 错名（GT -> 我方，前 %d）==" % args.top)
    for k, v in names.most_common(args.top):
        print("   %-28s %d" % (k, v))
    print("== component value 不符（前 %d）==" % args.top)
    for cid, k, gv, pv in val_mismatch[:args.top]:
        print("   case %-3s %-7s GT %-18r 我方 %r" % (cid, k, gv, pv))
    print("== component Name 不符（前 %d）==" % args.top)
    for cid, k, gv, pv in name_mismatch[:args.top]:
        print("   case %-3s %-7s GT %-22r 我方 %r" % (cid, k, gv, pv))
    print("== bbox 超 5px（前 %d）==" % args.top)
    for cid, k, e in sorted(bbox_big, key=lambda r: -(r[2] or 0))[:args.top]:
        print("   case %-3s %-7s max_err %.2f px" % (cid, k, e or -1))
    print("== 逐案通道 ==")
    for cid, sc in rows:
        print("   case %-3s comp %.4f (tp %d) pin %.4f (tp %d) conn %.4f wire %.4f score %.2f"
              % (cid, sc.component.f1, sc.component.tp, sc.pin.f1, sc.pin.tp,
                 sc.connectivity.f1, sc.wire.f1, sc.score))
    n = len(rows)
    print("   mean comp %.6f pin %.6f conn %.6f wire %.6f score %.4f"
          % (sum(s.component.f1 for _, s in rows) / n,
             sum(s.pin.f1 for _, s in rows) / n,
             sum(s.connectivity.f1 for _, s in rows) / n,
             sum(s.wire.f1 for _, s in rows) / n,
             sum(s.score for _, s in rows) / n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
