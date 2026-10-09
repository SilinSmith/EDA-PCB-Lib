"""用本地 metrics.py 复算官方评测机返回的 *_Pred.json，核对「本地尺 vs 官方尺」。

用法（在 Y:\\workspace\\EDA 下）：

    python -X utf8 work\\tools\\dev\\_offscore.py                       # 官方 pred，默认配置
    python -X utf8 work\\tools\\dev\\_offscore.py --run <fast_out 目录>  # 复算我们自己的某个臂
    python -X utf8 work\\tools\\dev\\_offscore.py --tol 5              # bbox_tol 对齐官方 5.0
    python -X utf8 work\\tools\\dev\\_offscore.py --cases 1,2,3

输出：逐案的 component / pin / net_hypergraph / net_line 的 F1，以及总分，
并且**把官方报告里的同名字段并排打出来**（官方口径只有 wire/connectivity 两个名字，
本地的 net_line ↔ 官方 wire_f1、net_hypergraph ↔ 官方 connectivity_f1）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402

GT_ROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
OFF_ROOT = (ROOT / "20261009引入" / "theSixthSubmission_44.29" / "scoring_result (5)")
OFF_PRED = OFF_ROOT / "final_results"
OFF_REPORT = OFF_ROOT / "report" / "submission" / "20261009T140206Z"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def official(cid: str):
    p = OFF_REPORT / ("%s.json" % cid)
    if not p.exists():
        return None
    d = load(p)
    s = d.get("scores") or {}
    return {
        "component_f1": s.get("component_f1"),
        "pin_f1": s.get("pin_f1"),
        "wire_f1": s.get("wire_f1"),
        "connectivity_f1": s.get("connectivity_f1"),
        "final_score": s.get("final_score"),
        "raw": d,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--run", default="", help="fast.py 的 --out 目录（相对 ROOT）")
    ap.add_argument("--tol", type=float, default=None, help="覆盖 EvalConfig.bbox_tol")
    ap.add_argument("--pin-tol", type=float, default=None)
    ap.add_argument("--unkeyed", default=None, help="off/gt/all")
    ap.add_argument("--pairs", action="store_true", help="逐案打印诊断键值")
    ap.add_argument("--engine", default="legacy", choices=("legacy", "official"))
    ap.add_argument("--net-mode", default="f1", choices=("f1", "iou", "cover", "inter"),
                    help="官方 connectivity 的网对得分公式（复刻用）")
    ap.add_argument("--no-collapse", action="store_true",
                    help="官方 connectivity 不折叠非 box 引脚号")
    args = ap.parse_args()

    cfg = M.EvalConfig()
    if args.tol is not None:
        cfg.bbox_tol = args.tol
    if args.pin_tol is not None:
        cfg.pin_tol = args.pin_tol
    if args.unkeyed is not None:
        cfg.unkeyed_match = args.unkeyed
    print("cfg: bbox_tol=%g pin_tol=%g line_tol=%g unkeyed=%s collapse_nonbox=%s"
          % (cfg.bbox_tol, cfg.pin_tol, cfg.line_tol, cfg.unkeyed_match,
             cfg.net_pin_collapse_nonbox))

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    rows = []
    for cid in cases:
        gt_p = GT_ROOT / cid / ("%s.json" % cid)
        if not gt_p.exists():
            print("skip %s（无 GT）" % cid)
            continue
        if args.run:
            pred_p = ROOT / args.run / "final_results" / ("%s_Pred.json" % cid)
        else:
            pred_p = OFF_PRED / ("%s_Pred.json" % cid)
        if not pred_p.exists():
            print("skip %s（无 pred %s）" % (cid, pred_p))
            continue
        gt, pred = load(gt_p), load(pred_p)
        if args.engine == "official":
            from schematic_parser import official as OFF
            sc = OFF.evaluate_case(gt, pred, cid,
                                   collapse_nonbox=not args.no_collapse,
                                   net_mode=args.net_mode)
            comp_f1, pin_f1 = sc.component.f1, sc.pin.f1
            netH_f1, netL_f1 = sc.connectivity.f1, sc.wire.f1
        else:
            sc = M.evaluate_case(gt, pred, cid, cfg)
            comp_f1, pin_f1 = sc.component.f1, sc.pin.f1
            netH_f1, netL_f1 = sc.net_hypergraph.f1, sc.net_line.f1
        off = official(cid)
        rows.append((cid, sc, off))
        line = ("case %-3s local: comp %.4f pin %.4f netH %.4f netL %.4f score %7.3f"
                % (cid, comp_f1, pin_f1, netH_f1, netL_f1, sc.score))
        if off and off["final_score"] is not None:
            line += ("  ||  off: comp %.4f pin %.4f conn %.4f wire %.4f score %7.3f"
                     % (off["component_f1"], off["pin_f1"], off["connectivity_f1"],
                        off["wire_f1"], off["final_score"]))
        print(line)
        if args.pairs:
            print("    detail:", {k: v for k, v in sorted(sc.diagnostic.items())})

    if rows:
        n = len(rows)

        def ch(sc, which):
            if args.engine == "official":
                return {'comp': sc.component.f1, 'pin': sc.pin.f1,
                        'netH': sc.connectivity.f1, 'netL': sc.wire.f1}[which]
            return {'comp': sc.component.f1, 'pin': sc.pin.f1,
                    'netH': sc.net_hypergraph.f1, 'netL': sc.net_line.f1}[which]

        agg = {k: sum(ch(r[1], k) for r in rows) / n
               for k in ("comp", "pin", "netH", "netL")}
        agg["score"] = sum(r[1].score for r in rows) / n
        print("LOCAL  mean: comp %.6f pin %.6f netH %.6f netL %.6f score %.4f"
              % (agg["comp"], agg["pin"], agg["netH"], agg["netL"], agg["score"]))
        offs = [r[2] for r in rows if r[2] and r[2]["final_score"] is not None]
        if len(offs) == n:
            print("OFFICIAL mean: comp %.6f pin %.6f conn %.6f wire %.6f score %.4f"
                  % (sum(o["component_f1"] for o in offs) / n,
                     sum(o["pin_f1"] for o in offs) / n,
                     sum(o["connectivity_f1"] for o in offs) / n,
                     sum(o["wire_f1"] for o in offs) / n,
                     sum(o["final_score"] for o in offs) / n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
