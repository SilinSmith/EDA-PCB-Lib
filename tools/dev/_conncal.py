"""校准官方 connectivity 公式：把官方 report 里的 soft_tp 与几种候选网对得分并排。

    python -X utf8 work\\tools\\dev\\_conncal.py [--collapse]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import official as OFF  # noqa: E402

GT_ROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
OFF_ROOT = ROOT / "20261009引入" / "theSixthSubmission_44.29" / "scoring_result (5)"
CASES = "1 2 3 4 5 6 7 8 9 10".split()
MODES = ("f1", "iou", "cover", "inter")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collapse", action="store_true", default=True)
    ap.add_argument("--no-collapse", dest="collapse", action="store_false")
    ap.add_argument("--soft-nonbox", action="store_true",
                    help="非 box 成员不折叠号、但把 GT 侧随机号换成通配（另类口径）")
    args = ap.parse_args()

    print("%-5s %-9s %-6s %-6s %s" % ("case", "off_soft", "gtn", "pdn",
                                      "  ".join("%-9s" % m for m in MODES)))
    tot = {m: 0.0 for m in MODES}
    off_tot = 0.0
    for cid in CASES:
        gt = json.loads((GT_ROOT / cid / ("%s.json" % cid)).read_text(encoding="utf-8"))
        pred = json.loads((OFF_ROOT / "final_results" / ("%s_Pred.json" % cid))
                          .read_text(encoding="utf-8"))
        rep = json.loads((OFF_ROOT / "report" / "submission" / "20261009T140206Z"
                          / ("%s.json" % cid)).read_text(encoding="utf-8"))
        cx = rep["connectivity"]
        soft = cx["f1"].get("soft_tp")
        off_tot += soft
        cells = []
        for m in MODES:
            c, pairs = OFF.connectivity_f1(gt, pred, args.collapse, m)
            cells.append(c.tp)
            tot[m] += c.tp
        print("%-5s %-9.4f %-6d %-6d %s" % (cid, soft,
                                            cx["f1"]["gt_net_count"],
                                            cx["f1"]["pred_net_count"],
                                            "  ".join("%-9.4f" % c for c in cells)))
    print("TOTAL off %.4f | %s" % (off_tot,
                                   "  ".join("%s %.4f" % (m, tot[m]) for m in MODES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
