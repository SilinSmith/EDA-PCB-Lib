"""**`pin`（权重 0.25）为什么爆 0** —— 在快测集上按官方判定口径逐条拆。

`fast.py` 的基线里 19 张有 14 张 `pin F1 = 0.000`。这个脚本回答「0 是哪来的」：
- 该图的 GT IC 框键配上了没有（没配上 → 全部引脚 `missing`，一分不得）；
- 配上了的话，`pin_f1` 的六个判决桶各有多少条（`missing` / `extra` / `name_mismatch` /
  `no_point` / `position_mismatch` / `malformed`）；
- 同号引脚点的位置误差分布（`position_mismatch` 是「编号错位」还是「位置噪声」）。

用法：
  python work\\tools\\diag\\_pinwhy.py                     # 快测集 19 张
  python work\\tools\\diag\\_pinwhy.py --pred work\\_submit_out220z\\final_results
  python work\\tools\\diag\\_pinwhy.py --cases 1,3,4
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                          # EDA
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "debug"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json   # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES, assign_components,     # noqa: E402
                                      canon_type)
import ledger as L                                                                 # noqa: E402

FAST = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
        "0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]


def load_pred(d, case_id):
    for nm in ("%s_Pred.json" % case_id, "%s.json" % case_id):
        p = d / nm
        if p.exists():
            return load_json(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="", help="逗号分隔；默认快测集 19 张")
    ap.add_argument("--all", action="store_true", help="跑 220 张全集")
    args = ap.parse_args()

    pred_dir = Path(args.pred)
    want = set(x.strip() for x in args.cases.split(",") if x.strip()) or set(FAST)
    cfg = EvalConfig()
    rows, tot = [], Counter()
    for c in discover(DEFAULT_DATA_ROOT, SPLITS):
        if not c.json_path:
            continue
        if not args.all and c.case_id not in want:
            continue
        gt = load_json(Path(c.json_path))
        pr = load_pred(pred_dir, c.case_id)
        if pr is None:
            continue
        _f1, km = assign_components(gt, pr, cfg, {})
        matched_gt = set(km.values())
        gt_ic = [k for k, v in (gt.get("components") or {}).items()
                 if canon_type(v.get("type")) in IC_TYPES]
        ok_ic = [k for k in gt_ic if k in matched_gt]
        led = L.build_ledger(gt, pr, case_id=c.case_id, cfg=cfg, split=c.split)
        part = next((p for p in led["score"]["parts"] if p["key"] == "pin"), None)
        bk = dict((part or {}).get("stats", {}).get("buckets", {}))
        rows.append((c.case_id, c.split, (part or {}).get("f1", 0.0), len(gt_ic), len(ok_ic), bk))
        tot.update(bk)
        tot["_gt_ic"] += len(gt_ic)
        tot["_ok_ic"] += len(ok_ic)

    rows.sort(key=lambda r: (r[2], r[0]))
    print("%-8s %-16s %7s  %-11s %s" % ("case", "split", "pinF1", "GT_IC/配上", "判决桶"))
    print("-" * 108)
    for cid, sp, f1, nic, nok, bk in rows:
        tag = ">> 0 <<" if f1 <= 0 else "%.4f" % f1
        print("%-8s %-16s %7s  %4d/%-5d %s"
              % (cid, sp, tag, nic, nok,
                 "  ".join("%s %d" % (k, v) for k, v in sorted(bk.items())) or "-"))

    zero = [r for r in rows if r[2] <= 0]
    nokey = [r for r in zero if r[4] == 0]
    print("\n共 %d 张｜爆 0 %d 张｜其中「GT IC 框一个都没配上」的 %d 张"
          % (len(rows), len(zero), len(nokey)))
    print("爆 0 但**键配上了**的：%s"
          % ", ".join("%s(%d/%d)" % (r[0], r[4], r[3]) for r in zero if r[4] > 0))
    print("全集判决桶合计：%s" % dict(sorted(tot.items())))


if __name__ == "__main__":
    main()
