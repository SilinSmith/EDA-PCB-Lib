"""两个 `final_results` 目录的逐项对比 —— 专为「全量 220 张跑完看净效果」写。

用法（**必须从仓库根 `Y:\\workspace\\EDA` 跑**）：

    python work\\tools\\eval\\cmp220.py work\\_submit_out220z\\final_results work\\_submit_out220w\\final_results

输出四块：
  1. 总均分 / 分 split 均分 / 四个指标的 F1 与 tp-fp-fn
  2. **爆 0 名单的差集** —— 哪几张从 0 变正（修好了）、哪几张从正变 0（弄坏了）
  3. 逐图增量最大的 20 张（正负各列）
  4. 失配桶（missing / extra / text_mismatch / bbox_mismatch / type_mismatch）合计对比

`CaseScore` 的属性是 `.component` / `.pin` / `.net_hypergraph` / `.net_line`（各是 `F1` 对象），
**不是** `.component_f1`。评测必须用官方 `evaluate_case`，与 `stage2_eval.py` 同一口径。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                     # eval -> tools -> work -> EDA
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import (DEFAULT_DATA_ROOT, build_index, load_json,  # noqa: E402
                         resolve)
from schematic_parser.metrics import EvalConfig, evaluate_case      # noqa: E402

METRICS = (("component", "元件", 0.30),
           ("pin", "引脚", 0.25),
           ("net_hypergraph", "网络(超图)", 0.35),
           ("net_line", "网络导线", 0.10))
BUCKETS = ("component.missing", "component.extra", "component.text_mismatch",
           "component.bbox_mismatch", "component.type_mismatch",
           "pin.missing", "pin.extra", "pin.name_mismatch",
           "pin.position_mismatch", "net.pred_pins", "net.gt_pins")


def pred_file(pred_dir: Path, case):
    for name in ("%s_Pred.json" % case.case_id, "%s.json" % case.case_id,
                 "_Pred.json"):
        f = pred_dir / name
        if f.exists():
            return f
    return None


def run(pred_dir: Path, cfg: EvalConfig, exact, digits):
    rows = {}
    for case in _all_cases(exact):
        f = pred_file(pred_dir, case)
        if f is None:
            continue
        gt = load_json(case.json_path)
        pr = load_json(str(f))
        try:
            sc = evaluate_case(gt, pr, case.case_id, cfg)
        except Exception as exc:                      # noqa: BLE001
            print("  !! %s %s" % (case.case_id, exc))
            continue
        rows[case.case_id] = {"split": case.split, "score": sc.score,
                              "cs": sc}
    return rows


def _all_cases(exact):
    seen, out = set(), []
    for key, case in sorted(exact.items()):
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        out.append(case)
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    a_dir, b_dir = Path(argv[1]), Path(argv[2])
    cfg = EvalConfig()
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    print("A = %s" % a_dir)
    print("B = %s\n" % b_dir)
    A, B = run(a_dir, cfg, exact, digits), run(b_dir, cfg, exact, digits)
    common = [k for k in A if k in B]
    print("A %d 张 / B %d 张 / 共有 %d 张\n" % (len(A), len(B), len(common)))

    def mean(rows, keys, metric=None):
        vals = []
        for k in keys:
            cs = rows[k]["cs"]
            vals.append(cs.score if metric is None
                        else getattr(cs, metric).f1)
        return sum(vals) / max(1, len(vals))

    print("%-26s %10s %10s %9s" % ("口径", "A", "B", "Δ"))
    print("-" * 60)
    ma, mb = mean(A, common), mean(B, common)
    print("%-26s %10.4f %10.4f %+9.4f" % ("总均分", ma, mb, mb - ma))
    for split in ("10GT", "10_GTcase", "200_train_cases"):
        ks = [k for k in common if A[k]["split"] == split]
        if not ks:
            continue
        x, y = mean(A, ks), mean(B, ks)
        print("%-26s %10.4f %10.4f %+9.4f  (n=%d)" % (split, x, y, y - x, len(ks)))
    print()
    for key, cn, w in METRICS:
        x, y = mean(A, common, key), mean(B, common, key)
        ta = sum(getattr(A[k]["cs"], key).tp for k in common)
        tb = sum(getattr(B[k]["cs"], key).tp for k in common)
        print("%-12s F1 %8.4f -> %8.4f %+8.4f | tp %6d -> %6d %+6d | 权重 %.2f"
              % (cn, x, y, y - x, ta, tb, tb - ta, w))

    print("\n--- 爆 0 名单的差集（F1 == 0）---")
    for key, cn, _w in METRICS:
        za = {k for k in common if getattr(A[k]["cs"], key).f1 == 0.0}
        zb = {k for k in common if getattr(B[k]["cs"], key).f1 == 0.0}
        fixed, broke = sorted(za - zb), sorted(zb - za)
        print("%-12s A 爆 0 %3d 张 -> B %3d 张   修好 %2d %s   弄坏 %2d %s"
              % (cn, len(za), len(zb), len(fixed), fixed[:14], len(broke),
                 broke[:14]))

    print("\n--- 逐图增量最大的 20 张 ---")
    d = sorted(common, key=lambda k: B[k]["score"] - A[k]["score"])
    for k in d[:10]:
        print("  %-18s %7.2f -> %7.2f  %+6.2f" % (k, A[k]["score"], B[k]["score"],
                                                  B[k]["score"] - A[k]["score"]))
    print("  ...")
    for k in d[-10:]:
        print("  %-18s %7.2f -> %7.2f  %+6.2f" % (k, A[k]["score"], B[k]["score"],
                                                  B[k]["score"] - A[k]["score"]))

    print("\n--- 失配桶合计 ---")
    for b in BUCKETS:
        va = sum(int(A[k]["cs"].net_line.detail.get(b, 0) or 0) for k in common)
        vb = sum(int(B[k]["cs"].net_line.detail.get(b, 0) or 0) for k in common)
        if va or vb:
            print("  %-28s %8d -> %8d  %+8d" % (b, va, vb, vb - va))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
