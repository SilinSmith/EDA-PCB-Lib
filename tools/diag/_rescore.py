# -*- coding: utf-8 -*-
"""把已存档的 A/B 臂**用新口径重打一遍**，看有没有结论会翻。

最新 QA（2026-10）说「非 box 元件 pinnumber 不计分，rcl 等在 hygraph 中相当于不区分
引脚」，`EvalConfig.net_pin_collapse_nonbox` 因此默认从 False 变 True。历史结论都是
旧口径量的，本工具用同一批预测目录跑两种口径，直接给出 Δ 和排名变化。

用法：
    python work\\tools\\diag\\_rescore.py --dirs "标签=路径[:glob],标签2=路径2[:glob2]"
    python work\\tools\\diag\\_rescore.py --preset f9            # 见 DEFAULTS
    python work\\tools\\diag\\_rescore.py --preset all --json ..\\work\\_rescore.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA\eda-pcb")
WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(WORK / "tools" / "eval"))

from schematic_parser.data import load_json                       # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case    # noqa: E402
import stage2_eval as SE                                          # noqa: E402

PRED_GLOB = "final_results/*_Pred.json"

DEFAULTS: dict[str, list[str]] = {
    # 引脚通道开关（官方 9 张）：A 58.6304 / B 61.5112 / C(mrg) 59.9468 / D(只号) 60.5816
    "pin": [f"{t}={WORK / d}" for t, d in [
        ("A 通道关", "_f9_A_base"), ("B 全开", "_f9_B_pin"), ("C 只名(mrg)", "_f9_C_mrg"),
        ("D 只号(n1m0)", "_f9_n1m0"), ("只名(n0m1)", "_f9_n0m1"),
        ("F 空跑(n0m0)", "_f9_n0m0")]],
    # VLM 引脚分数扫描（四臂旧口径全同：61.5112）
    "ps": [f"{t}={WORK / d}" for t, d in [
        ("ps090", "_f9_ps090"), ("ps130", "_f9_ps130"),
        ("ps160", "_f9_ps160"), ("ps250", "_f9_ps250")]],
    # 连号先验 / 阅读顺序（`_f9_ab_A..F` 就是那六臂）
    "order": [f"{t}={WORK / d}" for t, d in [
        ("A alt1/ord0 基线", "_f9_ab_A"), ("B alt0/ord0", "_f9_ab_B"),
        ("C alt0/ord1", "_f9_ab_C"), ("D alt1/ord1", "_f9_ab_D"),
        ("E alt0/ord1(home1)", "_f9g_ab_E"), ("E' ab_E", "_f9_ab_E"),
        ("F 空跑", "_f9_ab_F"), ("alt0", "_f9_alt0"), ("B_recheck", "_f9_B_recheck"),
        ("C workers=1", "_f9w1_C"), ("C2 workers=2(不可复现)", "_f9g2_C"),
        ("g_ab_C 1 0", "_f9g_ab_C 1 0")]],
    # 融合 220 张四臂
    "f220": [f"{t}={WORK / d}" for t, d in [
        ("A", "_f220_A"), ("B", "_f220_B"), ("C", "_f220_C"), ("n1m0", "_f220_n1m0")]],
    # tp 合并四臂（19 张）
    "tp19": [f"{t}={WORK / d}|{PRED_GLOB}" for t, d in [
        ("off", "_fast_out_tpmerge_D19off"), ("skel", "_fast_out_tpmerge_D19skel"),
        ("mrg", "_fast_out_tpmerge_D19mrg"), ("ms", "_fast_out_tpmerge_D19ms")]],
    # CRNN 换权重的两臂（官方 9 张，离线 stage2）
    "ckpt": [f"{t}={WORK / d}" for t, d in [
        ("出厂权重", r"_ckptab\base\runs\stage2"), ("mix66", r"_ckptab\mix66\runs\stage2")]],
    "reg9": [f"reg9={WORK / '_fast_out_reg9'}|{PRED_GLOB}"],
}


def score_dir(pred_dir: str, glob: str, cfg: EvalConfig):
    idx, digits = SE.build_index(SE.DEFAULT_DATA_ROOT)
    # ⚠️ 必须排掉 `<case>.pinindex.json` / `<case>.stage1.json` 这类**附属产物**：
    # `stage2_eval.resolve()` 最后有一条「剥掉非数字字符再查」的兜底，会把
    # `1.pinindex` 的数字别名 `1` 解析回 case 1 ⇒ 同一个 case 被计两次，
    # 第二次拿一个结构完全不同的 json 去打分（≈0）⇒ 均分被拖垮。
    files = [f for f in sorted(Path(pred_dir).glob(glob))
             if not f.name.startswith("_") and "." not in f.stem]
    scores, miss = [], 0
    for f in files:
        case, _how = SE.resolve(f.stem, idx, digits)
        if case is None:
            miss += 1
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        scores.append(evaluate_case(gt, pred, f.stem, cfg))
    return scores, miss


def main(argv=None):
    ap = argparse.ArgumentParser(description="A/B 臂用新旧口径重打")
    ap.add_argument("--dirs", default=None, help="标签=路径[:glob]，逗号分隔")
    ap.add_argument("--preset", default=None,
                    help="逗号分隔的预设名，或 all（可用：%s）" % "、".join(sorted(DEFAULTS)))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    specs = []
    if a.dirs:
        specs = [s for s in a.dirs.split(",") if s.strip()]
    for p in (a.preset.split(",") if a.preset else []):
        p = p.strip()
        if p == "all":
            specs += [s for v in DEFAULTS.values() for s in v]
        elif p in DEFAULTS:
            specs += DEFAULTS[p]
        elif p:
            ap.error("未知预设 %r（可用：%s、all）" % (p, "、".join(sorted(DEFAULTS))))
    if not specs:
        ap.error("给 --dirs 或 --preset（可用：%s、all）" % "、".join(sorted(DEFAULTS)))

    cfgs = {"旧(字面)": EvalConfig(net_pin_collapse_nonbox=False),
            "新(折叠)": EvalConfig(net_pin_collapse_nonbox=True)}
    rows = []
    print("%-22s %5s %9s %9s %8s %8s %8s" %
          ("臂", "n", "旧分", "新分", "Δ", "netH旧", "netH新"))
    print("-" * 76)
    for spec in specs:
        label, _, rest = spec.partition("=")
        # glob 用 '|' 分隔（':' 会被 Windows 盘符占用）
        if "|" in rest:
            path, _, glob = rest.partition("|")
        else:
            path, glob = rest, "*.json"
        rec = {"label": label, "dir": path, "glob": glob}
        for name, cfg in cfgs.items():
            scores, miss = score_dir(path, glob, cfg)
            n = max(1, len(scores))
            rec[name] = dict(
                n=len(scores), miss=miss,
                mean=sum(s.score for s in scores) / n,
                netH=sum(s.net_hypergraph.f1 for s in scores) / n,
                netL=sum(s.net_line.f1 for s in scores) / n,
                compF1=sum(s.component.f1 for s in scores) / n,
                pinF1=sum(s.pin.f1 for s in scores) / n,
            )
        o, w = rec["旧(字面)"], rec["新(折叠)"]
        rec["delta"] = w["mean"] - o["mean"]
        print("%-22s %5d %9.4f %9.4f %+8.4f %8.4f %8.4f" %
              (label, w["n"], o["mean"], w["mean"], rec["delta"],
               o["netH"], w["netH"]))
        rows.append(rec)

    # 每个 preset 内的排名变化
    print("\n排名（按新分降序，括号里是旧口径名次变化）：")
    groups: dict[str, list] = {}
    for spec, rec in zip(specs, rows):
        groups.setdefault("all", []).append(rec)
    old_rank = {r["label"]: i for i, r in
                enumerate(sorted(rows, key=lambda r: -r["旧(字面)"]["mean"]), 1)}
    for i, r in enumerate(sorted(rows, key=lambda r: -r["新(折叠)"]["mean"]), 1):
        was = old_rank[r["label"]]
        flag = "" if was == i else "  ← 旧第 %d" % was
        print("  %2d. %-22s 新 %.4f  旧 %.4f%s" %
              (i, r["label"], r["新(折叠)"]["mean"], r["旧(字面)"]["mean"], flag))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print("\n写到 %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
