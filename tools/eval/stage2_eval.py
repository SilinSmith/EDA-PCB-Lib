# -*- coding: utf-8 -*-
"""第二阶段的评测：把 `<pred_dir>\\<case>.json` 与官方 GT 对起来。

**为什么不用 `schematic_parser.evaluate.run`**：那个入口用 `_norm_id`（去掉前导零）
把预测和 GT 配起来，而本数据集里 `10GT\\1` 与 `200_train_cases\\0001` 规范化后
都是 `1` —— 两张完全不同的图会共用一个预测。这里改成**按文件名精确配对**。

**2025-10 修的一个真 bug（曾让 10 张评错 GT）**：旧的 `gt_index()` 一边记
`c.case_id`、一边顺手 `setdefault` 一个「只留数字」的别名。`10_GTcase\\0025-Altium
Designer` 的数字别名是 `0025`，而 `200_train_cases\\0025` 的 case_id 正好也是
`0025` —— 由于 `SPLITS` 里 `10_GTcase` 排在 `200_train_cases` **前面**，
`idx['0025']` 被旧的 golden case 抢走，于是我们为训练集 0025 生成的预测被拿去跟
另一张完全不同的图（且是旧版 GT）对比。现在改成：**先全表精确匹配，精确匹配不到
才退到数字别名，且别名必须唯一**。

用法：
    python work\\tools\\eval\\stage2_eval.py <pred_dir> [--data-root ...] [--no-text]
        [--profile latest] [--bbox-tol 20] [--pin-tol 20] [--line-tol 20]
        [--unkeyed-match {off,gt,all}] [--pin-all] [--no-canon-types]
        [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent


def TIERS() -> dict:
    """读 `tiers.json`：固定的「低分 / 中位 / 高分」样例档。

    ★ 用户定的规矩（m23636）：以后自主开的测试不超过 10 分钟，而且要**打中要害** ——
    只跑「上一轮低分的样例 + 少量正常样例 + 少量好样例」，不要每动一行就重跑 220 张。
    `low` 档就是从上一轮全量结果里取的最低分那批，改动有没有用看它最快。
    """
    p = _HERE / "tiers.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return {}
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, aggregate, evaluate_case  # noqa: E402


def build_index(data_root):
    """返回 ``(exact, digits)`` 两个索引。

    ``exact``: case_id -> Case（全表精确，跨 split 不会重名）
    ``digits``: 只留数字的别名 -> [Case, ...]（可能有多个，只用于兜底）
    """
    exact, digits = {}, {}
    for c in discover(data_root, SPLITS):
        if not c.json_path:
            continue
        exact.setdefault(c.case_id, c)
        d = re.sub(r"\D", "", c.case_id)
        if d:
            digits.setdefault(d, []).append(c)
    return exact, digits


def resolve(stem, exact, digits):
    r"""精确优先；找不到才用唯一数字别名。歧义时优先**纯数字 case_id**，仍歧义返回 ``None``。

    ⚠️ 这条兜底是必须的：`final_results` 导出的文件名是 `<case>_Pred.json`，而
    `10_GTcase` 里有形如 `0025-Altium Designer` 的 case（数字别名也是 `0025`），
    与 `200_train_cases\0025` 撞车。旧版直接判歧义 → **`final_results` 只配得上 200/220 张**，
    算出来的分与 `runs\stage2`（220/220）不可比（实测 `_submit_out220z`：
    `runs\stage3_vlm` **27.5464** vs 同 run 的 `final_results` **27.3264**，差的就是这 20 张）。
    如今歧义时优先「case_id 本身是纯数字」的那个（= 数据集里的正式编号）。
    """
    if stem in exact:
        return exact[stem], "exact"
    # ★ 2026-07 审计修复：先剥掉 `_Pred` 后缀再查一次 exact。
    # 以前这里直接掉进下面的「数字别名」兜底：`0025-Altium Designer_Pred` 的 stem 不在
    # `exact`（exact 的键是 `0025-Altium Designer`）⇒ `d = '0025'` ⇒
    # `cands = [10_GTcase\0025-Altium Designer, 200_train_cases\0025]` ⇒
    # `pure = ['0025']` 唯一 ⇒ 返回**train** 的 0025。后果：golden 预测拿去和一张无关的
    # GT 打分、`0025_Pred.json` 又命中同一个 Case（该 case 被计 2 次）、10 张 golden GT
    # 一次都没评，而 `len(scores)` **仍然是 220** ⇒「220/220」再也抓不到这个错。
    # 换句话说：上一版「优先纯数字」的修法只是把「配不上 20 张」换成了「配错 10 张」。
    if stem.endswith("_Pred") and stem[:-5] in exact:
        return exact[stem[:-5]], "exact^pred"
    d = re.sub(r"\D", "", stem) or stem
    cands = digits.get(d) or []
    if len(cands) == 1:
        return cands[0], "digits"
    pure = [c for c in cands if c.case_id.isdigit()]
    if len(pure) == 1:
        return pure[0], "digits^pure"
    return None, ("ambiguous(%d)" % len(cands) if cands else "none")


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage2 产物评测（按文件名精确配对）")
    ap.add_argument("pred")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--profile", choices=("latest", "protocol2", "protocol"), default="latest")
    ap.add_argument("--bbox-tol", type=float, default=None)
    ap.add_argument("--pin-tol", type=float, default=None)
    ap.add_argument("--line-tol", type=float, default=None)
    ap.add_argument("--no-text", action="store_true", help="不比器件名/参数（只考几何与键）")
    ap.add_argument("--unkeyed-match", default="gt", choices=("off", "gt", "all"),
                    help="无名元(∅NNN)的匹配方式，默认 gt=官方口径")
    ap.add_argument("--pin-all", action="store_true",
                    help="pin_f1 统计全部元件（旧口径），默认只统计 box")
    ap.add_argument("--net-literal", action="store_true",
                     help="历史对照：普通元件也按具体引脚号比较")
    ap.add_argument("--net-collapse", action="store_true",
                     help="普通元件按元件整体、IC按pin比较（latest默认）")
    ap.add_argument("--no-canon-types", action="store_true",
                    help="不按 type_list 归一化 type")
    ap.add_argument("--net-hard", action="store_true",
                    help="连通性通道回到旧口径「tp = Σ 引脚交集」（历史基线都是这个口径）；"
                         "默认是官方口径「tp = Σ 每对网的 pin-set 软 F1(Dice)」")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--cases", default=None,
                    help="只评这些 case（逗号分隔）；默认评目录里全部")
    ap.add_argument("--tier", default=None,
                    help="只评 tiers.json 里的某一档（low/mid/high/gt9）")
    args = ap.parse_args(argv)

    if args.net_literal and args.net_collapse:
        ap.error("--net-literal and --net-collapse are mutually exclusive")
    cfg = EvalConfig.for_profile(args.profile)
    for field in ("bbox_tol", "pin_tol", "line_tol"):
        if getattr(args, field) is not None:
            setattr(cfg, field, getattr(args, field))
    cfg.compare_text = not args.no_text
    cfg.unkeyed_match = args.unkeyed_match
    cfg.pin_box_only = not args.pin_all
    cfg.canon_types = not args.no_canon_types
    if args.net_literal or args.net_collapse:
        cfg.net_pin_collapse_nonbox = bool(args.net_collapse)
    if args.net_hard:
        cfg.net_soft = False
    exact, digits = build_index(args.data_root)

    pred_dir = Path(args.pred)
    files = [f for f in sorted(pred_dir.glob("*.json"))
             if not f.name.startswith("_")]
    # ---- 只评子集：`--tier` 是固定的低/中/高样例档，`--cases` 是临时点名单 ----
    # 目的是让「改动有没有用」能在**一分钟内**回答完，而不是每动一行就重跑 220 张
    # 或整条含 LLM 轮的链（那要 10 分钟以上）。
    want = None
    if args.tier:
        want = set(TIERS().get(args.tier) or [])
        if not want:
            print("tiers.json 里没有这一档: %s" % args.tier)
            return 1
    elif args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
    if want is not None:
        files = [f for f in files if f.stem in want]
        if not files:
            print("子集里没有匹配的预测文件")
            return 1
    if not files:
        print("没有预测文件: %s" % pred_dir)
        return 1

    scores, missing, used = [], [], {}
    for f in files:
        case, how = resolve(f.stem, exact, digits)
        if case is None:
            missing.append("%s(%s)" % (f.stem, how))
            continue
        used.setdefault(case.case_id, []).append(f.stem)
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        scores.append(evaluate_case(gt, pred, f.stem, cfg))

    if not scores:
        print("一个都没配上（输出目录名与 GT 的 case_id 不符）")
        return 1

    # ★ 2026-07 审计修复：`used` 原来只写不读（死代码）。有了它才能在张数仍然
    # 「正常」的情况下发现「两份预测打到同一张 GT、另一张 GT 没人评」这类错配
    # —— 那正是上面 `resolve()` 曾经犯的错（220 张看起来完全正常）。
    dup = {k: v for k, v in used.items() if len(v) > 1}
    if dup:
        print("⚠️ 有 %d 张 GT 被**多份预测**重复打分（张数会虚高、且这些 GT 的分数是混的）："
              % len(dup))
        for k, v in list(dup.items())[:10]:
            print("     %-28s <- %s" % (k, v))
    if len(used) != len(scores):
        print("⚠️ 去重后只覆盖 %d 张 GT，却有 %d 份预测被计分（差 %d）"
              % (len(used), len(scores), len(scores) - len(used)))

    if not args.quiet:
        print("%-22s %7s %7s %7s %7s %7s" %
              ("case", "score", "compF1", "pinF1", "netHF1", "lineF1"))
        print("-" * 62)
        for s in sorted(scores, key=lambda x: -x.score)[:60]:
            print("%-22s %7.2f %7.4f %7.4f %7.4f %7.4f" %
                  (s.case_id, s.score, s.component.f1, s.pin.f1,
                   s.net_hypergraph.f1, s.net_line.f1))
        if len(scores) > 60:
            print("... 还有 %d 张" % (len(scores) - 60))
        print("-" * 62)

    summary = aggregate(scores)
    print("配对 %d 张（配不上 %d 张）  平均权重得分 %.4f"
          % (len(scores), len(missing), summary.get("mean_score", 0.0)))
    # 注意：`aggregate()` 返回的是 `component_f1` / `pin_f1` / ... 这样的**均值浮点数**，
    # 不是 F1 对象；tp/fp/fn 要自己按 case 汇总。这里曾经错读过 `summary.get("component")`
    # （键不存在 -> None -> 打印全 0），别改回去。
    for k, attr in (("component", "component"), ("pin", "pin"),
                    ("net_hypergraph", "net_hypergraph"), ("net_line", "net_line")):
        tp = sum(getattr(s, attr).tp for s in scores)
        fp = sum(getattr(s, attr).fp for s in scores)
        fn = sum(getattr(s, attr).fn for s in scores)
        print("  %-14s F1 %.4f  P %.4f  R %.4f  tp %.6g fp %.6g fn %.6g"
              % (k, summary.get(k + "_f1", 0.0),
                 tp / max(1e-9, tp + fp), tp / max(1e-9, tp + fn), tp, fp, fn))
    # `evaluate_case` 里 detail 是**共享**的一份，四个 F1 各自快照；net_line 那份
    # 最全（含 component./pin./net. 三类键），所以只汇总它，别把四份加起来。
    det = {}
    for s in scores:
        for k, n in (s.net_line.detail or {}).items():
            det[k] = det.get(k, 0) + n
    if det:
        print("  失配明细: %s" % json.dumps(det, ensure_ascii=False))
    if missing:
        print("  配不上的: %s" % (missing[:10],))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "config": vars(cfg), "summary": summary,
            "cases": [s.as_dict() for s in scores],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print("写到 %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
