# -*- coding: utf-8 -*-
"""小规模快速回归 —— **跑全量 220 张要 78 分钟，改一次代码等不起。**

固定一小组 case（默认 19 张：10GT 全部 9 张 + 200 训练集里最差的 10 张），
端到端跑一遍，然后逐 case 打印**四个分项的 F1**，并把 `0.000` 高亮出来。

用法：
    python work\\tools\\eval\\fast.py                      # 跑 + 打分
    python work\\tools\\eval\\fast.py --score-only         # 只重新打分（复用上次产物）
    python work\\tools\\eval\\fast.py --cases 1,2,3        # 自定 case
    python work\\tools\\eval\\fast.py --no-vlm             # 跳过 VLM，最快
    python work\\tools\\eval\\fast.py --save base          # 存基线
    python work\\tools\\eval\\fast.py --cmp base           # 和基线比

产物：`work\\_fast_out\\`（`final_results\\` + `runs\\` + `decision_log.json`）
基线：`work\\_fast_base\\<名字>.json`
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # Y:\workspace\EDA  （eval -> tools -> work -> EDA）
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(HERE.parent))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, F1, evaluate_case  # noqa: E402
from stage2_eval import build_index, resolve  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# 默认集合：10GT 全部（官方说这 10 张会作为评测案例）+ 200 训练集里最差的那些
# ★ case "6" 曾经**不在**这个列表里（注释却写「10GT 全部 9 张」），后果是它从来没进过
#   任何一次默认 A/B：它是唯一一张「引脚文字旋转 90°」的图，引脚通道整张 0 分
#   （`--cases 6 --no-vlm` 总分 53.20 / pin 0.0000）。2026-10-13 补上，见
#   `docs/RESULTS.md` §2.13（`STAGE2_PINROT` 竖直脚旋转重读）。
DEFAULT_CASES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                 "0092", "0048", "0019", "0024", "0075", "0086",
                 "0002", "0001", "0042", "0103"]

INPUT_DIR = ROOT / "work" / "_fast_in"
OUT_DIR = ROOT / "work" / "_fast_out"
BASE_DIR = ROOT / "work" / "_fast_base"
METRICS = ["component", "pin", "net_hypergraph", "net_line"]
SHORT = {"component": "元件", "pin": "引脚", "net_hypergraph": "网络", "net_line": "导线"}

# ★ 打分引擎：`official` = `schematic_parser/official.py`，逐位复刻官方 scorer
#   （2026-10-13 用官方评测机返回的那批 `*_Pred.json` 标定：comp/pin/connectivity
#   三个通道的 10 案均值与逐案 F1 **精确命中**官方 report，wire 只在 1 案上有
#   0.0078 的差 —— 那是匈牙利配对并列最优时的打破方式不同，soft_tp 完全相同）。
#   `legacy` = 老的 `metrics.py`（`bbox_tol=20`、`unkeyed_match='gt'`、
#   非 box 引脚号折叠），在**同一批官方 pred** 上给出 comp 0.946 vs 官方 0.488、
#   总分 58.16 vs 44.29 ⇒ 老尺子松得离谱，**新 A/B 一律用 official**；
#   只有要和历史 `work/_fast_base/*.json` 对比时才用 `--engine legacy`。
ENGINE = "official"


def lookup(cid, exact, digits):
    """`0092` 这种带前导零的名字，`resolve` 只试原样和去非数字，试不出 `92`。"""
    case, how = resolve(cid, exact, digits)
    if case is None:
        case, how = resolve(cid.lstrip("0") or "0", exact, digits)
    return case, how


def ensure_inputs(cases):
    """把选中的 case 的 PNG 拷到一个平目录（`run.py` 按文件名配 GT）。"""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    want, names = [], []
    for cid in cases:
        case, how = lookup(cid, exact, digits)
        if case is None:
            print("  ! 找不到 case %s (%s)" % (cid, how))
            continue
        dst = INPUT_DIR / (case.case_id + ".png")
        if not dst.exists() or dst.stat().st_size != Path(case.png_path).stat().st_size:
            dst.write_bytes(Path(case.png_path).read_bytes())
        want.append((case, dst))
        names.append(case.case_id)
    return want, names


def run_pipeline(cases, extra):
    cmd = [sys.executable, str(EDA / "submission" / "run.py"),
           "-input", str(INPUT_DIR), "-output", str(OUT_DIR),
           "-cases", ",".join(cases), "-mode", "off"]
    cmd += extra
    print("$ " + " ".join(cmd[1:]))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(EDA))
    print("  退出码 %d，墙钟 %.1f s" % (p.returncode, time.time() - t0))
    return p.returncode == 0


def find_pred(case_id):
    for name in (case_id + "_Pred.json", case_id + ".json"):
        p = OUT_DIR / "final_results" / name
        if p.exists():
            return p
    return None


def score(cases, cfg):
    rows = []
    for cid in cases:
        exact, digits = build_index(DEFAULT_DATA_ROOT)
        case, _ = lookup(cid, exact, digits)
        if case is None:
            continue
        pp = find_pred(case.case_id)
        if pp is None:
            rows.append({"case": case.case_id, "split": case.split, "err": "没有预测"})
            continue
        gt = load_json(Path(case.json_path))
        pr = load_json(pp)
        if ENGINE == "official":
            from schematic_parser import official as OFF
            sc = OFF.evaluate_case(gt, pr, case_id=case.case_id)
            chans = {"component": sc.component, "pin": sc.pin,
                     "net_hypergraph": sc.connectivity, "net_line": sc.wire}
        else:
            sc = evaluate_case(gt, pr, case_id=case.case_id, cfg=cfg)
            chans = {m: getattr(sc, m) for m in METRICS}
        rows.append({
            "case": case.case_id, "split": case.split, "score": sc.score,
            "out": str(OUT_DIR), "engine": ENGINE,
            **{m: {"f1": chans[m].f1, "tp": chans[m].tp,
                   "fp": chans[m].fp, "fn": chans[m].fn} for m in METRICS},
            "diag": dict(sc.diagnostic or {}),
        })
    return rows


def table(rows, base=None):
    print()
    print("case      split            总分  |  " +
          "  ".join("%-13s" % SHORT[m] for m in METRICS))
    print("-" * 92)
    for r in rows:
        if r.get("err"):
            print("%-9s %-16s  %s" % (r["case"], r["split"], r["err"]))
            continue
        cells = []
        for m in METRICS:
            v = r[m]["f1"]
            txt = "%.4f" % v
            if v <= 0.0:
                txt = ">> 0.0000<<"          # 爆 0 的项：这就是要找的槽点
            cells.append("%-13s" % txt)
        d = ""
        if base:
            b = base.get(r["case"])
            if b and "score" in b:
                delta = r["score"] - b["score"]
                if abs(delta) >= 0.005:
                    d = "  %+.2f" % delta
        print("%-9s %-16s %6.2f%s |  %s" % (r["case"], r["split"], r["score"], d, "  ".join(cells)))
    print("-" * 92)

    ok = [r for r in rows if not r.get("err")]
    if not ok:
        return
    print("均值(%d 张)               %6.2f  |  " % (len(ok), sum(r["score"] for r in ok) / len(ok)) +
          "  ".join("%-13.4f" % (sum(r[m]["f1"] for r in ok) / len(ok)) for m in METRICS))
    for split in ("10GT", "10_GTcase", "200_train_cases"):
        sub = [r for r in ok if r["split"] == split]
        if sub:
            print("  %-14s (%2d)      %6.2f  |  " % (split, len(sub),
                  sum(r["score"] for r in sub) / len(sub)) +
                  "  ".join("%-13.4f" % (sum(r[m]["f1"] for r in sub) / len(sub)) for m in METRICS))
    print()
    zeros = {}
    for m in METRICS:
        z = [r["case"] for r in ok if r[m]["f1"] <= 0.0]
        if z:
            zeros[SHORT[m]] = z
    if zeros:
        print("★ 爆 0 的分项（这才是要修的槽点）：")
        for k, v in zeros.items():
            print("   %s F1 = 0 共 %d 张：%s" % (k, len(v), " ".join(v)))
    else:
        print("★ 没有爆 0 的分项 ✓")
    tot = {m: sum(r[m]["tp"] for r in ok) for m in METRICS}
    print("   tp 合计：" + "  ".join("%s %d" % (SHORT[m], tot[m]) for m in METRICS))


def main(argv=None):
    ap = argparse.ArgumentParser(description="小规模快速回归")
    ap.add_argument("--cases", default="", help="逗号分隔；留空用默认 19 张")
    ap.add_argument("--score-only", action="store_true", help="不跑管线，只重新打分")
    ap.add_argument("--no-vlm", action="store_true", help="关掉 VLM（最快）")
    ap.add_argument("--extra", default="", help="透传给 run.py 的额外参数")
    ap.add_argument("--out", default="", help="独立输出目录后缀 ⇒ work/_fast_out_<后缀>"
                                              "（默认共用 work/_fast_out，会跨跑累积，"
                                              "诊断脚本读它时必须先固定目录）")
    ap.add_argument("--save", default="", help="把本次成绩存为基线 <名字>")
    ap.add_argument("--cmp", default="", help="和基线 <名字> 比")
    ap.add_argument("--list", action="store_true", help="只打印默认 case 列表")
    ap.add_argument("--engine", default="official", choices=("official", "legacy"),
                    help="official（默认）= 与官方评测机逐位一致的尺子；"
                         "legacy = 老的 metrics.py（只用于和历史基线比）")
    a = ap.parse_args(argv)

    global OUT_DIR, ENGINE
    if a.out:
        OUT_DIR = ROOT / "work" / ("_fast_out_" + a.out.strip().rstrip("/\\"))
    ENGINE = a.engine

    cases = [c.strip() for c in a.cases.split(",") if c.strip()] or DEFAULT_CASES
    if a.list:
        print(" ".join(cases))
        return 0

    _, names = ensure_inputs(cases)
    print("case 集合（%d 张）：%s" % (len(names), " ".join(names)))
    print("打分引擎：%s" % ENGINE)
    if not a.score_only:
        extra = []
        if a.no_vlm:
            extra += ["-vlm", "off"]
        if a.extra:
            extra += a.extra.split()
        if not run_pipeline(names, extra):
            print("!! 管线没跑成功，下面是已有产物的分数")

    cfg = EvalConfig()
    rows = score(names, cfg)
    base = None
    if a.cmp:
        bp = BASE_DIR / (a.cmp + ".json")
        if bp.exists():
            base = {r["case"]: r for r in json.loads(bp.read_text(encoding="utf-8"))}
            be = {r.get("engine", "legacy") for r in base.values()}
            if be and ENGINE not in be:
                print("!! 基线 %s 是用 %s 尺子打的，本次是 %s ⇒ Δ 不可比，重打基线！"
                      % (a.cmp, "/".join(sorted(be)), ENGINE))
        else:
            print("! 找不到基线 %s" % bp)
    table(rows, base)
    print("\n产物：%s" % OUT_DIR)
    if a.save:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / (a.save + ".json")).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("已存基线：%s" % (BASE_DIR / (a.save + ".json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
