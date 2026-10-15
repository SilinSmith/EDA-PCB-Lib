# -*- coding: utf-8 -*-
"""一键「新 GT 复测矩阵」。

背景（`docs/RESULTS.md` §5.119）：赛题新 QA 明确说 **10GT 还会再更新一次**，而且主办方
承认现有 GT 的引脚坐标不合 PDF 附件三的规则（**红绿导线的交接处**）。
⇒ §5.65 / §5.111 / §5.118 那些「引脚几何不可修」的结论只对**当前 GT** 成立；
新 GT 落地当天必须能**一键**把几条臂 × 两个口径重跑一遍，并确认用的是新 GT。

本工具做的事：对每个（口径 × 臂）跑一次 `tools/eval/fast.py`，把成绩存成基线 JSON，
再读回来汇总成一张矩阵表 + 一份 markdown 报告，并打印 **GT 指纹**（GT json 的 sha1）
用来证明「这次跑的是哪一版 GT」。

用法：
    python -X utf8 tools/eval/gt_retest.py                       # 全部臂 × 全部口径（官方 10 张）
    python -X utf8 tools/eval/gt_retest.py --dry                 # 只打印将要执行的命令
    python -X utf8 tools/eval/gt_retest.py --arms default,color --rulers deterministic
    python -X utf8 tools/eval/gt_retest.py --cases 1,8 --tag smoke
    python -X utf8 tools/eval/gt_retest.py --pool               # 追加池子 10 张（只跑确定性口径）

产物：`work/_gt_retest/<tag>/`：`<slug>.log`（每条命令的完整 stdout）+ `report.md` + `summary.json`
基线：`work/_fast_base/<slug>.json`（fast.py 写的，可直接 `--cmp`）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # Y:\workspace\EDA
EDA = ROOT / "eda-pcb"
WORK = ROOT / "work"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(HERE.parent))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402
from stage2_eval import build_index  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

OFFICIAL = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
POOL = ["0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]

# 臂 = 环境变量覆盖（都只影响 stage1 的引脚链；每次跑之前会先把这些键从环境里清掉）
ARMS: "dict[str, dict]" = {
    "default": {},
    "color": {"STAGE1_PIN_COLOR": "1"},
    "color_cap": {"STAGE1_PIN_COLOR": "1", "STAGE1_PIN_COLOR_CAP": "12"},
    "color_keep": {"STAGE1_PIN_COLOR": "1", "STAGE1_PIN_COLOR_KEEP_MISS": "1"},
    "color_keep_cap": {"STAGE1_PIN_COLOR": "1", "STAGE1_PIN_COLOR_CAP": "12",
                       "STAGE1_PIN_COLOR_KEEP_MISS": "1"},
    "mono": {"STAGE1_MONO_FRAME_PINS": "1"},
    "nodesolve": {"PSKEL_NODE_SOLVE": "1"},
    "nodesolve_firm": {"PSKEL_NODE_SOLVE": "1", "PSKEL_NODE_SOLVE_LAM": "0.01"},
    "nodesolve_soft": {"PSKEL_NODE_SOLVE": "1", "PSKEL_NODE_SOLVE_LAM": "0.25"},
    "nodesolve_fit8": {"PSKEL_NODE_SOLVE": "1", "PSKEL_NODE_SOLVE_FIT": "8.0"},
    "nodesolve_fit14": {"PSKEL_NODE_SOLVE": "1", "PSKEL_NODE_SOLVE_FIT": "14.0"},
    # 2026-10-14 起 `PSKEL_NODE_SOLVE` 默认已是 1（§5.123），这条臂用来复现旧行为
    "nodesolve_off": {"PSKEL_NODE_SOLVE": "0"},
    # §5.126 端点归因消融：逐个关掉 stage2 端点链上的算子，看**自由端沿轴位移**
    "nltrim0": {"STAGE2_EDGE_TRIM": "0"},
    "nlcorner0": {"STAGE2_EDGE_CORNER": "0"},
    "nlext0": {"STAGE2_EDGE_EXT": "0", "STAGE2_EDGE_EXT_ANY": "0"},
    "nlpskel0": {"STAGE1_SPLIT_JUNC_NODES": "0"},
}
ARM_DOC = {
    "default": "现行默认：常数 12px 外伸猜测",
    "color": "交结实测（队友 `pincolor`）：新 GT 下的候选正解",
    "color_cap": "交结实测 + 距锚点上限 12px",
    "color_keep": "交结实测 + 保留未实测边的旧猜测",
    "color_keep_cap": "交结实测 + 上限 12px + 保留旧猜测",
    "mono": "黑白框实测（§5.114；新 QA 说最终评分集无黑白图）",
    "nodesolve": "结点位置重解（§5.123）：结点坐标从簇均值改成各分支拟合直线的交点",
    "nodesolve_firm": "同上，λ=0.01（更信交点、几乎不正则化）",
    "nodesolve_soft": "同上，λ=0.25（更靠近原簇均值）",
    "nodesolve_fit8": "同上，方向拟合从结点外 8px 起算",
    "nodesolve_fit14": "同上，方向拟合从结点外 14px 起算",
    "nodesolve_off": "关掉结点位置重解（复现 2026-10-14 之前的旧行为；两口径均 −0.14）",
    "nltrim0": "§5.126 归因：关掉自由端回缩 `STAGE2_EDGE_TRIM`（默认 3px）",
    "nlcorner0": "§5.126 归因：关掉拐角合拢 `STAGE2_EDGE_CORNER`（默认 12px）",
    "nlext0": "§5.126 归因：关掉两个外推算子 `STAGE2_EDGE_EXT`(20) + `STAGE2_EDGE_EXT_ANY`(25)",
    "nlpskel0": "§5.126 归因：关掉 `STAGE1_SPLIT_JUNC_NODES`（只在 pskel 真结点处切）",
}
# 口径 = 透传给 submission/run.py 的额外参数
RULERS: "dict[str, list]" = {
    "delivery": ["-vlm-pin", "off"],
    "deterministic": ["-offline", "-vlm-pin", "off"],
}
RULER_DOC = {
    "delivery": "交付口径（VLM 位号/值/名开、引脚通道关）",
    "deterministic": "确定性口径（关掉全部 LLM）",
}
METRICS = ["component", "pin", "net_hypergraph", "net_line"]
SHORT = {"component": "元件", "pin": "引脚", "net_hypergraph": "网络", "net_line": "导线"}
MANAGED = sorted({k for a in ARMS.values() for k in a})


def gt_fingerprint(cases):
    """对 case 列表里的 GT json 取指纹：能证明「这次跑的是哪一版 GT」。"""
    index = {}
    for case in discover(DEFAULT_DATA_ROOT):
        index.setdefault(case.case_id, case)
    h = hashlib.sha1()
    rows = []
    for cid in cases:
        case = index.get(cid)
        if case is None:
            rows.append((cid, "MISSING"))
            h.update(("%s:MISSING" % cid).encode())
            continue
        raw = Path(case.json_path).read_bytes()
        d = hashlib.sha1(raw).hexdigest()[:12]
        rows.append((cid, d))
        h.update(("%s:%s" % (cid, d)).encode())
    return h.hexdigest()[:16], rows


GT_INDEX = WORK / "_gt_retest" / "_gt_index.json"


def load_gt_index():
    if GT_INDEX.exists():
        try:
            data = json.loads(GT_INDEX.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        if isinstance(data, dict):
            return data
    return {}


def record_gt_index(rows):
    """把逐案 GT 指纹记进索引，返回「相对上次变了/新增」的 case 列表。

    逐 case 记账是有意的：整批指纹会随**跑了哪几张**变化，单张哈希不会。
    所以「主办方换了 GT」这条判断只在逐案哈希上成立。
    """
    old = load_gt_index()
    changed = [(cid, old.get(cid, "(新)"), d) for cid, d in rows if old.get(cid) != d]
    merged = dict(old)
    for cid, d in rows:
        merged[cid] = d
    GT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    GT_INDEX.write_text(json.dumps(merged, ensure_ascii=False, indent=1,
                                   sort_keys=True), encoding="utf-8")
    return changed


def run_one(ruler, arm, cases, tag, dry=False):
    slug = "%s_%s" % (ruler, arm)
    extra = " ".join(RULERS[ruler])
    cmd = [sys.executable, str(EDA / "tools" / "eval" / "fast.py"),
           "--cases", ",".join(cases), "--out", slug, "--extra", extra, "--save", slug]
    env = {k: v for k, v in os.environ.items() if k not in MANAGED}
    env.update(ARMS[arm])
    env["PYTHONIOENCODING"] = "utf-8"
    shown = " ".join("%s=%s" % (k, v) for k, v in sorted(ARMS[arm].items())) or "(无覆盖)"
    print("\n$ %s\n  env: %s" % (" ".join(cmd[1:]), shown))
    if dry:
        return None
    outdir = WORK / "_gt_retest" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / (slug + ".log")
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=str(EDA), env=env, stdout=fh, stderr=subprocess.STDOUT)
    print("  退出码 %d，墙钟 %.1f s ⇒ %s" % (p.returncode, time.time() - t0, log))
    bp = WORK / "_fast_base" / (slug + ".json")
    if not bp.exists():
        print("  ! 没有基线文件 %s" % bp)
        return None
    return json.loads(bp.read_text(encoding="utf-8"))


def summarise(rows):
    ok = [r for r in rows if "score" in r]
    if not ok:
        return None
    out = {"n": len(ok), "score": sum(r["score"] for r in ok) / len(ok)}
    for m in METRICS:
        out[m] = sum(r[m]["f1"] for r in ok) / len(ok)
        out[m + "_tp"] = sum(r[m]["tp"] for r in ok)
    out["pin_zero"] = [r["case"] for r in ok if r["pin"]["f1"] <= 0.0]
    out["per_case"] = {r["case"]: r["score"] for r in ok}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="一键新 GT 复测矩阵")
    ap.add_argument("--cases", default=",".join(OFFICIAL), help="官方案例列表（默认 10GT）")
    ap.add_argument("--arms", default=",".join(ARMS), help="臂（默认全部）：" + ",".join(ARMS))
    ap.add_argument("--rulers", default=",".join(RULERS), help="口径（默认全部）：" + ",".join(RULERS))
    ap.add_argument("--pool", action="store_true", help="追加池子 10 张（只跑确定性口径）")
    ap.add_argument("--tag", default="", help="本次跑的目录名（默认时间戳）")
    ap.add_argument("--dry", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--fingerprint-only", dest="fingerprint_only", action="store_true",
                    help="只算/记录 GT 指纹，不跑流水线")
    a = ap.parse_args(argv)

    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    rulers = [x.strip() for x in a.rulers.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            print("! 未知臂 %s" % x); return 2
    for x in rulers:
        if x not in RULERS:
            print("! 未知口径 %s" % x); return 2

    tag = a.tag or time.strftime("%Y%m%d-%H%M%S")
    fp, fp_rows = gt_fingerprint(cases)
    changed = [] if a.dry else record_gt_index(fp_rows)
    print("== 新 GT 复测矩阵 ==")
    print("case 集合（%d 张）：%s" % (len(cases), " ".join(cases)))
    print("GT 指纹：**%s**（逐案哈希见下；整批值随跑了哪几张变，别跨 case 集比）" % fp)
    for cid, d in fp_rows:
        mark = ""
        for ccid, was, now in changed:
            if ccid == cid:
                mark = "   ← 变了（上次 %s）" % was
        print("   %-8s %s%s" % (cid, d, mark))
    if changed:
        print("⇒ **检测到 GT 变更**：%s" % "、".join(c for c, _, _ in changed))
    else:
        print("⇒ 逐案 GT 与上次记录一致（%s）" % GT_INDEX.name)
    if a.fingerprint_only:
        return 0
    print("口径：" + "；".join("%s = %s" % (r, RULER_DOC[r]) for r in rulers))

    results = {}
    for ruler in rulers:
        for arm in arms:
            rows = run_one(ruler, arm, cases, tag, dry=a.dry)
            if rows is not None:
                results["%s|%s" % (ruler, arm)] = summarise(rows)
    pool_results = {}
    if a.pool and not a.dry:
        for arm in arms:
            rows = run_one("deterministic", arm, POOL, tag)
            if rows is not None:
                pool_results[arm] = summarise(rows)

    if a.dry:
        return 0

    # ---------------- 报告 ----------------
    lines = ["# 新 GT 复测矩阵（tag `%s`）" % tag, "",
             "- case 集合（%d）：`%s`" % (len(cases), ",".join(cases)),
             "- **GT 指纹：`%s`**（整批 sha1；逐案哈希：%s）" % (
                 fp, " ".join("%s=%s" % (c, d) for c, d in fp_rows)),
             "- GT 变更检测：%s" % ("**变了** → %s" % "、".join(c for c, _, _ in changed)
                                 if changed else "与 `%s` 记录一致（未换）" % GT_INDEX.name),
             "- 打分引擎：`protocol2`（`tools/eval/fast.py` 默认）", ""]
    for ruler in rulers:
        if not any(k.startswith(ruler + "|") for k in results):
            continue
        lines += ["## 口径：%s（`%s`）" % (RULER_DOC[ruler], " ".join(RULERS[ruler])), "",
                  "| 臂 | 说明 | 总分 | 元件 | 引脚 | 网络 | 导线 | 引脚 tp | 爆 0 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        base = results.get(ruler + "|default")
        for arm in arms:
            s = results.get(ruler + "|" + arm)
            if not s:
                continue
            d = "" if (base is None or arm == "default") else " (%+.2f)" % (s["score"] - base["score"])
            lines.append("| `%s` | %s | **%.2f**%s | %.4f | %.4f | %.4f | %.4f | %d | %s |" % (
                arm, ARM_DOC.get(arm, ""), s["score"], d, s["component"], s["pin"],
                s["net_hypergraph"], s["net_line"], s["pin_tp"],
                " ".join(s["pin_zero"]) or "—"))
        lines.append("")
    if pool_results:
        lines += ["## 池子 10 张（确定性口径；域外代理，只看方向）", "",
                  "| 臂 | 总分 | 元件 | 引脚 | 网络 | 导线 | 引脚 tp |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm in arms:
            s = pool_results.get(arm)
            if not s:
                continue
            lines.append("| `%s` | **%.2f** | %.4f | %.4f | %.4f | %.4f | %d |" % (
                arm, s["score"], s["component"], s["pin"], s["net_hypergraph"], s["net_line"], s["pin_tp"]))
        lines.append("")
    lines += ["## 怎么读", "",
              "- 新 GT 落地后先跑 `--fingerprint-only`：**逐案哈希**变了才说明这一版数据是新的"
              "（整批指纹会随跑了哪几张变，别跨 case 集比）。",
              "- 判据（§5.107 定的规矩）：**默认关 + 两口径 A/B + 有净增益才翻默认**；",
              "  只有 `default` 一列是当前交付路径，其余臂都是候选。",
              "- 单跑某条臂看逐案明细：`python -X utf8 tools/eval/fast.py --score-only "
              "--out <rulers>_<臂> --cases \"1,2,3\"`", ""]
    text = "\n".join(lines)
    outdir = WORK / "_gt_retest" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(text, encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(
        {"tag": tag, "gt_fingerprint": fp, "cases": cases,
         "gt_per_case": {c: d for c, d in fp_rows},
         "gt_changed": [{"case": c, "was": w, "now": n} for c, w, n in changed],
         "results": results, "pool": pool_results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + text)
    print("报告：%s" % (outdir / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
