# -*- coding: utf-8 -*-
"""合成集端到端尺子：把 `合成原理图数据集` 的图跑一遍**完整管线**，用项目自己的打分器对着
`gt/` 打分，并额外交出「绑定计量」——位号键命中率、引脚号命中率、引脚名命中率。

为什么专门做这把尺子（2026-10-17，用户 m57324 第⑤项 "合成资源投向绑定/文本通道"）：

  * `合成原理图数据集`（3000 张）的 GT 与赛题**同一 schema**（顶层就是
    `{components, pins, nets}`，`tools/data/check_synth_json.py` 已在守这条），
    所以 `schematic_parser.metrics` 可以直接吃；
  * 合成集的**位号/引脚号/引脚名是完美标注**，不存在「标错」噪声 ⇒ 这里丢的分
    一定是**我们自己读错/绑错**，正好把「位号绑定」这条瓶颈从检测噪声里隔离出来；
  * 反过来，它**不是**线上分布的替身：合成是干净渲染图，故它只能当**回归尺子**
    和**定点加压台**（要复现线上失分，得再用 `synth/style.py` 的模糊/旋转档造压力图）。

用法：

    python -X utf8 tools\\eval\\synth_bench.py --count 30 --no-vlm --out synthbench
    python -X utf8 tools\\eval\\synth_bench.py --count 30 --score-only --out synthbench
    python -X utf8 tools\\eval\\synth_bench.py --list --count 30
    python -X utf8 tools\\eval\\synth_bench.py --root "Y:\\workspace\\EDA\\合成原理图demo" --count 24

★★ 必须先归一化「文字尺度」再打分（`--scale`，默认 2.2）★
------------------------------------------------------------------
2026-10-17 实测：合成图的**文字只有真图的一半高**，直跑等于用一把坏尺子量识别：

    尺子            检出率    位号    引脚号    引脚名    值    网络标签
    scale 1.0       66.7%     66%     33%       76%      83%   57%
    scale 2.0       79.9%     82%     66%       82%      88%   43%
    scale 2.2       84.6%     88%     73%       81%      93%   71%
    scale 3.0       92.1%     91%     87%       97%      95%   71%

原因不是「检测器差」而是**渲染尺度差**：真图（`work/_fast_in/*.png`，13197 个文字框）
文字高度 p10/p50/p90 = 7/12/21 px，而合成图 `texts.jsonl` 里 `pin_number` 高 **4.3**、
`pin_name` 5.0、`designator`/`value` 5.7 px；`stage2/text.py` 的 CRNN 是在真图尺度上
训出来的，5px 的字喂进去当然读不出。所以本尺子默认 `--scale 2.2`（把合成图放大到
文字高度 ≈12px 再跑，并**同比例缩放 GT 的 bbox/point/edges**），否则会把「渲染器把字
画小了」误判成「我们识别不行」。

输出：逐案总分 + 四通道；末尾绑定计量（按元件类型分组）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]      # 仓库根 Y:\workspace\EDA
EDA = ROOT / "eda-pcb"
if str(EDA) not in sys.path:
    sys.path.insert(0, str(EDA))
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser import metrics as M  # noqa: E402

DEFAULT_ROOT = Path(r"Y:\workspace\EDA\合成原理图数据集")
IN_DIR = ROOT / "work" / "_synth_in"
METRICS = ("component", "pin", "net_hypergraph", "net_line")
SHORT = {"component": "元件", "pin": "引脚", "net_hypergraph": "网络", "net_line": "导线"}


def load(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_cases(root: Path, count: int):
    gts = sorted((root / "gt").glob("*.json"), key=lambda p: p.name)
    out = []
    for g in gts[:count]:
        cid = g.stem
        png = root / "images" / (cid + ".png")
        if png.exists():
            out.append((cid, png, g))
    return out


def scale_doc(doc, k: float):
    """把 GT 文档的所有几何坐标按同一系数缩放（与放大后的图对齐）。"""
    if k == 1.0:
        return doc
    out = json.loads(json.dumps(doc))
    for c in (out.get("components") or {}).values():
        b = (c or {}).get("bbox")
        if b:
            c["bbox"] = [float(v) * k for v in b]
    for pl in (out.get("pins") or {}).values():
        for p in (pl or {}).values():
            pt = (p or {}).get("point")
            if isinstance(pt, dict):
                pt["x"], pt["y"] = float(pt["x"]) * k, float(pt["y"]) * k
    for n in (out.get("nets") or {}).values():
        for e in ((n or {}).get("edges") or {}).values():
            for q in (e or []):
                if isinstance(q, dict):
                    q["x"], q["y"] = float(q["x"]) * k, float(q["y"]) * k
    return out


def ensure_inputs(cases, scale: float = 1.0):
    """按 `scale` 把图放进 `work/_synth_in/`（放大到文字尺度与真图一致）。"""
    IN_DIR.mkdir(parents=True, exist_ok=True)
    if scale == 1.0:
        for cid, png, _ in cases:
            dst = IN_DIR / (cid + ".png")
            if not dst.exists() or dst.stat().st_size != png.stat().st_size:
                shutil.copyfile(png, dst)
        return
    import cv2
    from stage2.text import imread_unicode          # 路径含中文 ⇒ 不能用 cv2.imread
    for cid, png, _ in cases:
        dst = IN_DIR / (cid + ".png")
        if dst.exists():
            continue
        img = imread_unicode(str(png))
        if img is None:
            continue
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(dst), big)                  # IN_DIR 一定是纯 ASCII 路径


def run_pipeline(names, out_dir, no_vlm, extra):
    cmd = [sys.executable, str(EDA / "submission" / "run.py"),
           "-input", str(IN_DIR), "-output", str(out_dir),
           "-cases", ",".join(names), "-mode", "off"]
    if no_vlm:
        cmd += ["-vlm", "off"]
    cmd += list(extra or [])
    print("$ " + " ".join(cmd[1:]))
    rc = subprocess.run(cmd, cwd=str(EDA)).returncode
    print("  退出码 %d" % rc)
    return rc == 0


def binding_stats(gt, pr, cfg):
    """绑定计量：位号键 / 引脚号 / 引脚名。"""
    gt_c, pr_c = gt.get("components") or {}, pr.get("components") or {}
    gt_p, pr_p = gt.get("pins") or {}, pr.get("pins") or {}
    # 与打分器同一套匹配（键优先 + 坐标兜底），保证口径一致
    _, keymap = M.assign_components(gt, pr, cfg, {})
    pr2 = M.rewrite_keys(pr, keymap)
    pr_c2, pr_p2 = pr2.get("components") or {}, pr2.get("pins") or {}
    by_type = defaultdict(lambda: Counter())
    for ck, gv in gt_c.items():
        t = (gv or {}).get("type") or "?"
        st = by_type[t]
        st["gt"] += 1
        hit = ck in pr_c2
        st["key_hit" if hit else "key_miss"] += 1
        gpp = gt_p.get(ck) or {}
        st["gt_pin"] += len(gpp)
        if not hit:
            continue
        have = pr_p2.get(ck) or {}
        nums = {M.pin_key_to_number(k) for k in have}
        names = {M.pin_key_to_number(k): (v or {}).get("pinname") for k, v in have.items()}
        for pk, gvv in gpp.items():
            num = M.pin_key_to_number(pk)
            if num in nums:
                st["num_hit"] += 1
                if names.get(num) == (gvv or {}).get("pinname"):
                    st["name_hit"] += 1
    return by_type


def main(argv=None):
    ap = argparse.ArgumentParser(description="合成集端到端尺子 + 绑定计量")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", default="synthbench")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--extra", default="")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--scale", type=float, default=2.2,
                    help="放大倍数（默认 2.2：把合成图文字高度归一到真图的 ~12px）")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    global IN_DIR
    if a.scale != 1.0:
        IN_DIR = ROOT / "work" / ("_synth_in_x%g" % a.scale)
    root = Path(a.root)
    cases = list_cases(root, a.count)
    if not cases:
        print("! %s 下没有可用的 <gt, images> 对" % root)
        return 1
    if a.list:
        print(" ".join(c[0] for c in cases))
        return 0

    out_dir = ROOT / "work" / ("_fast_out_" + a.out.strip().rstrip("/\\"))
    ensure_inputs(cases, a.scale)
    if not a.score_only:
        run_pipeline([c[0] for c in cases], out_dir, a.no_vlm, a.extra.split() if a.extra else [])

    cfg = M.EvalConfig()
    rows, agg, tot = [], defaultdict(Counter), Counter()
    for cid, _png, gpath in cases:
        pp = out_dir / "final_results" / (cid + "_Pred.json")
        if not pp.exists():
            print("%-14s 没有预测" % cid)
            continue
        gt, pr = scale_doc(load(gpath), a.scale), load(pp)
        sc = M.evaluate_case(gt, pr, case_id=cid, cfg=cfg)
        rows.append((cid, sc, {m: getattr(sc, m) for m in METRICS}))
        bs = binding_stats(gt, pr, cfg)
        for t, st in bs.items():
            agg[t].update(st)
            tot.update(st)
    if not rows:
        return 1

    print()
    print("case              总分  |  " + "  ".join("%-10s" % SHORT[m] for m in METRICS))
    print("-" * 78)
    for cid, sc, ch in rows:
        print("%-14s %6.2f  |  " % (cid, sc.score) +
              "  ".join("%-10.4f" % ch[m].f1 for m in METRICS))
    n = len(rows)
    print("-" * 78)
    print("均值(%d 张)      %6.2f  |  " % (n, sum(r[1].score for r in rows) / n) +
          "  ".join("%-10.4f" % (sum(r[2][m].f1 for r in rows) / n) for m in METRICS))
    print()
    print("绑定计量（合成集：位号/号/名都是完美标注 ⇒ 丢分必是我们读错/绑错）")
    print("类型         GT元件 键命中 键命中率 | GT脚  号命中 号命中率 名命中 名命中率")
    for t in sorted(agg, key=lambda k: -agg[k]["gt"]):
        st = agg[t]
        print("%-12s %5d %5d %7.1f%% | %5d %5d %7.1f%% %5d %7.1f%%" % (
            t, st["gt"], st["key_hit"], 100.0 * st["key_hit"] / max(1, st["gt"]),
            st["gt_pin"], st["num_hit"], 100.0 * st["num_hit"] / max(1, st["gt_pin"]),
            st["name_hit"], 100.0 * st["name_hit"] / max(1, st["gt_pin"])))
    print("%-12s %5d %5d %7.1f%% | %5d %5d %7.1f%% %5d %7.1f%%" % (
        "合计", tot["gt"], tot["key_hit"], 100.0 * tot["key_hit"] / max(1, tot["gt"]),
        tot["gt_pin"], tot["num_hit"], 100.0 * tot["num_hit"] / max(1, tot["gt_pin"]),
        tot["name_hit"], 100.0 * tot["name_hit"] / max(1, tot["gt_pin"])))
    print("产物：%s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
