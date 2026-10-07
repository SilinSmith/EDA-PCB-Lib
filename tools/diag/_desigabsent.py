"""决定性一问：那些「我们读不出位号」的 GT box，**图上到底有没有位号**？

已实证的三个例子（把裁片放大到几乎整张图，模型仍回 NONE）：
  * `0048-Datasheet.png`：IC 框旁只有引脚名 VCC/VSEN/…，**没有任何位号**；
  * `0057-Datasheet.png`：只有器件型号 `S-8225B`，**没有位号**；
  * `0102-jlc.png`：符号名画的是 `Flash`，型号 `W25Q16JVUXIQ`，**没有 `U3`**。
而 `0143-jlc.png`（真竞赛图）的 `U1` 画得清清楚楚。

本脚本把这件事**量化**：从「我们键是占位符 `pNNN`」的那一桶里抽样，对每个 GT 框取
`±--pad` 的裁片，用一个**明确区分位号 / 型号 / 符号名**的提示词去问。三种结局：
  * 回 `NONE`            ⇒ 图上真的没印 ⇒ **不可赢**，别再调读取链路了；
  * 回 GT 的键            ⇒ 图上印了、我们漏了 ⇒ **可赢**，去修裁剪/提示词；
  * 回别的合法位号（`K11` 而 GT 是 `U5`）⇒ 印了但 GT 被重编过 ⇒ **不可赢**。

用法：
  python work\\tools\\diag\\_desigabsent.py --n 45 --pad 150
  python work\\tools\\diag\\_desigabsent.py --source Datasheet --n 20
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

import cv2                                                                # noqa: E402
import numpy as np                                                        # noqa: E402
from vlm import client                                                    # noqa: E402
from vlm.enrich import DESIG_RE                                           # noqa: E402
from schematic_parser import data as D                                    # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, assign_components, canon_type, normalise_doc,
)

PLACEHOLDER = re.compile(r"^(?:p\d{3,4}|\u2205\d+)$", re.I)

#: 与 `enrich.DESIG_PROMPT` 的关键差别：①**明说位号可能画在框内**（实测 `0019 u1` 就是
#: 框内左上角）；②把「怎么和型号/符号名区分」讲清楚（`LM385`/`MAX232`/`S-8225B`/`Flash`
#: 都长得像位号，`DESIG_RE = ^[A-Za-z]{1,3}\d{1,4}$` 全都放行）。
PROMPT = (
    "这是电路原理图里**一个元器件**（芯片 / 模块 / 连接器）的局部放大图，"
    "画面中心就是它的外框。\n"
    "请**只找它的位号**（reference designator，形如 U1、IC2、J3、SW1、Q1、K11）："
    "**字母前缀 + 数字**，通常很短（2~4 个字符）。\n"
    "位号可能印在外框**左上角外侧**、**右上角外侧**、**正下方**，"
    "也可能印在**外框内部的左上角**。请看全。\n"
    "**不要**回下面这些东西：\n"
    "  · 器件型号 / part number（例如 S-8225B、LM385、MAX232、W25Q16JVUXIQ、RP2040）；\n"
    "  · 符号名（例如 Flash、Key）；\n"
    "  · 引脚名（VCC、GND、CLK）；\n"
    "  · 引脚号（纯数字，如 1、2、16）。\n"
    "只回位号本身，一行，不要解释。**真的一个位号都没有就回 NONE。**")


def _bb(v):
    b = (v or {}).get("bbox") or (v or {}).get("box")
    return [float(x) for x in b[:4]] if b and len(b) >= 4 else None


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    if i <= 0:
        return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def source_of(stem: str) -> str:
    return stem.split("-", 1)[1].rsplit("_target", 1)[0] if "-" in stem else "10GT(?)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fusion_out220no4"))
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--pad", type=float, default=150.0)
    ap.add_argument("--source", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    pdir = Path(a.dir)
    cfg = EvalConfig()
    pool = []
    for c in D.discover(D.DEFAULT_DATA_ROOT):
        if not c.json_path:
            continue
        src = source_of(Path(c.json_path).stem)
        if a.source and a.source.lower() not in src.lower():
            continue
        f = pdir / f"{c.case_id}.json"
        if not f.exists():
            continue
        gt = normalise_doc(D.load_json(c.json_path))
        pr = normalise_doc(__import__("json").loads(f.read_text(encoding="utf-8")))
        _f1, km = assign_components(gt, pr, cfg, {})
        matched = set(km.values())
        ours = {k: _bb(v) for k, v in pr["components"].items()}
        ours = {k: b for k, b in ours.items() if b}
        for k, v in gt["components"].items():
            if canon_type((v or {}).get("type")) not in D.IC_TYPES or k in matched:
                continue
            gb = _bb(v)
            hit = sorted(((_iou(gb, b), pk) for pk, b in ours.items() if _iou(gb, b) > 0.05),
                         reverse=True)
            if not hit or not PLACEHOLDER.match(hit[0][1]):
                continue
            pool.append((c, src, k, gb))

    rnd = random.Random(a.seed)
    rnd.shuffle(pool)
    sample = pool[:a.n]
    print("占位符桶共 %d 个 GT box（--dir %s），抽样 %d 个，pad=%.0f\n"
          % (len(pool), pdir.name, len(sample), a.pad))

    res = Counter()
    bysrc = defaultdict(Counter)
    for c, src, gk, gb in sample:
        bgr = cv2.imdecode(np.fromfile(c.png_path, dtype="uint8"), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        ib = [gb[0], H - gb[3], gb[2], H - gb[1]]
        x0 = max(0, int(ib[0] - a.pad)); y0 = max(0, int(ib[1] - a.pad))
        x1 = min(W, int(ib[2] + a.pad)); y1 = min(H, int(ib[3] + a.pad))
        crop = bgr[y0:y1, x0:x1]
        raw, dt = client.ask(crop, PROMPT, model="")
        first = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
        clean = re.sub(r"[^0-9A-Za-z?]", "", first)
        if clean.upper() == "NONE":
            verdict = "图上没有位号(不可赢)"
        elif clean.lower() == gk.lower():
            verdict = "印了、我们漏了(可赢)"
        elif DESIG_RE.match(clean):
            verdict = "印了别的(重编过)"
        else:
            verdict = "回了非位号: %s" % (clean[:14] or "(空)")
        res[verdict.split(":")[0]] += 1
        bysrc[src][verdict.split(":")[0]] += 1
        print("  %-6s %-16s GT=%-8s pad裁片=%dx%d  raw=%-14r → %s"
              % (c.case_id, src, gk, x1 - x0, y1 - y0, first[:14], verdict))

    print("\n=== 汇总（%d 个「我们读不出位号」的 GT box）===" % len(sample))
    for k, v in res.most_common():
        print("  %-22s %3d  %5.1f%%" % (k, v, 100.0 * v / max(1, len(sample))))
    print("\n按来源：")
    for src in sorted(bysrc, key=lambda s: -sum(bysrc[s].values())):
        n = sum(bysrc[src].values())
        print("  %-18s %3d | %s" % (src, n, "  ".join(
            "%s=%d" % (k, v) for k, v in bysrc[src].most_common())))


if __name__ == "__main__":
    main()
