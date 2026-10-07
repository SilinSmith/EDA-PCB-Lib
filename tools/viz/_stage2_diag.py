# -*- coding: utf-8 -*-
"""stage2 诊断：把 GT 的位号/参数/引脚名对照到 stage2 实际读到的文字上。

    cd Y:\workspace\EDA\eda-pcb
    python Y:\workspace\EDA\work\_stage2_diag.py 0059 [--n 6] [--ckpt ...]

回答的问题是：「位号赋值率只有 6%」到底是**认不出**（CRNN 读错）还是**对不上**
（几何/角色分配选错了框）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json  # noqa: E402
from stage2.text import TextEngine, imread_unicode, read_stage1_texts  # noqa: E402
from stage2.assign import _best_read, dist_to_rect, fit_designator, fit_value  # noqa: E402

STAGE1 = ROOT / "work" / "stage1_all14"


def gt_case(case, data_root=DEFAULT_DATA_ROOT):
    for c in discover(data_root, SPLITS):
        if c.case_id == case:
            return c
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--n", type=int, default=6, help="每个元件打印最近的 N 个文字框")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args(argv)

    st1 = json.loads((STAGE1 / ("%s.stage1.json" % args.case)).read_text(encoding="utf-8"))
    H = float(st1["image"]["h"])
    gt = load_json(gt_case(args.case).json_path)
    bgr = imread_unicode(st1["image"]["path"])
    boxes = read_stage1_texts(st1)

    eng = TextEngine(ckpt=args.ckpt) if args.ckpt else TextEngine()
    print("[diag] %s  %s" % (args.case, eng.describe()))
    reads = eng.read(bgr, boxes)
    print("文字框 %d 个" % len(reads))

    # GT 元件（过滤掉 ∅ 假号）
    gt_comps = []
    for key, c in (gt.get("components") or {}).items():
        b = c.get("bbox")
        if not b:
            continue
        real = not key.startswith("\u2205")
        box_img = (b[0], H - b[3], b[2], H - b[1])
        cx, cy = (box_img[0] + box_img[2]) / 2.0, (box_img[1] + box_img[3]) / 2.0
        gt_comps.append({"key": key, "type": c.get("type"), "name": c.get("Name"),
                         "value": c.get("value"), "box": box_img, "c": (cx, cy),
                         "real": real})
    # stage1 元件（用来查分配结果）
    pred = json.loads((ROOT / "work" / "stage2_t1" / ("%s.json" % args.case))
                      .read_text(encoding="utf-8")) if (ROOT / "work" / "stage2_t1" / ("%s.json" % args.case)).exists() else {}

    print("\n%-8s %-14s %-6s %-8s | 最近文字框读法" % ("GT位号", "GT Name", "type", "预测"))
    print("-" * 110)
    n_hit = 0
    for g in sorted(gt_comps, key=lambda x: (not x["real"], x["key"]))[:40]:
        if not g["real"]:
            continue
        ds = [dist_to_rect(g["c"][0], g["c"][1], (r["bbox"][0], r["bbox"][1],
                                                  r["bbox"][2], r["bbox"][3])) for r in reads]
        order = sorted(range(len(reads)), key=lambda i: ds[i])[: args.n]
        # 预测：找中心最近的预测元件
        pk = ""
        if pred.get("components"):
            best = None
            for k, v in pred["components"].items():
                b = v.get("bbox")
                if not b:
                    continue
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                d = ((cx - g["c"][0]) ** 2 + (cy - g["c"][1]) ** 2) ** 0.5
                if best is None or d < best[0]:
                    best = (d, k, v.get("Name"), v.get("value"), v.get("type"))
            if best:
                pk = "%s/%s/%s@%.0f" % (best[1], best[3], best[4], best[0])
        print("%-8s %-14s %-6s %-8s" % (g["key"], str(g["name"])[:14], g["type"], pk))
        for i in order:
            r = reads[i]
            rd = " | ".join("%s=%r(%.2f)" % (k, v[0], v[1]) for k, v in r["reads"].items())
            print("      d=%6.1f  %-46s  %s" % (ds[i], str(tuple(round(x) for x in r["bbox"])), rd))
            if ds[i] <= 6.0:
                n_hit += 1
    print("\n(<=6px 内有文字框的 GT 元件 %d 个)" % n_hit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
