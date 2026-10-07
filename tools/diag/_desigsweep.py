"""位号（器件键）识别精度的**直接**基准 + 裁剪外扩网格扫描。

为什么要这个：官方 9 张里大 IC 一个都没漏检（IoU 0.97~0.99），死的全是**位号**；
而位号是 `component_f1`（30%）与 `net_hypergraph_f1`（35%）的**共同闸门**。
2026-10 实测发现：喂进识别器的裁片**肉眼完全清晰**（`C160`/`C153`/`C155` …），
模型却读成 `C10`/`C53`/`C15`，所以这是**识别精度**问题，不是检测/取景问题。

做法：对官方 9 张，把「我们的元件」与 GT 元件按 bbox IoU 配对，
取出 `runs/stage2/_assigned/<case>.json` 里该元件的**位号原框**，
再按一组 `pad_box` 参数裁出来送 `Recognizer.read_many`，与 GT 键逐字比。
⇒ 一把尺子量「位号 exact」，并且能扫出外扩的最优配置。

用法：
    python work\\tools\\diag\\_desigsweep.py                 # 默认网格
    python work\\tools\\diag\\_desigsweep.py --dir work\\_fast_out_off9_pc0em
    python work\\tools\\diag\\_desigsweep.py --only 1        # 只看某张
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "eda-pcb" / "stage2"))

import text as T                                            # noqa: E402

DSROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
GT9 = ("1", "2", "3", "4", "5", "7", "8", "9", "10")


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_of(case):
    for p in DSROOT.rglob(case + ".json"):
        if p.parent.name == case:
            return json.load(open(p, encoding="utf-8"))
    return None


def collect(dirp: Path, only=None):
    """-> [(case, cid, gt_key, box_img, base_read)]   box 是**图像空间**。"""
    out = []
    for case in (only and [only] or GT9):
        g = gt_of(case)
        if g is None:
            print("  !! 找不到 GT", case)
            continue
        s1 = json.load(open(dirp / "runs" / "stage1" / (case + ".stage1.json"), encoding="utf-8"))
        H = float(s1["image"]["h"])
        try:
            a = json.load(open(dirp / "runs" / "stage2" / "_assigned" / (case + ".json"),
                              encoding="utf-8"))
        except FileNotFoundError:
            print("  !! 没有 _assigned，先跑一次 run.py：", case)
            continue
        bbox = {c["id"]: c.get("bbox_json") or c.get("bbox_img") for c in s1["components"]}
        gtbb = {}
        for k, v in (g.get("components") or {}).items():
            bb = v.get("bbox")
            if bb:
                gtbb[k] = bb
        for cid, rec in (a.get("comps") or {}).items():
            dsg = rec.get("designator") or {}
            box = dsg.get("box")
            if not box:
                continue
            bb = bbox.get(cid)
            if not bb:
                continue
            # 找 IoU 最大的 GT 元件当答案
            best, bk = 0.0, None
            for k, gb in gtbb.items():
                v = _iou(bb, gb)
                if v > best:
                    best, bk = v, k
            if bk is None or best < 0.5:
                continue
            out.append((case, cid, bk, tuple(float(v) for v in box),
                        (dsg.get("text") or "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out_off9_pc0em"))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    samples = collect(Path(args.dir), args.only)
    if not samples:
        print("没有样本")
        return
    print("样本 %d 个（官方 9 张里「有 GT 元件与之重叠」的位号）" % len(samples))
    from collections import Counter
    print("  按 case:", dict(Counter(s[0] for s in samples)))
    print("  基线（产物里已经写下的读数）exact = %.4f (%d/%d)"
          % (sum(1 for s in samples if s[4] == s[2]) / len(samples),
             sum(1 for s in samples if s[4] == s[2]), len(samples)))

    eng = T.TextEngine()
    gray = {}
    for case in sorted({s[0] for s in samples}):
        s1 = json.load(open(Path(args.dir) / "runs" / "stage1" / (case + ".stage1.json"),
                            encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(str(Path(s1["image"]["path"])), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        gray[case] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (水平定长, 水平比例, 垂直定长, 垂直比例)
    grid = [
        (2.0, 0.0, 3.0, 0.20),      # 现默认
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 2.0, 0.0),
        (2.0, 0.0, 4.0, 0.0),
        (2.0, 0.0, 6.0, 0.0),
        (2.0, 0.0, 3.0, 0.0),
        (4.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 6.0, 0.0),
        (6.0, 0.0, 6.0, 0.0),
        (0.0, 0.10, 0.0, 0.10),
        (0.0, 0.12, 0.0, 0.25),     # = crop_gray 自己的比例（不再二次外扩）
        (2.0, 0.12, 2.0, 0.25),
        (2.0, 0.0, 3.0, 0.10),
    ]
    print()
    print("%-26s %-8s %s" % ("pad(min_x,frac_x,min_y,frac_y)", "exact", "错例"))
    best = None
    for cfg in grid:
        mx, fx, my, fy = cfg
        items, keys = [], []
        for i, (case, cid, gk, box, base) in enumerate(samples):
            cb = T.pad_box(box, mx, fx, pad_min_y=my, pad_frac_y=fy)
            items.append((cb, "designator"))
            keys.append(i)
        got = eng.recognizer().read_many(gray[samples[0][0]], items) \
            if len({s[0] for s in samples}) == 1 else None
        if got is None:
            # 逐 case 批量（read_many 只吃一张图）
            got = [None] * len(samples)
            idx = {}
            for i, (case, cid, gk, box, base) in enumerate(samples):
                idx.setdefault(case, []).append(i)
            for case, ids in idx.items():
                sub = [(T.pad_box(samples[i][3], mx, fx, pad_min_y=my, pad_frac_y=fy),
                        "designator") for i in ids]
                res = eng.recognizer().read_many(gray[case], sub)
                for i, (txt, conf) in zip(ids, res):
                    got[i] = (txt, conf)
        ok = 0
        wrong = []
        for s, (txt, conf) in zip(samples, got):
            if txt == s[2]:
                ok += 1
            else:
                wrong.append("%s/%s>%s" % (s[0], s[2], txt))
        acc = ok / len(samples)
        line = "%-26s %-8.4f %s" % (str(cfg), acc, " ".join(wrong[:6]))
        print(line)
        if best is None or acc > best[0]:
            best = (acc, cfg)
    print()
    print("最优: %s  exact=%.4f" % (best[1], best[0]))


if __name__ == "__main__":
    main()
