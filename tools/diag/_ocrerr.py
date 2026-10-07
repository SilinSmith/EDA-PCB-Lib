# -*- coding: utf-8 -*-
"""CRNN 真实错误典例采集器（只读，不改任何产物）。

目的
----
把「识别器在真实图上到底错成什么样」逐条挖出来，**不经过 stage2 的归属/选框逻辑**，
只看「一个框 + 一次读取」这一层，方便按错误形态归类。

做法
----
1. 从官方 GT 取所有**真位号**元件（键形如 `C151`/`R164`/`LED1`）；
2. 把 GT 的 bbox（**y-UP**，`y_png = H - y1 .. H - y0`）投到 PNG 坐标；
3. 在 stage1 的文字框里挑**离这个元件框最近的那个**（不挑「读得最像的」，
   否则就是把答案喂进去）；
4. 用角色 `designator` 读一次；
5. 按编辑距离归类：`ok` / `del`（少字符）/ `ins`（多字符）/ `sub`（等长替换）/
   `far`（编辑距离 > 2 且不是前三类）；
6. 渲染接触表 `work/preview/_ocrerr_<role>.png`，每格标 `case:GT->pred`。

用法
----
    python work\\tools\\diag\\_ocrerr.py                      # 位号，全部 220 张
    python work\\tools\\diag\\_ocrerr.py --role value          # 换角色
    python work\\tools\\diag\\_ocrerr.py --limit 40 --per-class 24
    python work\\tools\\diag\\_ocrerr.py --pad 60 --show 200   # 控制台多打几条
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import cv2
import numpy as np

EDA = Path(__file__).resolve().parents[3] / "eda-pcb"
sys.path.insert(0, str(EDA))

import schematic_parser.data as D            # noqa: E402

DESIG_RE = re.compile(r"^[A-Za-z]{1,3}[0-9]{1,4}$")


# ---------------------------------------------------------------- 工具
def _dist_to_rect(cx: float, cy: float, r) -> float:
    x0, y0, x1, y1 = r
    dx = max(x0 - cx, 0.0, cx - x1)
    dy = max(y0 - cy, 0.0, cy - y1)
    return (dx * dx + dy * dy) ** 0.5


def _tolower(s: str) -> str:
    return (s or "").strip()


def edit_ops(gt: str, pr: str):
    """返回 (距离, 插入数, 删除数, 替换数)。标准 DP。"""
    n, m = len(gt), len(pr)
    INF = 10 ** 6
    d = [[0] * (m + 1) for _ in range(n + 1)]
    ins = [[0] * (m + 1) for _ in range(n + 1)]
    dele = [[0] * (m + 1) for _ in range(n + 1)]
    sub = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i; dele[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j; ins[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = (d[i - 1][j] + 1, ins[i - 1][j], dele[i - 1][j] + 1, sub[i - 1][j])
            if gt[i - 1] == pr[j - 1]:
                cand = (d[i - 1][j - 1], ins[i - 1][j - 1], dele[i - 1][j - 1], sub[i - 1][j - 1])
                if cand[0] < best[0]:
                    best = cand
            else:
                cand = (d[i - 1][j - 1] + 1, ins[i - 1][j - 1], dele[i - 1][j - 1], sub[i - 1][j - 1] + 1)
                if cand[0] < best[0]:
                    best = cand
            cand = (d[i][j - 1] + 1, ins[i][j - 1] + 1, dele[i][j - 1], sub[i][j - 1])
            if cand[0] < best[0]:
                best = cand
            d[i][j], ins[i][j], dele[i][j], sub[i][j] = best
    return d[n][m], ins[n][m], dele[n][m], sub[n][m]


def classify(gt: str, pr: str) -> str:
    if not pr:
        return "empty"
    if pr == gt:
        return "ok"
    if pr.lower() == gt.lower():
        return "case"
    dist, ni, nd, ns = edit_ops(gt, pr)
    if dist == 1 and nd == 1:
        return "del"          # 少读一个字符
    if dist == 1 and ni == 1:
        return "ins"          # 多读一个字符
    if dist == 1 and ns == 1:
        return "sub"          # 等长错一个字符
    if dist <= 2:
        return "near"
    return "far"


def collect(role: str, limit: int, pad: float, pred_stage1: Path):
    """-> list of dict(case, gt, pred, bbox, comp)"""
    cases = list(D.discover(D.DEFAULT_DATA_ROOT))
    if limit:
        cases = cases[:limit]
    jobs = []
    for c in cases:
        gt = D.load_json(Path(c.json_path))
        s1p = pred_stage1 / (c.case_id + ".stage1.json")
        if not s1p.is_file():
            continue
        import json
        s1 = json.loads(s1p.read_text(encoding="utf-8"))
        texts = s1.get("texts") or []
        if not texts:
            continue
        img = cv2.imread(str(c.png_path)) if Path(c.png_path).is_file() else None
        if img is None:
            continue
        H = img.shape[0]
        comps = gt.get("components") or {}
        for k, v in comps.items():
            if role == "designator":
                if not DESIG_RE.match(str(k)):
                    continue
                want = str(k)
            else:
                want = v.get("value") if role == "value" else None
                if not want:
                    continue
            bb = v.get("bbox")
            if not bb:
                continue
            x0, y0, x1, y1 = [float(t) for t in bb]
            rect = (min(x0, x1), H - max(y0, y1), max(x0, x1), H - min(y0, y1))
            cx, cy = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
            best = None
            for t in texts:
                tb = t.get("bbox_img")
                if not tb:
                    continue
                tx = (float(tb[0]) + float(tb[2])) / 2.0
                ty = (float(tb[1]) + float(tb[3])) / 2.0
                d = _dist_to_rect(tx, ty, rect)
                if d > pad:
                    continue
                if best is None or d < best[0]:
                    best = (d, tb)
            if best is None:
                continue
            jobs.append(dict(case=c.case_id, gt=want, bbox=list(best[1]),
                             d=round(best[0], 1), png=str(c.png_path),
                             img=img, comp=str(k)))
    return jobs


def read_all(jobs, role: str):
    from stage2.text import TextEngine
    eng = TextEngine()
    by_png = collections.OrderedDict()
    for j in jobs:
        by_png.setdefault(j["png"], []).append(j)
    for png, group in by_png.items():
        img = group[0]["img"]
        res = eng.read(img, [g["bbox"] for g in group], [role])
        for g, r in zip(group, res):
            rr = (r.get("reads") or {}).get(role)
            g["pred"] = rr[0] if rr else ""
            g["conf"] = rr[1] if rr else 0.0
    for j in jobs:
        j.pop("img", None)
    return jobs


def sheet(jobs, cls: str, out: Path, per_class: int, cols: int = 6, tile_h: int = 64):
    sel = [j for j in jobs if j["cls"] == cls]
    sel.sort(key=lambda j: (j["d"], j["case"]))
    sel = sel[:per_class]
    if not sel:
        return 0
    tiles = []
    for j in sel:
        x0, y0, x1, y1 = [int(round(v)) for v in j["bbox"]]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), x1, y1
        sub = j["_img"][y0:y1, x0:x1]
        if sub.size == 0:
            continue
        h = tile_h
        w = max(40, int(round(h * sub.shape[1] / max(1, sub.shape[0]))))
        sub = cv2.resize(sub, (w, h), interpolation=cv2.INTER_CUBIC)
        cv = np.full((h + 22, w + 8, 3), 255, np.uint8)
        cv[22:22 + h, 4:4 + w] = sub
        cv2.putText(cv, "%s %s>%s" % (j["case"], j["gt"], j["pred"]), (3, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(cv)
    if not tiles:
        return 0
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    out_img = np.full((rows * (th + 6), cols * (tw + 6), 3), 245, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out_img[r * (th + 6):r * (th + 6) + t.shape[0],
                 c * (tw + 6):c * (tw + 6) + t.shape[1]] = t
    cv2.imwrite(str(out), out_img)
    return len(tiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="designator",
                    choices=("designator", "value", "pin_name", "pin_number", "net_label"))
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 张 case")
    ap.add_argument("--pad", type=float, default=60.0, help="文字框中心到元件框的最大距离")
    ap.add_argument("--per-class", type=int, default=18, help="每类渲染多少格")
    ap.add_argument("--show", type=int, default=40, help="控制台打印多少条")
    ap.add_argument("--stage1", default=r"work\_submit_out220r36\runs\stage1")
    args = ap.parse_args()

    s1 = Path(args.stage1)
    if not s1.is_absolute():
        s1 = Path(__file__).resolve().parents[3] / args.stage1

    jobs = collect(args.role, args.limit, args.pad, s1)
    print("[1/3] 采集到 %d 个 (GT 文本, 最近文字框) 对" % len(jobs))
    read_all(jobs, args.role)
    for j in jobs:
        j["cls"] = classify(j["gt"], j["pred"])

    cnt = collections.Counter(j["cls"] for j in jobs)
    tot = sum(cnt.values())
    print("[2/3] 角色 %s 共 %d 条，分类：" % (args.role, tot))
    for k in ("ok", "case", "del", "ins", "sub", "near", "far", "empty"):
        if cnt.get(k):
            print("        %-6s %5d  (%.3f)" % (k, cnt[k], cnt[k] / tot))
    # 每个分类里的「错」样本：按 GT 是否含多位数字再切一刀
    multi = collections.Counter()
    for j in jobs:
        if j["cls"] == "ok":
            continue
        digits = sum(ch.isdigit() for ch in j["gt"])
        multi["0-2 位数字" if digits <= 2 else "3+ 位数字"] += 1
    if multi:
        print("        错例按 GT 数字位数：", dict(multi))

    # 渲染（需要原图，重新读一次，只在渲染时读）
    outdir = Path(__file__).resolve().parents[3] / "work" / "preview"
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for j in jobs:
        if j["png"] not in cache:
            cache[j["png"]] = cv2.imread(j["png"])
        j["_img"] = cache[j["png"]]
    print("[3/3] 渲染接触表：")
    for cls in ("del", "ins", "sub", "near", "far", "case", "empty"):
        n = sheet(jobs, cls, outdir / ("_ocrerr_%s_%s.png" % (args.role, cls)),
                  args.per_class)
        if n:
            print("        %-6s %3d 格 -> work\\preview\\_ocrerr_%s_%s.png" % (cls, n, args.role, cls))

    print("\n最严重的 %d 条（按编辑距离降序，只列错的）：" % args.show)
    bad = [j for j in jobs if j["cls"] not in ("ok",)]
    bad.sort(key=lambda j: -edit_ops(j["gt"], j["pred"])[0])
    for j in bad[:args.show]:
        print("   %-18s comp=%-10s d=%-6.1f GT=%-8s pred=%-10s conf=%.2f  %s"
              % (j["case"], j["comp"], j["d"], j["gt"], j["pred"], j["conf"], j["cls"]))


if __name__ == "__main__":
    main()
