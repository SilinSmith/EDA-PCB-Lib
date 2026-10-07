# -*- coding: utf-8 -*-
"""把 pincol 的**名字窗**（`in_win`）/ 号码窗（`out_win`）单独送 VLM，看它答得出来几个。

存在的理由：`eda-pcb/vlm/enrich.py:read_pins` 在 0143 的 U1（66 个脚）上**只回
13~19 个答案**，而 `work/tools/diag/_pincolread.py` 用本地 CRNN 读同样的窗口
读出来的是 `P31-TRAH1` / `R14C8_V` 这类垃圾。三件事必须拆开看：
  ① 窗口本身裁得对不对（几何）；
  ② VLM 收到的是几张图、几张图里有几个格子；
  ③ 它回的文本长什么样、`parse_grid` 有没有把行丢掉。

用法：
  python work/tools/diag/_vlmnamewin.py --case 0143 --key c0003 --per 8 --cols 4 --dump
  python work/tools/diag/_vlmnamewin.py --case 0143 --key c0003 --win out_win --side 220
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

import cv2  # noqa: E402

from vlm import client  # noqa: E402
from vlm import enrich as EN  # noqa: E402

DATA = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
S1_DIRS = [ROOT / "work" / "_submit_out220r36" / "runs" / "stage1",
           ROOT / "work" / "_fusion_out220no4" / "runs" / "stage1",
           ROOT / "work" / "_pin1_v5" / "runs" / "stage1",
           ROOT / "work" / "_pin1_vlm_ours" / "runs" / "stage1"]


def _s1(case: str, explicit: str = "") -> Path:
    if explicit:
        return Path(explicit)
    for d in S1_DIRS:
        p = d / ("%s.stage1.json" % case)
        if p.exists():
            return p
    raise SystemExit("找不到 %s 的 stage1 产物；用 --stage1 指路" % case)


def _png(case: str) -> Path:
    hits = [h for h in DATA.glob("*/*%s/*.png" % case)
            if h.stem.split("-")[0] == case]
    if not hits:
        hits = sorted(DATA.glob("*/*%s/*.png" % case))
    if not hits:
        raise SystemExit("找不到 case %s 的 png" % case)
    hits.sort(key=lambda p: (0 if "jlc" in p.stem else 1, -p.stat().st_size))
    return hits[0]


def _gt(case: str) -> dict:
    hits = sorted(DATA.glob("*/*%s/*_target_named.json" % case))
    if not hits:
        return {}
    return json.loads(hits[0].read_text(encoding="utf-8"))


def _gt_points(gt: dict, H: int) -> dict:
    """{元件键: {(x,y) 图像坐标: (号, 名)}}，用来给每个答案找「就近的 GT 引脚」。"""
    out = {}
    for ck, cv_ in (gt.get("components") or {}).items():
        if str(cv_.get("type") or "").lower() not in ("box", "ic", "boxic"):
            continue
        m = {}
        for pk, pv in (cv_.get("pins") or {}).items():
            pt = (pv or {}).get("point") or {}
            try:
                x, y = float(pt["x"]), H - float(pt["y"])
            except Exception:  # noqa: BLE001
                continue
            m[(round(x), round(y))] = (str(pk).replace("pin_", ""),
                                       str((pv or {}).get("pinname") or ""))
        out[ck] = m
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--key", default="", help="stage1 元件 id（如 c0003）；空=脚最多的")
    ap.add_argument("--stage1", default="")
    ap.add_argument("--win", default="in_win", choices=["in_win", "out_win"])
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--side", type=int, default=150)
    ap.add_argument("--pad", type=float, default=3.0, help="窗口外扩像素")
    ap.add_argument("--model", default="")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    s1 = json.loads(_s1(a.case, a.stage1).read_text(encoding="utf-8"))
    cols_doc = s1.get("pincol") or {}
    if not cols_doc:
        raise SystemExit("stage1 产物里没有 pincol（得开着 STAGE1_PINC 跑）")
    key = a.key or max(cols_doc, key=lambda k: len(cols_doc[k]))
    recs = cols_doc.get(key) or []
    if not recs:
        raise SystemExit("元件 %s 没有枚举出引脚" % key)
    png = _png(a.case)
    bgr = cv2.imread(str(png))
    if bgr is None:
        raise SystemExit("读不出 %s" % png)
    H, W = bgr.shape[:2]
    print("case %s  元件 %s  枚举 %d 个脚  png %s (%dx%d)"
          % (a.case, key, len(recs), png.name, W, H))

    crops, idx = [], []
    for i, r in enumerate(recs):
        w = r.get(a.win)
        if not w:
            continue
        x0, y0, x1, y1 = [float(v) for v in w]
        x0 = max(0.0, x0 - a.pad); y0 = max(0.0, y0 - a.pad)
        x1 = min(float(W), x1 + a.pad); y1 = min(float(H), y1 + a.pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        crops.append(bgr[int(y0):int(y1), int(x0):int(x1)].copy())
        idx.append(i)
    print("裁出 %d 张（%d 个没窗口）" % (len(crops), len(recs) - len(crops)))
    if not crops:
        return 1

    ans_all = {}
    for s in range(0, len(crops), a.per):
        sub = crops[s:s + a.per]
        subidx = idx[s:s + a.per]
        g = EN.tile(sub, cols=a.cols, side=a.side, pad=8, label_h=26)
        if a.out and s == 0:
            cv2.imwrite(a.out, g)
            print("格子图 -> %s  (%dx%d)" % (a.out, g.shape[1], g.shape[0]))
        txt, _dt = client.ask(g, EN.PINS_PROMPT, model=a.model or None)
        got = EN.parse_grid(txt or "")
        if a.dump:
            print("---- 第 %d 批（%d 格 → 回 %d 行）----"
                  % (s // a.per + 1, len(sub), len(got)))
            print(txt)
        for k, v in got.items():
            if 0 <= k < len(subidx):
                ans_all[subidx[k]] = v
    print("★ 回出 %d / %d 个答案" % (len(ans_all), len(crops)))
    if not ans_all:
        return 0

    gtp = _gt_points(_gt(a.case), H)
    # 把 GT 点按 y 归到「最近的枚举脚」上
    print("  #   side  y      模型 num        模型 name        最近 GT")
    for i, r in enumerate(recs):
        if i not in ans_all:
            continue
        num, name = ans_all[i]
        best, bd = "", 1e9
        for ck, m in gtp.items():
            for (gx, gy), (pn, pnm) in m.items():
                d = ((gx - r["x"]) ** 2 + (gy - r["y"]) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, "%s %s/%s" % (ck, pn, pnm)
        print("  %-3d %-5s %-7.1f %-14s %-16s %s" % (i, r.get("side"), r.get("y", 0),
                                                      num, name, best if bd < 12 else ""))
    print("GT 的 IC 元件脚数：%s" % {k: len(v) for k, v in gtp.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
