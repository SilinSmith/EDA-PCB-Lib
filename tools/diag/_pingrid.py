# -*- coding: utf-8 -*-
"""逐引脚问 VLM：把「一个引脚」的邻域裁片拼成网格，一次调用读回「编号:名称」。

动机（`_pincover.py` 实测）：GT 的 996 个 box 元件里 **987 个**我们的框 IoU>=0.5
（检测 0.991），GT 引脚点被 stage1 的几何点覆盖 **0.4634**，我们已输出 5030 个引脚
（GT 9356 的 0.538）—— 坏掉的只是「哪个点是几号脚」这个映射。

所以不让 VLM 列全表（拿不到坐标），而是**按点逐引脚问**：每个裁片用一个红圈
标出要读的那个引脚，编号（在外框外）与名称（在外框内）用不对称窗口一并裁进来。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_pingrid.py --case 1 --key U1 --n 8 --cols 4 --render
    python work\\tools\\diag\\_pingrid.py --splits 10GT --max-comp 8 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from schematic_parser.data import discover, load_json  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _vlmread import ask_vlm  # noqa: E402

PROMPT = (
    "这是一张由多个小格拼成的图，每一格是电路原理图里**一个引脚**附近的放大图。\n"
    "每一格里有一个**红色圆圈**，圈住的就是这一格要读的那个引脚。\n"
    "请对每一格读出两件事：\n"
    "1) 编号：紧贴该引脚、印在元器件外框**之外**的那串数字（引脚编号）\n"
    "2) 名称：元器件外框**内部**、与该引脚同一行的信号名文字（如 VDD、GND、SPI_MOSI）\n"
    "只回**一行**，格式 `格号:编号:名称`，多格之间用英文逗号分隔。例如：\n"
    "0:12:VDD,1:13:GND,2:44:SPI_CLK\n"
    "不要 JSON、不要换行、不要解释、不要任何其它字符。")


def crop_dir(bgr, pt, center, outer=34.0, inner=62.0, cross=42.0):
    h, w = bgr.shape[:2]
    px, py = float(pt[0]), float(pt[1])
    dx, dy = float(center[0]) - px, float(center[1]) - py
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-3:
        dx, dy, n = 1.0, 0.0, 1.0
    ux, uy = dx / n, dy / n
    ax0, ay0 = px - ux * outer, py - uy * outer
    ax1, ay1 = px + ux * inner, py + uy * inner
    x0 = max(0, int(min(ax0, ax1) - cross)); x1 = min(w, int(max(ax0, ax1) + cross))
    y0 = max(0, int(min(ay0, ay1) - cross)); y1 = min(h, int(max(ay0, ay1) + cross))
    if x1 <= x0 or y1 <= y0:
        return None, None
    return bgr[y0:y1, x0:x1].copy(), (px - x0, py - y0)


COLORS = {
    "red": (0, 0, 235),
    "magenta": (235, 0, 235),
    "cyan": (235, 235, 0),
    "blue": (235, 60, 0),
    "green": (0, 190, 0),
    "orange": (0, 140, 255),
}


def mark(im, pt, radius=4, thick=2, color=(235, 0, 235)):
    c = (int(round(pt[0])), int(round(pt[1])))
    cv2.circle(im, c, radius, color, thick, cv2.LINE_AA)
    return im


def tile(crops, cols=4, side=150, pad=8, label_h=26):
    if not crops:
        return None
    rows = (len(crops) + cols - 1) // cols
    cw = side + pad * 2
    ch = side + pad * 2 + label_h
    out = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
    for i, c in enumerate(crops):
        r, q = divmod(i, cols)
        if c is None:
            continue
        hh, ww = c.shape[:2]
        s = min(side / max(1, ww), side / max(1, hh))
        im = cv2.resize(c, (max(1, int(ww * s)), max(1, int(hh * s))),
                        interpolation=cv2.INTER_CUBIC)
        ih, iw = im.shape[:2]
        y = r * ch + label_h + pad + max(0, (side - ih) // 2)
        x = q * cw + pad + max(0, (side - iw) // 2)
        out[y:y + ih, x:x + iw] = im
        cv2.putText(out, "#%d" % i, (q * cw + pad, r * ch + label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.rectangle(out, (q * cw, r * ch), (q * cw + cw - 1, r * ch + ch - 1),
                      (200, 200, 200), 1)
    return out


def parse_grid(txt):
    """`0:12:VDD,1:13:GND` -> {0: ('12', 'VDD'), ...}；也容忍 JSON 数组。"""
    out = {}
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```\s*$", "", (txt or "").strip()).strip()
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            for it in json.loads(m.group(0)):
                if isinstance(it, dict):
                    i = it.get("id", it.get("index"))
                    out[int(i)] = (str(it.get("number") or ""), str(it.get("name") or ""))
            if out:
                return out
        except Exception:  # noqa: BLE001
            pass
    for seg in re.split(r"[,\n;]+", t):
        mm = re.match(r"\s*[\[\(]?\s*#?(\d{1,3})\s*[:：]\s*([^:：]*?)\s*[:：]\s*(.*?)\s*[\]\)]?\s*$",
                      seg)
        if mm:
            out[int(mm.group(1))] = (mm.group(2), mm.group(3))
        else:
            mm = re.match(r"\s*#?(\d{1,3})\s*[:：]\s*(.*?)\s*$", seg)
            if mm:
                out[int(mm.group(1))] = ("", mm.group(2))
    return out


def norm(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s or "")).upper()


def adaptive_window(bbox, base_inner=78.0, base_cross=58.0, base_outer=34.0, ref=762.0):
    """按元件尺寸缩放裁片窗口。

    固定窗口在大 IC（`1/U1`，bbox 352x762）上调得好（outer34/inner78/cross58），
    但套到只有 4 个脚的 `HS96L03W2C03`（bbox ~84x84）上就把整颗 IC 连同周围
    连线都框进一格，标记淹没在走线里 —— 实测 `2/U1` 掉到 0.25、`10/U1` 0.44。
    按 `max(w,h)/ref` 缩放后小 IC 拿到紧得多的窗口。
    """
    w = abs(float(bbox[2]) - float(bbox[0]))
    h = abs(float(bbox[3]) - float(bbox[1]))
    k = max(0.35, min(1.0, max(w, h) / ref))
    return (max(12.0, base_outer * k), max(30.0, base_inner * k),
            max(20.0, base_cross * k))


def build_crops(bgr, gp, ctr, outer, inner, cross, mr=4, mt=2, color=(235, 0, 235)):
    """按 y（图像坐标）排序取引脚，返回 (裁片列表, [(pk, num, name)], [点])。"""
    H = bgr.shape[0]
    items = sorted(gp.items(), key=lambda kv: (kv[1]["point"]["y"], kv[1]["point"]["x"]))
    crops, refs, pts = [], [], []
    for pk, info in items:
        pt = info["point"]
        q = (float(pt["x"]), H - float(pt["y"]))
        im, rel = crop_dir(bgr, q, ctr, outer, inner, cross)
        if im is None:
            continue
        mark(im, rel, mr, mt, color)
        crops.append(im)
        refs.append((pk, norm(pk.split("_", 1)[-1]), norm(info.get("pinname"))))
        pts.append(q)
    return crops, refs, pts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--key", default="")
    ap.add_argument("--splits", default="10GT")
    ap.add_argument("--max-comp", type=int, default=8)
    ap.add_argument("--min-pins", type=int, default=3)
    ap.add_argument("--per", type=int, default=8, help="一次调用拼几格")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--outer", type=float, default=34.0)
    ap.add_argument("--inner", type=float, default=78.0)
    ap.add_argument("--cross", type=float, default=58.0)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--detail", action="store_true", help="逐引脚打印 真值 vs 读数")
    ap.add_argument("--mr", type=int, default=4, help="标记半径")
    ap.add_argument("--mt", type=int, default=2, help="标记线宽")
    ap.add_argument("--color", default="red", choices=tuple(COLORS),
                    help="标记颜色（原理图自带的连接点是红色，所以要避开红）")
    ap.add_argument("--adapt", action="store_true", help="按元件尺寸缩放裁片窗口")
    ap.add_argument("--ref", type=float, default=762.0, help="自适应窗口的参考尺寸")
    a = ap.parse_args(argv)

    jobs = []
    for c in discover():
        if a.case and c.case_id != a.case:
            continue
        if not c.json_path:
            continue
        gt = load_json(c.json_path)
        comps = gt.get("components") or {}
        pins = gt.get("pins") or {}
        for k, v in comps.items():
            if str(v.get("type") or "").lower() not in ("box", "ic", "boxic"):
                continue
            if a.key and k != a.key:
                continue
            gp = pins.get(k) or {}
            if len(gp) < a.min_pins:
                continue
            jobs.append(dict(case=c.case_id, key=k, bbox=v.get("bbox"), Name=v.get("Name"),
                             gt=gp, png=c.png_path))
            break
        if len(jobs) >= a.max_comp:
            break
    print("待测元件 %d 个" % len(jobs))
    if not jobs:
        return 1

    def work(j):
        bgr = imread_unicode(j["png"])
        H = bgr.shape[0]
        bb = j["bbox"]
        ctr = ((bb[0] + bb[2]) / 2.0, H - (bb[1] + bb[3]) / 2.0)
        outer, inner, cross = a.outer, a.inner, a.cross
        if a.adapt:
            outer, inner, cross = adaptive_window(bb, a.inner, a.cross, a.outer, a.ref)
        crops, refs, pts = build_crops(bgr, j["gt"], ctr, outer, inner, cross, a.mr, a.mt,
                                       COLORS[a.color])
        if not crops:
            return None
        ans = {}
        dt_tot = 0.0
        for s in range(0, len(crops), a.per):
            sub = crops[s:s + a.per]
            g = tile(sub, cols=a.cols)
            if a.render:
                p = ROOT / "work" / "preview" / ("_pingrid_%s_%s_%d.png" % (j["case"], j["key"], s))
                cv2.imwrite(str(p), g)
            txt, dt = ask_vlm(g, prompt=PROMPT, model=a.model)
            dt_tot += dt
            got = parse_grid(txt)
            if os.environ.get("VLM_SHOW"):
                print("   raw[%d]: %r" % (s, (txt or "")[:260]))
            for i, (num, name) in got.items():
                if 0 <= i < len(sub):
                    ans[s + i] = (num, name)
        return j, refs, ans, dt_tot

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(work, jobs) if r]
    wall = time.time() - t0

    tn = tp = tot = 0
    print("\n%-7s %-6s %5s %5s %5s %6s" % ("case", "key", "GT", "num", "pair", "T(s)"))
    for j, refs, ans, dt in res:
        gn = gp = 0
        for i, (pk, num, name) in enumerate(refs):
            tot += 1
            if i not in ans:
                continue
            an, aw = norm(ans[i][0]), norm(ans[i][1])
            if an == num:
                gn += 1
            if an == num and aw == name:
                gp += 1
        tn += gn; tp += gp
        print("%-7s %-6s %5d %5d %5d %6.0f" % (j["case"], j["key"], len(refs), gn, gp, dt))
        if a.detail:
            for i, (pk, num, name) in enumerate(refs):
                an, aw = ans.get(i, ("", ""))
                ok = "OK " if (norm(an) == num and norm(aw) == name) else "   "
                print("     %s#%-3d %-10s GT=%-10s %-14s READ=%s:%s"
                      % (ok, i, pk, num, name, an, aw))
    print("\n墙钟 %.1f s（%.2f s/元件）" % (wall, wall / max(1, len(res))))
    print("  GT 引脚 %d｜编号对上 %d = %.4f｜编号+名称对上 %d = %.4f"
          % (tot, tn, tn / max(1, tot), tp, tp / max(1, tot)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
