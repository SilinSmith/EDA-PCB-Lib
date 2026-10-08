# -*- coding: utf-8 -*-
"""大 IC 补救：**YOLO 检不出的大矩形，用几何把它找回来。**

为什么必须补
------------
`pin_f1` 只统计 GT 里 `IC_TYPES`（`box`/`boxic`/`ic`）那些元件的引脚。**只要那颗 IC 的
键对不上，它下面所有引脚就全变成 `pin.missing`** ⇒ 那张图的引脚分（权重 0.25）直接归零。

实测（`work\\tools\\diag\\_boxmiss.py`，220 张）：
    有 GT box 的图 213 张｜GT box 元件 996 个｜**键对上了 425（0.427）｜整颗没输出 571（0.573）**
    这些漏掉的 box 一共带走 **4598 个 GT 引脚**
    至少漏一颗 box 的图：**181 / 213 = 0.850**
在快速回归集（19 张）上，引脚 F1 = 0 的有 **14 张**。

为什么不是「调低 YOLO 的 conf」
-------------------------------
在 10GT case `1` 上把 conf 从 0.25 降到 **0.01**，`box` 类的输出始终是 **0 个**
（只有 `gnd` 多了一个）。⇒ **这是模型的能力缺口，不是阈值问题。**

怎么找
------
IC 体在原理图里一律画成**轴对齐的实线矩形**（KiCad / Altium / EasyEDA 都是），
所以用「长直线 → 轮廓 → 矩形」的老办法能稳定找到。case `1` 的实测：
    找出来 x=961 y=56 w=353 h=765  vs  GT U1 x=962 y=59 w=352 h=762   （差 1~3 px）

**精度靠「引脚桩」把关**：IC 的边上有许多向外伸出的短横线（引脚）。在边线外侧取一列像素，
数「暗→亮」的跳变次数（`n_stub`）—— 真 IC 有 ≥3 个，标题栏/装饰框基本为 0。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# ---- 可调参数（全部可用环境变量覆盖，方便 A/B） ----
MIN_SIDE = int(os.environ.get("BIGIC_MIN_SIDE", "60"))       # 矩形最短边
MIN_AREA = int(os.environ.get("BIGIC_MIN_AREA", "8000"))     # 最小面积
MAX_FRAC = float(os.environ.get("BIGIC_MAX_FRAC", "0.55"))   # 最大面积占比
FILL_MIN = float(os.environ.get("BIGIC_FILL_MIN", "0.88"))   # 轮廓面积 / 外接矩形
STUB_MIN = int(os.environ.get("BIGIC_STUB_MIN", "3"))        # 边缘引脚桩数下限
STUB_OFF = int(os.environ.get("BIGIC_STUB_OFF", "10"))       # 取列的位置：边线外多少像素
INK = int(os.environ.get("BIGIC_INK", "140"))                # 暗像素阈值
STUB_BAND = float(os.environ.get("BIGIC_STUB_BAND", "0.6"))  # 只数中间这一段（避开四角）
BORDER_MIN = float(os.environ.get("BIGIC_BORDER_MIN", "0.9"))


def _runs(col: np.ndarray) -> int:
    """一列 0/1 里「连续 1 的段」有几段。"""
    if col.size == 0:
        return 0
    d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
    return int((d == 1).sum())


def _edge_stubs(dark: np.ndarray, x: int, y1: int, y2: int) -> int:
    """在竖直线 `x` 附近数水平的引脚桩。

    引脚桩是**向外伸出**的短横线，所以取边线外侧一点点的那一列来数跳变。
    """
    H, W = dark.shape[:2]
    x = int(round(x))
    if x < 0 or x >= W:
        return 0
    pad = max(2, int((y2 - y1) * (1.0 - STUB_BAND) / 2.0))
    a, b = max(0, y1 + pad), min(H, y2 - pad)
    if b - a < 4:
        return 0
    sub = dark[a:b, max(0, x - 2):min(W, x + 3)]
    return _runs((sub.max(axis=1) > 0).astype(np.uint8))


def _edge_stubs_h(dark: np.ndarray, y: int, x1: int, x2: int) -> int:
    """横放 IC 的上/下边：数竖直的引脚桩。"""
    H, W = dark.shape[:2]
    y = int(round(y))
    if y < 0 or y >= H:
        return 0
    pad = max(2, int((x2 - x1) * (1.0 - STUB_BAND) / 2.0))
    a, b = max(0, x1 + pad), min(W, x2 - pad)
    if b - a < 4:
        return 0
    sub = dark[max(0, y - 2):min(H, y + 3), a:b]
    return _runs((sub.max(axis=0) > 0).astype(np.uint8))


def _border_support(dark, box, pad=2, trim=4):
    """Ink coverage of each expected side; box right/bottom are exclusive.

    A contour enclosing a wire loop can have high area fill while missing part
    of its rectangular boundary. Such a region must not erase a real component
    and its wires as an IC. Narrow bands tolerate contour/stroke rounding.
    """
    height, width = dark.shape[:2]
    x0, y0, x1, y1 = map(int, box)
    xa, xb = max(0, x0 + trim), min(width, x1 - trim)
    ya, yb = max(0, y0 + trim), min(height, y1 - trim)
    bands = (
        dark[max(0, y0 - pad):min(height, y0 + pad + 1), xa:xb].any(axis=0),
        dark[ya:yb, max(0, x1 - pad - 1):min(width, x1 + pad)].any(axis=1),
        dark[max(0, y1 - pad - 1):min(height, y1 + pad), xa:xb].any(axis=0),
        dark[ya:yb, max(0, x0 - pad):min(width, x0 + pad + 1)].any(axis=1),
    )
    return [float(side.mean()) if side.size else 0.0 for side in bands]


def detect_rects(bgr: np.ndarray, min_side=None, min_area=None, max_frac=None,
                 fill_min=None, stub_min=None, border_min=None):
    """返回候选 IC 体：`[{x1,y1,x2,y2,fill,stub,w,h}]`（图像坐标，y 向下）。"""
    min_side = MIN_SIDE if min_side is None else min_side
    min_area = MIN_AREA if min_area is None else min_area
    max_frac = MAX_FRAC if max_frac is None else max_frac
    fill_min = FILL_MIN if fill_min is None else fill_min
    stub_min = STUB_MIN if stub_min is None else stub_min
    # Some real ICs have square terminal pads interrupting their body outline.
    # Keep standalone contours; add_components uses borders with detector context.
    border_min = 0.0 if border_min is None else border_min

    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < INK).astype(np.uint8) * 255

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, W // 40), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, H // 40)))
    hor = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
    ver = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for ct in cnts:
        x, y, w, h = cv2.boundingRect(ct)
        if w < min_side or h < min_side or w * h < min_area:
            continue
        if w * h > W * H * max_frac:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue                                     # 贴边的是图纸外框
        fill = cv2.contourArea(ct) / float(w * h + 1e-6)
        if fill < fill_min:
            continue
        border = _border_support(dark, (x, y, x + w, y + h))
        if min(border) < border_min:
            continue
        ns = max(_edge_stubs(dark, x - STUB_OFF, y, y + h),
                 _edge_stubs(dark, x + w + STUB_OFF, y, y + h),
                 _edge_stubs_h(dark, y - STUB_OFF, x, x + w),
                 _edge_stubs_h(dark, y + h + STUB_OFF, x, x + w))
        if ns < stub_min:
            continue
        out.append({"x1": float(x), "y1": float(y), "x2": float(x + w),
                    "y2": float(y + h), "w": float(w), "h": float(h),
                    "fill": round(float(fill), 4), "stub": int(ns),
                    "border": [round(v, 4) for v in border]})
    out.sort(key=lambda r: -r["w"] * r["h"])
    return out


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-9)


def _straddles_body_boundary(box, body, margin=3.0):
    """A proposed IC edge cuts through another detected IC's body interior."""
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = body
    overlap_x = min(x1, bx1) - max(x0, bx0)
    overlap_y = min(y1, by1) - max(y0, by0)
    return ((overlap_x > 2 * margin and any(by0 + margin < y < by1 - margin for y in (y0, y1)))
            or (overlap_y > 2 * margin and any(bx0 + margin < x < bx1 - margin for x in (x0, x1))))


def add_components(bgr, comps, hyps, iou_skip=0.35) -> list:
    """把找到的大 IC 追加进 `comps`（同时登记一个假设槽位）。

    和已有的框 IoU 超过 `iou_skip` 就跳过 —— YOLO 已经检出的别重复加。
    """
    from stage1.detect import to_json_box          # 延迟导入避免环

    H = bgr.shape[0]
    have = [c["bbox_img"] for c in comps]
    added = []
    for r in detect_rects(bgr):
        box = [r["x1"], r["y1"], r["x2"], r["y2"]]
        if any(_iou(box, hb) >= iou_skip for hb in have):
            continue
        # Low border coverage alone cannot reject a pad-interrupted real IC.
        # A second independent contradiction is required: the proposed body
        # cuts through an already detected IC. Wire loops plus a nearby symbol
        # can otherwise erase that symbol's pins and steal its designator.
        if (min(r["border"]) < BORDER_MIN
                and any(c.get("type") in ("box", "bigic")
                        and _straddles_body_boundary(box, c["bbox_img"])
                        for c in comps)):
            continue
        cid = "b%04d" % len(added)
        hyp_id = f"{cid}.type"
        # 置信度按证据强度给：桩越多越像 IC
        score = round(min(0.9, 0.55 + 0.05 * r["stub"]), 4)
        hyps.add(hyp_id, "box", score, source="cv:bigic",
                 evidence={"fill": r["fill"], "stub": r["stub"],
                           "w": r["w"], "h": r["h"], "border": r["border"]},
                 question=f"{cid}（{int(r['w'])}x{int(r['h'])} 的大矩形）是 IC 吗？",
                 context={"w": r["w"], "h": r["h"], "stub": r["stub"]})
        comps.append({
            "id": cid, "type": "box", "proposal": "box", "score": score,
            "source": "cv:bigic", "subtype": "bigic",
            "bbox_img": [round(v, 2) for v in box],
            "bbox_json": [round(float(v), 2) for v in to_json_box(box, H)],
            "bbox_raw_img": [round(v, 2) for v in box],
            "center_img": [round((box[0] + box[2]) / 2, 2), round((box[1] + box[3]) / 2, 2)],
            "key": None, "n_alts": 1, "pending": False,
        })
        have.append(box)
        added.append(cid)
    return added


# ------------------------------------------------------------------ 自评
def main(argv=None):
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    HERE = Path(__file__).resolve()
    ROOT = HERE.parents[1]                                   # eda-pcb
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.stdout.reconfigure(encoding="utf-8")
    from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, load_json
    from schematic_parser.metrics import IC_TYPES, canon_type

    ap = argparse.ArgumentParser(description="大 IC 检测器自评（只看 YOLO 漏掉的那些 box）")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--split", default="200_train_cases")
    ap.add_argument("--cases", default="", help="逗号分隔；给了就只跑这些 case")
    ap.add_argument("--stage1", default="", help="已有的 stage1 产物目录（当作 YOLO 基线）")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--min-side", type=int, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--fill-min", type=float, default=None)
    ap.add_argument("--stub-min", type=int, default=None)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--sweep", action="store_true", help="扫阈值，看召回/精度怎么变")
    ap.add_argument("--sizes", action="store_true", help="打印 GT box 的尺寸分布")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT.parent / "work" / "tools" / "eval"))
    from stage2_eval import build_index, resolve as _resolve

    exact, digits = build_index(a.data_root)

    def cases_of():
        if a.cases:
            out = []
            for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
                c, _how = _resolve(cid, exact, digits)
                if c is None:
                    c, _how = _resolve(cid.lstrip("0"), exact, digits)
                if c is not None:
                    out.append(c)
            return out
        return [c for c in discover(a.data_root, SPLITS)
                if c.split == a.split and c.json_path][:a.limit]

    cases = cases_of()
    kw = dict(min_side=a.min_side, min_area=a.min_area,
              fill_min=a.fill_min, stub_min=a.stub_min)

    def load(c, key):
        """GT box（图像坐标）+ YOLO 已有的框。"""
        gt = load_json(Path(c.json_path))
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        H = bgr.shape[0]
        gimg = [(k, [v["bbox"][0], H - v["bbox"][3], v["bbox"][2], H - v["bbox"][1]])
                for k, v in (gt.get("components") or {}).items()
                if canon_type(v.get("type")) in IC_TYPES and v.get("bbox")]
        base = []
        if a.stage1:
            p = Path(a.stage1) / ("%s.stage1.json" % c.case_id)
            if p.exists():
                import json as _json
                doc = _json.loads(p.read_text(encoding="utf-8"))
                # 只有**我们自己输出成 IC 类型**的框才算覆盖 —— 一颗标成 `c` 的元件压在
                # IC 上，键永远不会是 `U1`（assign 的 strict 前缀会把 U1 掐掉）。
                base = [cc["bbox_img"] for cc in (doc.get("components") or [])
                        if canon_type(cc.get("type")) in IC_TYPES]
        return bgr, gimg, base

    if a.sizes:
        sides = []
        for c in cases:
            _bgr, gimg, _base = load(c, None)
            for k, g in gimg:
                w, h = int(g[2] - g[0]), int(g[3] - g[1])
                sides.append((min(w, h), (w, h), k))
        sides.sort()
        print("GT box %d 个，最短边分位：" % len(sides))
        for q in (0, 5, 10, 25, 50, 75, 90, 100):
            i = min(len(sides) - 1, int(len(sides) * q / 100))
            print("   p%-3d 最短边 %4d   实际 %s" % (q, sides[i][0], sides[i][1]))
        for t in (24, 40, 60):
            print("   最短边 <%-3d 的：%d" % (t, sum(1 for s in sides if s[0] < t)))
        return 0

    def run(kw):
        """核心指标：只看 YOLO 没覆盖的 GT box，bigic 补回多少、误加多少。"""
        ng = nmiss = nfix = 0
        nadd = nbad = 0
        det = []
        for c in cases:
            bgr, gimg, base = load(c, None)
            ai = a.iou
            rects = detect_rects(bgr, **kw)
            used_g = set()
            for r in rects:
                rb = [r["x1"], r["y1"], r["x2"], r["y2"]]
                if any(_iou(rb, bb) >= ai for bb in base):
                    continue                              # 和已有元件重合，不算新增
                nadd += 1
                if any(_iou(rb, g) >= ai for _k, g in gimg):
                    used_g.add(next(_k for _k, g in gimg if _iou(rb, g) >= ai))
                else:
                    nbad += 1                             # 谁也没配上 = 纯误加
            for k, g in gimg:
                ng += 1
                cov_base = any(_iou(g, bb) >= ai for bb in base)
                cov_new = any(_iou(g, [r["x1"], r["y1"], r["x2"], r["y2"]]) >= ai
                              for r in rects)
                if not cov_base:
                    nmiss += 1
                    if cov_new:
                        nfix += 1
            det.append((c.case_id, len(gimg), sum(1 for _k, g in gimg
                                                  if any(_iou(g, bb) >= ai
                                                         for bb in base)),
                        len(used_g)))
        return dict(ng=ng, nmiss=nmiss, nfix=nfix, nadd=nadd, nbad=nbad, det=det)

    if a.sweep:
        import itertools
        print("min_side min_area fill_min stub_min | YOLO漏  bigic补回  补回率  新增  误加")
        for ms, ma, fm, sm in itertools.product(
                [16, 24, 40], [400, 1500, 4000], [0.70, 0.85], [0, 2, 3]):
            r = run(dict(min_side=ms, min_area=ma, fill_min=fm, stub_min=sm))
            print("%8d %8d %8.2f %8d | %6d %9d %8.3f %5d %5d"
                  % (ms, ma, fm, sm, r["nmiss"], r["nfix"],
                     r["nfix"] / max(1, r["nmiss"]), r["nadd"], r["nbad"]))
        return 0

    import time as _t
    t0 = _t.time()
    r = run(kw)
    print("%d 张  %.3f s/张   IoU 门槛 %.2f" % (len(cases), (_t.time() - t0) / max(1, len(cases)), a.iou))
    print("GT box %d 个｜YOLO 漏 %d（%.3f）｜**bigic 补回 %d（%.3f）**｜仍漏 %d"
          % (r["ng"], r["nmiss"], r["nmiss"] / max(1, r["ng"]), r["nfix"],
             r["nfix"] / max(1, r["nmiss"]), r["nmiss"] - r["nfix"]))
    print("bigic 新增 %d 个框：命中 GT %d，**纯误加 %d（%.3f）**"
          % (r["nadd"], r["nadd"] - r["nbad"], r["nbad"], r["nbad"] / max(1, r["nadd"])))
    print("按图（GT / YOLO 已有 / 补后覆盖）：")
    for cid, ng, nb, na in sorted(r["det"], key=lambda x: -(x[1] - x[3]))[:a.show]:
        print("   %-9s GT %2d  YOLO %2d  补后 %2d" % (cid, ng, nb, na))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
