# -*- coding: utf-8 -*-
"""把电源/地/端口符号的框长到「整颗」——用符号色（暗红）连通块在**窄窗口**内补齐。

病（`work/_compok2.py o13pk 1`）：case1 里配上对的 47 组元件有 15 组死在 bbox
（`bbox_tol=20` 逐坐标），全是 `gnd` / `net_input` / `led` 这类**小符号**：

| 族 | GT 框 | 我们 |
|---|---|---|
| `gnd` `∅141` | `[990,983,1027,1016]` 37×33 | `[997,1008,1017,1027]` 20×19 |
| 端口 `BAT` | `[1383,894,1425,916]` 42×22 | `[1412,899,1442,910]` 30×10 |

GT 的符号框覆盖**整个符号**（横杠 + 竖杆 + 箭头），stage1 给的是**内腔/标签附近**。
这两类符号的图形都是暗红色（HSV `h<=10 or h>=170, s>=60, v>=40`），所以取「离当前框
中心最近的红色连通块」、把它在**窄窗口**内与当前框求并，就落到 GT 约定上。

实测（`work/_symbb2.py`，13 张 131 个符号，逐坐标 ≤20px 算过）：
窗口 34/26/22/20/18/16/14 对应 88%/91%/95%/97%/98%/98%/98%（原样 89%）。
窗口越大越容易把相邻的 IC 框线/引脚号并进来，14px 是拐点。

※ 只治「框小了一圈」，治不了「框根本在别处」（`led` 的 `LED2` 差 93px、`gnd` 的
`∅141` 这类窗口里没有可用红色块）——那要检测器自己给整颗框。

※ 落在 **stage1**（而不是 stage2）：`fusion/run.py::_assemble` 会用原始 `s1_doc`
重新 `build_doc`，Stage2 里改 `doc` 不留痕（§5.426/§5.427 同一条教训）。
"""
import os

import cv2
import numpy as np

SYM_BOX = int(os.environ.get("STAGE1_SYM_BOX", "1") or 0)
SYM_WIN = float(os.environ.get("STAGE1_SYM_BOX_WIN", "14"))
SYM_CAP = float(os.environ.get("STAGE1_SYM_BOX_CAP", "70"))
SYM_MINPX = int(os.environ.get("STAGE1_SYM_BOX_MINPX", "60"))
SYM_TYPES = frozenset((
    "gnd", "gnda", "gndd", "agnd", "dgnd", "pgnd", "gnd_ref", "earth", "vss",
    "v", "vcc", "vdd", "vbus_in", "net_input", "net_output", "net_short",
    "net_bidirection",
))
# ★ 被动件/二极管也一样「内腔 ≠ 整颗」。逐类合格率实测（`work/_bboxall.py o13sb`，13 张）：
#   `led 29%→71%`、`l 80%→90%`、`r 98%→100%`、`c 99%→99%`。
#   但整轮 A/B 不以「合格率」为准、只认总分：
#   `r,c,l,d,led` 77.02 ≡ 只符号 77.02 > `r,c,l` 76.95，`led,d` 77.01/77.05（两个方向各跑一次），
#   官方口径 75.72（带 `d,led`）< 75.79（只符号）⇒ **默认关**（`STAGE1_SYM_BOX_EXTRA=''`）。
#   把 `r` 的框拉长会改 `STAGE2_PIN_LEG_FILL_TARGET` 的证据样本，case1 反而掉 1.9 分。
#   IC/大框/开关/测试点任何时候都不动（它们本来就整颗）。
SYM_EXTRA = frozenset(
    t.strip() for t in os.environ.get("STAGE1_SYM_BOX_EXTRA", "").split(",")
    if t.strip()
)


def _red_cc(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (((h <= 10) | (h >= 170)) & (s >= 60) & (v >= 40)).astype(np.uint8)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    return lab, stats, n, m


PORT_TYPES = frozenset(("net_input", "net_output", "net_short", "net_bidirection"))
# ★ 端口符号单独一套规则（§5.446）：端口是**实心红箭头**（长 80~85px、厚 10~16px），
#   而导线只有 2~3px 粗且**接在箭头尖端**（同一个连通块）。用「红色连通块外扩」会被导线
#   一路带出去（`SYM_CAP=70` 是为了挡这个，但 81px 的箭头也因此被整颗拒绝）。
#   所以端口改用**厚度剖面**：沿主轴逐列统计带内红像素，只保留厚度 ≥ 中位厚度×frac 的
#   连续段（箭头厚、导线薄），段长受 `SYM_PORT_CAP` 限制。
SYM_PORT_CAP = float(os.environ.get("STAGE1_SYM_PORT_CAP", "60"))
SYM_PORT_FRAC = float(os.environ.get("STAGE1_SYM_PORT_FRAC", "0.6"))
SYM_PORT_PAD = int(os.environ.get("STAGE1_SYM_PORT_PAD", "4"))
SYM_PORT = int(os.environ.get("STAGE1_SYM_PORT", "1") or 0)


def _port_extent(mask, box):
    """沿主轴取「厚度 ≥ 阈值」的最大连续段（包含当前框中心）。box 为图像坐标。"""
    H, W = mask.shape
    ix0, iy0, ix1, iy1 = [float(v) for v in box]
    w, h = ix1 - ix0, iy1 - iy0
    if w <= 0 or h <= 0:
        return None
    horiz = w >= h
    pad = SYM_PORT_PAD
    if horiz:
        y0 = max(0, int(iy0) - pad)
        y1 = min(H, int(iy1) + pad + 1)
        prof = mask[y0:y1, :].sum(axis=0).astype(np.float32)
        c0, c1 = int(ix0), int(ix1)
        lim = int(max(0, ix0 - SYM_PORT_CAP)), int(min(W - 1, ix1 + SYM_PORT_CAP))
    else:
        x0 = max(0, int(ix0) - pad)
        x1 = min(W, int(ix1) + pad + 1)
        prof = mask[:, x0:x1].sum(axis=1).astype(np.float32)
        c0, c1 = int(iy0), int(iy1)
        lim = int(max(0, iy0 - SYM_PORT_CAP)), int(min(H - 1, iy1 + SYM_PORT_CAP))
    inner = prof[max(0, c0):max(1, c1)]
    if inner.size == 0 or not (inner > 0).any():
        return None
    thr = max(3.0, float(np.median(inner[inner > 0])) * SYM_PORT_FRAC)
    lo_lim, hi_lim = lim
    # 从当前框内最厚的一列向两侧走，遇到连续 3 列低于阈值就停
    core = int(np.argmax(prof[max(0, c0):max(1, c1)])) + max(0, c0)
    a, b = core, core
    thin = 0
    i = core
    while i + 1 <= hi_lim:
        i += 1
        if prof[i] >= thr:
            thin, a = 0, i
        else:
            thin += 1
            if thin >= 3:
                break
    thin = 0
    i = core
    while i - 1 >= lo_lim:
        i -= 1
        if prof[i] >= thr:
            thin, b = 0, i
        else:
            thin += 1
            if thin >= 3:
                break
    if horiz:
        return [float(b), iy0, float(a), iy1]
    return [ix0, float(b), ix1, float(a)]


def _grab(mask, lab, stats, n, box, W, H):
    """在 `box` 周围 `SYM_WIN` 窗口里找与符号墨色同块且离框心最近的连通块；返回其紧框或 None。"""
    ix0, iy0, ix1, iy1 = [float(v) for v in box]
    wx0, wx1 = ix0 - SYM_WIN, ix1 + SYM_WIN
    wy0, wy1 = iy0 - SYM_WIN, iy1 + SYM_WIN
    cx, cy = (ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0
    best = None
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < SYM_MINPX:
            continue
        bx0, by0 = max(float(x), wx0), max(float(y), wy0)
        bx1, by1 = min(float(x + w), wx1), min(float(y + hh), wy1)
        if bx1 - bx0 < 6 or by1 - by0 < 5:
            continue
        if (bx1 - bx0) > SYM_CAP or (by1 - by0) > SYM_CAP:
            continue
        sub = lab[int(by0):int(by1) + 1, int(bx0):int(bx1) + 1]
        hit = (sub == i)
        if int(hit.sum()) < SYM_MINPX:
            continue
        ys, xs = np.nonzero(hit)
        nx0, nx1 = bx0 + xs.min(), bx0 + xs.max()
        ny0, ny1 = by0 + ys.min(), by0 + ys.max()
        d = ((nx0 + nx1) / 2.0 - cx) ** 2 + ((ny0 + ny1) / 2.0 - cy) ** 2
        if best is None or d < best[0]:
            best = (d, nx0, ny0, nx1, ny1)
    if best is None:
        return None
    return best[1:]


def grow(bgr, comps):
    """就地改 `comps[*]["bbox_img"]` / `bbox_json`；返回统计。"""
    stat = {"on": bool(SYM_BOX), "n": 0, "grew": 0}
    if not SYM_BOX or bgr is None or not comps:
        return stat
    lab, stats, n, mask = _red_cc(bgr)
    H, W = bgr.shape[:2]
    for c in comps:
        _ct = str(c.get("type") or "").strip().lower()
        if _ct not in SYM_TYPES and not _ct.startswith("gnd") and _ct not in SYM_EXTRA:
            continue
        bb = c.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        if SYM_PORT and _ct in PORT_TYPES:
            ext = _port_extent(mask, bb)
            if os.environ.get("STAGE1_SYM_BOX_DBG"):
                print("[symbox port] id=%s type=%s bb=%s ext=%s" % (c.get("id"), _ct, bb, ext))
            _pb = [float(v) for v in bb]
            if ext:
                _pb = [max(0.0, min(_pb[0], ext[0])), max(0.0, min(_pb[1], ext[1])),
                       min(float(W - 1), max(_pb[2], ext[2])), min(float(H - 1), max(_pb[3], ext[3]))]
            _cc = _grab(mask, lab, stats, n, _pb, W, H)
            if _cc:
                _pb = [min(_pb[0], _cc[0]), min(_pb[1], _cc[1]), max(_pb[2], _cc[2]), max(_pb[3], _cc[3])]
                _ex2 = _port_extent(mask, _pb)
                if _ex2:
                    _pb = [max(0.0, min(_pb[0], _ex2[0])), max(0.0, min(_pb[1], _ex2[1])),
                           min(float(W - 1), max(_pb[2], _ex2[2])), min(float(H - 1), max(_pb[3], _ex2[3]))]
            if abs(_pb[0] - bb[0]) + abs(_pb[1] - bb[1]) + abs(_pb[2] - bb[2]) + abs(_pb[3] - bb[3]) >= 1.0:
                c["bbox_img"] = [round(_pb[0], 2), round(_pb[1], 2), round(_pb[2], 2), round(_pb[3], 2)]
                if c.get("bbox_json"):
                    c["bbox_json"] = [round(_pb[0], 2), round(float(H) - _pb[3], 2),
                                      round(_pb[2], 2), round(float(H) - _pb[1], 2)]
                stat["grew"] += 1
            stat["n"] += 1
            continue
        _cc = _grab(mask, lab, stats, n, [float(v) for v in bb], W, H)
        if _cc is None:
            continue
        nx0, ny0, nx1, ny1 = _cc
        ix0, iy0, ix1, iy1 = [float(v) for v in bb]
        ux0, ux1 = max(0.0, min(ix0, nx0)), min(float(W - 1), max(ix1, nx1))
        uy0, uy1 = max(0.0, min(iy0, ny0)), min(float(H - 1), max(iy1, ny1))
        if abs(ux0 - ix0) + abs(uy0 - iy0) + abs(ux1 - ix1) + abs(uy1 - iy1) < 1.0:
            continue
        c["bbox_img"] = [round(ux0, 2), round(uy0, 2), round(ux1, 2), round(uy1, 2)]
        if c.get("bbox_json"):
            c["bbox_json"] = [round(ux0, 2), round(float(H) - uy1, 2),
                              round(ux1, 2), round(float(H) - uy0, 2)]
        stat["grew"] += 1
    return stat
