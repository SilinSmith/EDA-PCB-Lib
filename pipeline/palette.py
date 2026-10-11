# -*- coding: utf-8 -*-
"""图片 → 颜色：**唯一**的颜色入口（用户 m56713/m56805 要求的那条接口）。

## 为什么单独做一个模块

历史形态是「颜色躲在导线提取里当**判据**」：`pipeline/pskel.py:242 wire_colour_mask()` +
`:542-574 foreground()` 的 `STAGE1_WIRE_UNCONF=veto` 会把**色相不对的细笔画整条删掉**
（`thin = cv2.bitwise_and(thin, cm)`），默认值是 `hint` + `veto` ⇒ 交付路径**一直在用
颜色判「这是不是导线」**，而用户已经指出：颜色只是小样本里恰好红绿，**黑白扫描件照样
会来**（实测无彩图这条链路会自失效：`chroma p90 < 22` ⇒ `wire_colour_mask` 返回 `None`、
Di Zenzo 彩色梯度恒 0）。

本模块的定位：**把「图里有什么颜色」这件事变成一次可断言、可缓存的测量**，其余模块只
消费它的输出（当证据、当权重），不再自己 `cv2.cvtColor` + `os.environ`。

## 契约（改这个文件的人请守住）

  * **纯函数**：不读环境变量、不写全局、不碰文件系统；同样输入必须给同样输出。
  * 返回**可 JSON 序列化**的 dict（产物里能直接落盘、能进日志）。
  * 不知道就返回 `None`/`0.0`，**不要猜**：`wire_hue=None` 表示「没测出可信的主色」。

## 输出

    {
      "has_chroma": bool,          # 图里是否有成规模的有彩墨迹
      "ink_chroma_p95": float,     # 墨迹内彩度 p95（灰度图 ≈ 0~3）
      "palette": [ {hue, sat, val, frac, rgb, n} ... ],   # 按占比降序，只留 frac>=MIN_FRAC
      "wire_hue": float | None,    # 主色相（OpenCV 0~179），不可信时为 None
      "confidence": float,         # 主色占比（= palette[0].frac），没有则 0
      "source": "histogram" | "none"
    }

    python -X utf8 -m pipeline.palette                 # 自检（合成图 + 真图，若在）
    python -X utf8 -m pipeline.palette <png> [...]     # 直接量几张图
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# ---- 阈值（都在这里，别再散到各模块）----------------------------------------
INK = 170            # 灰度 < INK 算墨（与 `pipeline/pskel.py` 的约定一致）
CHROMA_MIN = 22.0    # 单像素「有彩」的彩度下限（OpenCV 惯例：max-min，0~255）
CHROMA_STRONG = 40.0  # 进调色板要的彩度
HUE_BIN = 15         # 色相分桶宽度（0~179 共 12 桶）
MIN_FRAC = 0.02      # 一个色桶至少占墨迹的 2% 才进 palette
CHROMA_FRAC = 0.005  # 有彩像素占墨迹的比例下限（低于它判「灰度图」）


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """自带的墨迹掩膜（Otsu 反向），刻意不依赖 pskel，保持本模块可独立使用。"""
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    simple = gray < INK
    # Otsu 在纯白底上偶尔把整幅判成前景；两票取交集更稳，空了就退回灰度阈值。
    both = cv2.bitwise_and(m, (simple * 255).astype(np.uint8))
    if int((both > 0).sum()) >= 50:
        return both > 0
    return simple


def estimate(bgr: np.ndarray, ink: Optional[np.ndarray] = None) -> Dict:
    """量一张图的颜色。`ink` 可给现成掩膜（阶段一已经有 `_ink`，别重复算）。"""
    if bgr is None or bgr.size == 0:
        return {"has_chroma": False, "ink_chroma_p95": 0.0, "palette": [],
                "wire_hue": None, "confidence": 0.0, "source": "none"}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    if bgr.ndim == 2:                                   # 灰度图：没有颜色可言
        return {"has_chroma": False, "ink_chroma_p95": 0.0, "palette": [],
                "wire_hue": None, "confidence": 0.0, "source": "none"}
    m = np.asarray(ink, dtype=bool) if ink is not None else _ink_mask(gray)
    if m.shape[:2] != gray.shape[:2]:
        m = _ink_mask(gray)
    n_ink = int(m.sum())
    if n_ink < 50:
        return {"has_chroma": False, "ink_chroma_p95": 0.0, "palette": [],
                "wire_hue": None, "confidence": 0.0, "source": "none"}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    chroma = bgr.astype(np.int16).max(axis=2) - bgr.astype(np.int16).min(axis=2)
    ch_ink = chroma[m]
    p95 = float(np.percentile(ch_ink, 95))
    n_col = int((ch_ink >= CHROMA_MIN).sum())
    has = (n_col >= CHROMA_FRAC * n_ink) and (p95 >= CHROMA_MIN)
    if not has:
        return {"has_chroma": False, "ink_chroma_p95": round(p95, 2), "palette": [],
                "wire_hue": None, "confidence": 0.0, "source": "none"}

    strong = m & (chroma >= CHROMA_STRONG)
    # 每个色桶：像素数 / 平均 HSV / 代表 BGR
    h = hsv[:, :, 0][strong].astype(np.int32)
    s = hsv[:, :, 1][strong].astype(np.float32)
    v = hsv[:, :, 2][strong].astype(np.float32)
    palette: List[Dict] = []
    if h.size:
        bins = np.clip(h // HUE_BIN, 0, 179 // HUE_BIN)
        for b in np.unique(bins):
            sel = bins == b
            n = int(sel.sum())
            frac = n / float(max(1, n_ink))
            if frac < MIN_FRAC:
                continue
            hue = float((b + 0.5) * HUE_BIN)
            sat = float(s[sel].mean())
            val = float(v[sel].mean())
            px = np.uint8([[[hue, sat, val]]])
            bgr_px = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
            palette.append({"hue": round(hue, 1), "sat": round(sat, 1), "val": round(val, 1),
                            "frac": round(frac, 4), "n": n,
                            "rgb": [int(bgr_px[2]), int(bgr_px[1]), int(bgr_px[0])]})
    palette.sort(key=lambda d: -d["frac"])
    hue0 = palette[0]["hue"] if palette else None
    return {"has_chroma": True, "ink_chroma_p95": round(p95, 2), "palette": palette,
            "wire_hue": hue0, "confidence": float(palette[0]["frac"] if palette else 0.0),
            "source": "histogram"}


def is_greyscale(bgr: np.ndarray) -> bool:
    """黑白扫描件的判据（本模块唯一的布尔门；调用方不该自己写 `chroma < 22`）。"""
    return not estimate(bgr)["has_chroma"]


def describe(img: Dict, max_items: int = 4) -> str:
    """一行日志用。"""
    if not img or not img.get("has_chroma"):
        return "颜色：无彩（灰度/单色图）chroma_p95=%s" % (img or {}).get("ink_chroma_p95", 0)
    top = ", ".join("h%.0f(%.0f%%,rgb%s)" % (p["hue"], 100 * p["frac"], p["rgb"])
                    for p in img["palette"][:max_items])
    return "颜色：主色 h%s 置信 %.2f｜%s" % (img["wire_hue"], img["confidence"], top)


# ------------------------------------------------------------------ 自检
def _selftest() -> int:
    ok = 0
    bad = []

    def chk(cond, msg):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(msg)

    # 1) 纯灰图：必须判无彩
    g = np.full((200, 300, 3), 255, np.uint8)
    cv2.line(g, (20, 100), (280, 100), (40, 40, 40), 2)
    cv2.putText(g, "GND", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    r = estimate(g)
    chk(r["has_chroma"] is False and r["wire_hue"] is None, "灰图被判有彩：%s" % r)
    chk(is_greyscale(g), "is_greyscale 对灰图返回假")
    chk(r["source"] == "none", "灰图 source 不是 none")

    # 2) 单通道灰度数组（ndim==2）：也必须无彩
    chk(estimate(np.full((50, 50), 255, np.uint8))["has_chroma"] is False, "ndim==2 未被识别")

    # 3) 绿导线 + 红桩：主色应当是绿（OpenCV 绿 ≈ h60），且识别出红
    c = np.full((200, 300, 3), 255, np.uint8)
    cv2.line(c, (20, 60), (280, 60), (60, 160, 60), 2)     # BGR 绿
    cv2.line(c, (20, 120), (120, 120), (60, 60, 200), 2)   # BGR 红
    cv2.putText(c, "R1", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    rc = estimate(c)
    chk(rc["has_chroma"] is True, "彩图被判无彩：%s" % rc)
    hues = [p["hue"] for p in rc["palette"]]
    chk(any(45 <= h <= 75 for h in hues), "没测出绿色桶：%s" % hues)
    chk(any(h <= 10 or h >= 170 for h in hues), "没测出红色桶：%s" % hues)
    chk(0.0 < rc["confidence"] <= 1.0, "confidence 越界：%s" % rc["confidence"])

    # 4) 现成掩膜与自带掩膜结果一致（同一张图上）
    r2 = estimate(c, ink=_ink_mask(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)))
    chk(r2["wire_hue"] == rc["wire_hue"], "给了 ink 掩膜后主色变了：%s vs %s"
        % (r2["wire_hue"], rc["wire_hue"]))

    # 5) 幂等（纯函数）
    chk(estimate(c) == rc, "同一输入两次结果不同（不是纯函数）")

    # 6) 真图（若有）：只要求不抛异常、字段齐
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "..", "work", "_fast_in", "1.png")
    if os.path.exists(p):
        from stage2.text import imread_unicode
        real = imread_unicode(p)
        rr = estimate(real)
        chk(set(rr) == {"has_chroma", "ink_chroma_p95", "palette", "wire_hue",
                        "confidence", "source"}, "真图字段不全：%s" % sorted(rr))
        print("   真图 1.png：%s" % describe(rr))

    if bad:
        print("自检失败 %d 项：" % len(bad))
        for b in bad:
            print("  - " + b)
        return 1
    print("自检 %d 项：全部通过" % ok)
    return 0


def _main(argv: Sequence[str]) -> int:
    if not argv:
        return _selftest()
    from stage2.text import imread_unicode
    for p in argv:
        img = estimate(imread_unicode(p))
        print("%-40s %s" % (p, describe(img, max_items=6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
