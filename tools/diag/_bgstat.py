# -*- coding: utf-8 -*-
"""背景估计体检：解释 bglab 在 0037-Datasheet 上把导线通道打崩的原因。

只读，不改仓库。用法：
    python -X utf8 work\tools\dev\_bgstat.py --dir work\_fast_in --cases 1,0037-Datasheet,2
"""
import argparse
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb\tools\eval")
sys.path.insert(0, r"Y:\workspace\EDA\work\tools\dev")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from pipeline import pskel as P  # noqa: E402


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def stat(tag, bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = P.estimate_bg(bgr)
    bgl = P.estimate_bg_local(bgr, 51)
    d_bgr = np.abs(bgr.astype(np.float32) - np.asarray(bg, np.float32)).max(axis=2)
    d_lab = P.bg_distance(bgr, None, "lab", 51)
    d_labl = P.bg_distance(bgr, bgl, "lab", 51)
    frac = lambda m: float(m.mean())
    m_ad = P.ink_mask(bgr, "adaptive")
    m_ot = P.ink_mask(bgr, "otsu")
    m_lab = P.ink_mask(bgr, "bglab")
    m_bd = P.ink_mask(bgr, "bgdiff")
    print(
        "[%s] shape=%s ring_bg(BGR)=%s gray_p50=%.0f\n"
        "    dE_global p50=%.1f p90=%.1f p99=%.1f | dE_local p50=%.1f p90=%.1f p99=%.1f\n"
        "    ink frac: adaptive=%.4f otsu=%.4f bglab=%.4f bgdiff=%.4f\n"
        "    dE_global>26 frac=%.4f  dE_local>26 frac=%.4f  |deltabg|=%s"
        % (
            tag,
            bgr.shape,
            tuple(int(v) for v in np.asarray(bg).ravel()[:3]),
            float(np.percentile(g, 50)),
            float(np.percentile(d_lab, 50)),
            float(np.percentile(d_lab, 90)),
            float(np.percentile(d_lab, 99)),
            float(np.percentile(d_labl, 50)),
            float(np.percentile(d_labl, 90)),
            float(np.percentile(d_labl, 99)),
            frac(m_ad),
            frac(m_ot),
            frac(m_lab),
            frac(m_bd),
            frac(d_bgr > 26),
            frac(d_labl > 26),
            np.abs(np.asarray(bg, np.float32).ravel()[:3] - np.median(bgl.reshape(-1, 3).astype(np.float32), axis=0)).round(1).tolist(),
        )
    )
    # 局部背景的空间变化：把图切成 4x4 看每块中位色
    h, w = bgr.shape[:2]
    rows = []
    for i in range(4):
        r = []
        for j in range(4):
            blk = bgr[i * h // 4 : (i + 1) * h // 4, j * w // 4 : (j + 1) * w // 4]
            r.append("%3d,%3d,%3d" % tuple(int(v) for v in np.median(blk.reshape(-1, 3), axis=0)))
        rows.append("    " + " | ".join(r))
    print("    4x4 块中位色(BGR):")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"Y:\workspace\EDA\work\_fast_in")
    ap.add_argument("--cases", default="1,0037-Datasheet,2")
    a = ap.parse_args()
    for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
        p = os.path.join(a.dir, c + ".png")
        if not os.path.exists(p):
            print("[skip] %s" % p)
            continue
        bgr = imread_unicode(p)
        if bgr is None:
            print("[skip] decode fail %s" % p)
            continue
        stat(c, bgr)


if __name__ == "__main__":
    main()
