"""为 CV testpoint 找判别门槛。

tp_diag 已经暴露了信号：真 testpoint 是**三层同心环**，半径紧贴 14.1 / 20.1；
误报是半径 3~5px 的双环（字母 o/0/8、小连接器圆、度符号）。

本脚本把阈值洗一遍，看有没有能把精确率拉起来的工作点。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "eda-pcb"))
sys.path.insert(0, str(HERE))
import schematic_parser as S  # noqa: E402
from cvparts.testpoint import detect_rings  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 220


def main():
    groups = []          # (r_min, r_max, n_rings, circ, is_tp, case)
    n_gt = 0
    ncase = 0
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        if ncase >= LIMIT:
            break
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        ncase += 1
        H = bgr.shape[0]
        gts = []
        for k, v in (doc.get("components") or {}).items():
            if str(v.get("type", "")).strip() != "testpoint":
                continue
            b = v.get("bbox")
            if not b:
                continue
            x1, y1, x2, y2 = [float(t) for t in b]
            gts.append(((x1 + x2) / 2.0, H - (y1 + y2) / 2.0))
        n_gt += len(gts)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for g in detect_rings(gray):
            cx, cy = g["center"]
            hit = any(max(abs(cx - gx), abs(cy - gy)) <= 20.0 for gx, gy in gts)
            groups.append((g["r_min"], g["r_max"], g["n_rings"], g["circ"], hit, c.case_id))
    print(f"{ncase} 张   GT {n_gt}   候选组 {len(groups)}   其中命中 {sum(g[4] for g in groups)}")

    def ev(pred):
        tp = sum(1 for g, p in zip(groups, pred) if p and g[4])
        fp = sum(1 for g, p in zip(groups, pred) if p and not g[4])
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / n_gt if n_gt else 0.0
        f = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        return tp, fp, p_, r_, f

    print("\n=== 扫描 r_max 下限 × 环数下限 ===")
    print(f"{'r_max>=':>8}{'n_ring>=':>9}{'TP':>6}{'FP':>8}{'P':>8}{'R':>8}{'F1':>8}{'FP/张':>8}")
    best = None
    for rmax_min in (0, 6, 8, 10, 12, 14, 16, 18):
        for nmin in (2, 3):
            pred = [g[1] >= rmax_min and g[2] >= nmin for g in groups]
            tp, fp, p_, r_, f = ev(pred)
            print(f"{rmax_min:>8}{nmin:>9}{tp:>6}{fp:>8}{p_:>8.3f}{r_:>8.3f}{f:>8.3f}{fp/max(1,ncase):>8.2f}")
            if best is None or f > best[0]:
                best = (f, rmax_min, nmin)
    print(f"\n最佳 F1: {best[0]:.3f}   r_max>={best[1]}  n_rings>={best[2]}")

    print("\n=== 再加「半径比」下限 r_max/r_min ===")
    print(f"{'ratio>=':>8}{'TP':>6}{'FP':>8}{'P':>8}{'R':>8}{'F1':>8}")
    for ratio_min in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        pred = [g[1] >= best[1] and g[2] >= best[2] and (g[1] / max(1e-6, g[0])) >= ratio_min
                for g in groups]
        tp, fp, p_, r_, f = ev(pred)
        print(f"{ratio_min:>8.1f}{tp:>6}{fp:>8}{p_:>8.3f}{r_:>8.3f}{f:>8.3f}")

    print("\n=== 半径分布（r_max >= 10 的组）===")
    big = [g for g in groups if g[1] >= 10]
    print(f"共 {len(big)} 组，命中 {sum(g[4] for g in big)}，误报 {sum(1 for g in big if not g[4])}")
    print("r_max 直方图（命中）:", dict(sorted(Counter(round(g[1]) for g in big if g[4]).items())))
    print("r_max 直方图（误报）:", dict(sorted(Counter(round(g[1]) for g in big if not g[4]).items())))
    print("r_min 直方图（误报）:", dict(sorted(Counter(round(g[0]) for g in big if not g[4]).items())))
    print("按 case 的误报:", dict(Counter(g[5][-4:] for g in big if not g[4]).most_common(12)))


if __name__ == "__main__":
    main()
