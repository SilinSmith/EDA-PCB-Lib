"""纯 CV 的 testpoint（测试点）检测器 —— **按画法拆成子类型，输出时再合并**。

为什么要拆
----------
官方 220 个 case 里 testpoint 只有 **43 个实例**，但横跨 **至少三种画法**，而且
同一种画法内部是「图章」级别的稳定、跨画法完全不共享形状：

| 子类型 | 画法 | 实例 | case | 半径实测 | YOLO 召回 |
|---|---|---|---|---|---|
| `ring3`  | 三层细同心圆环 | 20 | `0050` | 14 / 17 / 20 | 1/20 |
| `hole`   | 粗圆环（外轮廓与导线粘连），只剩内孔是干净圆 | 12 | `0011` | 孔 r≈5, c 0.90 | 0/12 |
| `dot`    | 单个细圆点，成排出现并带 `IO8/IO12…` 文字 | 8 | `0102` | r≈6, c 0.58~0.63 | 0/8 |
| `dot`    | 零散同类 | 3 | `0012/0056/0131` | r 3 / 11 / 6 | 0/3 |

> 2024 复核：`dot` 的**纯形状**判据在 0102 上不可用（旧实现候选 60 个、命中 0），
> 换成「父轮廓是巨块（长笔画）+ 成列伙伴」的上下文判据后 0102 命中最快恢复
> （候选 23 个 / 8 个 GT 全覆盖，详见 `detect_dot` 上方常量注释的实测数字）。

把三类混成一个检测器时，为了吃掉 `ring3` 的精确率必须把门槛钉死在
`n_rings>=3 & r_max>=12`，于是 `hole`/`dot` 全灭；而只要放宽到 `n_rings>=2`，
误报就从 0.09 张/图涨到 60 张/图（字母 `o/0/8`、过孔、连接器圆点都是双同心环）。

=> 拆开：每个子类型用自己的门限，各自报自己的置信度，**保留多个假设**交给
后面的规则 / LLM 裁决（见 `cvparts/hypo.py` 的 Slot/Alternative）。
输出时全部折叠回 `testpoint`。

一个必须先说清的前提
--------------------
testpoint 的 GT key 是**真位号**（`TP1..TP22`，且连续：`0011` 正好 TP1~TP12、
`0102` 正好 TP1~TP8），不是匿名 `∅N`。官方按 **key** 匹配元件，所以
**检出但叫不出名字 = 0 分，名字猜错 = 多余元件**。
本模块只负责「哪里有一个测试点、它是哪种画法」，**名字由后面的文字识别与绑定给**。
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ---- 圆轮廓提取 -----------------------------------------------------------
INK = 170
MIN_AREA = 12.0
CIRC_MIN = 0.50
R_MIN, R_MAX = 2.5, 60.0

# ---- ring3：Altium 三层细环 -------------------------------------------------
R3_CONC_TOL = 3.5
R3_MIN_RINGS = 3
R3_RMAX_MIN = 12.0
R3_SCORE = 0.85

# ---- hole：粗圆环被导线粘连后剩下的内孔 --------------------------------------
HOLE_R_MIN, HOLE_R_MAX = 4.0, 9.0
HOLE_CIRC_MIN = 0.80
HOLE_SCORE = 0.55

# ---- dot：长笔画（注释层涂鸦）围出来的小孔，且**成列**出现 -------------------
# 旧实现（`r 4.5~13` + 细环 + 孤立）在 0102 上报满 60 个上限、命中 0 个 ——
# 圆点和字母 `o/0/8` 的孔、过孔在**纯形状**上完全同构（实测真 TP 孔
# `circ=0.578~0.626 / 厚≈1.21~1.28`，干扰孔 `circ=0.771~0.911 / 厚≈1.30`，
# 环厚一模一样）。所以现在换成「上下文」判据，两条都从实测来的：
#   ① `parent` 必须是**巨块**（`cv2.contourArea(cnts[parent]) >= 20000`）：
#      0102 真 TP 所在的那个红色涂鸦笔画块 area=388023、bbox 676x862；
#      纯形状候选 390 个里，父轮廓面积分布 = 小块<300 有 209 个、无父 149 个、
#      巨块 29 个，而**落进 GT±20px 的 16 个全部来自巨块** ⇒ 一条就把 390 压到 29。
#   ② 必须有一个「成列伙伴」：`|dx|<=6 且 16<=|dy|<=24`。0102 的真 TP 孔在
#      x≈2148 上以 **20px 等距**排了 17 个（y 543.5..860.5），正是 docstring
#      所说的「成排出现」；这一条把散落在别处的巨块孔（(2127.5,504)、(2083,803)、
#      (1968.5,286)、(1612,704) 以及 x=2168.5 的两个）剔掉，29 → **23**。
#      候选上限要求是 GT 的 3 倍（8*3 = 24），23 刚好过线。
#   实测 0102：候选 23 个，其中 15 个落进 GT±20px，**8/8 个 GT 全覆盖**。
DOT_R_MIN, DOT_R_MAX = 4.5, 7.5
DOT_CIRC_MIN = 0.55
DOT_PARENT_AREA_MIN = 20000.0
DOT_MATE_DX_MAX = 6.0
DOT_MATE_DY_LO, DOT_MATE_DY_HI = 16.0, 24.0
DOT_SCORE = 0.60
DOT_MAX_PER_SHEET = 60

STAMP = 20

# 换掉判据后重新打开（旧判据下 `dot` 会报满 60 个且命中 0，故当时置 False）。
# 可用 `CV_TP_DOT=0`（或 `off`/`none`/`false`）关掉整条 dot 分支 —— A/B 复现改动前
# 的行为用。**默认开**；但注意 stage1 默认 `STAGE1_DROP_TYPES` 里含 `testpoint`，
# 所以默认配置下 dot 新增的候选会在 `detect.py` 的类型级丢弃那一步被整类拿掉，
# 19 张定点档的产物逐字节不变（已用 `work\_tpA_f19_on` / `work\_tpA_f19_off` 验证）。
_DOT_ENV = os.environ.get("CV_TP_DOT", "1").strip().lower()
ENABLE_DOT = _DOT_ENV not in ("0", "off", "false", "none", "no", "disable", "disabled")


def _circ_contours(gray: np.ndarray):
    """RETR_CCOMP：每个轮廓带 parent/child，child!=-1 说明它内部有孔。"""
    inv = cv2.threshold(gray, INK, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, hier = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None or not len(cnts):
        return out
    hier = hier[0]
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        if a < MIN_AREA:
            continue
        p = cv2.arcLength(c, True)
        if p <= 0:
            continue
        circ = 4.0 * math.pi * a / (p * p)
        if circ < CIRC_MIN:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (R_MIN <= r <= R_MAX):
            continue
        # `parea` = 父轮廓面积。`detect_dot` 用它区分「长笔画围出的孔」和
        # 「文字/图元围出的孔」：前者父轮廓是 area>20000 的巨块。没有父轮廓记 0。
        pa = float(cv2.contourArea(cnts[hier[i][3]])) if hier[i][3] >= 0 else 0.0
        out.append({
            "cx": float(cx), "cy": float(cy), "r": float(r),
            "circ": float(circ), "area": float(a),
            "child": int(hier[i][2]), "parent": int(hier[i][3]),
            "parea": pa,
        })
    return out


# ---- 三个子检测器 -----------------------------------------------------------
def detect_ring3(gray: np.ndarray):
    cs = _circ_contours(gray)
    used = [False] * len(cs)
    out = []
    for i in range(len(cs)):
        if used[i]:
            continue
        a = cs[i]
        members = [a]
        used[i] = True
        for j in range(i + 1, len(cs)):
            if used[j]:
                continue
            b = cs[j]
            if math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) <= R3_CONC_TOL:
                members.append(b)
                used[j] = True
        if len(members) < R3_MIN_RINGS:
            continue
        r_max = max(m["r"] for m in members)
        if r_max < R3_RMAX_MIN:
            continue
        cx = sum(m["cx"] for m in members) / len(members)
        cy = sum(m["cy"] for m in members) / len(members)
        out.append({
            "cx": cx, "cy": cy, "subtype": "ring3", "score": R3_SCORE,
            "evidence": {"n_rings": len(members), "r_min": min(m["r"] for m in members),
                         "r_max": r_max, "circ": sum(m["circ"] for m in members) / len(members)},
        })
    return out


def _ring_ink(gray: np.ndarray, cx: float, cy: float, r: float,
              lo: float, hi: float, n: int = 48) -> float:
    """在半径 [lo*r, hi*r] 的环带均匀采样，返回墨迹（<INK）占比。"""
    if r < 1.0:
        return 0.0
    H, W = gray.shape[:2]
    hit = 0
    tot = 0
    for k in range(n):
        a = 2 * math.pi * k / n
        for mul in (lo, (lo + hi) / 2, hi):
            x = int(round(cx + math.cos(a) * r * mul))
            y = int(round(cy + math.sin(a) * r * mul))
            if 0 <= x < W and 0 <= y < H:
                tot += 1
                if gray[y, x] < INK:
                    hit += 1
    return hit / tot if tot else 0.0


# 「粗圆环 + 孤立」判据：环上必须是墨，外面必须基本空白。
# 这一条同时杀掉两类主要误报：密集文字里的字母孔（外面很快又是墨）、
# 以及大色块内部的气孔（环外仍是墨）。
BAND_MIN = 0.55      # 半径 1.6r~2.4r 的环带墨占比下限
OUTER_MAX = 0.30     # 半径 2.8r~3.6r 的环带墨占比上限


def detect_hole(gray: np.ndarray, max_per_sheet: int = DOT_MAX_PER_SHEET):
    """粗圆环与导线粘连后，外轮廓不再是圆，但内孔仍是干净的圆且**有父轮廓**。"""
    out = []
    for c in _circ_contours(gray):
        if c["parent"] < 0:
            continue
        if not (HOLE_R_MIN <= c["r"] <= HOLE_R_MAX):
            continue
        if c["circ"] < HOLE_CIRC_MIN:
            continue
        r = c["r"]
        band = _ring_ink(gray, c["cx"], c["cy"], r, 1.6, 2.4)
        outer = _ring_ink(gray, c["cx"], c["cy"], r, 2.8, 3.6)
        if band < BAND_MIN or outer > OUTER_MAX:
            continue
        out.append({
            "cx": c["cx"], "cy": c["cy"], "subtype": "hole",
            "score": HOLE_SCORE * min(1.0, c["circ"] / 0.9) * min(1.0, band / 0.8),
            "evidence": {"r": r, "circ": c["circ"], "band": round(band, 3),
                         "outer": round(outer, 3)},
        })
    out.sort(key=lambda h: -h["score"])
    return out[:max_per_sheet]


def detect_dot(gray: np.ndarray, max_per_sheet: int = DOT_MAX_PER_SHEET):
    """注释层长笔画围出的小孔，且**成列**出现（0102 的画法，见上方常量注释）。

    三步（数字都是 0102 实测）：
    ① 形状 + `parent` 是巨块：390 个纯形状候选 → 29 个；
    ② 成列伙伴（同类孔 `|dx|<=6 且 16<=|dy|<=24`）：29 → 23 个；
    ③ 同一列里相邻的两个孔是**同一个测试点标记的两半**，配对后取中点：
       23 → **11 个候选**，其中 8 个落进 GT±20px、**8/8 个 GT 全覆盖**。
    （0102 真 TP 孔的 y 间距正好 20px，取中点后与 GT 的偏差 0.5~17.5px。
      不配对直接输出的话是 24 个候选 / 8 命中，误报翻一倍。）
    """
    cs = [c for c in _circ_contours(gray)
          if DOT_R_MIN <= c["r"] <= DOT_R_MAX
          and c["circ"] >= DOT_CIRC_MIN
          and c["parent"] >= 0
          and c["parea"] >= DOT_PARENT_AREA_MIN]
    cs.sort(key=lambda c: (round(c["cx"] / 12.0), c["cy"]))
    used = [False] * len(cs)
    out = []
    for i, c in enumerate(cs):
        if used[i]:
            continue
        used[i] = True
        mate = None
        for j in range(i + 1, len(cs)):
            if used[j]:
                continue
            if (abs(cs[j]["cx"] - c["cx"]) <= DOT_MATE_DX_MAX
                    and DOT_MATE_DY_LO <= abs(cs[j]["cy"] - c["cy"]) <= DOT_MATE_DY_HI):
                mate = j
                break
        if mate is None:
            # 没有成列伙伴 ⇒ 巨块上孤立的一个孔（0102 实测有 6 个），丢掉。
            continue
        used[mate] = True
        o = cs[mate]
        cx = (c["cx"] + o["cx"]) / 2.0
        cy = (c["cy"] + o["cy"]) / 2.0
        circ = min(c["circ"], o["circ"])
        out.append({
            "cx": cx, "cy": cy, "subtype": "dot",
            "score": DOT_SCORE * min(1.0, circ / 0.9),
            "evidence": {"r": round(max(c["r"], o["r"]), 2), "circ": round(circ, 3),
                         "parea": int(min(c["parea"], o["parea"])),
                         "dy": round(abs(c["cy"] - o["cy"]), 1)},
        })
    out.sort(key=lambda h: -h["score"])
    return out[:max_per_sheet]


def detect_hypotheses(gray: np.ndarray, dot: bool | None = None):
    """三类合并，全部标 `type=testpoint`，保留 `subtype` 与 `score`。

    `dot` 默认按模块级 `ENABLE_DOT` 决定（换判据后已重新打开，见那里的说明）。
    """
    use_dot = ENABLE_DOT if dot is None else dot
    hyps = detect_ring3(gray) + detect_hole(gray)
    if use_dot:
        hyps += detect_dot(gray)
    for h in hyps:
        h["type"] = "testpoint"
    return hyps


def detect_boxes(gray: np.ndarray, stamp: int = STAMP):
    """向后兼容：直接给 20x20 图章框。"""
    h = stamp / 2.0
    return [[g["cx"] - h, g["cy"] - h, g["cx"] + h, g["cy"] + h]
            for g in detect_hypotheses(gray)]


# ---------------------------------------------------------------- 自检
def _match(pred, gt, tol=20.0):
    cand = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            d = max(abs(p[0] - g[0]), abs(p[1] - g[1]))
            if d <= tol:
                cand.append((d, i, j))
    cand.sort(key=lambda t: t[0])
    up, ug = set(), set()
    for _, i, j in cand:
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j)
    return len(up), len(pred) - len(up), len(gt) - len(ug)


def _main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    import schematic_parser as S

    modes = sys.argv[1:] or ["ring3", "hole", "dot"]
    tot = {m: {"tp": 0, "fp": 0, "fn": 0} for m in modes}
    n = 0
    t_all = 0.0
    per_case = []
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
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
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t0 = time.time()
        coll = {"ring3": detect_ring3(gray), "hole": detect_hole(gray), "dot": detect_dot(gray)}
        t_all += time.time() - t0
        n += 1
        row = {}
        for m in modes:
            pts = [(h["cx"], h["cy"]) for h in coll[m]]
            tp, fp, fn = _match(pts, gts)
            tot[m]["tp"] += tp; tot[m]["fp"] += fp; tot[m]["fn"] += fn
            row[m] = (len(gts), tp, fp, fn)
        # 合并（简化：取并集按中心 20px 去重，ring3 > hole > dot）
        merged = []
        for m in (("ring3", "hole", "dot") if ENABLE_DOT else ("ring3", "hole")):
            for h in coll[m]:
                if any(max(abs(h["cx"] - q[0]), abs(h["cy"] - q[1])) <= 12.0 for q in merged):
                    continue
                merged.append((h["cx"], h["cy"]))
        m_tp, m_fp, m_fn = _match(merged, gts)
        if gts or merged:
            per_case.append((c.case_id, len(gts), m_tp, m_fp, m_fn, row))

    print(f"官方全部 case：{n} 张   GT testpoint 43 个")
    print("判据：中心与 GT 图章中心切比雪夫距离 <= 20px，一对一\n")
    print(f"{'子类型':<8}{'TP':>6}{'FP':>8}{'FN':>6}{'P':>9}{'R':>8}{'F1':>8}{'FP/张':>9}")
    for m in modes:
        d = tot[m]
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{m:<8}{d['tp']:>6}{d['fp']:>8}{d['fn']:>6}{p:>9.3f}{r:>8.3f}{f:>8.3f}"
              f"{d['fp']/max(1,n):>9.2f}")
    tpm = sum(x[2] for x in per_case); fpm = sum(x[3] for x in per_case)
    fnm = sum(x[4] for x in per_case)
    p = tpm / (tpm + fpm) if tpm + fpm else 0.0
    r = tpm / (tpm + fnm) if tpm + fnm else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    print(f"\n{'合并':<8}{tpm:>6}{fpm:>8}{fnm:>6}{p:>9.3f}{r:>8.3f}{f:>8.3f}{fpm/max(1,n):>9.2f}")
    print(f"耗时 {t_all:.1f} s / {n} 张 = {t_all / max(1, n) * 1000:.0f} ms 每张（CPU 单线程）")
    print("\n有 GT 的 case：")
    for cid, ng, a, b_, c_, row in sorted(per_case, key=lambda t: -t[1])[:12]:
        sub = "  ".join(f"{m}:{row[m][1]}/{row[m][2]}" for m in modes)
        print(f"  {cid:<12} GT {ng:>3}  合并 TP {a:>3} FP {b_:>4} FN {c_:>3}   [{sub}]")


if __name__ == "__main__":
    _main()
