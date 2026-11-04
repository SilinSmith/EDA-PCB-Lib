# -*- coding: utf-8 -*-
"""网线端点吸附到「电源/地符号」的连接点（`FUSION_PWRCLIP`，默认 **1** = 开；`0` 关，A/B 用）。

## 病：我们画的网线把电源符号自己那截引线也圈进来了

`net_line_f1`（权重 0.10，10 张均值 0.4984，case 2 只有 0.1707）里最大的一类错，是
**同一条网线的两端对不上**：`schematic_parser/metrics.py:266` 的 `_seg_distance` 取两端
距离的**最大值**，任一端差 >5px（`cfg.line_tol`）这条边就算 fp+fn。

case 2 的取证（`work/_c2overlay.png`，GT 蓝线 / 我方橙线）：

| 网 | GT | 我方 |
| --- | --- | --- |
| `net_12 (U2.2,∅144.142)` | 边 `(933,787)-(1021,787)` | `net_2 (U2.2)` 边 `(907,785.81)-(1020,786.52)` |

右端（`U2.2` 的脚点 `(1018,786.5)`）两边一致；左端我们**多画了 26px** —— GT 的网线从
**电源符号的连接点**起画，我们把符号自己那截引线也当成了导线。

## 连接点的规律（case 1/2 的 7 颗 `∅` 元件全部吻合）

电源/地符号的脚点 = **bbox 四边之一向外 10px、且在那一侧居中**的那个点：

| GT 元件 | bbox | 脚点 | 方向 |
| --- | --- | --- | --- |
| `∅202` v | `[1074,286,1084,296]` | `(1079,276)` | 上边外 10 |
| `∅109` gnd | `[450,425,470,444]` | `(460,454)` | 下边外 10 |
| `∅104` gnd | `[1096,423,1115,443]` | `(1125,433)` | 右边外 10 |
| `∅144` v | `[905,781,915,791]` | `(925,786)` | 右边外 10 |

我们**检出**了这些符号（case 2 的 `p020 v [908.8,781.4,918.8,791.4]` 对 GT `∅144`，尺寸一样、
整体偏 3.8px），只是**不给它们发脚**（`pin.missing` 那 114 根里 `gnd` 59 / `v` 51）——
所以这里不依赖脚点，直接用 bbox 生成 4 个候选。

## 规则（纯几何，只碰我们自己的产物）

对每条网线的每个端点 `E`（另一端 `O`）：

1. 生成候选点：每颗 `gnd`/`v` 元件的 bbox 四边各外 10px、居中的点；
2. 候选点必须落在边**内侧**（沿 `E→O` 方向投影 `t ∈ (0.5, NEAR]`）、垂距 `≤ PERP`；
3. **守卫**：端点必须贴在这颗符号上（到 bbox 的距离 `≤ BOXTOL`）—— 没有这条守卫，
   T 型接头处的中段端点会被误吸（123 次移动 vs 93 次，且净分更低）；
4. 取最近的候选点，把端点吸过去。

## 实测（离线试算 `work/_pwrclip.py` + `fast.py --score-only`，交付态 `work/_fast_out_pnrn` = 82.72）

| 守卫 | 吸端点 | 10 张均值 | 导线 F1 |
| --- | --- | --- | --- |
| 无守卫 | 123 | 82.87（+0.15） | 0.5135 |
| `BOXTOL=4` | 20 | 82.87（+0.15） | 0.5136 |
| **`BOXTOL=8`** | 93 | **83.03（+0.31）** | **0.5291** |
| `BOXTOL=12` | 99 | 83.03（+0.31） | 0.5291 |
| `BOXTOL=8, PERP=2` | 92 | 83.03（+0.31） | 0.5291 |

逐案（`BOXTOL=8`）**全部非负**：1 +0.10、2 +0.73、4 +0.33、5 +0.52、6 +0.58、8 +0.57、
10 +0.24，3/7/9 不动。

## 不做的事

- **不给 `gnd`/`v` 补脚**：`∅` 元件的脚号是标注工具的全局计数器（case 1 的 8 颗是
  107/111/115/119/123/198/202/206，步长 4），图像里没有、复现不了 ⇒ 网节点集合永远
  配不上（`netH` 的 ①，占成员数 22%）；
- **不碰别的通道**：只改 `nets[*].edges` 的端点坐标，`components`/`pins`/`hyperGraph`
  一字不动。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from envflag import env_on

# ★ 2026-11-04 曾翻默认 1 → 0（13 页 +0.2110：2 +1.88 / 5 +1.27 / 8 −0.81 / 3 −0.42）。
#   池侧只值 −0.09（`onoffShip3` 29.5856 → `onoffShip4` 29.6782），收益比不划算 ⇒ 回退到 1。
#   注：`0058` 那个 118 → 275 成员的巨型网**不是它**造成的（单案复现证明是 `STAGE1_CROSS_TAGS`），
#   别把那条证据记到这里。
ON = env_on("FUSION_PWRCLIP", "1")

PWR_TYPES = ("v", "gnd", "ground", "power", "earth")
STEM = 10.0          # 连接点在 bbox 边外多少 px
NEAR = 45.0          # 端点最多内吸多少 px
PERP = 4.0           # 候选点到边的垂距上限
BOXTOL = 8.0         # 端点必须贴在符号框上（外扩 px）


def _canon(t) -> str:
    t = str(t or "").strip().lower()
    return {"ground": "gnd", "power": "v", "earth": "gnd"}.get(t, t)


def _cands(doc):
    """[(x, y, key, bbox)]：每颗电源/地符号的 4 个连接点候选。"""
    out = []
    for key, c in (doc.get("components") or {}).items():
        if _canon((c or {}).get("type")) not in PWR_TYPES:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in (c.get("bbox") or c.get("box"))[:4]]
        except Exception:                                       # noqa: BLE001
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        bb = (x0, y0, x1, y1)
        out.append((cx, y1 + STEM, str(key), bb))
        out.append((cx, y0 - STEM, str(key), bb))
        out.append((x0 - STEM, cy, str(key), bb))
        out.append((x1 + STEM, cy, str(key), bb))
    return out


def _box_dist(p, bb) -> float:
    """点到 bbox 的距离（框内为 0）。"""
    dx = max(bb[0] - p[0], 0.0, p[0] - bb[2])
    dy = max(bb[1] - p[1], 0.0, p[1] - bb[3])
    return math.hypot(dx, dy)


def clip_doc(doc, near=NEAR, perp=PERP, boxtol=BOXTOL):
    """就地吸附；返回 trace 列表。"""
    cands = _cands(doc)
    trace = []
    if not cands:
        return trace
    for nid, net in (doc.get("nets") or {}).items():
        for eid, raw in ((net or {}).get("edges") or {}).items():
            try:
                pts = [((float(q["x"]), float(q["y"])) if isinstance(q, dict)
                        else (float(q[0]), float(q[1]))) for q in raw]
            except Exception:                                   # noqa: BLE001
                continue
            if len(pts) < 2:
                continue
            for end in (0, len(pts) - 1):
                E = pts[end]
                O = pts[1 - end if end == 0 else -2]
                dx, dy = O[0] - E[0], O[1] - E[1]
                L = math.hypot(dx, dy)
                if L < 1e-6:
                    continue
                ux, uy = dx / L, dy / L
                best = None
                for px, py, key, bb in cands:
                    vx, vy = px - E[0], py - E[1]
                    t = vx * ux + vy * uy
                    if t <= 0.5 or t > near:
                        continue
                    off = abs(vx * (-uy) + vy * ux)
                    if off > perp:
                        continue
                    if boxtol is not None and _box_dist(E, bb) > boxtol:
                        continue
                    if best is None or t < best[0]:
                        best = (t, (px, py), key, off)
                if best is None:
                    continue
                trace.append({"net": nid, "edge": eid, "end": end,
                              "from": [round(E[0], 2), round(E[1], 2)],
                              "to": [round(best[1][0], 2), round(best[1][1], 2)],
                              "sym": best[2], "d": round(best[0], 1)})
                pts[end] = best[1]
            net["edges"][eid] = [{"x": round(x, 2), "y": round(y, 2)} for x, y in pts]
    return trace


def apply_file(src, dst, **kw):
    """读 `src` 的 `<case>_Pred.json`，吸端点后写 `dst`；失败返回 None（导出退化成原样）。

    ★ 原子落盘：`dst` 常常**就是** `final_results` 里已经发布的那个 `<case>_Pred.json`
    （`submission/run.py` 的 `_pwrclip_step` 就地改写），而判题机按墙钟硬杀是现实场景 ——
    直接 `write_text` 被杀在半路就留下半截 JSON、该 case 直接 0 分。先写同目录 `.tmp`
    再 `os.replace`，任何时刻被读到的都是完整的旧文件或完整的新文件。
    """
    src, dst = Path(src), Path(dst)
    doc = json.loads(src.read_text(encoding="utf-8"))
    trace = clip_doc(doc, **kw)
    if trace:
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(dst))
    return {"moved": len(trace), "trace": trace}
