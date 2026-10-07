"""第一阶段：导线 / 元器件 的互相校验（用户 m03802 的第二条指令）。

⚠️⚠️ **2026-07 审计：本模块目前全仓库无人 import，是一段「写好了但从不执行」的代码。**
改 `stage1` 前请先确认它是不是你想要的那条路 —— 照着它改会**看不到任何效果**。
（保留而不删除的理由：里面的判定逻辑本身是可用的设计素材，删掉就找不回来了。
 真要启用它，注意两处已知的坑：①`reconcile.py` 内部调用连通/合并时**没传 `weld`/`ink`**，
 与 `nets.build_nets` 的连通定义不一致；②`:219-226` 只改 `segments`/`n_segments`，
 不动 `endpoints`/`junctions`。）

论证基础（用户原话）：
  * "导线是用来连接元器件的" —— 正面：导线可以介导切图去帮 YOLO 认元件；
    反面：YOLO 认出来的元件周围**必然**存在为连接它而画的导线。
  * 所以「悬空导线」或「悬空元器件」只有两种解释：
      ① 对侧漏检了  → 可以召回（bridge / 补检）
      ② 它根本不是元件或导线（例如嘉立创 logo、标题栏的花边）
                    → 应该删掉（提升精度）

本模块只做**判定与建议**，不改产物；调用方（评测脚本或 pipeline）决定要不要执行。

坐标：全部用图像坐标（左上原点，y 向下），与 stage1 产物里的 bbox_img / segments 一致。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


# ------------------------------------------------------------------ 几何

def box_dist(pt: Point, box: Box) -> float:
    """点到矩形的最短距离；点在矩形内返回 0。"""
    x, y = pt
    x1, y1, x2, y2 = min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(math.hypot(dx, dy))


def seg_box_dist(a: Point, b: Point, box: Box) -> float:
    """线段到矩形的最短距离（端点、以及线段与矩形边的相交都算）。"""
    if box_dist(a, box) == 0.0 or box_dist(b, box) == 0.0:
        return 0.0
    x1, y1, x2, y2 = min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    best = min(box_dist(a, box), box_dist(b, box))
    for c, d in edges:
        best = min(best, _seg_seg(a, b, c, d))
    return float(best)


def _seg_seg(a: Point, b: Point, c: Point, d: Point) -> float:
    """两条线段的最短距离；相交返回 0。"""
    if _intersect(a, b, c, d):
        return 0.0
    return min(_pt_seg(a, c, d), _pt_seg(b, c, d), _pt_seg(c, a, b), _pt_seg(d, a, b))


def _pt_seg(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return float(math.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return float(math.hypot(px - (ax + t * vx), py - (ay + t * vy)))


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


# ------------------------------------------------------------------ 导线图
#
# 关键教训（本项目已踩过两次）：连通性**不能**建在线段端点是否严格相等上。
# pskel 的 rdp + snap 会把一条走线切成很多段，端点常有 1~2px 误差，
# 早期按 `tol=0` 建图会把整条走线拆成几百个几像素的小分量，导致过滤失效。
# 这里用 tol=2.0 的并查集。

def wire_graph(segments: Sequence[Sequence[Point]], tol: float = None,
               ext: float = None, ext_nearest: bool = None):
    """把线段聚成连通分量。

    **委托给 `stage1.labels.wire_groups`** —— 这里原本是项目里第四份图定义，而且
    只做了「端点-端点」，**没有 T 型结点规则**（一条走线拐到另一条走线中间是最常见
    的接线之一），也没有端点朝前延伸规则，所以连接数系统性偏低。现在只有一份实现。

    返回 (groups, seg_of_group)：
      groups       = [{"segs": [i...], "n_ends": k, "ends": [(x,y)...], "length": L}]
      seg_of_group = {seg_idx: group_idx}（按 length 从大到小排序后的组号）
    """
    from .labels import (WIRE_EXT, WIRE_EXT_NEAREST, WIRE_EXT_PERP, WIRE_TOL,
                         wire_groups)
    tol = WIRE_TOL if tol is None else tol
    ext = WIRE_EXT if ext is None else ext
    ext_nearest = WIRE_EXT_NEAREST if ext_nearest is None else ext_nearest
    n = len(segments)
    if not n:
        return [], {}
    pts = []
    for s in segments:
        pts.append((float(s[0][0]), float(s[0][1]), float(s[-1][0]), float(s[-1][1])))
    roots = wire_groups(np.asarray(pts, np.float64), tol=tol, ext=ext,
                        ext_perp=WIRE_EXT_PERP, ext_nearest=ext_nearest)

    buckets: Dict[int, Dict] = {}
    for i, s in enumerate(segments):
        g = buckets.setdefault(roots[i], {"segs": [], "ends": [], "length": 0.0,
                                          "raw_ends": []})
        g["segs"].append(i)
        g["length"] += math.hypot(s[-1][0] - s[0][0], s[-1][1] - s[0][1])
        g["raw_ends"].append((float(s[0][0]), float(s[0][1])))
        g["raw_ends"].append((float(s[-1][0]), float(s[-1][1])))
    for g in buckets.values():
        # 端点 = 只出现一次的端点位置（度=1）；出现两次以上的是内部连接点
        cell = max(tol, 0.5)
        seen: Dict[Tuple[int, int], int] = {}
        for p in g["raw_ends"]:
            key = (int(round(p[0] / cell)), int(round(p[1] / cell)))
            seen[key] = seen.get(key, 0) + 1
        deg1 = [p for p in g["raw_ends"]
                if seen[(int(round(p[0] / cell)), int(round(p[1] / cell)))] == 1]
        merged: List[Point] = []
        for p in deg1:
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) <= tol for q in merged):
                merged.append(p)
        g["ends"] = merged
        g["n_ends"] = len(merged)
        g.pop("raw_ends", None)
    out = sorted(buckets.values(), key=lambda g: -g["length"])
    seg_of_group = {}
    for gi, g in enumerate(out):
        for i in g["segs"]:
            seg_of_group[i] = gi
    return out, seg_of_group


# ------------------------------------------------------------------ 判定

def reconcile(doc: dict,
              anchor_dist: float = 12.0,
              dangling_dist: float = 24.0,
              text_boxes: Optional[Sequence[Box]] = None,
              suspect_max_len: float = 160.0,
              tol: float = 2.0) -> dict:
    """对一份 stage1 产物做互相校验。

    anchor_dist   —— 端点离元件框（或文字框）多近算「被锚定」
    dangling_dist —— 元件离最近的导线多远算「悬空元件」
    suspect_max_len —— 端点全悬空的导线分量，总长小于它才建议删除
    """
    comps = doc.get("components", [])
    segments = [s for s in doc.get("wire_detail", {}).get("segments", [])]
    boxes = [(c["bbox_img"][0], c["bbox_img"][1], c["bbox_img"][2], c["bbox_img"][3]) for c in comps]
    anchors: List[Box] = list(boxes) + [tuple(b) for b in (text_boxes or [])]

    groups, seg_of_group = wire_graph(segments, tol=tol)

    # --- 1) 导线分量：端点有没有锚
    wgroups = []
    drop_segs: List[int] = []
    for gi, g in enumerate(groups):
        anchored = 0
        for e in g["ends"]:
            if any(box_dist(e, b) <= anchor_dist for b in anchors):
                anchored += 1
        suspect = (g["n_ends"] >= 2 and anchored == 0 and g["length"] <= suspect_max_len)
        wgroups.append({
            "group": gi, "n_seg": len(g["segs"]), "length": round(g["length"], 1),
            "n_ends": g["n_ends"], "anchored_ends": anchored, "suspect": suspect,
        })
        if suspect:
            drop_segs.extend(g["segs"])

    # --- 2) 元件：离最近的导线多远
    comp_flags = []
    for c in comps:
        b = (c["bbox_img"][0], c["bbox_img"][1], c["bbox_img"][2], c["bbox_img"][3])
        near = min((seg_box_dist(s[0], s[1], b) for s in segments), default=1e9)
        n_text = 0
        if text_boxes:
            n_text = sum(1 for t in text_boxes if seg_box_dist((t[0], t[1]), (t[2], t[3]), b) <= 2.0
                         or box_dist(((t[0] + t[2]) / 2, (t[1] + t[3]) / 2), b) <= dangling_dist)
        comp_flags.append({
            "id": c.get("id"), "type": c.get("type"), "source": c.get("source"),
            "nearest_wire_px": round(float(near), 1),
            "dangling": bool(near > dangling_dist and n_text == 0),
        })

    n_susp = sum(1 for w in wgroups if w["suspect"])
    stats = {
        "n_components": len(comps),
        "n_segments": len(segments),
        "n_wire_groups": len(groups),
        "n_suspect_groups": n_susp,
        "n_suspect_segments": len(drop_segs),
        "suspect_len_frac": round(
            sum(w["length"] for w in wgroups if w["suspect"]) / max(1.0, sum(w["length"] for w in wgroups)), 4),
        "n_dangling_components": sum(1 for c in comp_flags if c["dangling"]),
    }
    return {
        "wire_groups": wgroups,
        "drop_segments": sorted(drop_segs),
        "components": comp_flags,
        "stats": stats,
    }


def apply_drop(doc: dict, drop_segments: Iterable[int]) -> dict:
    """返回一个把 drop_segments 里的线段删掉的浅拷贝产物（用于评测删除的收益）。"""
    drop = set(int(i) for i in drop_segments)
    src = doc.get("wire_detail", {}).get("segments", [])
    keep = [s for i, s in enumerate(src) if i not in drop]
    new = {k: v for k, v in doc.items() if k != "wire_detail"}
    wd = dict(doc.get("wire_detail", {}))
    wd["segments"] = keep
    new["wire_detail"] = wd
    w = dict(new.get("wires", {}))
    w["n_segments"] = len(keep)
    new["wires"] = w
    return new
