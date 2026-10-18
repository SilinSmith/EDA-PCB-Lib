# -*- coding: utf-8 -*-
"""Reverse-engineered implementation of the competition's four F1 metrics.

The official scorer has not been released yet (the problem statement says it
"will be provided together with the code"), so this module follows the written
specification **plus the answers in the published Q&A** and keeps every
ambiguous choice behind a switch in :class:`EvalConfig`.

Metrics and weights (``Score_Accuracy_SingleCase``)::

    100 * (0.30*ComponentF1 + 0.25*PinF1 + 0.35*NetHypergraphF1 + 0.10*NetLineF1)

官方口径的三条关键答案（`Y:\\workspace\\EDA\\docs\\QA_2025.txt`，2025-10 版）：

* **数据 Q3/Q4** —— type 以随包发布的 ``type_list.md`` 为准；该表**不区分引脚数
  与极性**（只有 ``crystal``/``amp``/``mosfet``/``bjt``，没有
  ``crystal_2pin``/``amp_3pin``/``mosfet_npn``/``bjt_pnp``）。所以比较前要用
  :func:`canon_type` 归一化。
* **评测 Q5/Q6** —— 「Vcc 和 gnd 是 ∅，**直接是坐标匹配**」「GND 的序号名字不影响
  评分」。⇒ 键形如 ``∅NNN`` 的 GT 元件**不走键匹配，走坐标匹配**。
* **评测 Q2** —— 「**非 box 类型元器件 pin name, pin number 不参与评分**」。
  ⇒ ``pin_f1`` 只统计 **GT 里 type 为 box 的元件**的引脚。
* **评测 Q3.1** —— LineF1 是「按照端点进行匈牙利匹配」。
* **数据 Q6** —— 电源/地那类 ∅ 的 pin number 是「**随机生产的**」，不必也无法重建。
* **最新 QA（2026-10，用户转述）** —— 「非 box 元件 pinnumber 不计分，所以 rcl 等在
  hygraph 中相当于**不区分引脚** 1.2；gnd/vcc 同理；vcc 和 gnd 默认属于同一个 net；
  输出规范**参考 10gt**」。⇒ 净 hyperGraph 的节点对非 box 元件折叠成
  ``组件.*``（:attr:`EvalConfig.net_pin_collapse_nonbox`，官方 9 张上 +1.08 分）。
  输出侧：电源符号按 10GT 的样子给（单个 `∅NNN` 元件 + 一个 pin + `value` 写 `GND`/`VCC`）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .data import (IC_TYPES, VALUE_TYPES, net_pin_set, normalise_doc,
                   parse_hypergraph, pin_key_to_number, build_hypergraph)

try:  # scipy ships with ultralytics, but stay importable without it
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

WEIGHTS = {'component': 0.30, 'pin': 0.25, 'net_hypergraph': 0.35, 'net_line': 0.10}

# ---------------------------------------------------------------------------
# 官方 type 归一化（`10GT/type_list.md`，33 类，不区分引脚数与极性）

TYPE_ALIASES = {
    # 引脚数不分
    'crystal': 'crystal', 'crystal_2pin': 'crystal', 'crystal_3pin': 'crystal',
    'crystal_4pin': 'crystal',
    'amp': 'amp', 'amp_3pin': 'amp', 'amp_5pin': 'amp', 'amp_7pin': 'amp',
    'mosfet': 'mosfet', 'mosfet_npn': 'mosfet', 'mosfet_pnp': 'mosfet',
    'mosfet_n': 'mosfet', 'mosfet_p': 'mosfet',
    'bjt': 'bjt', 'bjt_npn': 'bjt', 'bjt_pnp': 'bjt',
    # 题面/旧数据里的别名
    'resistor': 'r', 'capacitor': 'c', 'ic': 'box', 'boxic': 'box',
    'battery': 'battary', 'transformer': 'transfomer', 'antanna': 'ant',
    'screw': 'm3螺丝', 'm3': 'm3螺丝',
}

# GT 里「无名对象」的键：直径符号 + 数字（就是源文档的 componentID）
UKEY_RE = re.compile(r'^\u2205\d+$')
# 我们自己生成的兜底键（assemble.py 的 `"p%03d" % i`）
FKEY_RE = re.compile(r'^p\d{3,4}$')


def canon_type(t) -> str:
    s = (t or '').strip()
    return TYPE_ALIASES.get(s, s)


def is_unkeyed(key: str) -> bool:
    """True for ``∅NNN`` (GT) and ``pNNN`` (our fallback) keys."""
    return bool(UKEY_RE.match(key or '') or FKEY_RE.match(key or ''))


# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    bbox_tol: float = 20.0
    pin_tol: float = 5.0
    line_tol: float = 5.0
    # 'corners'  all four bbox numbers within bbox_tol
    # 'center'   centre distance within bbox_tol (size ignored)
    # 'iou'      IoU >= bbox_iou_min
    bbox_mode: str = 'corners'
    bbox_iou_min: float = 0.5
    # net comparison: 'micro' counts pins inside matched nets, 'exact' requires
    # the whole pin set of a matched net to be identical.
    net_mode: str = 'micro'
    # 连通性通道的 tp 口径（2026-10-14 用官方判分机产物实测标定，见 RESULTS §5.21）：
    #   官方 `connectivity.f1.soft_tp` = Σ_{配对网} 2|P∩G|/(|P|+|G|)（**每对网的
    #   pin-set 软 F1 之和**，配对 = 以该 Dice 为目标最大化的匈牙利），随后
    #   precision = soft_tp/pred_net_count、recall = soft_tp/gt_net_count、
    #   f1 = 2*soft_tp/(pred+gt)。用我们第 6 次提交的真实预测复算，
    #   case 1/7/2/3 的 soft_tp 与 f1 与官方报告**逐位相同**。
    #   本文件早先按「tp = Σ|P∩G| 交集」实现（偏乐观：官方 10 张上高估 1.73 分），
    #   设 net_soft=False 可回到那个旧口径（历史基线都是旧口径）。
    net_soft: bool = True
    # compare text for IC (Name) and R/C (value)?
    compare_text: bool = True
    # do edges of a matched net pair have to be compared unordered?
    line_unordered: bool = True
    # 无名元的匹配方式（评测 Q5「Vcc 和 gnd 是 ∅，直接是坐标匹配」）
    #   'off' 一律按键匹配（旧口径）
    #   'gt'  只有 GT 侧是 ∅NNN 的元件走坐标匹配（**官方口径**）
    #   'all' 所有键没配上的元件都再走一轮坐标匹配（宽松上界）
    unkeyed_match: str = 'gt'
    # 坐标匹配的可用距离（中心距，px）；超过就置 INF 不配
    coord_gate: float = 40.0
    # 坐标匹配用**中心距**还是左上角距：2026-11 新口径（protocol3）按「几何位置」
    # 配 GND/电源符号，而我们的符号框常比 GT 的小一圈，左上角距会被尺寸差放大
    # （实测 44/108 个未配上的 ∅ 符号，中心距 p50 13.4 / p90 18.7px，全都够 20px 门槛）。
    coord_center: bool = False
    # 类型不同时的代价惩罚，让匈牙利优先配同类
    coord_type_penalty: float = 1.0e4
    # pin_f1 是否只统计 box（评测 Q2「非 box 类型元器件 pin name, pin number
    # 不参与评分」）；设 False 回到旧口径（全部元件）
    pin_box_only: bool = True
    # 净 hyperGraph 的**节点口径**：把非 box 元件的引脚号折成通配 `*`。
    # ★ 2026-10-14 官方产物标定：官方**不折算**（case 1/7/2/3 的 `soft_tp`/`f1`
    # 只在「字面 refs」下逐位复现；折算后 netH 差 0.033/张）⇒ 默认关。
    # 折算会把「我们把 c/r 的脚号读错」这一事实掩盖掉，官方 10 张上虚高 7.4 分。
    # 设 True 回到旧口径（2026-10-14 之前的基线都是它）。
    net_pin_collapse_nonbox: bool = False
    # type 是否按 type_list 归一化
    canon_types: bool = True


@dataclass
class F1:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    detail: Dict[str, int] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {'tp': self.tp, 'fp': self.fp, 'fn': self.fn,
                'precision': round(self.precision, 6),
                'recall': round(self.recall, 6),
                'f1': round(self.f1, 6)}


@dataclass
class CaseScore:
    case_id: str
    component: F1
    pin: F1
    net_hypergraph: F1
    net_line: F1
    diagnostic: Dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return 100.0 * (
            WEIGHTS['component'] * self.component.f1 +
            WEIGHTS['pin'] * self.pin.f1 +
            WEIGHTS['net_hypergraph'] * self.net_hypergraph.f1 +
            WEIGHTS['net_line'] * self.net_line.f1)

    def as_dict(self) -> dict:
        return {
            'case_id': self.case_id,
            'score': round(self.score, 4),
            'component_f1': round(self.component.f1, 6),
            'pin_f1': round(self.pin.f1, 6),
            'net_hypergraph_f1': round(self.net_hypergraph.f1, 6),
            'net_line_f1': round(self.net_line.f1, 6),
            'counts': {
                'component': self.component.as_dict(),
                'pin': self.pin.as_dict(),
                'net_hypergraph': self.net_hypergraph.as_dict(),
                'net_line': self.net_line.as_dict(),
            },
            'diagnostic': {k: round(v, 6) for k, v in self.diagnostic.items()},
        }


# ---------------------------------------------------------------------------
# helpers

def _box(raw) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _centre(b) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_ok(a, b, cfg: EvalConfig) -> Optional[bool]:
    """``None`` when either box is unusable (treated as a mismatch)."""
    ba, bb = _box(a), _box(b)
    if ba is None or bb is None:
        return None
    if cfg.bbox_mode == 'iou':
        ix = max(0.0, min(ba[2], bb[2]) - max(ba[0], bb[0]))
        iy = max(0.0, min(ba[3], bb[3]) - max(ba[1], bb[1]))
        inter = ix * iy
        ua = (ba[2] - ba[0]) * (ba[3] - ba[1])
        ub = (bb[2] - bb[0]) * (bb[3] - bb[1])
        union = ua + ub - inter
        return (inter / union if union > 0 else 0.0) >= cfg.bbox_iou_min
    if cfg.bbox_mode == 'center':
        cxa, cya = _centre(ba)
        cxb, cyb = _centre(bb)
        return math.hypot(cxa - cxb, cya - cyb) <= cfg.bbox_tol
    return max(abs(ba[i] - bb[i]) for i in range(4)) <= cfg.bbox_tol


def component_text_ok(gt: dict, pred: dict, cfg: EvalConfig) -> bool:
    if not cfg.compare_text:
        return True
    ctype = canon_type(gt.get('type')) if cfg.canon_types else gt.get('type')
    if ctype in IC_TYPES:
        return pred.get('Name') == gt.get('Name')
    if ctype in VALUE_TYPES:
        return pred.get('value') == gt.get('value')
    return True


def _point(raw) -> Optional[Tuple[float, float]]:
    if isinstance(raw, dict):
        x, y = raw.get('x'), raw.get('y')
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        x, y = raw
    else:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _edge_points(raw) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(raw, (list, tuple)):
        return None
    pts = [_point(p) for p in raw]
    if not pts or any(p is None for p in pts):
        return None
    return pts


def _seg_distance(p: Sequence[Tuple[float, float]],
                  q: Sequence[Tuple[float, float]],
                  unordered: bool) -> float:
    def m(a, b):
        return max(math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1]),
                   math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]))
    forward = m(p, q)
    if not unordered:
        return forward
    return min(forward, m(p, q[::-1]))


def _assign(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """Maximum-weight one-to-one assignment given a *cost* matrix (minimised).

    Returns index pairs; entries whose cost is ``INF`` are dropped.
    """
    if not cost or not cost[0]:
        return []
    INF = float('inf')
    if _HAVE_SCIPY:
        import numpy as np
        m = np.array([[1e12 if c == INF else c for c in row] for row in cost],
                     dtype=float)
        rows, cols = linear_sum_assignment(m)
        return [(int(r), int(c)) for r, c in zip(rows, cols)
                if m[r, c] < 1e11]
    # greedy fallback: repeatedly take the cheapest remaining pair
    pairs = sorted(((cost[i][j], i, j)
                    for i in range(len(cost)) for j in range(len(cost[i]))
                    if cost[i][j] != INF))
    used_r, used_c, out = set(), set(), []
    for c, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i); used_c.add(j); out.append((i, j))
    return out


# ---------------------------------------------------------------------------
# the four metrics

def assign_components(gt: dict, pred: dict, cfg: EvalConfig,
                      detail: Optional[Dict[str, int]] = None
                      ) -> Tuple[F1, Dict[str, str]]:
    """Match components and score them.

    Returns ``(component_F1, keymap)`` where ``keymap`` maps a *prediction*
    component key onto the *ground-truth* key it was matched to.  The caller
    rewrites the prediction with ``keymap`` before the pin/net metrics, so the
    coordinate-matched ``∅NNN`` objects line up downstream too.
    """
    gc, pc = gt['components'], pred['components']
    res = F1()
    keymap: Dict[str, str] = {}

    def ctype_of(rec):
        t = (rec or {}).get('type')
        return canon_type(t) if cfg.canon_types else t

    pred_used = set()
    gt_used = set()
    pairs: List[Tuple[str, str]] = []           # (gt_key, pred_key)

    # ---- 1) 键匹配 -------------------------------------------------------
    for key in set(gc) & set(pc):
        pairs.append((key, key))
        gt_used.add(key)
        pred_used.add(key)
    for key in gc:
        if key not in gt_used:
            res.fn += 1
            _bump(detail, 'component.missing')
    for key in pc:
        if key not in pred_used:
            res.fp += 1
            _bump(detail, 'component.extra')

    # ---- 2) 坐标匹配（无名元） -------------------------------------------
    if cfg.unkeyed_match != 'off':
        if cfg.unkeyed_match == 'gt':
            left_g = [k for k in gc if k not in gt_used and is_unkeyed(k)]
            left_p = [k for k in pc if k not in pred_used]
        else:                                    # 'all'
            left_g = [k for k in gc if k not in gt_used]
            left_p = [k for k in pc if k not in pred_used]
        if left_g and left_p:
            INF = float('inf')
            cost = []
            for pk in left_p:
                pb = _box((pc.get(pk) or {}).get('bbox'))
                pt = ctype_of(pc.get(pk))
                row = []
                for gk in left_g:
                    gb = _box((gc.get(gk) or {}).get('bbox'))
                    if pb is None or gb is None:
                        row.append(INF); continue
                    if cfg.coord_center:
                        d = math.hypot((pb[0] + pb[2]) / 2.0 - (gb[0] + gb[2]) / 2.0,
                                       (pb[1] + pb[3]) / 2.0 - (gb[1] + gb[3]) / 2.0)
                    else:
                        d = math.hypot(pb[0] - gb[0], pb[1] - gb[1])
                    if bbox_ok(gb, pb, cfg) is not True and d > cfg.coord_gate:
                        row.append(INF); continue
                    row.append(d + (0.0 if pt == ctype_of(gc.get(gk))
                                    else cfg.coord_type_penalty))
                cost.append(row)
            for i, j in _assign(cost):
                gk, pk = left_g[j], left_p[i]
                cost_ij = cost[i][j]
                if cost_ij >= cfg.coord_type_penalty:
                    continue                      # 类型都对不上，别硬配
                pairs.append((gk, pk))
                gt_used.add(gk); pred_used.add(pk)
                # 撤销第 1 步里为这两个键记下的 fp/fn
                if gk not in pc:
                    res.fn -= 1
                    _bump(detail, 'component.missing', -1)
                if pk not in gc:
                    res.fp -= 1
                    _bump(detail, 'component.extra', -1)
                _bump(detail, 'component.coord_matched')

    # ---- 3) 逐对判定 -----------------------------------------------------
    for gk, pk in pairs:
        g, p = gc.get(gk), pc.get(pk)
        if g is None or p is None:
            continue
        keymap[pk] = gk
        if ctype_of(g) != ctype_of(p):
            res.fp += 1; res.fn += 1
            _bump(detail, 'component.type_mismatch')
            continue
        if bbox_ok(g.get('bbox'), p.get('bbox'), cfg) is not True:
            res.fp += 1; res.fn += 1
            _bump(detail, 'component.bbox_mismatch')
            continue
        if not component_text_ok(g, p, cfg):
            res.fp += 1; res.fn += 1
            _bump(detail, 'component.text_mismatch')
            continue
        res.tp += 1
    res.detail = dict(detail or {})
    return res, keymap


def component_f1(gt: dict, pred: dict, cfg: EvalConfig,
                 detail: Optional[Dict[str, int]] = None) -> F1:
    """兼容旧调用点的薄包装（丢掉 keymap）。"""
    return assign_components(gt, pred, cfg, detail)[0]


def rewrite_keys(doc: dict, keymap: Dict[str, str]) -> dict:
    """把预测里被坐标匹配上的元件键改写成对应的 GT 键。

    ``components`` / ``pins`` 的顶层键，以及 ``nets[*].hyperGraph`` 里的元件名
    都会被改写，这样坐标匹配上的 ``∅NNN`` 在 pin / net 两处也能对上。
    """
    if not keymap:
        return doc
    out = {}
    out['components'] = {keymap.get(k, k): v for k, v in doc['components'].items()}
    pins = {}
    for comp, group in doc['pins'].items():
        tgt = keymap.get(comp, comp)
        # 目标键下已经有引脚就把两边合起来（坐标匹配理论上不会撞）
        pins.setdefault(tgt, {})
        if isinstance(group, dict):
            pins[tgt].update(group)
    out['pins'] = pins
    nets = {}
    for name, net in doc['nets'].items():
        if not isinstance(net, dict):
            nets[name] = net; continue
        nn = dict(net)
        hg = net.get('hyperGraph')
        if isinstance(hg, str) and hg:
            refs = [(keymap.get(c, c), p) for c, p in parse_hypergraph(hg)]
            nn['hyperGraph'] = build_hypergraph(refs)
        nets[name] = nn
    out['nets'] = nets
    return out


def pin_f1(gt: dict, pred: dict, cfg: EvalConfig,
           detail: Optional[Dict[str, int]] = None) -> F1:
    """官方口径：**只统计 GT 里 type 为 box 的元件**的引脚（评测 Q2）。"""
    box_comps = None
    if cfg.pin_box_only:
        box_comps = {k for k, v in gt['components'].items()
                     if canon_type((v or {}).get('type')) in IC_TYPES}

    def flat(section):
        out = {}
        for comp, group in section.items():
            if box_comps is not None and comp not in box_comps:
                continue
            if not isinstance(group, dict):
                continue
            for pk, info in group.items():
                out[(comp, pin_key_to_number(pk))] = info
        return out
    gf, pf = flat(gt['pins']), flat(pred['pins'])

    res = F1()
    for key in set(gf) | set(pf):
        g, p = gf.get(key), pf.get(key)
        if g is None:
            res.fp += 1; _bump(detail, 'pin.extra'); continue
        if p is None:
            res.fn += 1; _bump(detail, 'pin.missing'); continue
        if not isinstance(p, dict) or not isinstance(g, dict):
            res.fp += 1; res.fn += 1; _bump(detail, 'pin.malformed'); continue
        if p.get('pinname') != g.get('pinname'):
            res.fp += 1; res.fn += 1; _bump(detail, 'pin.name_mismatch'); continue
        a, b = _point(g.get('point')), _point(p.get('point'))
        if a is None or b is None:
            res.fp += 1; res.fn += 1; _bump(detail, 'pin.no_point'); continue
        if math.hypot(a[0] - b[0], a[1] - b[1]) > cfg.pin_tol:
            res.fp += 1; res.fn += 1; _bump(detail, 'pin.position_mismatch'); continue
        res.tp += 1
    res.detail = dict(detail or {})
    return res


def _net_sets(section: dict, comps: Optional[dict] = None,
              collapse_nonbox: bool = False) -> List[Tuple[str, frozenset]]:
    """每个 net 的节点集合。

    ``collapse_nonbox=True`` 时把**非 box 元件**的引脚号折成通配 ``*``
    （最新 QA：「rcl 等在 hygraph 中相当于不区分引脚」；电源引脚号是随机生产的，
    字面比永远配不上）。box（IC）的引脚号照旧逐位比较。
    """
    out = []
    for name, net in section.items():
        if not isinstance(net, dict):
            out.append((name, frozenset()))
            continue
        refs = set(net_pin_set(net))
        if collapse_nonbox:
            c = comps or {}
            refs = {(comp, pin if canon_type((c.get(comp) or {}).get('type')) in IC_TYPES
                     else '*')
                    for comp, pin in refs}
        out.append((name, frozenset(refs)))
    return out


def match_nets(gt_nets, pred_nets, soft: bool = False) -> List[Tuple[int, int]]:
    """Hungarian matching of predicted nets onto ground-truth nets.

    ``soft=True``（官方口径）以**每对网的 pin-set 软 F1（Dice）**
    2|P∩G|/(|P|+|G|) 为目标最大化 —— 官方 protocol 的
    ``connectivity_match = 'Hungarian one-to-one pin-set soft F1'``；
    ``net_hypergraph_f1`` 会按 ``cfg.net_soft`` 传这个开关。
    ``soft=False``（本函数默认，保持历史探针语义）是旧口径：
    交集（外加 Jaccard 千分之一破平）。
    """
    if not gt_nets or not pred_nets:
        return []
    cost = []
    for _, pset in pred_nets:
        row = []
        for _, gset in gt_nets:
            inter = len(pset & gset)
            if soft:
                union = len(pset) + len(gset)
                sim = (2.0 * inter / union) if union else 0.0
            else:
                union = len(pset | gset)
                sim = inter + (inter / union if union else 0.0) * 1e-3
            row.append(-sim)
        cost.append(row)
    return _assign(cost)


def _net_sim(pset, gset, soft: bool = True) -> float:
    """一对网的相似度：soft = 软 F1（Dice），否则 = 交集条数。"""
    inter = len(pset & gset)
    if soft:
        union = len(pset) + len(gset)
        return (2.0 * inter / union) if union else 0.0
    return float(inter)


def net_hypergraph_f1(gt: dict, pred: dict, cfg: EvalConfig,
                      detail: Optional[Dict[str, int]] = None,
                      ) -> Tuple[F1, List[Tuple[int, int]]]:
    g_nets = _net_sets(gt['nets'], gt.get('components'),
                       cfg.net_pin_collapse_nonbox)
    p_nets = _net_sets(pred['nets'], pred.get('components'),
                       cfg.net_pin_collapse_nonbox)
    pairs = match_nets(g_nets, p_nets, soft=bool(getattr(cfg, 'net_soft', True)))
    res = F1()
    if cfg.net_mode == 'exact':
        tp = sum(1 for i, j in pairs if p_nets[i][1] == g_nets[j][1])
        res.tp = tp
        res.fp = len(p_nets) - tp
        res.fn = len(g_nets) - tp
    elif bool(getattr(cfg, 'net_soft', True)):
        # 官方口径：tp = Σ 软 F1（Dice），fp/fn 由网数扣掉 soft_tp
        # ⇒ f1 = 2*soft_tp/(pred_count + gt_count)，与官方 report 逐位一致
        soft = sum(_net_sim(p_nets[i][1], g_nets[j][1], True) for i, j in pairs)
        res.tp = soft
        res.fp = len(p_nets) - soft
        res.fn = len(g_nets) - soft
        if detail is not None:
            detail['net.soft_tp'] = soft
            detail['net.pred_count'] = len(p_nets)
            detail['net.gt_count'] = len(g_nets)
            detail['net.pred_pins'] = sum(len(s) for _, s in p_nets)
            detail['net.gt_pins'] = sum(len(s) for _, s in g_nets)
    else:
        tp = sum(len(p_nets[i][1] & g_nets[j][1]) for i, j in pairs)
        total_pred = sum(len(s) for _, s in p_nets)
        total_gt = sum(len(s) for _, s in g_nets)
        res.tp = tp
        res.fp = total_pred - tp
        res.fn = total_gt - tp
        _bump(detail, 'net.pred_pins', total_pred)
        _bump(detail, 'net.gt_pins', total_gt)
        _bump(detail, 'net.pred_count', len(p_nets))
        _bump(detail, 'net.gt_count', len(g_nets))
    return res, pairs


def net_line_f1(gt: dict, pred: dict, cfg: EvalConfig,
                pairs: List[Tuple[int, int]],
                detail: Optional[Dict[str, int]] = None) -> F1:
    g_nets = list(gt['nets'].items())
    p_nets = list(pred['nets'].items())
    res = F1()
    INF = float('inf')
    matched_pred_nets, matched_gt_nets = set(), set()
    for i, j in pairs:
        matched_pred_nets.add(i); matched_gt_nets.add(j)
        p_edges = [(_edge_points(v)) for v in (p_nets[i][1].get('edges') or {}).values()]
        g_edges = [(_edge_points(v)) for v in (g_nets[j][1].get('edges') or {}).values()]
        p_edges = [e for e in p_edges if e is not None]
        g_edges = [e for e in g_edges if e is not None]
        if not p_edges or not g_edges:
            res.fp += len(p_edges); res.fn += len(g_edges)
            continue
        cost = []
        for pe in p_edges:
            row = []
            for ge in g_edges:
                if len(pe) < 2 or len(ge) < 2:
                    row.append(INF); continue
                d = _seg_distance(pe[:2], ge[:2], cfg.line_unordered)
                row.append(d if d <= cfg.line_tol else INF)
            cost.append(row)
        got = _assign(cost)
        res.tp += len(got)
        res.fp += len(p_edges) - len(got)
        res.fn += len(g_edges) - len(got)
    # nets with no counterpart contribute their edges as errors
    for i, (_, net) in enumerate(p_nets):
        if i not in matched_pred_nets:
            res.fp += len([v for v in (net.get('edges') or {}).values()
                           if _edge_points(v) is not None])
    for j, (_, net) in enumerate(g_nets):
        if j not in matched_gt_nets:
            res.fn += len([v for v in (net.get('edges') or {}).values()
                           if _edge_points(v) is not None])
    res.detail = dict(detail or {})
    return res


def _bump(detail: Optional[Dict[str, int]], key: str, amount: int = 1) -> None:
    if detail is not None:
        detail[key] = detail.get(key, 0) + amount


# ---------------------------------------------------------------------------

def evaluate_case(gt_doc: dict, pred_doc: dict, case_id: str = '',
                  cfg: Optional[EvalConfig] = None) -> CaseScore:
    cfg = cfg or EvalConfig()
    gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
    detail: Dict[str, float] = {}

    comp, keymap = assign_components(gt, pred, cfg, detail)
    # 坐标匹配上的元件，把预测的键改写成 GT 的键，pin / net 才能对上
    pred2 = rewrite_keys(pred, keymap)
    pin = pin_f1(gt, pred2, cfg, detail)
    net, pairs = net_hypergraph_f1(gt, pred2, cfg, detail)
    line = net_line_f1(gt, pred2, cfg, pairs, detail)

    # how good the recovered net *membership* is, ignoring edge geometry
    detail['net.pairs'] = len(pairs)
    detail['keymap.size'] = len(keymap)
    return CaseScore(case_id=case_id, component=comp, pin=pin,
                     net_hypergraph=net, net_line=line, diagnostic=detail)


def aggregate(scores: List[CaseScore]) -> dict:
    if not scores:
        return {'cases': 0, 'mean_score': 0.0}
    n = len(scores)
    return {
        'cases': n,
        'mean_score': round(sum(s.score for s in scores) / n, 4),
        'component_f1': round(sum(s.component.f1 for s in scores) / n, 6),
        'pin_f1': round(sum(s.pin.f1 for s in scores) / n, 6),
        'net_hypergraph_f1': round(sum(s.net_hypergraph.f1 for s in scores) / n, 6),
        'net_line_f1': round(sum(s.net_line.f1 for s in scores) / n, 6),
    }
