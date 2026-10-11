"""官方评测机口径的本地复刻（`eda-pcb/schematic_parser/official.py`）。

## 为什么要有这个文件

我们原来的本地尺 `metrics.py` 比官方**松**（同一批 `*_Pred.json`：本地 component
0.9463 / 官方 0.4878，本地 connectivity 0.5616 / 官方 0.4968），而 pin 又比官方**紧**
（本地 0.3720 / 官方 0.4582）。用错的尺子排序优化方向会整条偏掉，所以这里按官方
`report/*.json` 里的 `protocol` 字段逐条复刻：

| 通道 | 官方原文 |
|---|---|
| component | `component key/type/Name/value exact; ordered bbox, every coordinate within threshold` |
| pin | `component key and pin_<number> exact; owner exists in both components; pinname exact; point Euclidean threshold` |
| wire | `one-to-one direction-invariant two-point edge matching within Hungarian-matched nets` |
| connectivity | `Hungarian one-to-one pin-set soft F1` |
| 阈值 | `spatial_match_threshold_px = 5.0`（**引脚点距 / bbox 四坐标 / 导线端点全用它**，见下第 6 条） |
| 权重 | component .30 / pin .25 / connectivity .35 / wire .10 |

## 从官方报告里反推出来的判定细节（每条都有 `20261009引入` 的实证）

1. **无名元（`∅NNN`）不参与坐标匹配**：case 1 的官方 `component.error_examples` 里，
   GT 的 8 个 `∅109/∅113/∅117/∅121/∅125/∅200/∅202/∅208` 全部是 `missing`，而我们输出的
   `p012…p020` 全部是 `extra` ⇒ 键就是字符串相等，没有兜底。
2. **属性不符既算 fp 也算 fn**：GT `C162` 的键明明配上（`key_f1` tp 7），却因
   `failed_checks: ["value"]` 记成 `attribute_or_position_mismatch` ⇒ component tp 6。
3. **pin 的分母是全部引脚**：case 1 `id_f1` tp 28 / fp 45 / fn 32，而 pin f1
   tp 23 / fp 50 / fn 37 ⇒ 28 = 23 + 5（4 个 `tip_position`、1 个 `pin_name`），
   且 45 = 全部预测脚 − 28、32 = 全部 GT 脚 − 28 ⇒ **没配上的元件，其引脚照样计入 fp**。
4. **wire 的分母也是全部边**：`pred_segment_count 207` / `gt_segment_count 102` /
   `matched_segment_count 2` ⇒ tp 2 / fp 205 / fn 100。
5. `component_stage` / `wire_stage` / `connectivity_stage` **不进总分**（case 1:
   `0.3*0.272727 + 0.25*0.345865 + 0.35*0.295378 + 0.1*0.012945 = 0.272961` = 27.2961 ✓），
   所以本文件不复刻它们。
6. **bbox 的容差是 5px**（2026-10 曾按「业主指正」写成 20px，**已被真实判题机报告推翻**）。
   `OFFICIAL_BBOX_PX` 扫 5/8/12/16/20：只有 `=5` 时 component 逐案等于判题机
   （case 3 0.4615 / case 7 0.6250 / case 9 0.5294；均值 0.487847 = 官方 0.487847）。
7. **连通性按字面比，非 box 脚号不折叠**（`collapse_nonbox=False`）：`=True` 时连通性
   均值 0.529815、逐案高于判题机；`=False` 时 0.496836 = 官方 0.496836，**10 案逐案全等**，
   且 `matched_net_pairs` 与报告逐对一致（case 1 的 7 对）。可用 `OFFICIAL_COLLAPSE=1` 复现旧口径。

## 与判题机的对账状态（2026-10 复核 `report/submission/20261009T140206Z`）

| 通道 | 复刻 vs 判题机 |
|---|---|
| component | **逐案全等**（均值 0.487847） |
| pin | **逐案全等**（均值 0.458210） |
| connectivity | **逐案全等**（均值 0.496836） |
| wire | 9/10 案逐位相同；case 6 差 1 条段（0.086275 vs 0.094118，判题机多认一条 `max_endpoint_error_px = 5.0` 的边界段） |

## 用法

    python -X utf8 work\\tools\\dev\\_offscore.py --engine official
    python -X utf8 work\\tools\\dev\\_offscore.py --engine official --run work/_fast_out_maj2
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

THRESHOLD_PX = 5.0
# ★★ **bbox 的容差就是 5px（= 官方 `protocol` 的 `spatial_match_threshold_px`）**。
# 2026-10 曾按「业主指正」把它写成 20px，但**拿真实判题机报告逐案复核时被推翻**：
#   `work\tools\dev\_offscore.py --engine official`（pred = `scoring_result (5)\final_results`，
#   GT = `10GT`，官方值读 `report/submission/20261009T140206Z/*.json` 的 `scores`）
#     OFFICIAL_BBOX_PX=5  → component 均值 0.487847 = 官方 0.487847，**10 案逐案全等**
#     OFFICIAL_BBOX_PX≥8  → case 3 0.5385 / case 7 0.6667 / case 9 0.5882 都**高于**判题机
#                           （0.4615 / 0.6250 / 0.5294）⇒ 20px 太松。
# 也就是说 `protocol` 里那**唯一一个** 5.0 同时管引脚点距、bbox 四坐标与导线端点。
BBOX_PX = float(os.environ.get("OFFICIAL_BBOX_PX", "5.0"))
# 引脚**点距**（`tip_position`）的容差，默认 5.0 = 官方 `spatial_match_threshold_px`。
# ★ 这个开关只为「量一下改成 20px 会翻回多少」而存在，**默认不动**。要试：
# `$env:OFFICIAL_PIN_PX='20'` 再跑 `fast.py --score-only`。
PIN_PX = float(os.environ.get("OFFICIAL_PIN_PX", "5.0"))
# 连通性是否把非 box 元件的脚号折成 `*`。★ **判题机不折**（字面比）：
# 同一把尺子上 `collapse_nonbox=True` 时连通性均值 0.529815、**逐案高于**判题机
# （case 3 0.6578 vs 0.6088、case 7 0.5303 vs 0.4443、case 10 0.4861 vs 0.4012），
# 改成 `False` 后 0.496836 = 官方 0.496836 且 **10 案逐案全等**，
# 连 `connectivity.diagnostics.matched_net_pairs`（case 1 的 7 对）都与报告逐对一致。
# 要复现旧的折叠口径：`$env:OFFICIAL_COLLAPSE='1'`。
COLLAPSE_NONBOX = os.environ.get("OFFICIAL_COLLAPSE", "0") not in ("0", "false", "False", "", "no")
WEIGHTS = {'component': 0.30, 'pin': 0.25, 'connectivity': 0.35, 'wire': 0.10}

# `hyperGraph` 里的成员引用：`U1.101` / `C151.2` / `∅208.206`
REF_RE = re.compile(r'([^,()\s]+?)\.([^,()\s]+)')
BOX_TYPES = frozenset({'box', 'boxic', 'ic'})


# ---------------------------------------------------------------------------
# 基本结构

@dataclass
class F1:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    detail: Dict[str, object] = field(default_factory=dict)

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
                'recall': round(self.recall, 6), 'f1': round(self.f1, 6)}


@dataclass
class CaseScore:
    case_id: str
    component: F1
    pin: F1
    connectivity: F1
    wire: F1
    diagnostic: Dict[str, object] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return 100.0 * (WEIGHTS['component'] * self.component.f1 +
                        WEIGHTS['pin'] * self.pin.f1 +
                        WEIGHTS['connectivity'] * self.connectivity.f1 +
                        WEIGHTS['wire'] * self.wire.f1)

    def as_dict(self) -> dict:
        return {'case': self.case_id, 'score': round(self.score, 6),
                'component': self.component.as_dict(),
                'pin': self.pin.as_dict(),
                'connectivity': self.connectivity.as_dict(),
                'wire': self.wire.as_dict(),
                'diagnostic': self.diagnostic}


def _box(raw) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


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


def _s(v) -> str:
    """属性比较用的字符串：缺字段当空串。"""
    return '' if v is None else str(v)


# ---------------------------------------------------------------------------
# component

def component_f1(gt: dict, pred: dict) -> F1:
    """键精确 + type/Name/value 精确 + 四坐标都在 5px 内。"""
    gc = gt.get('components') or {}
    pc = pred.get('components') or {}
    res = F1()
    bad: List[dict] = []
    for key in pc:
        prec = pc.get(key) or {}
        grec = gc.get(key)
        if grec is None:
            res.fp += 1
            bad.append({'component_id': key, 'error': 'extra'})
            continue
        failed = []
        if _s(prec.get('type')) != _s(grec.get('type')):
            failed.append('type')
        if _s(prec.get('Name')) != _s(grec.get('Name')):
            failed.append('Name')
        if _s(prec.get('value')) != _s(grec.get('value')):
            failed.append('value')
        gb, pb = _box(grec.get('bbox')), _box(prec.get('bbox'))
        if gb is None or pb is None:
            failed.append('bbox')
            bmax = None
        else:
            bmax = max(abs(gb[i] - pb[i]) for i in range(4))
            if bmax > BBOX_PX:
                failed.append('bbox')
        if failed:
            res.fp += 1
            bad.append({'component_id': key, 'error': 'attribute_or_position_mismatch',
                        'failed_checks': failed,
                        'bbox_max_coordinate_error_px': bmax})
        else:
            res.tp += 1
    for key in gc:
        if key not in pc:
            res.fn += 1
            bad.append({'component_id': key, 'error': 'missing'})
        elif any(e.get('component_id') == key and e.get('error') for e in bad):
            res.fn += 1                       # 属性不符：既 fp 也 fn
    res.detail['examples'] = bad[:30]
    res.detail['n_examples'] = len(bad)
    return res


# ---------------------------------------------------------------------------
# pin

def _pins_of(doc: dict) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for key, rec in (doc.get('pins') or {}).items():
        if isinstance(rec, dict):
            out[key] = {str(k): (v if isinstance(v, dict) else {}) for k, v in rec.items()}
    return out


def pin_f1(gt: dict, pred: dict) -> F1:
    """（元件键，pin_<号>）精确 + pinname 精确 + 尖端点 5px 内。"""
    gp, pp = _pins_of(gt), _pins_of(pred)
    res = F1()
    n_pred_pins = sum(len(v) for v in pp.values())
    n_gt_pins = sum(len(v) for v in gp.values())
    bad: List[dict] = []
    for key, pins in pp.items():
        gpins = gp.get(key) or {}
        for pid, prec in pins.items():
            grec = gpins.get(pid)
            if grec is None:
                bad.append({'pin_id': [key, pid], 'component_id': key, 'error': 'extra'})
                continue
            failed = []
            if _s(prec.get('pinname')) != _s(grec.get('pinname')):
                failed.append('pin_name')
            a, b = _point(grec.get('point')), _point(prec.get('point'))
            dist = math.hypot(a[0] - b[0], a[1] - b[1]) if (a and b) else None
            if dist is None or dist > PIN_PX:
                failed.append('tip_position')
            if failed:
                bad.append({'pin_id': [key, pid], 'component_id': key,
                            'failed_checks': failed, 'tip_distance_px': dist})
            else:
                res.tp += 1
    for key, pins in gp.items():
        for pid in pins:
            if pid not in (pp.get(key) or {}):
                res.fn += 1
                bad.append({'pin_id': [key, pid], 'component_id': key, 'error': 'missing'})
    res.fp = float(n_pred_pins) - res.tp
    res.fn = float(n_gt_pins) - res.tp
    res.detail['n_pred_pins'] = n_pred_pins
    res.detail['n_gt_pins'] = n_gt_pins
    res.detail['examples'] = bad[:30]
    res.detail['n_examples'] = len(bad)
    return res


# ---------------------------------------------------------------------------
# 网成员

def net_members(doc: dict, collapse_nonbox: bool = True) -> Dict[str, Set[str]]:
    """把 `nets[*].hyperGraph` 解成 `{net_id: {成员引用}}`。

    `collapse_nonbox=True` 时，非 box 元件的引脚号折成 `*`（官方 QA：非 box 的
    pinnumber 不计分，且 GT 里电源脚号是随机生成的，字面比永远配不上）。
    """
    types = {k: _s((v or {}).get('type')) for k, v in (doc.get('components') or {}).items()}
    out: Dict[str, Set[str]] = {}
    for nid, rec in (doc.get('nets') or {}).items():
        hg = (rec or {}).get('hyperGraph')
        if hg is None:
            hg = (rec or {}).get('hypergraph')
        members: Set[str] = set()
        if isinstance(hg, str):
            body = hg.strip()
            if body.startswith('(') and body.endswith(')'):
                body = body[1:-1]
            for m in REF_RE.finditer(body):
                comp, num = m.group(1), m.group(2)
                if collapse_nonbox and types.get(comp) not in BOX_TYPES:
                    members.add('%s.*' % comp)
                else:
                    members.add('%s.%s' % (comp, num))
        elif isinstance(hg, (list, tuple)):
            for item in hg:
                if isinstance(item, str) and '.' in item:
                    members.add(item)
        out[str(nid)] = members
    return out


def _pair_score(a: Set[str], b: Set[str], mode: str = 'f1') -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    if mode == 'f1':
        return 2.0 * inter / (len(a) + len(b))
    if mode == 'iou':
        return inter / len(a | b)
    if mode == 'cover':
        return inter / min(len(a), len(b))
    return float(inter)


def _assign_max(score: List[List[float]]) -> List[Tuple[int, int]]:
    """最大权一对一（把负分/零分当不可选）。"""
    if not score or not score[0]:
        return []
    n, m = len(score), len(score[0])
    big = 0.0
    for row in score:
        for v in row:
            big = max(big, abs(v))
    cost = [[(big * 10.0 - v) if v > 0 else big * 100.0 for v in row] for row in score]
    if _HAVE_SCIPY:
        import numpy as np
        pairs = _linear(np.array(cost, dtype=float))
    else:
        pairs = _greedy(score)
    return [(i, j) for i, j in pairs if score[i][j] > 0]


def _linear(m):
    rows, cols = linear_sum_assignment(m)
    return list(zip([int(r) for r in rows], [int(c) for c in cols]))


def _greedy(score: List[List[float]]) -> List[Tuple[int, int]]:
    cand = sorted(((score[i][j], i, j) for i in range(len(score))
                   for j in range(len(score[i])) if score[i][j] > 0), reverse=True)
    ur, uc, out = set(), set(), []
    for _, i, j in cand:
        if i in ur or j in uc:
            continue
        ur.add(i)
        uc.add(j)
        out.append((i, j))
    return out


def connectivity_f1(gt: dict, pred: dict, collapse_nonbox: bool = False,
                    mode: str = 'f1') -> Tuple[F1, List[Tuple[str, str]]]:
    g, p = net_members(gt, collapse_nonbox), net_members(pred, collapse_nonbox)
    gk, pk = list(g), list(p)
    score = [[_pair_score(g[gk[i]], p[pk[j]], mode) for j in range(len(pk))]
             for i in range(len(gk))]
    pairs = _assign_max(score)
    soft_tp = sum(score[i][j] for i, j in pairs)
    res = F1(tp=soft_tp, fp=float(len(pk)) - soft_tp, fn=float(len(gk)) - soft_tp)
    res.detail['gt_net_count'] = len(gk)
    res.detail['pred_net_count'] = len(pk)
    res.detail['matched_net_pairs'] = [(gk[i], pk[j]) for i, j in pairs]
    return res, [(gk[i], pk[j]) for i, j in pairs]


# ---------------------------------------------------------------------------
# wire

def _edges_of(doc: dict) -> Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    out: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
    for nid, rec in (doc.get('nets') or {}).items():
        segs = []
        for _eid, pts in ((rec or {}).get('edges') or {}).items():
            if isinstance(pts, (list, tuple)) and len(pts) >= 2:
                a, b = _point(pts[0]), _point(pts[-1])
                if a and b:
                    segs.append((a, b))
        out[str(nid)] = segs
    return out


def _edge_cost(a, b) -> float:
    def m(x, y):
        return max(math.hypot(x[0][0] - y[0][0], x[0][1] - y[0][1]),
                   math.hypot(x[1][0] - y[1][0], x[1][1] - y[1][1]))
    return min(m(a, b), m(a, (b[1], b[0])))


def wire_f1(gt: dict, pred: dict, pairs: Sequence[Tuple[str, str]]) -> F1:
    """只在「连边匹配上的网对」里做方向无关的一对一边匹配。"""
    ge, pe = _edges_of(gt), _edges_of(pred)
    n_gt = sum(len(v) for v in ge.values())
    n_pred = sum(len(v) for v in pe.values())
    tp = 0
    examples: List[dict] = []
    for gid, pid in pairs:
        gsegs, psegs = ge.get(gid) or [], pe.get(pid) or []
        if not gsegs or not psegs:
            continue
        cost = [[_edge_cost(g, p) for p in psegs] for g in gsegs]
        pairs2 = _assign_max([[1.0 if c <= THRESHOLD_PX else 0.0 for c in row] for row in cost])
        for i, j in pairs2:
            tp += 1
            if len(examples) < 8:
                examples.append({'gt_net_id': gid, 'pred_net_id': pid,
                                 'gt_edge_index': i, 'pred_edge_index': j,
                                 'max_endpoint_error_px': round(cost[i][j], 6)})
    res = F1(tp=float(tp), fp=float(n_pred - tp), fn=float(n_gt - tp))
    res.detail.update({'gt_segment_count': n_gt, 'pred_segment_count': n_pred,
                       'matched_segment_count': tp, 'assignment_examples': examples})
    return res


# ---------------------------------------------------------------------------

def evaluate_case(gt_doc: dict, pred_doc: dict, case_id: str = '',
                  collapse_nonbox: bool = COLLAPSE_NONBOX,
                  net_mode: str = 'f1') -> CaseScore:
    comp = component_f1(gt_doc, pred_doc)
    pin = pin_f1(gt_doc, pred_doc)
    conn, pairs = connectivity_f1(gt_doc, pred_doc, collapse_nonbox, net_mode)
    wire = wire_f1(gt_doc, pred_doc, pairs)
    return CaseScore(case_id=case_id, component=comp, pin=pin, connectivity=conn,
                     wire=wire, diagnostic={'net_pairs': len(pairs)})


def aggregate(scores: Sequence[CaseScore]) -> dict:
    if not scores:
        return {'cases': 0, 'mean_score': 0.0}
    n = len(scores)
    return {
        'cases': n,
        'mean_score': round(sum(s.score for s in scores) / n, 4),
        'component_f1': round(sum(s.component.f1 for s in scores) / n, 6),
        'pin_f1': round(sum(s.pin.f1 for s in scores) / n, 6),
        'connectivity_f1': round(sum(s.connectivity.f1 for s in scores) / n, 6),
        'wire_f1': round(sum(s.wire.f1 for s in scores) / n, 6),
    }
