# -*- coding: utf-8 -*-
"""文字锚定的补件：位号文字读到了、附近却没有任何元件时，就地按红色墨迹补一颗。

动机（`work/_textanchor.py` 在 13 页上的实测）：`/`missing` 的 16 个 GT 件里，
有 3 个是「文字读得很准、元件根本没检出」——
  case1  `L2`  位号 0.99，距最近元件 35.8px（GT 那一颗就在文字旁 7px）
  case11 `SW1` 位号 1.00，距最近元件 45.8px（GT 10px）
  case12 `VOUT` 标签 `VOUT` 0.75，距最近元件 96px（GT 那一颗就压在文字上）
检测器对这三族字形（四段半圆的电感、双触点的轻触开关、五边形输出端口）系统性失明。

做法：位号前缀 → 类型（`L→l`、`R→r`、`C→c`、`D→d`、`LED→led`、`SW→switch`、
`TP→testpoint`），几何取**文字窗口内最近的一块红色墨迹**的连通域外框，中心不动、
尺寸按墨迹。只追加，不改任何既有件；同一页同一键只补一次。

上限（离线上界，见 `work/_rescueorc.py`）：补 GT 框 +1.97（三页合计，均值 +0.15）；
按墨迹框 +1.61（VOUT 因为墨迹连着导线，框宽 121 vs GT 82 ⇒ −0.51，故端口暂不启用）。
"""
from __future__ import annotations

import re

import cv2
import numpy as np

# 位号前缀 → 元件类型（顺序有意义：先长后短，`LED` 不能被 `L` 抢走）
PREFIX_TYPES = (
    ("LED", "led"),
    ("SW", "switch"),
    ("TP", "testpoint"),
    ("L", "l"),
    ("R", "r"),
    ("C", "c"),
    ("D", "d"),
)
MIN_LONG = {"switch": 40.0, "l": 40.0, "led": 30.0, "d": 30.0,
            "testpoint": 20.0}

DESIG_RE = re.compile(r"^([A-Za-z]{1,3})(\d{1,4})$")
# 值文字：必须带单位字母（`4.7uH`/`20k`/`22uf`）或带小数点的长串，避免把脚号 `1`/`2`
# 当成值抄上去（GT 的 `r/c/l` 都带 value，抄错一个字符 = 该件 fp+fn）。
VALUE_RE = re.compile(r"^\d{1,4}\.\d{1,2}\s*[A-Za-zµΩ]{0,3}$|^\d{1,4}\s*[A-Za-zµΩ]{1,3}$")


def _rect_dist(box, bb):
    x0, y0, x1, y1 = box
    a0, b0, a1, b1 = bb
    dx = max(0.0, max(a0 - x1, x0 - a1))
    dy = max(0.0, max(b0 - y1, y0 - b1))
    return (dx * dx + dy * dy) ** 0.5


def _sym_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((((h <= 12) | (h >= 168)) & (s >= 60) & (v >= 40))).astype(np.uint8)


def _ink_box(mask, cx, cy, half, min_area=40):
    h, w = mask.shape[:2]
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    if x1 <= x0 or y1 <= y0:
        return None
    sub = mask[y0:y1, x0:x1]
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(sub, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        gx, gy = x + ww / 2.0 + x0, y + hh / 2.0 + y0
        d = ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, [x + x0, y + y0, x + ww + x0, y + hh + y0], int(area))
    return best


def _type_of(text):
    m = DESIG_RE.match(text)
    if not m:
        return None
    pre = m.group(1).upper()
    for p, ty in PREFIX_TYPES:
        if pre == p:
            return ty
    return None


def _pick_read(reads):
    """同一个文字框有 5 种读法；位号优先，其次网名，最后值。"""
    for ch in ("designator", "net_label", "value"):
        v = reads.get(ch)
        if v and v[0]:
            return v[0].strip(), float(v[1] or 0.0), ch
    return None, 0.0, None


def _value_near(reads, bb, win=60.0, min_conf=0.5):
    """在墨迹框周围捞一条像值的文字（`4.7uH`/`20k`）。

    GT 的 `r/c/l` 都带 `value`，`component_text_ok` 会严格比它 ⇒ 补件不抄值等于白补
    （case1 `L2` 的 `value` 是 `4.7uH`）。只认 `VALUE_RE` 的形状，宁缺毋滥。
    """
    best = None
    for r in reads:
        v = (r.get("reads") or {}).get("value")
        if not v or not v[0]:
            continue
        txt, conf = str(v[0]).strip(), float(v[1] or 0.0)
        if conf < min_conf or not VALUE_RE.match(txt):
            continue
        box = r.get("bbox")
        if not box or len(box) != 4:
            continue
        d = _rect_dist(box, bb)
        if d > win:
            continue
        if best is None or d < best[0]:
            best = (d, txt, conf)
    return best


def _overlap(a, b):
    """两框交集面积 / 较小框面积。"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    amin = (a[2] - a[0]) * (a[3] - a[1])
    bmin = (b[2] - b[0]) * (b[3] - b[1])
    m = min(amin, bmin)
    return (inter / m) if m > 0 else 0.0


def desig_candidates(reads, *, min_conf=0.80):
    """收集「像位号」的文字读法，交给交付层做 `pNNN` 回填（图像坐标）。"""
    out = []
    for rec in reads:
        box = rec.get("bbox")
        if not box or len(box) != 4:
            continue
        t, c, ch = _pick_read(rec.get("reads") or {})
        if not t or c < min_conf or not DESIG_RE.match(t):
            continue
        if _type_of(t) is None:
            continue
        out.append({"text": t, "conf": round(c, 4),
                    "box": [float(v) for v in box], "channel": ch})
    return out


def rescue(doc, a, reads, bgr, *, gap=30.0, min_conf=0.80, win=140.0,
           pad=2.0, max_side=260.0, min_side=8.0):
    """Returns {"added": [{key,type,box,d}], "skipped_*": int}. 纯追加。"""
    out = {"added": [], "near": 0, "low_conf": 0, "no_ink": 0, "bad_size": 0,
           "overlap": 0, "dup": 0, "no_type": 0}
    comps = doc.get("components") or []
    boxes = [(c.get("id"), c.get("bbox_img")) for c in comps if c.get("bbox_img")]
    used_keys = set()
    for cid, rec in (a.get("comps") or {}).items():
        dg = (rec.get("designator") or {}).get("canon") or (rec.get("designator") or {}).get("text")
        if dg:
            used_keys.add(str(dg))
    for c in comps:
        if c.get("key"):
            used_keys.add(str(c["key"]))
    mask = None
    for r in reads:
        box = r.get("bbox")
        if not box or len(box) != 4:
            continue
        text, conf, _ch = _pick_read(r.get("reads") or {})
        if not text:
            continue
        ty = _type_of(text)
        if ty is None:
            out["no_type"] += 1
            continue
        ty, wd = hit
        if conf < min_conf:
            out["low_conf"] += 1
            continue
        dmin = min((_rect_dist(box, bb) for _cid, bb in boxes), default=1e9)
        if dmin < gap:
            out["near"] += 1
            continue
        key = text
        if key in used_keys:
            out["dup"] += 1
            continue
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        if mask is None:
            mask = _sym_mask(bgr)
        ink = _ink_box(mask, cx, cy, win)
        if not ink:
            out["no_ink"] += 1
            continue
        _d, bb, _area = ink
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        if bw < min_side or bh < min_side or bw > max_side or bh > max_side:
            out["bad_size"] += 1
            continue
        # 形状最小尺寸（按类型）：已知类型就该像那个符号，不能拿一小块墨迹充数。
        # 池 0050 把连接器引脚名 `SW1`/`SW2` 补成两颗 25x18 的 switch（各 −0.08），
        # 而真 switch（case11 SW1）是 28x113、真电感（case1 L2）是 77x11 ⇒ 加类型闸门。
        _ml = MIN_LONG.get(ty, 0.0)
        if _ml and max(bw, bh) < _ml:
            out["bad_size"] += 1
            continue
        # 墨迹和任何既有件重叠 ⇒ 那个符号已经被检出了，补上去只会多一颗错键
        # （case1 `C1` 164x145 压着 U1、case11 `LED1` 20x20 落在已发出的 LED2 上，
        #  实测这两颗各 −0.20 / −0.95，而 L2/SW1/C4 都在 35px 外 ⇒ 全留）。
        if any(_overlap(bb, cbb) > 0.25 for _c, cbb in boxes):
            out["overlap"] += 1
            continue
        used_keys.add(key)
        h = bgr.shape[0]
        val = _value_near(reads, bb)
        cid = "txt%03d" % (len(comps) + 1)
        comps.append({
            "id": cid, "type": ty, "proposal": ty, "score": round(conf, 4),
            "source": "textanchor", "subtype": None,
            "bbox_img": [float(bb[0] - pad), float(bb[1] - pad),
                         float(bb[2] + pad), float(bb[3] + pad)],
            "bbox_json": [float(bb[0] - pad), float(h - bb[3] - pad),
                          float(bb[2] + pad), float(h - bb[1] + pad)],
            "bbox_raw_img": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
            "center_img": [float(cx), float(cy)], "key": key, "n_alts": 1,
            "pending": False, "dual_fine": 0.0,
        })
        a.setdefault("comps", {})[cid] = {
            "designator": {"text": key, "canon": key, "conf": round(conf, 4),
                           "box": [float(v) for v in box], "d": round(dmin, 2)},
            "partname": None,
            "value": ({"text": val[1], "conf": round(val[2], 4)} if val else None),
            "pins": {},
        }
        doc.setdefault("pins", {})[cid] = []
        out["added"].append({"key": key, "type": ty, "cid": cid,
                             "value": (val[1] if val else None),
                             "box": [round(float(v), 1) for v in bb],
                             "bbox": [round(float(bb[0] - pad), 2), round(float(h - bb[3] - pad), 2),
                                      round(float(bb[2] + pad), 2), round(float(h - bb[1] + pad), 2)],
                             "score": round(conf, 4), "d": round(dmin, 1)})
    return out

# ---------------------------------------------------------------------------
# 「文字 → 类型 → 导线末端/墨迹 → 按类型先验补件」（`STAGE2_TEXT_WORDS`，默认关）
# 与上面的位号锚定同一动机，但覆盖**没有位号**的符号：GND / VCC / VIN / VOUT 这类网标
# 文字读得到，符号却检不出（case11 的两颗竖排 GND：`('GNDGND', 0.994)`、离最近元件 96px）。
# 位号锚定靠「文字周围的红色墨迹」定框，这里必须换锚点：实测最近悬空导线末端会把框放到
# 21.5px 外（`work/_wireorc.py`，case11 −0.95），所以取「导线末端 → 附近暗墨迹块 → 按类型
# 先验取并集」，并要求文字与既有件 ≥30px（否则那是连接器里的引脚名，池 0050 的 `SW1`）。
WORD_TYPES = (
    ("VOUT", "net_output"),
    ("GND", "gnd"),
    ("VSS", "gnd"),
    ("VIN", "net_input"),
    ("VCC", "v"),
    ("VDD", "v"),
    ("3V3", "v"),
)
WORD_SIZE = {"gnd": (26.0, 34.0), "v": (22.0, 26.0),
             "net_input": (54.0, 20.0), "net_output": (54.0, 20.0)}


def _word_type(text):
    # 竖排整条读出来的串带 OCR 噪声：case11 那串真值是 `GND GND`，读出 `NDGND`
    # （首字被邻字吃掉了），所以用**包含**而不是 startswith。
    t = re.sub(r"[^0-9A-Za-z]", "", (text or "").upper())
    if len(t) < 2:
        return None
    for w, ty in WORD_TYPES:
        if w in t:
            return ty, w
    return None


def _dark_mask(bgr):
    """红色墨迹 ∪ 暗墨迹：GND 这类符号的笔画大半是黑色，只取红色会漏（case11 只剩 2x49 竖杆）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return ((gray < 110).astype(np.uint8) | _sym_mask(bgr))


def _seg_ends(doc):
    out = []
    for s in ((doc.get("wire_detail") or {}).get("segments") or []):
        try:
            out.append((float(s[0][0]), float(s[0][1])))
            out.append((float(s[1][0]), float(s[1][1])))
        except (TypeError, IndexError, ValueError):
            continue
    return out


def anchor_box(doc, box, kind, mask, *, win=45.0, lim=45.0, min_area=60):
    """文字框 → 锚点 → 按类型先验定框。

    顺序是**墨迹优先、导线末端兜底**（实测：反过来的话 case11 会落到 21.5px 外的
    悬空末端上，框中心离 GT 33px，只算 fp，−0.95）。符号的笔画常常是一根细杆
    （GND 的竖杆 2x49），所以细杆只取它的中心、尺寸用类型先验；块状墨迹才直接用外框。
    """
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    w, hh = WORD_SIZE.get(kind, (26.0, 30.0))
    # 文字本身也是墨迹：先把文字框从掩膜里挖掉，否则最近的"墨迹块"就是这行字自己
    # （case11 实测会得到 d=0.5 的自命中，框把文字和符号一起圈进去）。
    m = mask
    if mask is not None:
        m = mask.copy()
        tx0, ty0, tx1, ty1 = [int(round(v)) for v in box]
        m[max(0, ty0 - 1):ty1 + 2, max(0, tx0 - 1):tx1 + 2] = 0
    ink = _ink_box(m, cx, cy, win, min_area=min_area)
    if ink and ink[0] <= win:
        d, ib, _a = ink
        gx, gy = (ib[0] + ib[2]) / 2.0, (ib[1] + ib[3]) / 2.0
        bw, bh = ib[2] - ib[0], ib[3] - ib[1]
        if bw < w * 0.6 or bh < hh * 0.6:
            return [gx - w / 2.0, gy - hh / 2.0, gx + w / 2.0, gy + hh / 2.0], d, "ink"
        return [min(ib[0], gx - w / 2.0), min(ib[1], gy - hh / 2.0),
                max(ib[2], gx + w / 2.0), max(ib[3], gy + hh / 2.0)], d, "inkbox"
    best = None
    for p in _seg_ends(doc):
        d = ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, p)
    if best is None or best[0] > lim:
        return None, None, None
    dw, (ex, ey) = best
    return [ex - w / 2.0, ey - hh / 2.0, ex + w / 2.0, ey + hh / 2.0], dw, "wire"


def rescue_words(doc, a, reads, bgr, *, gap=30.0, min_conf=0.60, lim=45.0,
                 pad=2.0, max_side=260.0, min_side=8.0):
    """网标词（GND/VCC/VIN/VOUT/3V3）→ 类型 → 导线末端锚定补件。键用 `p9NN`（判题按几何配）。"""
    out = {"added": [], "near": 0, "low_conf": 0, "no_wire": 0, "bad_size": 0,
           "overlap": 0, "no_type": 0}
    comps = doc.get("components") or []
    boxes = [c.get("bbox_img") for c in comps if c.get("bbox_img")]
    mask = None
    n = 0
    for r in reads:
        box = r.get("bbox")
        if not box or len(box) != 4:
            continue
        text, conf, _ch = _pick_read(r.get("reads") or {})
        if not text:
            continue
        hit = _word_type(text)
        if hit is None:
            out["no_type"] += 1
            continue
        ty, wd = hit
        if conf < min_conf:
            out["low_conf"] += 1
            continue
        dmin = min((_rect_dist(box, bb) for bb in boxes), default=1e9)
        if dmin < gap:
            out["near"] += 1
            continue
        if mask is None:
            mask = _dark_mask(bgr)
        bb, dw, how = anchor_box(doc, box, ty, mask, lim=lim)
        if bb is None:
            out["no_wire"] += 1
            continue
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        if bw < min_side or bh < min_side or bw > max_side or bh > max_side:
            out["bad_size"] += 1
            continue
        if any(_overlap(bb, cbb) > 0.25 for cbb in boxes):
            out["overlap"] += 1
            continue
        n += 1
        key = "p9%02d" % n
        cid = "txt%03d" % (len(comps) + 1)
        h = bgr.shape[0]
        comps.append({
            "id": cid, "type": ty, "proposal": ty, "score": round(conf, 4),
            "source": "textwords", "subtype": None,
            "bbox_img": [float(bb[0] - pad), float(bb[1] - pad),
                         float(bb[2] + pad), float(bb[3] + pad)],
            "bbox_json": [float(bb[0] - pad), float(h - bb[3] - pad),
                          float(bb[2] + pad), float(h - bb[1] + pad)],
            "bbox_raw_img": [float(v) for v in bb],
            "center_img": [float((bb[0] + bb[2]) / 2.0), float((bb[1] + bb[3]) / 2.0)],
            "key": key, "n_alts": 1, "pending": False, "dual_fine": 0.0,
        })
        a.setdefault("comps", {})[cid] = {"designator": None, "partname": None,
                                          "value": {"text": wd, "conf": round(conf, 4)},
                                          "pins": {}}
        doc.setdefault("pins", {})[cid] = []
        boxes.append([float(bb[0] - pad), float(bb[1] - pad),
                      float(bb[2] + pad), float(bb[3] + pad)])
        out["added"].append({"key": key, "type": ty, "cid": cid, "value": wd,
                             "box": [round(float(v), 1) for v in bb],
                             "bbox": [round(float(bb[0] - pad), 2), round(float(h - bb[3] - pad), 2),
                                      round(float(bb[2] + pad), 2), round(float(h - bb[1] - pad), 2)],
                             "score": round(conf, 4), "d": round(dmin, 1),
                             "word": text, "wire": round(dw, 1), "how": how})
    return out
