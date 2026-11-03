"""电源符号网名补读（默认开）。

## 病

官方 GT 的电源符号**都带 `value`**（`open_GT-1020`：gnd 83 / v 51 全有；`10GT`：gnd 59 /
v 49 全有），值是印在符号旁的网名（`+5V` / `VCC` / `3.3V`）。我们原来一律发 `None`
（`work/_strictpow.py` 按 `评测要求.pdf`「其余元件比 type/name/value」实测：local 74.02
→ official 63.18，130 个缺口里 gnd 78 / v 50）。`assemble.py` 的 `POWER_VALUE` 已经用上
「OCR 当成元件 `partname` 的那批」，但 `assign` 只把网名挂给了约 16/51 个符号，剩下的：

1. **横排网名，检出了、读对了，但 `assign` 没把它归给这个符号**（`open_GT` case2
   `∅727`：25×10 的框距符号中心 13.6px，读数是 `+5V`）——这批不用重读，主链 `reads`
   里就有；
2. **竖排网名**（旋转 90° 印，`+5V` 底→顶）：CRNN 只读得出 `'R'` 这类碎片
   （`work/_vlab2.png` 目视；case2 `t0014` 0° 0.47 → 转 90° 顺时针 **`+5V` 0.95**）。

## 做法

只对「本来没有网名的电源符号」做，纯追加：

- 第一遍（0°）复用手里的 `reads`：框未被任何元件认领（不是别人的位号/值/型号）＋
  字符形如网名 ＋ 置信度过闸 ＋ 中心在 `NEAR` 内 ⇒ 写 `partname`；
- 第二遍（旋转帧）只对**高瘦**框重读一次，同一套闸门。

0° 永不改写既有字段，识别不出就不写 ⇒ 关掉或闸门不命中时下游逐位不变。
纯数字读数一律拒绝（实测会写出 `'5'` 这种垃圾）。

复现：`work/_strictpow.py <tag> 1 … 13`（official 口径）。
"""

from __future__ import annotations

import os
import re

import cv2

from envflag import env_on

CONF = float(os.environ.get("STAGE2_POWER_LABEL_CONF", "0.70"))
NEAR = float(os.environ.get("STAGE2_POWER_LABEL_NEAR", "46.0"))
NEAR_H = float(os.environ.get("STAGE2_POWER_LABEL_NEAR_H", "26.0"))
# ★ 端口符号（net_input/output/...）的标签比电源符号印得更远（§5.446）：
#   case12 三颗端口的标签中心距 32~36px、读数很干净（VIN 0.92~1.00），却因 NEAR_H=26 一颗都没认领。
PORT_NEAR = float(os.environ.get("STAGE2_POWER_LABEL_NEAR_PORT", "46.0"))
PORT_GAP = float(os.environ.get("STAGE2_POWER_LABEL_PORT_GAP", "26.0"))
PORT_TYPES = frozenset(("net_input", "net_output", "net_short", "net_bidirection"))

DBG = env_on("STAGE2_POWER_LABEL_DBG", "0")

#: 要补网名的类型（地符号不在内：它的 value 恒为字面 `GND`，`assemble.py` 直接给）。
TYPES = frozenset(("v", "vcc", "vdd", "vbus_in", "net_input", "net_output", "power"))

#: 网名允许的字符集：`+5V` / `-12V` / `VCC` / `3.3V` / `VDD_IO` / `+3V3`。故意收紧，
#: 宁可漏也不要把 `R19`、`GND` 这类误当网名写进去（本地指标不看这个字段，官方口径
#: 每个错值 = 一个 fp+fn 对，代价与收益同量级）。
LABEL_RE = re.compile(r"^[+\-]?[A-Za-z0-9][A-Za-z0-9_.+\-]{0,15}$")
DIGITS_RE = re.compile(r"^\d{1,3}$")

#: 竖排文字的框形：窄而高。`open_GT` 的 `t0014` 是 10×24（比值 2.4）；横排小字
#: （位号/值）比值 < 1，不会被选中。
MIN_ASPECT = float(os.environ.get("STAGE2_POWER_LABEL_ASPECT", "1.5"))
MAX_W = float(os.environ.get("STAGE2_POWER_LABEL_MAXW", "26"))
MIN_H = float(os.environ.get("STAGE2_POWER_LABEL_MINH", "12"))
#: 横排网名的框形上限（`+5V` 25×10、`VCC` 29×10、`VIN` 32×15）
MAX_W_H = float(os.environ.get("STAGE2_POWER_LABEL_MAXW_H", "40"))
MAX_H_H = float(os.environ.get("STAGE2_POWER_LABEL_MAXH_H", "22"))

ROLES = ("net_label", "designator", "value", "pin_name")

#: 网名最短长度（`STAGE2_POWER_LABEL_MINLEN`，默认 3）。实测转帧会把 `VCC` 只读出尾部
#: `CC`、把无关笔画读成 `C`/`Q`/`R`（`work/_valbad.py` 逐条列出）—— 每个错值在官方口径
#: 里等于 fp+fn 双罚，所以宁可只要长度够的：`+5V`/`VCC`/`VIN`/`VOUT`/`3.3V`/`VDD_1.9V`。
MIN_LEN = int(os.environ.get("STAGE2_POWER_LABEL_MINLEN", "3"))


def _center(box):
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def _key(box):
    return tuple(round(float(v), 1) for v in box[:4])


def _gap(a, b):
    """两轴对齐矩形的边到边间距（相交＝0）。"""
    dx = max(0.0, max(float(a[0]), float(b[0])) - min(float(a[2]), float(b[2])))
    dy = max(0.0, max(float(a[1]), float(b[1])) - min(float(a[3]), float(b[3])))
    return (dx * dx + dy * dy) ** 0.5 if dx and dy else max(dx, dy)


def _has_label(rec):
    if not isinstance(rec, dict):
        return False
    for fld in ("partname", "value"):
        o = rec.get(fld) or {}
        if str(o.get("text") or "").strip():
            return True
    return False


def _pick(reads):
    """从一次读数里挑网名。

    按**角色优先级**取第一个过闸的（`net_label` → `designator` → `value` → `pin_name`），
    不按置信度取最大：`+5V` 这类标签四个角色读得都对，只有 `pin_number` 那遍因为走的是
    **收紧裁片**（`tight_ink_box`，切掉有色描边）会把 `+` 切掉读成 `45V` / `5V`
    （实测 `[483,562,508,573]`：net_label `+5V` 0.876、pin_number `5V` 0.920）。
    纯数字、长度不足、字符集不符的一律拒绝。
    """
    for role in ROLES:
        txt, conf = (reads or {}).get(role) or ("", 0.0)
        txt = str(txt or "").strip()
        if not txt or conf < CONF:
            continue
        if DIGITS_RE.match(txt) or not LABEL_RE.match(txt):
            continue
        # 两字符只在带数字时接受（`5V` / `12V` 可以，`CC` = `VCC` 的残读不行）
        if len(txt) < MIN_LEN and not (len(txt) >= 2 and any(c.isdigit() for c in txt)):
            continue
        return txt, float(conf)
    return "", 0.0


def _claimed(a, comps):
    """已被**别的元件**认领的文字框。

    ⚠ 电源符号自己的引脚槽不算认领：它的 `pin_0` 号是内部随机号，OCR 常把旁边的网名
    读成它的「引脚号」（case2 `c0027.pin_0.number='5V'`、`c0021='45V'`），把那批框
    当已认领就会漏掉 9 个 `+5V`（`work/_pwdbg.py` 实测）。
    """
    out = set()
    for cid, rec in (a.get("comps") or {}).items():
        if not isinstance(rec, dict):
            continue
        for fld in ("designator", "value", "partname"):
            o = rec.get(fld) or {}
            b = o.get("box")
            if b:
                out.add(_key(b))
        c = comps.get(str(cid)) or {}
        if str(c.get("type") or "").strip().lower() in TYPES:
            continue
        for slot in (rec.get("pins") or {}).values():
            for fld in ("number", "name"):
                o = (slot or {}).get(fld) or {}
                b = o.get("box")
                if b:
                    out.add(_key(b))
    return out


def _map_box(box, shape):
    """原图帧 → 顺时针 90° 帧（与 `pinrot._map_box` 同一套半开区间映射）。"""
    h = shape[0]
    x0, y0, x1, y1 = [float(v) for v in box]
    return [h - y1, x0, h - y0, x1]


def fill(bgr, doc, a, engine, reads=None) -> dict:
    """给缺网名的电源符号补 `partname`。返回统计（纯追加，不改任何既有读数）。"""
    stat = {"on": True, "rot": 0, "flat": 0, "written": 0, "labels": []}
    comps = {str(c.get("id")): c for c in (doc.get("components") or [])}
    need = {}
    for cid, rec in (a.get("comps") or {}).items():
        c = comps.get(str(cid))
        if not c:
            continue
        if str(c.get("type") or "").strip().lower() not in TYPES:
            continue
        if _has_label(rec):
            continue
        bb = c.get("bbox_img") or c.get("bbox_json")
        if not bb or len(bb) < 4:
            continue
        need[str(cid)] = _center(bb) + (
            str(c.get("type") or "").strip().lower() in PORT_TYPES,) + (
            [float(v) for v in bb],)
    if not need:
        return stat
    claimed = _claimed(a, comps)
    taken = set()

    def _write(cid, txt, conf, box, how):
        rec = (a.get("comps") or {}).get(cid)
        if not isinstance(rec, dict):
            return False
        rec["partname"] = {"text": txt, "conf": float(conf),
                           "box": [float(v) for v in box], "by": how}
        taken.add(cid)
        need.pop(cid, None)
        stat["written"] += 1
        stat["labels"].append({"cid": cid, "text": txt,
                               "conf": round(float(conf), 4), "by": how})
        if DBG:
            print("[pwlabel] %s cid=%s text=%r conf=%.2f" % (how, cid, txt, float(conf)))
        return True

    # ---- 第一遍：复用手里的读数（横排网名已经读对了，只是没归给这个符号）
    cands = []
    for r in (reads or []):
        b = r.get("bbox")
        if not b or len(b) < 4 or _key(b) in claimed:
            continue
        w, h = float(b[2]) - float(b[0]), float(b[3]) - float(b[1])
        if w <= 0 or h <= 0 or w > MAX_W_H or h > MAX_H_H:
            continue
        txt, conf = _pick(r.get("reads"))
        if not txt:
            continue
        bx, by = _center(b)
        best = None
        for cid, ctr in need.items():
            d = ((bx - ctr[0]) ** 2 + (by - ctr[1]) ** 2) ** 0.5
            _port = len(ctr) > 3 and ctr[2]
            ok = d <= (PORT_NEAR if _port else NEAR_H)
            if not ok and _port and _gap(b, ctr[3]) <= PORT_GAP:
                # 端口框可能已被撑到整支箭头，中心飘走 ⇒ 只要标签贴着框边就算
                ok = True
            if ok and (best is None or d < best[0]):
                best = (d, cid)
        if best is not None:
            cands.append((best[0], _key(b), txt, conf, b, best[1]))
    cands.sort(key=lambda t: t[0])
    for _d, kb, txt, conf, b, cid in cands:
        if cid in taken or kb in claimed:
            continue
        if _write(cid, txt, conf, b, "flat"):
            stat["flat"] += 1
            claimed.add(kb)

    # ---- 第二遍：高瘦框（竖排网名）→ 旋转帧重读
    if not need or engine is None:
        return stat
    tall = []
    for t in (doc.get("texts") or []) + (doc.get("texts_wide") or []):
        b = t.get("bbox_img")
        if not b or len(b) < 4 or _key(b) in claimed:
            continue
        w, h = float(b[2]) - float(b[0]), float(b[3]) - float(b[1])
        if w <= 0 or h <= 0:
            continue
        if h / max(w, 0.1) < MIN_ASPECT or w > MAX_W or h < MIN_H:
            continue
        tall.append(b)
    if not tall:
        return stat
    pairs = []
    for b in tall:
        bx, by = _center(b)
        best = None
        for cid, ctr in need.items():
            d = ((bx - ctr[0]) ** 2 + (by - ctr[1]) ** 2) ** 0.5
            if d <= NEAR and (best is None or d < best[0]):
                best = (d, cid)
        if best is not None:
            pairs.append((best[0], b, best[1]))
    pairs.sort(key=lambda t: t[0])
    if not pairs:
        return stat
    rot = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    rot2 = None
    for _d, b, cid in pairs:
        if cid in taken:
            continue
        best = ("", 0.0)
        try:
            got = engine.read(rot, [_map_box(b, bgr.shape)])[0].get("reads") or {}
            best = _pick(got)
        except Exception:                        # noqa: BLE001
            pass
        stat["rot"] += 1
        # 逆时针再试一次：竖排文字有「底→顶」和「顶→底」两种印法，实测两种都出现过
        # （`open_GT` 的 `+5V` 需要顺时针，某些页面的标签相反），取置信度高的那个。
        if best[1] < 0.99:
            try:
                if rot2 is None:
                    rot2 = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                w = bgr.shape[1]
                x0, y0, x1, y1 = [float(v) for v in b]
                got2 = engine.read(rot2, [[y0, w - x1, y1, w - x0]])[0].get("reads") or {}
                alt = _pick(got2)
                if alt[1] > best[1]:
                    best = alt
            except Exception:                    # noqa: BLE001
                pass
        if best[0]:
            _write(cid, best[0], best[1], b, "rot")
    return stat
