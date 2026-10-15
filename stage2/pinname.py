# -*- coding: utf-8 -*-
"""引脚名救援（`STAGE2_PINNAME_RESCUE`，2026-10-16 起**默认开**）：号读出来了、名字没读到 ⇒ 以前整条丢。

★ 默认值依据（端到端 A/B，`tools\\eval\\fast.py`，两套装置都不为负）
  * 官方 10 张 `--no-vlm`：**60.5854 → 61.0000（+0.4146/张，3 涨 0 跌）** ——
    case 4 58.75 → 60.14（+1.39）、case 5 53.06 → 53.40（+0.35）、case 6 58.09 → 60.46
    （+2.38），其余 7 张逐位不变；引脚 tp 297 → 301、网络 tp 112 → 115、导线 tp 241 → 249。
  * 派生 val 19 张（泄漏无关留出集）`--no-vlm`：**17.0047 → 17.0322（+0.0274，4 涨 0 跌）**
    —— 0176 +0.33、0011 +0.10、0166 +0.08、0050 +0.01。
  * 实际补名（与关闭臂逐槽位 diff）：case 4 `U1`（c0001）号 5 补 `PROG`；case 5 `U1`
    号 7 补 `P`、号 1 补 `GND`；case 6 两颗 IC 号 3 补 `FAULT`、号 8 补 `EN` —— 共 6 处。
  关闭时（`STAGE2_PINNAME_RESCUE=0`）行为与不挂接这条通道**逐位一致**。


问题（可复现读数）
------------------
`eda-pcb/tools/diag/_pinledger.py --run <交付 run> --split 10GT` 在官方 9 张上给出
`tp=37 / 缺号=51 / 多号=6 / 名错=16 / 位置错=16`，其中缺号的 **23 根（45%）** 成因是
「号读对了，但名字没读到，于是被 `stage2/assemble.py` 的 `BOX_PIN_NEED_NAME` 闸门
整条丢掉」。`eda-pcb/tools/diag/_namerescue.py` 进一步量出：这 23 根里 **21 根在
体内该位置确实印着字**（体内墨像素 > 0）——名字并不是没印刷，而是 **文字检测器
没把这个体内名字检出**，于是 `assign` 的候选池里根本没有它，与读得好不好无关。

做法
----
从脚点出发、朝 IC 体内切一条带，在带内取墨迹连通域：

1. **先剔线状连通块**（`_tight_ink_grp`）：IC 体描边是长直线，只要落进带里就会把
   紧致外接框撑满整条带，crop 里于是同时有描边与字 ⇒ 读数被打碎。实测把「紧致框」
   换成「剔线后的行分组框」，同一位置的名字读取准确率 **0.19 → 0.33**。
2. **按 y 重叠贪心成行**，只取行心离脚点最近的那一组（避免读到邻脚的名字）。
3. 用阶段二同一个 CRNN **只读一次**（`pin_name` 角色），过四道闸门后才写回：
   `conf >= CONF_MIN`、字符集干净、不得等于该脚自己的号、长度上限。

实测（10GT 9 张，`_namerescue.py` + `tools/diag/_namerule.py` 离线规则比较）
-----------------------------------------------------------------------------
* 单臂（行分组、剔线、内缩 4px）：出 21 根、对 11 根（**准确率 0.524**）；
* 加 `conf >= 0.70` 门槛：出 19 根、对 11 根、错 8 根 ⇒ pinF1 0.3915 → 0.4615
  （**+0.0700**），折合总分 **+1.75**（= 0.25 × +0.07 × 100）；
* 两个独立臂一致（`行分组 == 剔线臂`）时 5/5 全对，但只有 5 根 ⇒ 留作后续加固；
* 官方元件库那条路**无效**：这些 IC 的 `partname` 大多没读出来（字段里是 `VOUT`
  / `CHR5` 这类值文本），`libpins.key_for` 解析不到型号，库名集合为空。

代价与安全
----------
* 只对 `type == "box"` 且「**号已读到、名为空**」的脚动手；不动任何已有名字。
* 判错的名字在下游是 **fp+fn 双罚**（`schematic_parser/metrics.py:466`），所以闸门
  宁可少发：`CONF_MIN` 默认 0.70（离线比过 0.5/0.6/0.7/0.8，0.7 与 0.8 并列最好）。
* 读不出就什么都不写，行为与今天逐位一致（`STAGE2_PINNAME_RESCUE=0` 也一样）。
"""
from __future__ import annotations

import os
import re
import sys

import cv2
import numpy as np

try:                                    # 作为包导入（submission\run.py 把 eda-pcb\ 加进 sys.path）
    from envflag import env_on
except ImportError:                     # 直接 `python stage2\pinname.py` 自检时
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envflag import env_on

#: 总开关（默认开，2026-10-16；依据见模块 docstring：官方 10 +0.4146、派生 val 19 +0.0274，两套装置都 0 跌）
ON = env_on("STAGE2_PINNAME_RESCUE", "1")
#: 采信门槛
CONF_MIN = float(os.environ.get("STAGE2_PINNAME_RESCUE_CONF", "0.70"))
#: 两参考点（脚点 y / 该脚号码文字框的 y）必须取到**同一行**才发（默认开）
AGREE = env_on("STAGE2_PINNAME_RESCUE_AGREE", "1")
#: 带的内缩（跳过体描边）与深度
INSET = float(os.environ.get("STAGE2_PINNAME_RESCUE_INSET", "4.0"))
DEPTH = float(os.environ.get("STAGE2_PINNAME_RESCUE_DEPTH", "52.0"))
HALF_LO, HALF_HI, HALF_FRAC = 7.0, 26.0, 0.68
INK_GRAY = int(os.environ.get("STAGE2_PINNAME_RESCUE_GRAY", "170"))
INK_SAT = int(os.environ.get("STAGE2_PINNAME_RESCUE_SAT", "70"))
MIN_INK = 6
MAX_LEN = 14
GRP_OVERLAP = 0.45
CLEAN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._\-/]{0,13}$")


def _side_of(pt, bb):
    """pt 落在 bbox 的哪条边（L/R/T/B）。"""
    x1, y1, x2, y2 = bb
    ds = {"L": abs(pt[0] - x1), "R": abs(pt[0] - x2),
          "T": abs(pt[1] - y1), "B": abs(pt[1] - y2)}
    return min(ds, key=lambda k: ds[k])


def _strip_box(pt, bb, half, depth=DEPTH, inset=INSET):
    """贴边、以脚点为中心、朝体内 depth 的带（图像帧）。"""
    s = _side_of(pt, bb)
    x1, y1, x2, y2 = bb
    if s in ("L", "R"):
        xe = x1 + inset if s == "L" else x2 - inset
        xa, xb = (xe, xe + depth) if s == "L" else (xe - depth, xe)
        return (min(xa, xb), pt[1] - half, max(xa, xb), pt[1] + half)
    ye = y1 + inset if s == "T" else y2 - inset
    ya, yb = (ye, ye + depth) if s == "T" else (ye - depth, ye)
    return (pt[0] - half, min(ya, yb), pt[0] + half, max(ya, yb))


def _ink_mask(bgr, box, gray_max=INK_GRAY, sat_max=INK_SAT):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, (x1, y1)
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    mx, mn, g = sub.max(2), sub.min(2), sub.mean(2)
    m = (((mx - mn) < sat_max) & (g < gray_max)).astype(np.uint8)
    if int(m.sum()) < MIN_INK:
        return None, (x1, y1)
    return m, (x1, y1)


def _ink_count(bgr, box):
    m, _o = _ink_mask(bgr, box)
    return 0 if m is None else int(m.sum())


def _tight_ink_grp(bgr, box, near_y=None, pad=1.0):
    """剔线 → 按 y 重叠成行 → 取行心离 `near_y` 最近的那一行，返回紧致框。"""
    m, (ox, oy) = _ink_mask(bgr, box)
    if m is None:
        return None
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    H, W = m.shape
    comps = []
    for i in range(1, n):
        bw_, bh_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        ar = st[i, cv2.CC_STAT_AREA]
        if ar < 4:
            continue
        if (bw_ >= 0.75 * W and bh_ <= 4) or (bh_ >= 0.75 * H and bw_ <= 4):
            continue                      # 体描边（横线 / 竖线）
        if ar >= 0.55 * H * W:
            continue                      # 整块填充
        comps.append([int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                      int(bw_), int(bh_)])
    if not comps:
        return None
    comps.sort(key=lambda c: (c[1], c[0]))
    groups = []
    for c in comps:
        placed = False
        for gp in groups:
            ov = min(gp[1] + gp[3], c[1] + c[3]) - max(gp[1], c[1])
            if ov > GRP_OVERLAP * min(gp[3], c[3]):
                gp[0] = min(gp[0], c[0]); gp[1] = min(gp[1], c[1])
                gp[2] = max(gp[2], c[0] + c[2]); gp[3] = max(gp[3], c[1] + c[3])
                placed = True
                break
        if not placed:
            groups.append([c[0], c[1], c[0] + c[2], c[1] + c[3]])
    if near_y is None:
        near_y = oy + H / 2.0
    best = min(groups, key=lambda gp: abs(oy + (gp[1] + gp[3]) / 2.0 - float(near_y)))
    return (float(ox + best[0] - pad), float(oy + best[1] - pad),
            float(ox + best[2] + pad), float(oy + best[3] + pad))


def _cbox_cy(b):
    """文字框（图像帧 [x0,y0,x1,y1]）的 y 中心；拿不到返回 None。"""
    try:
        return (float(b[1]) + float(b[3])) / 2.0
    except Exception:                                                       # noqa: BLE001
        return None


def rescue(bgr, doc, a, engine, dbg_cid=None) -> dict:
    """把「号读对、名没读到」的 box 脚补上名字；就地改 `a`，返回统计。"""
    st = {"on": bool(ON), "conf_min": CONF_MIN, "n_cand": 0, "n_emit": 0,
          "emit": [], "skip": {}}
    if not ON:
        return st
    try:
        rec = engine.recognizer()
    except Exception as e:                     # noqa: BLE001
        st["skip"]["recognizer_error"] = 1
        st["error"] = "%s: %s" % (type(e).__name__, e)
        return st
    if rec is None:
        st["skip"]["no_recognizer"] = 1
        return st
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if getattr(bgr, "ndim", 2) == 3 else bgr

    def _skip(k):
        st["skip"][k] = st["skip"].get(k, 0) + 1

    comps = {c.get("id"): c for c in (doc.get("components") or [])}
    plans, items, slot = [], [], []
    for cid, cc in (a.get("comps") or {}).items():
        c = comps.get(cid) or {}
        if (c.get("type") or "") != "box":      # 与 `assemble.py:953` 的闸门同域
            continue
        bb = c.get("bbox_img") or cc.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        bb = [float(v) for v in bb]
        pts = (doc.get("pins") or {}).get(cid) or []
        tl = list((cc.get("pins") or {}).items())
        ys = sorted(float(pts[int(k)][1]) for k, _p in tl
                    if str(k).isdigit() and int(k) < len(pts))
        dd = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        dd = [d for d in dd if d > 3]
        pitch = float(np.median(dd)) if dd else 20.0
        half = max(HALF_LO, min(HALF_HI, pitch * HALF_FRAC)) * 1.1
        for k, pin in tl:
            if not isinstance(pin, dict) or not str(k).isdigit():
                continue
            i = int(k)
            if pin.get("name"):
                continue                        # 已有名字一律不动
            num = (((pin.get("number") or {}) or {}).get("text") or "").strip()
            if not num:
                continue                        # 号都没读到：不归本机制管
            if i >= len(pts):
                _skip("no_point")
                continue
            pt = (float(pts[i][0]), float(pts[i][1]))
            sb = _strip_box(pt, bb, half)
            if _ink_count(bgr, sb) < MIN_INK:
                _skip("no_ink")
                continue
            # 两个参考点：脚点自己，以及该脚「号码文字框」的 y（号码印在脚边上，
            # 比脚点更不容易被几何偏置带跑）。两参考点各取一行，稍后要求一致。
            refs = [pt[1]]
            ncy = _cbox_cy((pin.get("number") or {}).get("box"))
            if ncy is not None and abs(ncy - pt[1]) > 1.0:
                refs.append(ncy)
            tbs = []
            for ry in refs:
                t = _tight_ink_grp(bgr, sb, near_y=ry)
                if t is not None and t not in tbs:
                    tbs.append(t)
            if not tbs:
                _skip("no_group")
                continue
            st["n_cand"] += 1
            plans.append((cid, i, pin, num, pt, tbs))
            for t in tbs:
                items.append((t, "pin_name"))
                slot.append(len(plans) - 1)
    if not items:
        return st
    try:
        got = rec.read_many(gray, items)
    except Exception as e:                      # noqa: BLE001
        st["error"] = "%s: %s" % (type(e).__name__, e)
        return st
    per = {}
    for pi, (txt, conf) in zip(slot, got):
        per.setdefault(pi, []).append(((txt or "").strip(), float(conf)))
    for pi, (cid, i, pin, num, pt, tbs) in enumerate(plans):
        cands = per.get(pi) or []
        if AGREE and len(cands) > 1:
            if cands[0][0].lower() != cands[1][0].lower():
                _skip("disagree")
                continue
        if not cands:
            _skip("no_read")
            continue
        txt, conf = max(cands, key=lambda c: c[1])
        tb = tbs[cands.index(max(cands, key=lambda c: c[1]))]
        t = (txt or "").strip()
        conf = float(conf)
        if not t:
            _skip("empty")
            continue
        if conf < CONF_MIN:
            _skip("low_conf")
            continue
        if not CLEAN.match(t):
            _skip("dirty")
            continue
        if t == num:
            _skip("same_as_number")
            continue
        if len(t) > MAX_LEN:
            _skip("too_long")
            continue
        if pin.get("name"):                     # 双保险
            continue
        pin["name"] = {"text": t, "conf": conf, "box": [float(v) for v in tb],
                       "d": 0.0, "src": "rescue"}
        st["n_emit"] += 1
        st["emit"].append({"cid": cid, "pin": i, "num": num, "text": t,
                           "conf": round(conf, 3)})
        try:                                    # 「不删除只削弱」账本（可选依赖）
            from cvparts.hypo import LEDGER
            LEDGER.record("evidence", where="pinname:%s:%s" % (cid, i),
                          before={"name": None, "number": num},
                          after={"text": t, "conf": round(conf, 3),
                                 "src": "rescue"},
                          reason="体内行分组裁片救回引脚名（原本会被 NEED_NAME 闸门整条丢）",
                          by="stage2/pinname.py:rescue")
        except Exception:                       # noqa: BLE001
            pass
        if dbg_cid and dbg_cid == cid:
            print("[dbg] pinname rescue cid=%s pin=%d num=%s -> %s (%.2f)"
                  % (cid, i, num, t, conf))
    return st


def _selftest() -> int:
    """合成图夹具：体内印 `VOUT`、体描边很粗、外面印号。"""
    import numpy as _np
    img = _np.full((200, 300, 3), 255, _np.uint8)
    cv2.rectangle(img, (100, 40), (220, 160), (0, 0, 0), 3)   # 粗描边
    cv2.putText(img, "VOUT", (112, 96), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    bb = [100.0, 40.0, 220.0, 160.0]
    sb = _strip_box((100.0, 96.0), bb, 14.0)
    tb = _tight_ink_grp(img, sb, near_y=96.0)
    ok = tb is not None and _ink_count(img, sb) > 0
    print("pinname 自检 1 项：%s（strip=%s tb=%s）"
          % ("全部通过" if ok else "失败", [round(v, 1) for v in sb],
             None if tb is None else [round(v, 1) for v in tb]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
