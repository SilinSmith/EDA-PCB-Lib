# -*- coding: utf-8 -*-
"""竖直（上下边）引脚的**旋转重读**（`STAGE2_PINROT`，默认由 `submission/run.py` 按
「VLM 引脚通道是否会跑」自动门控；直接调 stage2 时默认关）。

问题（可复现读数）
-----------------
官方 10GT 里有一类符号把引脚文字**旋转 90°** 印（SOT-23 之类，脚在元件上下边）：
`eda-pcb/tools/diag/_pinshift.py` 式的普查给出 case 6 有 6 根（U1/U3）、case 10 有 3 根
（U2）、case 7 有 2 根（U1）。`stage2/text.py` 的裁片是轴对齐的，`textrec` 也只认水平
文本 ⇒ 这些脚读到的是 `'L'`/`'O'`/`'D'`/`'33'` 这类碎片，**case 6 的引脚通道整张 0 分**
（`fast.py --cases 6 --no-vlm` ⇒ 总分 53.20 / pin 0.0000）。

做法
----
只对「脚点落在元件框上下边」的脚动手；**在元件局部窗口里**按 0°/90° 顺/逆时针三个参照帧
各裁一轮：体内条带（名字）与体外条带（号），`pinname._tight_ink_grp` 剔线取紧致框，每帧批量
读一轮。选帧不走「逐字段取 conf 最大」，而是**按元件外形分族投票**（同宽高 ⇒ 同一符号 ⇒
文字朝向一致）：族内统计每个朝向读出多少「像引脚名」的串，只有旋转帧严格赢过 0° 帧时才启用
该帧，且该帧对族内所有元件一致生效（详见 `fix()` 里的注释）。

实测（官方 10GT，`eda-pcb/tools/eval/fast.py`）
---------------------------------------------
无 VLM 口径（`--no-vlm`，`fast.py` 默认的 9 张官方图）**63.02 → 63.69（+0.67）且九张零回归**
（case 7 +3.85、case 10 +2.17，其余 7 张逐位不变；tp 引脚 39 → 41）；case 6（10GT 第 6 张，
不在 `fast.py` 默认 9 张里）53.20 → **56.06（+2.86）**，**引脚通道从 0.0000 变 0.0833，不再爆 0**。
交付口径（VLM 补强 + 融合 LLM 轮）里它**净负**：case 6 59.94 → 59.67（引脚 +0.0033 但网络
−0.0101 —— 修好的引脚名会参与电源轨挂标签），case 7/10 逐位不变（VLM 自己就读对了）。
⇒ 默认值走**门控**（见 `submission/run.py` 的「VLM 预探测 + 竖直脚旋转重读的自动兜底」）：
预探测 VLM 端点，只有「VLM 引脚通道不会跑」（`-vlm off` / 端点不支持图片 / 预探测失败）
**且没人显式设过 `STAGE2_PINROT`** 时才自动打开。这正是用户 m42543 要的灾难现场兜底。

判据细节（`eda-pcb/tools/diag/_pinrotview.py` 的三帧读数）
--------------------------------------------------------
* U1（SOT-23）cw 帧：内条带 `OUT`(0.96)/`GND`(0.96)，外条带 `3`(0.99)/`2`(1.00)；0° 帧只有
  `'Go'`/`'X'`/`'C7'` 这类碎片。U3 是同型符号（bbox 79×118 ≈ U1 的 79×119）⇒ 同族借票。
* U2/U4 的脚在 0° 就是对的，旋转帧却读出 `'T1'`/`'G 0'`/`'Q 2'`；**逐字段取 conf 最大**会把
  `U4 p12` 本来正确的 `EN` 覆盖成 `'2'` ⇒ 改成族级投票后这两族整体跳过（`no_rot_vote`）。

闸门与安全
----------
* 只碰 `type == "box"` 且 `_side_of(pt, bb) in ("T", "B")` 的脚；其余逐位不变。
* 号必须是 1~3 位纯数字（先去掉 `LlIi|[]()` 这类读串碎片字符）；名字必须过 `pinname.CLEAN`
  且不等于该脚的号；两者都要 `conf >= STAGE2_PINROT_CONF`（默认 0.70）。
* **族级投票门**（`STAGE2_PINROT_MIN_NAME`，默认 2）：旋转帧至少读出 2 个「像引脚名」的串，
  且「名 + 号」总数**严格大于** 0° 帧，才允许整族换帧；否则整族不动（`no_rot_vote`）。
  这一条是防「把本来读对的脚改坏」的关键（首版逐字段版在 case 6 覆盖 15 处、多为垃圾）。
* 已有值不是「不动」而是**覆盖**：竖直脚的旧值本来就是
  碎片，留着必错；覆盖会写一条 `LEDGER.record("overwrite", …)` 账本，可回溯 before/after。
* 名字与号都写回（读仍然照读，投票要用）。
"""
from __future__ import annotations

import os
import re
import sys

import cv2

try:                                    # 作为包导入（submission\run.py 把 eda-pcb\ 加进 sys.path）
    from envflag import env_on
except ImportError:                     # 直接 `python stage2\pinrot.py` 自检时
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envflag import env_on

try:
    from . import pinname as PN
except ImportError:                     # noqa: BLE001
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pinname as PN

#: 总开关（默认关：先 A/B 再决定默认值）
ON = env_on("STAGE2_PINROT", "0")
#: 采信门槛
CONF_MIN = float(os.environ.get("STAGE2_PINROT_CONF", "0.70"))
#: 名字（体内）与号（体外）的条带深度
NAME_DEPTH = float(os.environ.get("STAGE2_PINROT_NAME_DEPTH", "46.0"))
NUM_DEPTH = float(os.environ.get("STAGE2_PINROT_NUM_DEPTH", "34.0"))
#: 条带半宽（沿元件边方向）
HALF = float(os.environ.get("STAGE2_PINROT_HALF", "14.0"))
#: 元件局部窗口外扩量（旋转只在这个窗口里做，见 `fix()` 里 `INK_YNORM` 的注释）
WIN_PAD = float(os.environ.get("STAGE2_PINROT_WIN_PAD", "60.0"))
#: 元件分族的宽高取整粒度（同宽高 ⇒ 同一符号 ⇒ 文字朝向一致）
SHAPE_ROUND = float(os.environ.get("STAGE2_PINROT_SHAPE_ROUND", "8.0"))
#: 旋转帧要被选中，至少得读出几个「像引脚名」的串
ROT_MIN_NAME = int(os.environ.get("STAGE2_PINROT_MIN_NAME", "2"))

#: 读串里常见的碎片字符（竖线/括号/引号/连字），去号时先剥掉
JUNK = "LlIi|[](){}.,'\"-_/\\:;`~ "
NUM_RE = re.compile(r"^\d{1,3}$")
SIDES = ("T", "B")

#: 三个参照帧：**0° 是必须的**（`_c6num.py` 实测：U2/U4 的脚在 0° 就是对的，
#: 90° 会读成 `'G'`/`'2'` 这类碎片 ⇒ 只有旋转帧更强时才覆盖；0° 胜出即视为「本就对」）
_ROT = {"0": None, "cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}


def digits(s: str) -> str:
    """剥掉两端的碎片字符（`L1` → `1`、`2.` → `2`）；内部残留一律保留以便被 `NUM_RE` 拒绝。"""
    t = str(s or "").strip()
    while t and (t[0] in JUNK or t[0].isalpha()):
        t = t[1:]
    while t and (t[-1] in JUNK or t[-1].isalpha()):
        t = t[:-1]
    return t


def _map_box(box, shape, which):
    """把原图帧的框映射到旋转帧（`cv2.rotate` 的像素映射）。

    ⚠️ 必须用**半开区间**：`cv2.rotate` 把 src 像素 (x,y) 送到 dst 的
    `cw: (H-1-y, x)` / `ccw: (y, W-1-x)`（dst 形状 = (W,H)）。写成
    `H-1-y1` 这种「用最后一个像素下标当边界」的写法会**整体错 1px**，
    实测同一块裁片错 1px 就把 `GND` 读成 `GD`（`work/tools/dev/_rotdiff.py`
    打印 maxdiff=250 定位到）。所以这里统一用 `H-y1`/`W-x1`。
    """
    h, w = shape[0], shape[1]                 # 原图高/宽
    x0, y0, x1, y1 = [float(v) for v in box]
    if which == "0":
        return [x0, y0, x1, y1]
    if which == "cw":                          # (x,y) -> (H-1-y, x)
        return [h - y1, x0, h - y0, x1]
    return [y0, w - x1, y1, w - x0]            # ccw: (x,y) -> (y, W-1-x)


def _map_pt(pt, shape, which):
    h, w = shape[0], shape[1]
    x, y = float(pt[0]), float(pt[1])
    if which == "0":
        return (x, y)
    if which == "cw":
        return (h - 1 - y, x)
    return (y, w - 1 - x)


def _pitch_of(xs, ys, side):
    """沿边方向的引脚间距（上下边用 x，左右边用 y）：取最小的正间隔做基准，
    丢掉离群（`_map_box` 时代的教训：y 方向算 pitch 会把远端伪脚算进来 ⇒ half 爆到 28）。"""
    vals = sorted(xs if side in ("T", "B") else ys)
    dd = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    dd = [d for d in dd if d > 3]
    if not dd:
        return 20.0
    base = min(dd)
    dd = [d for d in dd if d <= max(4.0 * base, base + 6.0)]
    return float(sorted(dd)[len(dd) // 2])


def _out_box(pt, bb, half, depth, inset=2.0):
    """朝**体外**的条带（引脚号印在元件体外面）。"""
    x0, y0, x1, y1 = [float(v) for v in bb]
    s = PN._side_of((float(pt[0]), float(pt[1])), [x0, y0, x1, y1])
    if s == "B":
        return [pt[0] - half, y1 + inset, pt[0] + half, y1 + inset + depth]
    if s == "T":
        return [pt[0] - half, y0 - inset - depth, pt[0] + half, y0 - inset]
    if s == "L":
        return [x0 - inset - depth, pt[1] - half, x0 - inset, pt[1] + half]
    return [x1 + inset, pt[1] - half, x1 + inset + depth, pt[1] + half]


def fix(bgr, doc, a, engine, dbg_cid=None) -> dict:
    """对上下边引脚做旋转重读；就地改 `a`，返回统计。"""
    st = {"on": bool(ON), "conf_min": CONF_MIN, "n_comp": 0, "n_pin": 0,
          "num_set": 0, "name_set": 0, "emit": [], "skip": {}, "raw": []}
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
    comps = {c.get("id"): c for c in (doc.get("components") or [])}

    def _skip(k):
        st["skip"][k] = st["skip"].get(k, 0) + 1

    # ---- 收集候选：只收上下边的脚 ----
    jobs = []                                  # (cid, i, rec, field, box0, pt, num)
    bbs = {}                                   # cid -> 元件框（旋转窗口用）
    for cid, cc in (a.get("comps") or {}).items():
        c = comps.get(cid) or {}
        if (c.get("type") or "") != "box":
            continue
        bb = c.get("bbox_img") or cc.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        bb = [float(v) for v in bb]
        bbs[cid] = bb
        pts = (doc.get("pins") or {}).get(cid) or []
        tl = list((cc.get("pins") or {}).items())
        idxs = [int(k) for k, _p in tl if str(k).isdigit() and int(k) < len(pts)]
        xs_all = [float(pts[i][0]) for i in idxs]
        ys_all = [float(pts[i][1]) for i in idxs]
        n_here = 0
        for k, pin in tl:
            if not isinstance(pin, dict) or not str(k).isdigit():
                continue
            i = int(k)
            if i >= len(pts):
                _skip("no_point")
                continue
            pt = (float(pts[i][0]), float(pts[i][1]))
            side = PN._side_of(pt, bb)
            if side not in SIDES:
                continue
            # 沿边方向算间距（上下边看 x、左右边看 y），再定条带半宽
            pitch = _pitch_of(xs_all, ys_all, side)
            half = max(PN.HALF_LO, min(PN.HALF_HI, pitch * PN.HALF_FRAC)) * 1.1
            num = (((pin.get("number") or {}) or {}).get("text") or "").strip()
            jobs.append((cid, i, pin, "name",
                         PN._strip_box(pt, bb, half, depth=NAME_DEPTH), pt, num))
            jobs.append((cid, i, pin, "num",
                         _out_box(pt, bb, half, NUM_DEPTH), pt, num))
            n_here += 1
        if n_here:
            st["n_comp"] += 1
            st["n_pin"] += n_here
    if not jobs:
        return st

    # ---- 按元件分组，在「元件局部窗口」里旋转后读 ----
    # ★ 为什么不是整图旋转：`textrec/dataset.py:crop_gray` 里 `C.INK_YNORM` 会调
    #   `_ink_band()` 在裁片上下各探一段（`INK_PROBE_PAD`），按墨迹行重新定垂直取景。
    #   同一个框在**整图旋转**与**先裁小块再旋转**两张输入上，探带看到的邻域不同 ⇒
    #   归一后的裁片不同 ⇒ 同一块像素读出 `'GD'`(0.25) 与 `'GND'`(0.78)。
    #   实测（`work/tools/dev/_rotdiff.py`）局部窗口口径与探针口径读数一致，故用窗口。
    out = {}                                   # (ji, which) -> (text, conf)
    by_cid = {}
    for ji, job in enumerate(jobs):
        by_cid.setdefault(job[0], []).append(ji)
    for cid, jis in by_cid.items():
        bb = bbs[cid]
        wx0 = max(0, int(bb[0] - WIN_PAD))
        wy0 = max(0, int(bb[1] - WIN_PAD))
        wx1 = min(bgr.shape[1], int(bb[2] + WIN_PAD))
        wy1 = min(bgr.shape[0], int(bb[3] + WIN_PAD))
        if wx1 - wx0 < 8 or wy1 - wy0 < 8:
            continue
        win = bgr[wy0:wy1, wx0:wx1]
        for which, code in _ROT.items():
            rimg = win if code is None else cv2.rotate(win, code)
            rgray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
            items, keys = [], []
            for ji in jis:
                _cid, _i, _pin, field, box0, pt, _num = jobs[ji]
                lb = [box0[0] - wx0, box0[1] - wy0, box0[2] - wx0, box0[3] - wy0]
                lpt = (pt[0] - wx0, pt[1] - wy0)
                rbox = _map_box(lb, win.shape, which)
                rpt = _map_pt(lpt, win.shape, which)
                x0, y0, x1, y1 = [int(round(v)) for v in rbox]
                x0, y0 = max(0, x0), max(0, y0)
                x1 = min(rimg.shape[1], x1)
                y1 = min(rimg.shape[0], y1)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                tb = PN._tight_ink_grp(rimg, [float(x0), float(y0), float(x1), float(y1)],
                                       near_y=rpt[1])
                if tb is None:
                    continue
                items.append((tb, "pin_name" if field == "name" else "pin_number"))
                keys.append(ji)
            if not items:
                continue
            try:
                got = rec.read_many(rgray, items)
            except Exception as e:                 # noqa: BLE001
                st["error"] = "%s: %s" % (type(e).__name__, e)
                return st
            for ji, (txt, conf) in zip(keys, got):
                out[(ji, which)] = ((txt or "").strip(), float(conf))

    # ---- 元件族级投票选帧 ----
    # ★ 为什么不是「逐字段取 conf 最大」：竖着的碎片也常常 conf 很高，逐字段取胜出侧会把
    #   本来读对的脚覆盖掉。实测（case 6，`work/tools/dev/_stview.py`）逐字段版发出 12 个，
    #   其中 `c0000 p12 name 'EN' → '2'` 是**把正确名改坏**（U2/U4 的文字本来就是水平的，
    #   只是脚点多为 stage1 伪脚 ⇒ 0° 帧读出的也是碎片）。
    #   判据改成：竖直文字只出现在**某些符号**上 ⇒ 先按外形（bbox 宽高取整）把元件分族，
    #   族内统计各朝向能读出多少「像引脚名」的串；只有旋转帧严格赢过 0° 帧（且至少
    #   `ROT_MIN_NAME` 个）才启用该帧，选中的帧对族内**所有**元件一致生效。
    def _name_like(t):
        t = (t or "").strip()
        return (bool(t) and 2 <= len(t) <= PN.MAX_LEN and bool(PN.CLEAN.match(t))
                and any(ch.isalpha() for ch in t))

    def _num_like(t):
        return bool(NUM_RE.match(digits(t))) and len(str(t or "").strip()) <= 3

    shape_of = {}
    score = {}                                  # shape -> frame -> [n_name, n_num]
    for _cid, _bb in bbs.items():
        k = (int(round((_bb[2] - _bb[0]) / SHAPE_ROUND)),
             int(round((_bb[3] - _bb[1]) / SHAPE_ROUND)))
        shape_of[_cid] = k
        score.setdefault(k, {})
    for ji, (_cid, _i, _pin, field, _b, _p, _n) in enumerate(jobs):
        sc = score.get(shape_of.get(_cid))
        if sc is None:
            continue
        for w in _ROT:
            if (ji, w) not in out:
                continue
            cell = sc.setdefault(w, [0, 0])
            if field == "name" and _name_like(out[(ji, w)][0]):
                cell[0] += 1
            elif field == "num" and _num_like(out[(ji, w)][0]):
                cell[1] += 1
    pick = {}                                   # shape -> 选中的旋转帧
    for k, sc in score.items():
        base = sc.get("0", [0, 0])
        best_w, best_v = None, None
        for w in ("cw", "ccw"):
            v = sc.get(w, [0, 0])
            if v[0] >= ROT_MIN_NAME and (v[0] + v[1]) > (base[0] + base[1]):
                if best_v is None or (v[0] + v[1]) > (best_v[0] + best_v[1]):
                    best_w, best_v = w, v
        if best_w is None:
            _skip("no_rot_vote")
        else:
            pick[k] = best_w
            st.setdefault("vote", []).append(
                {"shape": [k[0], k[1]], "rot": best_w, "name": best_v[0],
                 "num": best_v[1], "base": [base[0], base[1]]})

    # ---- 用选中的帧覆盖（族内一致） ----
    for ji, (cid, i, pin, field, box0, _pt, num) in enumerate(jobs):
        which = pick.get(shape_of.get(cid))
        if which is None:
            _skip("no_rot_vote_" + field)
            continue
        if (ji, which) not in out:
            _skip("no_read_" + field)
            continue
        txt, conf = out[(ji, which)]
        if field == "num":
            d = digits(txt)
            if not _num_like(txt):
                _skip("bad_number")
                continue
            val = d
        else:
            if not _name_like(txt):
                _skip("bad_name")
                continue
            if num and txt.lower() == str(num).lower():
                _skip("same_as_number")
                continue
            val = txt
        if conf < CONF_MIN:
            _skip("low_conf_" + field)
            continue
        old = pin.get(field)
        old_txt = ((old or {}).get("text") if isinstance(old, dict) else old) or ""
        if str(old_txt).strip() == str(val).strip():
            _skip("same_" + field)
            continue
        before = {"number": (((pin.get("number") or {}) or {}).get("text") or ""),
                  "name": (((pin.get("name") or {}) or {}).get("text") or "")}
        pin[field] = {"text": val, "conf": round(conf, 3),
                      "box": [float(v) for v in box0], "d": 0.0,
                      "src": "rot%s" % which}
        st["num_set" if field == "num" else "name_set"] += 1
        after = {"number": (((pin.get("number") or {}) or {}).get("text") or ""),
                 "name": (((pin.get("name") or {}) or {}).get("text") or "")}
        st["emit"].append({"cid": cid, "pin": i, "field": field, "text": val,
                           "conf": round(conf, 3), "rot": which, "before": before})
        try:                                    # 「不删除只削弱」账本（可选依赖）
            from cvparts.hypo import LEDGER
            LEDGER.record("overwrite", where="pinrot:%s:%s:%s" % (cid, i, field),
                          before=before, after=after,
                          reason="竖直脚（上下边）旋转 90° 重读（%s）：旧值是轴对齐裁片读出的碎片" % which,
                          by="stage2/pinrot.py:fix")
        except Exception:                       # noqa: BLE001
            pass
        if dbg_cid and dbg_cid == cid:
            print("[dbg] pinrot %s pin=%d %s %r -> %r (%.2f, %s)"
                  % (cid, i, field, old_txt, val, conf, which))
    return st


def _selftest() -> int:
    """合成图：把 `OUT` 竖着画在框内，检查旋转重读能拿回来。"""
    import numpy as _np
    img = _np.full((200, 300, 3), 255, _np.uint8)
    cv2.rectangle(img, (100, 40), (220, 160), (0, 0, 0), 3)
    cv2.putText(img, "OUT", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA)
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    mapbox = _map_box([148.0, 172.0, 192.0, 198.0], img.shape, "cw")
    tb = PN._tight_ink_grp(rot, mapbox, near_y=None)
    ok = tb is not None
    d1 = digits("L1") == "1" and digits("33") == "33" and not NUM_RE.match(digits("1K2"))
    print("pinrot 自检：映射 %s（%s）| digits %s"
          % ("通过" if ok else "失败",
             None if tb is None else [round(v, 1) for v in tb],
             "通过" if d1 else "失败"))
    return 0 if (ok and d1) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
