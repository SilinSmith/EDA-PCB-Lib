# -*- coding: utf-8 -*-
"""脚号 / 脚名的多口径投票重读（全边，不依赖旋转）。

背景（实测，`work/_pinvote.py` 与 `work/_pinscale2.py` 探针）：
- 存进 `a["comps"][cid]["pins"][slot]` 的框是**单字紧框**：`GND` 的框只覆盖到 `G`，
  `SWT` 只覆盖到 `T`，`LED1` 只覆盖到 `E` ⇒ 在这些框上怎么加 padding 都读不回整串。
- 所以这里像 `pinrot` 一样**按条带重定框**：体内条带读名字、体外条带读号。
- 单口径仍会读出碎片（`G` / `T` / `SV`），必须多口径投票：
  `{原样, 2× 放大, 反色}` 三路，≥2 路口径一致且 conf ≥ CONF_MIN 才采用。
- 绝不在「名字仍是空」的槽上写号：判分器对 `pin_<num>` 缺失只记 1 个 fn，而
  `pinname` 不符记 fp+fn 双扣（`schematic_parser/metrics.py:454-492`）。
"""
import os

import cv2

from . import pinname as PN
from .pinrot import _out_box

try:                                   # 仓库脚本与包内两种入口
    from envflag import env_on
except ImportError:                    # pragma: no cover
    from ..envflag import env_on

# 默认开（§5.589）：只改**脚名**、不改号（改号会移动 pin 键，case4 实测 -2.7 分）。
ON = env_on("STAGE2_PINVOTE", "1")
NUM_ON = env_on("STAGE2_PINVOTE_NUM", "0")   # 改号会移动 pin 键（case4 实测 -2.7）
CONF_MIN = float(os.environ.get("STAGE2_PINVOTE_CONF", "0.90"))
NAME_DEPTH = float(os.environ.get("STAGE2_PINVOTE_NAME_DEPTH", "46.0"))
NUM_DEPTH = float(os.environ.get("STAGE2_PINVOTE_NUM_DEPTH", "34.0"))
ZOOM = float(os.environ.get("STAGE2_PINVOTE_ZOOM", "2.0"))
PAD = float(os.environ.get("STAGE2_PINVOTE_PAD", "3.0"))
SIDES = ("T", "B", "L", "R")


def _digits(s):
    return "".join(ch for ch in (s or "") if ch.isdigit())


def _name_like(t):
    t = (t or "").strip()
    return (bool(t) and 2 <= len(t) <= PN.MAX_LEN and bool(PN.CLEAN.match(t))
            and any(ch.isalpha() for ch in t))


def _num_like(t):
    d = _digits(t)
    return bool(d) and len(d) <= 3 and bool(PN.CLEAN.match((t or "").strip() or "x"))


def _crop(img, box, pad):
    x0 = max(0, int(round(box[0] - pad)))
    y0 = max(0, int(round(box[1] - pad)))
    x1 = min(img.shape[1], int(round(box[2] + pad)))
    y1 = min(img.shape[0], int(round(box[3] + pad)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, None
    return img[y0:y1, x0:x1], (x0, y0)


def _read_one(rec, gray, box, pad=0.0, zoom=1.0, invert=False):
    """在 box（图像坐标）上裁一小块，按需放大/反色，再读一次。返回 (text, conf)。"""
    crop, off = _crop(gray, box, pad)
    if crop is None:
        return "", 0.0
    if zoom != 1.0:
        crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    if invert:
        crop = 255 - crop
    h, w = crop.shape[:2]
    try:
        got = rec.read_many(crop, [([0.0, 0.0, float(w), float(h)], "pin")])
    except Exception:                      # noqa: BLE001
        return "", 0.0
    if not got:
        return "", 0.0
    txt, conf = got[0]
    return (txt or "").strip(), float(conf)


def _vote(cands, like):
    """cands: [(text, conf, variant)] -> (text, conf, variant) | None（≥2 路一致）。"""
    good = [(t, c, v) for (t, c, v) in cands if like(t)]
    if not good:
        return None
    tally = {}
    for t, c, v in good:
        key = _digits(t) if like is _num_like else t
        tally.setdefault(key, []).append((t, c, v))
    best = max(tally.items(), key=lambda kv: (len(kv[1]), max(x[1] for x in kv[1])))
    hits = best[1]
    if len(hits) < 2:
        return None
    t, c, v = max(hits, key=lambda x: x[1])
    if c < CONF_MIN:
        return None
    return t, c, v


def fix(bgr, doc, a, engine, dbg_cid=None):
    """对 box 元件的所有脚做条带重定框 + 三口径投票重读；就地改 `a`。"""
    st = {"on": bool(ON), "conf_min": CONF_MIN, "n_comp": 0, "n_slot": 0,
          "num_set": 0, "name_set": 0, "emit": [], "skip": {}}
    if not ON:
        return st
    try:
        rec = engine.recognizer()
    except Exception as e:                 # noqa: BLE001
        st["skip"]["recognizer_error"] = 1
        st["error"] = "%s: %s" % (type(e).__name__, e)
        return st
    if rec is None:
        st["skip"]["no_recognizer"] = 1
        return st
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    comps = {c.get("id"): c for c in (doc.get("components") or [])}

    def _skip(k):
        st["skip"][k] = st["skip"].get(k, 0) + 1

    for cid, cc in (a.get("comps") or {}).items():
        c = comps.get(cid) or {}
        if (c.get("type") or "") != "box":
            continue
        bb = c.get("bbox_img") or cc.get("bbox_img")
        if not bb or len(bb) != 4:
            continue
        bb = [float(v) for v in bb]
        pts = (doc.get("pins") or {}).get(cid) or []
        tl = list((cc.get("pins") or {}).items())
        idxs = [int(k) for k, _p in tl if str(k).isdigit() and int(k) < len(pts)]
        if not idxs:
            continue
        _used = set()
        for _k, _p in tl:
            if isinstance(_p, dict):
                _t = (((_p.get("name") or {}) or {}).get("text") or "").strip().lower()
                if _t:
                    _used.add(_t)
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
                _skip("side_" + str(side))
                continue
            xs = [float(pts[j][0]) for j in idxs if PN._side_of((pts[j][0], pts[j][1]), bb) in ("T", "B")]
            ys = [float(pts[j][1]) for j in idxs if PN._side_of((pts[j][0], pts[j][1]), bb) in ("L", "R")]
            pitch = 20.0
            vals = sorted(xs if side in ("T", "B") else ys)
            dd = [vals[q + 1] - vals[q] for q in range(len(vals) - 1)]
            dd = [d for d in dd if d > 3]
            if dd:
                base = min(dd)
                dd = [d for d in dd if d <= max(4.0 * base, base + 6.0)]
                pitch = float(sorted(dd)[len(dd) // 2])
            half = max(PN.HALF_LO, min(PN.HALF_HI, pitch * PN.HALF_FRAC)) * 1.1
            n_here += 1
            for field, box0 in (("name", PN._strip_box(pt, bb, half, depth=NAME_DEPTH)),
                                ("num", _out_box(pt, bb, half, NUM_DEPTH))):
                if field == "num" and not NUM_ON:
                    continue
                tb = PN._tight_ink_grp(bgr, [float(v) for v in box0], near_y=pt[1])
                base_box = tb if tb is not None else [float(v) for v in box0]
                cands = []
                for variant, kw in (("pad%d" % int(PAD), {"pad": PAD}),
                                    ("zoom", {"pad": PAD, "zoom": ZOOM}),
                                    ("inv", {"pad": PAD, "invert": True})):
                    txt, conf = _read_one(rec, gray, base_box, **kw)
                    if txt:
                        cands.append((txt, conf, variant))
                hit = _vote(cands, _num_like if field == "num" else _name_like)
                if hit is None:
                    _skip("no_vote_" + field)
                    continue
                txt, conf, variant = hit
                if field == "num":
                    val = _digits(txt)
                    num_now = (((pin.get("number") or {}) or {}).get("text") or "").strip()
                    name_now = (((pin.get("name") or {}) or {}).get("text") or "").strip()
                    if not name_now and not num_now:
                        _skip("no_name_yet")   # 有号无名 = fp+fn 双扣，宁可不发
                        continue
                else:
                    val = txt
                    num_now = (((pin.get("number") or {}) or {}).get("text") or "").strip()
                    if num_now and val.lower() == num_now.lower():
                        _skip("same_as_number")
                        continue
                    if val.strip().lower() in _used:
                        # 同一颗件里已有同名脚 ⇒ 再写一次只会把错名复制到相邻脚
                        _skip("dup_name")
                        continue
                old = pin.get("number" if field == "num" else "name")
                old_txt = ((old or {}).get("text") if isinstance(old, dict) else old) or ""
                if str(old_txt).strip() == str(val).strip():
                    _skip("same_" + field)
                    continue
                pin["number" if field == "num" else "name"] = {
                    "text": val, "conf": round(conf, 3),
                    "box": [float(v) for v in base_box], "d": 0.0, "src": "vote:%s" % variant}
                st["num_set" if field == "num" else "name_set"] += 1
                st["emit"].append({"cid": cid, "pin": i, "field": field, "text": val,
                                   "conf": round(conf, 3), "variant": variant,
                                   "before": old_txt})
                if dbg_cid and dbg_cid == cid:
                    print("[dbg] pinvote %s pin=%d %s %r -> %r (%.2f, %s)"
                          % (cid, i, field, old_txt, val, conf, variant))
        if n_here:
            st["n_comp"] += 1
            st["n_slot"] += n_here
    if st.get("error"):
        print("[stage2] 脚号名投票：异常 %s" % st["error"])
    elif st["n_slot"]:
        top = sorted(st["skip"].items(), key=lambda kv: -kv[1])[:4]
        print("[stage2] 脚号名投票：槽 %d / 改号 %d / 改名 %d / 跳过 %s"
              % (st["n_slot"], st["num_set"], st["name_set"],
                 "、".join("%s %d" % kv for kv in top) or "无"))
    return st
