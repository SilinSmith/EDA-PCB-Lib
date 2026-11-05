# -*- coding: utf-8 -*-
"""位号回填：把 `pNNN` 兜底键改成邻近的孤立位号文字（容灾）。

纯函数，便于离线单测与在交付层复用；调用方负责给出候选文字、页高与（可选的）模板框。
"""
import re

PREFIX = {"r": ("R",), "c": ("C",), "l": ("L",), "d": ("D",),
          "led": ("LED", "D"), "switch": ("SW",), "testpoint": ("TP",)}
DESIG = re.compile(r"^([A-Za-z]{1,3})\d{1,4}$")


def _dist(box, bb):
    dx = max(0.0, max(bb[0] - box[2], box[0] - bb[2]))
    dy = max(0.0, max(bb[1] - box[3], box[1] - bb[3]))
    return (dx * dx + dy * dy) ** 0.5


def rename_components(doc, cand, hgt, *, radius=12.0, tmpl=None):
    """就地改名；返回 [(old, new, d)]。只动键名与引用，不增删元件。"""
    comps = doc.get("components")
    if not isinstance(comps, dict) or not cand or not hgt:
        return []
    keys = set(str(k) for k in comps)
    taken = set()
    renamed = []
    for old in list(comps):
        if not re.fullmatch(r"p\d+", str(old)):
            continue
        v = comps.get(old)
        if not isinstance(v, dict):
            continue
        want = PREFIX.get(str(v.get("type") or ""))
        if not want:
            continue
        bb = v.get("bbox")
        tv = (tmpl or {}).get(old)
        if isinstance(tv, dict) and isinstance(tv.get("bbox"), (list, tuple)):
            bb = tv["bbox"]
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        ib = [float(bb[0]), hgt - float(bb[3]), float(bb[2]), hgt - float(bb[1])]
        best = None
        for c in cand:
            t = str(c.get("text") or "")
            m = DESIG.match(t)
            if not m or t in keys or t in taken:
                continue
            if m.group(1).upper() not in want:
                continue
            box = c.get("box") or []
            if len(box) != 4:
                continue
            d = _dist([float(x) for x in box], ib)
            if d > radius:
                continue
            if best is None or d < best[0]:
                best = (d, t)
        if best is None:
            continue
        d, new = best
        comps[new] = comps.pop(old)
        if isinstance(comps[new], dict) and comps[new].get("Name") == old:
            comps[new]["Name"] = new
        keys.discard(str(old))
        keys.add(new)
        taken.add(new)
        pins = doc.get("pins")
        if isinstance(pins, dict) and old in pins:
            pins[new] = pins.pop(old)
        for net in (doc.get("nets") or {}).values():
            refs = net.get("_refs")
            if isinstance(refs, list):
                for r in refs:
                    if isinstance(r, list) and r and str(r[0]) == str(old):
                        r[0] = new
            hg = net.get("hyperGraph")
            if isinstance(hg, str) and ("%s." % old) in hg:
                net["hyperGraph"] = hg.replace("%s." % old, "%s." % new)
        renamed.append((old, new, round(d, 2)))
    return renamed
