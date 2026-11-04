# -*- coding: utf-8 -*-
"""Pin-name row rebind for a side whose leading digit was never read.

The lane readers bind a printed name to the pin row it is drawn on, but a name
is only useful when the *digit* of that row identifies the pin.  When the
leading digit of a side is missing (thin ``1``, wire over the glyph), every
name on the side still sits on its own row and the digit slots carry the next
numbers up -- so slot ``2`` inherits the name printed for pin 1 and the whole
side is off by one row.

The signature is exact, so this pass only acts on it:

* the side carries exactly one more name box than digit boxes;
* the digit texts form a consecutive ascending run (``2,3,4``);
* their rows ascend in the same direction as the numbers;
* the surplus name sits past the low-numbered end (above ``2`` on a
  top-to-bottom numbered side).

Nothing else is touched: no name is created, moved, renamed or deleted, and the
digit that never existed stays missing.  Measured on the 13-page set: fires
once (``case 11 U1`` left side, ``pin_2`` gets ``BAT`` instead of ``OUT``) for
+1.5152 on that page, +0.1166 mean, every other page byte-identical.
"""
from __future__ import annotations

import math


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _side_key(name_box, number_box):
    nx, ny = _center(name_box)
    mx, my = _center(number_box)
    if abs(nx - mx) >= abs(ny - my):
        return ("v", "left" if nx > mx else "right")
    return ("h", "top" if ny > my else "bottom")


def _missing_leading_digit(digits, names):
    """``True`` when the side lost its first digit (see module docstring)."""
    if len(digits) < 3 or len(names) != len(digits) + 1:
        return False
    digits = sorted(digits)
    names = sorted(names)
    rows = [row for row, _slot, _text in digits]
    if any(b - a < 3 for a, b in zip(rows, rows[1:])):
        return False
    if any(not text.isdigit() for _row, _slot, text in digits):
        return False
    values = [int(text) for _row, _slot, text in digits]
    if any(b - a != 1 for a, b in zip(values, values[1:])):
        return False
    if (rows[0] < rows[-1]) != (values[0] < values[-1]):
        return False
    return names[0][0] < rows[0]


def rebind(assigned):
    """Rebind shifted names in place; return a small report dict."""
    changed = []
    for _cid, comp in (assigned.get("comps") or {}).items():
        pins = comp.get("pins") or {}
        designator = ((comp.get("designator") or {}).get("text") or "").strip()
        if not designator or not isinstance(pins, dict):
            continue
        anchored = {}
        for slot, pin in pins.items():
            number, name = pin.get("number"), pin.get("name")
            if (number and number.get("box") and number.get("text")
                    and name and name.get("box") and name.get("text")):
                anchored[slot] = _side_key(name["box"], number["box"])
        if not anchored:
            continue
        sides = {}
        for slot, key in anchored.items():
            number, name = pins[slot]["number"], pins[slot]["name"]
            tangent = 1 if key[0] == "v" else 0
            entry = sides.setdefault(key, {"digit": [], "name": []})
            entry["digit"].append((_center(number["box"])[tangent], slot, str(number["text"])))
            entry["name"].append((_center(name["box"])[tangent], slot))
        # A lane whose partner box was never cropped still belongs to the side
        # of the anchored slot it sits next to.
        for slot, pin in pins.items():
            if slot in anchored:
                continue
            number, name = pin.get("number"), pin.get("name")
            if number and number.get("box") and number.get("text"):
                kind, box = "digit", number["box"]
            elif name and name.get("box") and name.get("text"):
                kind, box = "name", name["box"]
            else:
                continue
            best = None
            for other, key in anchored.items():
                ref = pins[other].get(kind)
                if not ref or not ref.get("box"):
                    continue
                distance = math.dist(_center(box), _center(ref["box"]))
                if best is None or distance < best[0]:
                    best = (distance, key)
            if best is None:
                continue
            key = best[1]
            tangent = 1 if key[0] == "v" else 0
            entry = sides.setdefault(key, {"digit": [], "name": []})
            if kind == "digit":
                entry["digit"].append((_center(box)[tangent], slot, str(number["text"])))
            else:
                entry["name"].append((_center(box)[tangent], slot))
        for key, entry in sides.items():
            if not _missing_leading_digit(entry["digit"], entry["name"]):
                continue
            digits = sorted(entry["digit"])
            names = sorted(entry["name"])
            for offset, target in enumerate(digits):
                source = names[offset + 1]
                if source[1] == target[1]:
                    continue
                name = pins[source[1]].get("name")
                if not name or not name.get("text"):
                    continue
                old = (pins[target[1]].get("name") or {}).get("text")
                if old == name["text"]:
                    continue
                pins[target[1]]["name"] = dict(name)
                changed.append({"component": designator, "number": target[2],
                                "old": old, "new": name["text"]})
    return {"changed": len(changed), "details": changed}
