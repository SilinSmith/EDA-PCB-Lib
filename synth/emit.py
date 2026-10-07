"""Turn a synthetic Scene into the four artefacts the two routes need.

Route A (competition ground truth)  -> `<case>.json`
    exactly the shape the official dataset uses: {components, pins, nets}, bbox / point
    coordinates in the corpus' left-bottom origin (`y_json = H - y_image`).
Route A (component detector)        -> `labels_comp/<case>.txt`
    YOLO boxes in YOLO's own left-top normalised form, one class id per component type.
Route C (designator / text detector) -> `labels_text/<case>.txt` + `texts.jsonl`
    YOLO boxes for the six text roles, plus the string each box contains so a recogniser
    can be trained on the crops.  This is the route that matters: ComponentF1 keys on the
    designator string, and one wrong character costs the component, its pins and every net
    hanging off it.

Everything here reads coordinates in image space (left-top) and flips exactly once, at the
boundary.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .symbols import PREFIX, Comp, TextItem
from .style import PALETTES

#: Text roles the route-C detector has to separate.  Kept in one place because the class ids
#: written into the YOLO labels and the ids read back by the trainer must agree.
TEXT_ROLES = ["designator", "value", "pin_name", "pin_number", "net_label", "note"]
TEXT_ROLE_ID = {r: i for i, r in enumerate(TEXT_ROLES)}

#: Component types that carry their part number in `Name` instead of their designator.  The
#: corpus agrees: of 764 components where Name != key, 749 are `box`.
NAME_IS_PARTNO = frozenset({"box", "block"})

#: Types whose `value` is null in the corpus as a matter of course.
VALUE_NULL = frozenset({"box", "block"})


@dataclass
class Emitted:
    """One generated case, in every form the pipeline wants."""

    name: str
    scene: Any
    case: dict = field(default_factory=dict)
    comp_labels: list[str] = field(default_factory=list)
    text_labels: list[str] = field(default_factory=list)
    texts: list[dict] = field(default_factory=list)


def _suffixes(comp: Comp) -> list[str]:
    """Pin-key suffixes for one component, de-duplicated the way the corpus does it.

    A pin key is always `pin_<suffix>` and the net hyperGraph token is always `<key>.<suffix>`
    -- the same string in both places (checked on all 22691 pins).  When a part exposes the same
    pad number twice the corpus renames *both* to `<suffix>_1` / `<suffix>_2`, e.g. `U1.19_1`.
    """
    counts = Counter(p.number for p in comp.pins)
    seen: dict[str, int] = {}
    out = []
    for p in comp.pins:
        if counts[p.number] > 1:
            seen[p.number] = seen.get(p.number, 0) + 1
            out.append(f"{p.number}_{seen[p.number]}")
        else:
            out.append(p.number)
    return out


def case_json(scene) -> dict:
    """The competition-format ground truth for one scene."""
    h = float(scene.h)

    def pt(p) -> dict:
        return {"x": round(float(p[0]), 3), "y": round(h - float(p[1]), 3)}

    components: dict[str, dict] = {}
    pins: dict[str, dict] = {}
    suffix_of: dict[tuple[str, str], str] = {}

    for c in scene.comps:
        x1, y1, x2, y2 = c.bbox
        components[c.key] = {
            "Name": c.name or c.key,
            "type": c.ctype,
            "value": None if c.ctype in VALUE_NULL else c.value,
            "bbox": [round(float(x1), 3), round(h - float(y2), 3),
                     round(float(x2), 3), round(h - float(y1), 3)],
        }
        entry: dict[str, dict] = {}
        for p, suf in zip(c.pins, _suffixes(c)):
            entry[f"pin_{suf}"] = {"pinname": p.name or p.number, "point": pt(p.tip)}
            # first occurrence wins: a duplicate pad number can only be referenced once by the
            # hyperGraph, and the corpus' own `_1` form is what a reader sees there
            suffix_of.setdefault((c.key, p.number), suf)
        pins[c.key] = entry

    nets: dict[str, dict] = {}
    for n in scene.nets:
        tokens = []
        for ck, pn in n.pins:
            suf = suffix_of.get((ck, pn))
            if suf is None:
                continue
            tok = f"{ck}.{suf}"
            if tok not in tokens:
                tokens.append(tok)
        edges = {}
        for i, seg in enumerate(n.edges, 1):
            edges[f"edge_{i}"] = [pt(seg[0]), pt(seg[1])]
        nets[n.key] = {"hyperGraph": "(" + ",".join(tokens) + ")", "edges": edges}

    if not nets:
        raise ValueError("scene has no nets; the corpus never has a sheet without one")
    return {"components": components, "pins": pins, "nets": nets}


def _norm(bbox, w: float, h: float) -> str | None:
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw, bh = x2 - x1, y2 - y1
    if bw < 2.0 or bh < 2.0:
        return None
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    if not (0 < cx < 1 and 0 < cy < 1):
        return None
    return f"{cx:.6f} {cy:.6f} {min(bw / w, 1.0):.6f} {min(bh / h, 1.0):.6f}"


def yolo_comp_labels(scene, class_id: dict[str, int]) -> list[str]:
    """Component boxes in YOLO's own left-top normalised coordinates."""
    out = []
    for c in scene.comps:
        cid = class_id.get(c.ctype)
        if cid is None:
            continue
        box = _norm(c.bbox, scene.w, scene.h)
        if box:
            out.append(f"{cid} {box}")
    return out


def yolo_text_labels(scene) -> list[str]:
    """Text-role boxes for the designator route, same coordinates as YOLO expects."""
    out = []
    for t in scene.texts:
        if not t.content or t.role not in TEXT_ROLE_ID:
            continue
        box = _norm(t.bbox, scene.w, scene.h)
        if box:
            out.append(f"{TEXT_ROLE_ID[t.role]} {box}")
    return out


def text_records(scene) -> list[dict]:
    """The recogniser's supervision: what each text box actually says."""
    return [
        {"role": t.role, "content": t.content, "comp": t.comp,
         "bbox": [round(float(v), 2) for v in t.bbox]}
        for t in scene.texts if t.content and t.role in TEXT_ROLE_ID
    ]


def emit(scene, name: str, class_id: dict[str, int]) -> Emitted:
    return Emitted(
        name=name,
        scene=scene,
        case=case_json(scene),
        comp_labels=yolo_comp_labels(scene, class_id),
        text_labels=yolo_text_labels(scene),
        texts=text_records(scene),
    )


def write(em: Emitted, root: Path, save_image: bool = True) -> None:
    """Lay one case out on disk in the same skeleton the team's trainer already reads."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "gt").mkdir(parents=True, exist_ok=True)
    (root / "labels_comp").mkdir(parents=True, exist_ok=True)
    (root / "labels_text").mkdir(parents=True, exist_ok=True)
    if save_image:
        em.scene.image.save(root / "images" / f"{em.name}.png", optimize=True)
    (root / "gt" / f"{em.name}.json").write_text(
        json.dumps(em.case, ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "labels_comp" / f"{em.name}.txt").write_text(
        "\n".join(em.comp_labels) + "\n", encoding="utf-8")
    (root / "labels_text" / f"{em.name}.txt").write_text(
        "\n".join(em.text_labels) + "\n", encoding="utf-8")
    with (root / "texts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"case": em.name, "texts": em.texts}, ensure_ascii=False) + "\n")


def write_class_lists(root: Path, class_names: list[str]) -> None:
    """YOLO reads the class order from the label ids, so the name list must match it exactly."""
    (root / "classes_comp.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (root / "classes_text.txt").write_text("\n".join(TEXT_ROLES) + "\n", encoding="utf-8")
    (root / "data_comp.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images\nval: images\nnc: {len(class_names)}\n"
        "names:\n" + "\n".join(f"- {n}" for n in class_names) + "\n", encoding="utf-8")
    (root / "data_text.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images\nval: images\nnc: {len(TEXT_ROLES)}\n"
        "names:\n" + "\n".join(f"- {n}" for n in TEXT_ROLES) + "\n", encoding="utf-8")


def palettes() -> dict[str, Any]:
    return PALETTES


_ = (PREFIX, TextItem, Path)  # re-exported for callers that only import this module
