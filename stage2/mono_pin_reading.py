"""Read trusted monochrome pin glyphs through two fixed white-margin views.

These are role-only additional observations. Missing optional OCR leaves the
existing pipeline intact. Agreement is never a dictionary or numbering repair.
"""

from __future__ import annotations


import math

import re


import cv2

import numpy as np


from .mono_pin_crops import _finite, mono_observations, mono_pin_boxes

from .pin_name_ocr import get_pin_name_ocr

from .pin_identifier import pin_identifier_key

from .intact_pin_numbers import intact_number_line_box

from .name_fringe import antialiased_name_candidate, FRINGE_PADDINGS


ACCEPTANCE_THRESHOLD = .90

PADDINGS = (2, 4)


def _text(value, role):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if role == "pin_number":
        return text if pin_identifier_key(text, allow_alphanumeric=True) is not None else None
    return text if text and any(ch.isalpha() for ch in text) else None


def _agreement(views, role, threshold, paddings=PADDINGS):
    if len(views) != len(paddings):
        return None
    values = []
    for padding, view in zip(paddings, views):
        text = _text(view.get("text"), role)
        try:
            confidence = float(view.get("confidence", 0))
        except (TypeError, ValueError, OverflowError):
            return None
        if (view.get("padding") != padding or text is None or not math.isfinite(confidence)
                or not threshold <= confidence <= 1):
            return None
        values.append((text, confidence))
    return (values[0][0], min(v[1] for v in values)) if values[0][0] == values[1][0] else None


def _small_vertical_number_retry(glyph, rotations, provider, initial, threshold):
    """Require both fixed scales and both margins, without overriding strong ink reads."""
    trials,qualified=[],[]
    for rotation in rotations:
        pixels=cv2.rotate(glyph,cv2.ROTATE_90_COUNTERCLOCKWISE) if rotation==-90 else glyph
        if not (3<=pixels.shape[0]<=12 and pixels.shape[1]>=2
                and max(pixels.shape)*3+2*max(PADDINGS)<=384):
            continue
        pairs=[]
        for scale in (2,3):
            resized=cv2.resize(pixels,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            pair=[dict(provider.read_glyph(np.pad(resized,p,constant_values=255)),
                       padding=p,rotation=rotation,source_scale=scale) for p in PADDINGS]
            trials.append(pair);pairs.append(pair)
        agreements=[_agreement(pair,'pin_number',threshold) for pair in pairs]
        if (all(agreements) and agreements[0][0]==agreements[1][0]
                and agreements[0][0].isascii() and agreements[0][0].isdecimal()):
            qualified.append((agreements[0][0],min(a[1] for a in agreements),pairs[0]))
    if not qualified or len({q[0] for q in qualified})!=1:
        return None,trials
    chosen=qualified[0]
    for pair in initial+trials:
        for view in pair:
            confidence=view.get('confidence',0)
            text=_text(view.get('text'),'pin_number')
            if (isinstance(confidence,(int,float)) and math.isfinite(confidence)
                    and threshold<=confidence<=1 and text and text!=chosen[0]):
                return None,trials
    return chosen,trials


def _overlap(first, second):
    return (max(0, min(first[2], second[2]) - max(first[0], second[0]))
            * max(0, min(first[3], second[3]) - max(first[1], second[1])))


def _incomplete_name_replacement(old_name, observation, candidate, bgr):
    """Hold a suffix/prefix crop that visibly drops a strong existing word.

    Existing OCR does not define or extend the new crop. This guard only
    rejects a replacement: both rectangles must describe the same observed
    word inside the current physical row cell, and actual adjacent source
    ink must remain outside the proposed smaller rectangle.
    """
    if not isinstance(old_name, dict) or not isinstance(candidate, dict):
        return None
    proposed = (observation.get("reads") or {}).get("pin_name")
    if not proposed:
        return None
    old_text, new_text = old_name.get("text"), proposed[0]
    if (not isinstance(old_text, str) or not isinstance(new_text, str)
            or len(new_text) >= len(old_text)
            or not (old_text.startswith(new_text) or old_text.endswith(new_text))):
        return None
    try:
        confidence = float(old_name.get("conf", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(confidence) or not ACCEPTANCE_THRESHOLD <= confidence <= 1:
        return None
    old = _finite(old_name.get("box"), 4)
    new = _finite(candidate.get("bbox"), 4)
    cleanup = candidate.get("cleanup_evidence") or {}
    region = _finite(cleanup.get("physical_search_region"), 4)
    if old is None or new is None or region is None:
        return None
    if (old[2] <= old[0] or old[3] <= old[1]
            or old[0] < region[0] - 1 or old[1] < region[1] - 1
            or old[2] > region[2] + 1 or old[3] > region[3] + 1
            or new[0] < old[0] - 1 or new[1] < old[1] - 1
            or new[2] > old[2] + 1 or new[3] > old[3] + 1):
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = bgr.max(2).astype(np.int16) - bgr.min(2).astype(np.int16)
    height, width = gray.shape
    gap = max(2, int(cleanup.get("word_gap_pixels", 3)) + 2)
    # Inspect only ink immediately adjacent to this glyph. A distant opposite
    # name inside an overwide old rectangle cannot trigger the guard.
    x0, y0 = max(0, int(math.floor(max(old[0], new[0]-gap)))), max(0, int(math.floor(max(old[1], new[1]-2))))
    x1, y1 = min(width, int(math.ceil(min(old[2], new[2]+gap)))), min(height, int(math.ceil(min(old[3], new[3]+2))))
    if x0 >= x1 or y0 >= y1:
        return None
    ink = (gray[y0:y1, x0:x1] < 190) & (chroma[y0:y1, x0:x1] < 30)
    xa, ya = max(x0, int(math.floor(new[0]))), max(y0, int(math.floor(new[1])))
    xb, yb = min(x1, int(math.ceil(new[2]))), min(y1, int(math.ceil(new[3])))
    ink[ya-y0:yb-y0, xa-x0:xb-x0] = False
    pixels = int(ink.sum())
    if pixels < 2:
        return None
    return {"reason": "strong_existing_word_has_adjacent_source_ink_outside_new_crop",
            "previous_word": old_text, "proposed_fragment": new_text,
            "previous_box": old, "proposed_box": new,
            "adjacent_excluded_source_pixels": pixels}


def _ambiguous_complete_name_replacement(old_name, observation, candidate):
    """Hold a one-character ambiguity between two reads of the same full word.

    Neither recognizer's confidence proves that a complete, strong old word
    is wrong. A new marker can change the pin index; the original word bounds
    still establish whether this is the same physical printed label.
    """
    if not isinstance(old_name,dict) or not isinstance(candidate,dict):return None
    proposed=(observation.get('reads') or {}).get('pin_name')
    if not proposed:return None
    old_text,new_text=old_name.get('text'),proposed[0]
    if not isinstance(old_text,str) or not isinstance(new_text,str) or len(old_text)!=len(new_text):return None
    differences=[(a,b) for a,b in zip(old_text,new_text) if a!=b]
    if len(differences)!=1 or set(differences[0]) not in ({'O','0'},{'I','1'},{'l','1'}):return None
    try:confidence=float(old_name.get('conf',0))
    except (TypeError,ValueError,OverflowError):return None
    if not math.isfinite(confidence) or not .90<=confidence<=1:return None
    old,new=_finite(old_name.get('box'),4),_finite(candidate.get('bbox'),4)
    region=_finite((candidate.get('cleanup_evidence') or {}).get('physical_search_region'),4)
    if old is None or new is None or region is None:return None
    if any(old[i]<region[i]-1 for i in (0,1)) or any(old[i]>region[i]+1 for i in (2,3)):return None
    if max(abs(old[i]-new[i]) for i in range(4))>2:return None
    area=(new[2]-new[0])*(new[3]-new[1])
    intersection=max(0,min(old[2],new[2])-max(old[0],new[0]))*max(0,min(old[3],new[3])-max(old[1],new[1]))
    if area<=0 or intersection/area<.85:return None
    return dict(reason='strong_complete_same_word_has_unresolved_character_ambiguity',
                previous_word=old_text,proposed_word=new_text,previous_box=old,proposed_box=new)


def _old_number_geometry_reasons(doc, number, candidate, target, bgr):
    """Audit the OLD winning numeral, never infer dirt from PP's failure.

    A held or missing new digit is not evidence that the legacy digit is bad.
    This function needs observed mixed geometry or actual ink continuing
    across the old box within the same new physical word cell.
    """
    old = _finite(number.get("box"), 4) if isinstance(number, dict) else None
    if old is None or old[2] <= old[0] or old[3] <= old[1]:
        return []
    cid, edge, anchor = target["component_id"], target["edge"], target["anchor_img"]
    reasons = []
    if old[0] < anchor[0] < old[2]:
        reasons.append("winning_number_box_crosses_actual_frame")
    peers = [ob for ob in mono_observations(doc)
             if ob["component_id"] == cid and ob["edge"] == edge]
    covered = [ob["pin_index"] for ob in peers if old[1] <= ob["anchor_img"][1] < old[3]]
    if len(covered) > 1:
        reasons.append("winning_number_box_contains_multiple_physical_rows")
    if candidate is None:
        return reasons
    glyph = _finite(candidate.get("bbox"), 4)
    region = _finite(candidate.get("region"), 4)
    if glyph is None or region is None:
        return reasons
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = bgr.max(2).astype(np.int16) - bgr.min(2).astype(np.int16)
    ink = (gray < 190) & (chroma < 30)
    height, width = ink.shape
    if doc.get("pin_text_view_rotation") == 90:
        cell = _finite((candidate.get("cleanup_evidence") or {}).get("physical_search_region"), 4)
        if cell is not None:
            xa, ya = max(0, int(math.floor(old[0]))), max(0, int(math.floor(old[1])))
            xb, yb = min(width, int(math.ceil(old[2]))), min(height, int(math.ceil(old[3])))
            if xa < xb and ya < yb:
                outside = ink[ya:yb, xa:xb].copy()
                ca, cb = max(xa, int(math.floor(cell[0]))-1), max(ya, int(math.floor(cell[1]))-1)
                cc, cd = min(xb, int(math.ceil(cell[2]))+1), min(yb, int(math.ceil(cell[3]))+1)
                if ca < cc and cb < cd:
                    outside[cb-ya:cd-ya, ca-xa:cc-xa] = False
                if int(outside.sum()) >= 2:
                    reasons.append("winning_number_has_source_ink_outside_current_physical_cell")
    # A real horizontal stroke must connect this winning box to the measured
    # frame probe. Merely having a baseline close to the pin is not enough.
    stroke = (candidate.get("cleanup_evidence") or {}).get("stroke_probe")
    if isinstance(stroke, dict):
        top, bottom = stroke.get("stroke_top"), stroke.get("stroke_bottom")
        if isinstance(top, int) and isinstance(bottom, int):
            xa = int(math.ceil(old[2])) - 1 if edge == "left" else int(math.floor(anchor[0])) + 2
            xb = int(math.ceil(anchor[0])) - 2 if edge == "left" else int(math.floor(old[0])) + 1
            lo, hi = max(0, min(xa, xb)), min(width, max(xa, xb) + 1)
            if hi - lo >= 3:
                for row in range(max(top, 0, int(math.ceil(old[1]))),
                                 min(bottom, height, int(math.floor(old[3])))):
                    if np.mean(ink[row, lo:hi]) >= .90:
                        reasons.append("attached_pin_stroke_enters_winning_number_box")
                        break
    if _overlap(old, glyph) <= 0:
        return reasons
    # A crop can lose a leading disconnected 1 without crossing an ink
    # component. That alone does not prove a bad old read when PP is held.
    # Require an actual 8-connected stroke cut by an OLD rectangle edge,
    # bounded by the same new physical word cell and observed word bbox.
    x0, y0, x1, y1 = [int(math.ceil(old[0])), int(math.ceil(old[1])),
                       int(math.floor(old[2])), int(math.floor(old[3]))]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return reasons
    support = [max(region[0], glyph[0]), max(region[1], glyph[1]),
               min(region[2], glyph[2]), min(region[3], glyph[3])]
    cut = False
    for border, outside in ((x0, x0 - 1), (x1 - 1, x1)):
        if not support[0] <= outside < support[2]:
            continue
        ya, yb = max(y0, int(math.ceil(support[1]))), min(y1, int(math.floor(support[3])))
        active = np.flatnonzero(ink[ya:yb, border]) + ya
        for delta in (-1, 0, 1):
            neighboring = active + delta
            neighboring = neighboring[(neighboring >= support[1]) & (neighboring < support[3])]
            cut = cut or bool(len(neighboring) and np.any(ink[neighboring, outside]))
    for border, outside in ((y0, y0 - 1), (y1 - 1, y1)):
        if not support[1] <= outside < support[3]:
            continue
        xa, xb = max(x0, int(math.ceil(support[0]))), min(x1, int(math.floor(support[2])))
        active = np.flatnonzero(ink[border, xa:xb]) + xa
        for delta in (-1, 0, 1):
            neighboring = active + delta
            neighboring = neighboring[(neighboring >= support[0]) & (neighboring < support[2])]
            cut = cut or bool(len(neighboring) and np.any(ink[outside, neighboring]))
    if cut:
        reasons.append("winning_number_box_cuts_observed_connected_glyph")
    return reasons


def _guard_recovered_names(doc, reads, output, candidates, bgr, min_conf=0.0):
    names = [rec for rec in output if rec.get("name_only")]
    if not names:
        return
    # Lazy import avoids introducing an import cycle into the reader modules.
    from .assign import assign
    from .pin_lane_reading import _threshold_reads
    eligible_reads = _threshold_reads(reads, min_conf)
    existing = assign(doc.get("components", []), doc.get("pins", {}), eligible_reads,
                      pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
    accepted_numbers = [r for r in output if r.get("number_only") and r.get("reads")
                        and r["mono_pin_ocr"].get("accepted") is True]
    directional = doc.get("pin_text_view_rotation") == 90
    rebound = (assign(doc.get("components", []), doc.get("pins", {}), eligible_reads + output,
                      pincol=doc.get("pincol"), pin_evidence=doc.get("pin_evidence"))
               if accepted_numbers or directional else {})
    confirmed = set()
    for observed in accepted_numbers:
        target = observed["pin_target"]
        key = (target["component_id"], target["pin_index"])
        actual = rebound.get("comps", {}).get(key[0], {}).get("pins", {}).get(key[1], {})
        number = actual.get("number") or {}
        text = (observed.get("reads") or {}).get("pin_number")
        # OCR acceptance alone is insufficient: assignment can still reject
        # its crop's frame/row geometry. Only the independently-read ACTUAL
        # winner for this same physical pin can release the partner guard.
        if (actual.get("number_independently_read") is True and text
                and number.get("text") == text[0]
                and _finite(number.get("box"), 4) == _finite(observed.get("bbox"), 4)
                and _finite(observed.get("bbox"), 4) is not None):
            confirmed.add(key)
    digit_boxes = {(c["pin_target"]["component_id"], c["pin_target"]["pin_index"]): c
                   for c in candidates if c["role"] == "pin_number"}
    name_boxes = {(c["pin_target"]["component_id"], c["pin_target"]["pin_index"]): c
                  for c in candidates if c["role"] == "pin_name"}
    for rec in names:
        target, meta = rec["pin_target"], rec["mono_pin_ocr"]
        key = (target["component_id"], target["pin_index"])
        old = existing.get("comps", {}).get(key[0], {}).get("pins", {}).get(key[1], {})
        meta["previous_name"], meta["previous_number"] = old.get("name"), old.get("number")
        # Multiple independent name readers must retain the original ranking
        # observation until this row's number is independently replaced.
        meta["previous_number_selection_name"] = old.get("number_selection_name", old.get("name"))
        same_word_names=[p.get('name') for p in existing.get('comps',{}).get(key[0],{}).get('pins',{}).values()
                         if isinstance(p,dict) and p.get('name')]
        ambiguity=[guard for name in same_word_names
                   if (guard:=_ambiguous_complete_name_replacement(name,rec,name_boxes.get(key)))]
        if ambiguity and len({g['previous_word'] for g in ambiguity})==1:
            meta.update(accepted=False,held_reason='unresolved_complete_word_character_ambiguity',
                        name_replacement_guard=ambiguity[0])
            rec['reads']={}
            continue
        completeness = _incomplete_name_replacement(old.get("name"), rec, name_boxes.get(key), bgr)
        if completeness:
            meta.update(accepted=False, held_reason="incomplete_name_replacement",
                        name_replacement_guard=completeness)
            rec["reads"] = {}
            continue
        actual = rebound.get("comps", {}).get(key[0], {}).get("pins", {}).get(key[1], {})
        partner = actual.get("number") if directional else old.get("number")
        if (not rec.get("reads") or not partner
                or (old.get("name") and not directional)):
            continue
        reasons = [] if key in confirmed else _old_number_geometry_reasons(
            doc, partner, digit_boxes.get(key), target, bgr)
        meta["partner_guard"] = {"new_number_confirmed": key in confirmed,
                                  "winning_old_number_geometry_reasons": reasons,
                                  "winning_number_after_rebinding": partner}
        if reasons:
            meta.update(accepted=False, held_reason="unresolved_dirty_partner_number_geometry")
            rec["reads"] = {}


def _read_horizontal_mono_pin_texts(doc, reads, engine, bgr, min_conf=0.0, mode="auto",
                        include_names=True, include_numbers=True):
    """Return role-limited targeted records, including held audit observations.

    ``engine`` is accepted for the pipeline interface. PP sees actual complete
    source pixels directly, with pads 2 and 4, without intermediate gray32
    resampling or cross-crop atlases. Input objects are never modified.
    """
    if mode not in ("off", "auto", "on"):
        raise ValueError("mono pin OCR mode must be off, auto or on")
    min_conf = float(min_conf)
    if not math.isfinite(min_conf):
        raise ValueError("min_conf must be finite")
    if mode == "off":
        return []
    if not include_names and not include_numbers:
        return []
    all_candidates, gray = mono_pin_boxes(doc, bgr, reads)
    if doc.get("pin_text_view_rotation") == 90:
        from .vertical_pin_crops import complete_vertical_pin_boxes
        all_candidates = complete_vertical_pin_boxes(doc, bgr, all_candidates)
    candidates = [c for c in all_candidates if
                  (include_names if c["role"] == "pin_name" else include_numbers)]
    if not candidates:
        return []
    provider = get_pin_name_ocr(mode)
    if provider is None:
        return []
    threshold, output = max(ACCEPTANCE_THRESHOLD, min_conf), []
    for candidate in candidates:
        candidate = intact_number_line_box(candidate, gray) or candidate
        role = candidate["role"]
        x0, y0, x1, y1 = candidate["bbox"]
        glyph = gray[y0:y1, x0:x1].copy()
        for x, y in (candidate.get("cleanup_evidence") or {}).get("known_line_exclusions", []):
            if x0 <= x < x1 and y0 <= y < y1:
                glyph[y - y0, x - x0] = 255
        views, held, alternatives = [], None, []
        if "glyph_integrity_unresolved" in candidate["flags"]:
            held = "glyph_integrity_unresolved"
        elif max(glyph.shape) + 2 * max(PADDINGS) > 384:
            held = "glyph_exceeds_verified_recognizer_size"
        else:
            rotations = candidate.get("recognition_rotations", [0])
            alternatives = []
            for rotation in rotations:
                pixels = (cv2.rotate(glyph, cv2.ROTATE_90_COUNTERCLOCKWISE)
                          if rotation == -90 else glyph)
                trial = [dict(provider.read_glyph(np.pad(pixels, p, constant_values=255)),
                              padding=p, rotation=rotation) for p in PADDINGS]
                alternatives.append(trial)
            qualified = [trial for trial in alternatives if _agreement(trial, role, threshold)]
            texts = {_agreement(trial, role, threshold)[0] for trial in qualified}
            if len(texts) > 1:
                held = "independent_glyph_orientations_disagree"
            views = qualified[0] if len(texts) == 1 else alternatives[0]
        agreed = _agreement(views, role, threshold)
        initial_views = list(views)
        retry_views = []
        scaled_retry = False
        paddings = PADDINGS
        if (held is None and agreed is None and role=='pin_number'
                and doc.get('pin_text_view_rotation')==90
                and not candidate.get('recognition_view')):
            retry,scaled_trials=_small_vertical_number_retry(
                glyph,candidate.get('recognition_rotations',[0]),provider,alternatives,threshold)
            alternatives.extend(scaled_trials)
            if retry is not None:
                agreed=(retry[0],retry[1]);views=retry[2];scaled_retry=True
        if held is None and agreed is None and role == 'pin_name':
            fringe = antialiased_name_candidate(candidate, gray, bgr)
            if fringe is not None:
                xa, ya, xb, yb = fringe['bbox']
                pixels = gray[ya:yb, xa:xb].copy()
                if max(pixels.shape)+2*max(PADDINGS) <= 384:
                    # This is a geometrically different complete source crop,
                    # not a search through arbitrary OCR thresholds or pads.
                    retry_views = [dict(provider.read_glyph(np.pad(pixels, p, constant_values=255)),
                                        padding=p, rotation=0) for p in PADDINGS]
                    retry = _agreement(retry_views, role, threshold)
                    if retry is None:
                        narrow = [dict(provider.read_glyph(np.pad(pixels, 0, constant_values=255)),
                                       padding=0, rotation=0), retry_views[0]]
                        narrow_read = _agreement(narrow, role, threshold, FRINGE_PADDINGS)
                        # A strong contradictory read of this SAME complete
                        # crop stays held even if another margin pair agrees.
                        conflict = bool(narrow_read and any(
                            float(v.get('confidence', 0)) >= threshold
                            and _text(v.get('text'), role) != narrow_read[0] for v in retry_views))
                        if narrow_read and not conflict:
                            retry, retry_views, paddings = narrow_read, narrow, FRINGE_PADDINGS
                    if retry is not None:
                        agreed, candidate, views = retry, fringe, retry_views
        if (agreed and role == 'pin_number' and candidate.get("recognition_view") and
                not (agreed[0].isascii() and agreed[0].isdecimal())):
            held = "attached_line_view_requires_decimal_identifier"
        if held is None and agreed is None:
            held = "white_margin_views_disagree_or_below_threshold"
        text, confidence = agreed if agreed else (None, 0.0)
        proposed = {role: (text, confidence)} if agreed else {}
        metadata = {"role": role, "backend": "ppocr_v4_direct", "accepted": held is None,
                    "held_reason": held, "acceptance_threshold": threshold,
                    "rotation": 0, "orientation_source": "horizontal_physical_pin_row",
                     "view": candidate.get("recognition_view", "original_complete_glyph"), "region": list(candidate["region"]),
                    "flags": list(candidate["flags"]),
                    "cleanup_evidence": candidate["cleanup_evidence"],
                    "selected_views": views, "raw_reads": views,
                    "paddings": list(paddings),
                    "proposed_read": proposed,
                    "confidence_definition": "minimum of two agreeing PP character-probability means"}
        if candidate.get("recognition_rotations") or scaled_retry:
            metadata["orientation_trials"] = alternatives
        if scaled_retry:
            metadata['small_number_scale_consensus']={'scales':[2,3],'paddings':list(PADDINGS),
                'interpolation':'cubic','minimum_confidence_all_four':agreed[1],
                'strong_original_conflicts_rejected':True,'source_geometry_unchanged':True}
            metadata['confidence_definition']='minimum of four agreeing fixed-scale and margin views'
        if views:
            metadata["sampling"] = views[0].get("sampling")
        if retry_views:
            metadata['initial_core_views'] = initial_views
            metadata['fringe_retry_views'] = retry_views
        output.append({"bbox": list(candidate["bbox"]), "geometry_bbox": list(candidate["bbox"]),
                       "glyph_bbox": list(candidate["bbox"]), "pin_target": dict(candidate["pin_target"]),
                       "reads": proposed if held is None else {}, "name_only": role == "pin_name",
                       "number_only": role == "pin_number", "source": "mono_" + role + "_ppocr",
                       "mono_pin_ocr": metadata})
    if doc.get('pin_text_view_rotation') == 90:
        _guard_vertical_number_collisions(doc, reads, output, min_conf=min_conf)
    _guard_recovered_names(doc, reads, output, all_candidates, bgr, min_conf=min_conf)
    return output


def _guard_vertical_number_collisions(doc, reads, output, min_conf=0.0):
    """A weaker new digit cannot displace an established complete PP digit.

    Both readings must come from this same recognizer, pass their independent
    geometry checks and use agreeing fixed margins. Confidence from a legacy
    recognizer is never compared with PP. A stronger new digit remains eligible.
    """
    from .assign import assign, _qualified_mono_views
    from .pin_lane_reading import _threshold_reads
    eligible = _threshold_reads(reads,min_conf)
    existing = assign(doc.get('components',[]),doc.get('pins',{}),eligible,
                      pincol=doc.get('pincol'),pin_evidence=doc.get('pin_evidence'))
    def same_provider(record):
        if _qualified_mono_views(record,'pin_number'):
            return True
        meta=record.get('pin_number_ocr') or {}
        agreed=_agreement(meta.get('selected_views') or [],'pin_number',ACCEPTANCE_THRESHOLD)
        value=(record.get('reads') or {}).get('pin_number')
        return (record.get('source')=='color_pin_number_ppocr'
                and meta.get('backend')=='ppocr_v4_direct' and meta.get('accepted') is True
                and not meta.get('held_reason') and 'glyph_integrity_unresolved' not in meta.get('flags',[])
                and set(record.get('reads') or {})=={'pin_number'}
                and agreed is not None and value and agreed[0]==value[0])
    for record in output:
        if not record.get('number_only') or not record.get('reads'):
            continue
        text,certainty = record['reads']['pin_number']
        target = record['pin_target']
        cid,index = target['component_id'],target['pin_index']
        for other,pin in existing.get('comps',{}).get(cid,{}).get('pins',{}).items():
            number = pin.get('number') or {}
            if (other==index or pin.get('number_independently_read') is not True
                    or number.get('text')!=text):
                continue
            qualified = [r for r in eligible if r.get('number_only')
                         and (r.get('pin_target') or {}).get('component_id')==cid
                         and (r.get('pin_target') or {}).get('pin_index')==other
                         and r.get('bbox')==number.get('box')
                         and same_provider(r)]
            if qualified and max(r['reads']['pin_number'][1] for r in qualified)>=certainty:
                record['mono_pin_ocr'].update(accepted=False,held_reason='stronger_existing_independent_number_collision',
                    number_collision_guard=dict(other_pin_index=other,number=text,
                        same_backend='ppocr_v4_direct',existing_confidence=number.get('conf'),
                        proposed_confidence=certainty,existing_box=number.get('box')))
                record['reads']={}
                break


def read_mono_pin_texts(doc, reads, engine, bgr, *, min_conf=0.0, mode="auto",
                        include_names=True, include_numbers=True):
    """Read original horizontal glyphs and a local clockwise vertical view."""
    options = dict(min_conf=min_conf, mode=mode, include_names=include_names,
                   include_numbers=include_numbers)
    horizontal = _read_horizontal_mono_pin_texts(doc, reads, engine, bgr, **options)
    if mode == "off" or not (include_names or include_numbers):
        return horizontal
    from .vertical_pin_reading import read_vertical_pin_texts
    return horizontal + read_vertical_pin_texts(
        doc, list(reads) + horizontal, engine, bgr, _read_horizontal_mono_pin_texts, **options)
