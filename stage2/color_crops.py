"""Re-read external pin digits without bringing internal markers into crops."""

# ─────────────────────────────────────────────────────────────────────────────
# 【接口】统一约定（队友分支 13 个模块，2026-10-16 统一；只改签名不改识别逻辑）
#   读数器   read_xxx(doc, reads, engine, bgr, *, min_conf=0.0, **opts) -> List[dict]
#   裁片器   xxx_boxes(doc, image, reads=None, **opts)                 -> List[dict]
#   返回     纯追加：只返回**新增**候选（{"bbox":..., "reads":{role:(text,conf)}}），
#            绝不修改入参、绝不删除既有读数；关掉开关时下游逐位不变
#   开关     一律由 `stage2/run.py` 的 `PIN_READERS` 注册表读取，模块自身不读 env
#   挂载     workflow 节点 ⑦c/⑦d/⑦e/⑦f/⑦g（`tools/flow/graph.json`）
#   来源     队友分支 `origin/codex/improve-pin-network-parsing`
#   本模块入口：
#     color_number_boxes(doc, image, reads)
#     read_color_numbers(doc, reads, engine, bgr, *, min_conf)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import re

import cv2
import numpy as np

COLOR_SOURCES = frozenset(("red_green_transition", "red_stub_endpoint",
                           "red_green_transition_uncertain"))


def _finite_vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(v) for v in result) else None


def _rectangle(value):
    result = _finite_vector(value, 4)
    return result if result and result[2] > result[0] and result[3] > result[1] else None


def _ink_stats(box, black, colored):
    """Measure a source window, not the recognizer's padded sampling area."""
    height, width = black.shape
    x0, y0, x1, y1 = box
    xa, ya = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    xb, yb = min(width, int(math.ceil(x1))), min(height, int(math.ceil(y1)))
    nblack = int(np.count_nonzero(black[ya:yb, xa:xb]))
    ncolor = int(np.count_nonzero(colored[ya:yb, xa:xb]))
    return {"black_pixels": nblack, "colored_pixels": ncolor,
            "color_contamination_ratio": round(ncolor / max(1, ncolor + nblack), 6)}


def _digit_row_owner(glyph, edge, pins, evidence):
    """A re-crop belongs to the nearest observed row, never an adjacent one.

    Large legacy OCR boxes can be within the broad search window of two pins.
    Resolve their owner after ink extraction, before assigning an entity ID.
    This uses observed contacts only and does not inspect OCR digit values.
    """
    peers = []
    for index, (raw_point, ev) in enumerate(zip(pins, evidence)):
        if not isinstance(ev, dict) or ev.get("edge") != edge:
            continue
        point = _finite_vector(raw_point, 2)
        observed = _finite_vector(ev.get("point_img"), 2)
        if (point is None or observed is None
                or not isinstance(ev.get("source"), str)
                or ev.get("source") not in COLOR_SOURCES
                or ev.get("trust") != "trusted" or ev.get("frame_confirmed") is not True
                or math.dist(point, observed) > .001):
            continue
        distance = (abs(point[1]-glyph[3]) if edge in ("left", "right")
                    else max(glyph[0]-point[0], point[0]-glyph[2], 0))
        peers.append((distance,index))
    if not peers:
        return None
    distance, owner = min(peers)
    tolerance = max(4, .75*(glyph[3]-glyph[1]))
    return owner if distance <= tolerance else None


def _remove_numeric_specks(ink):
    """Remove only isolated 2x2 specks beside a substantial digit component.

    Keep every substantial component, including a narrow 1 and all digits of
    a multi-digit number. A tiny low-resolution sample without a real body is
    left unchanged. These rules inspect pixels, never the recognized number.
    """
    mask = np.asarray(ink, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    substantial = any(int(s[4]) >= 8 and max(int(s[2]), int(s[3])) >= 5
                      for s in stats[1:])
    removed = np.zeros(mask.shape, dtype=bool)
    if substantial:
        for index in range(1, count):
            width, height, area = (int(stats[index, k]) for k in (2, 3, 4))
            if width <= 2 and height <= 2 and area <= 4:
                removed |= labels == index
    return mask.astype(bool) & ~removed, removed


def _clip_external(box, edge, anchor, shape):
    x0,y0,x1,y1 = map(float,box)
    if edge == "left": x1 = min(x1,anchor[0])
    elif edge == "right": x0 = max(x0,anchor[0])
    elif edge == "top": y1 = min(y1,anchor[1])
    elif edge == "bottom": y0 = max(y0,anchor[1])
    h,w = shape[:2]
    x0,y0,x1,y1 = max(0,x0),max(0,y0),min(w,x1),min(h,y1)
    return [x0,y0,x1,y1] if x1-x0>=2 and y1-y0>=3 else None


def color_number_boxes(doc, image, reads=None):
    """Return external, ink-tight windows; existing OCR boxes supply glyph rows.

    No GT or expected numbers participate. Every window is constrained by an
    observed color pin and an actual frame side. Only low-chroma dark pixels
    are glyph evidence; blue labels and colored symbol/wire strokes are removed.
    """
    reads = reads or []  # 契约参数：old/new 调用点都用位置参数传 (doc, image, reads)
    if image is None or image.ndim != 3:
        return [], None
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    chroma = image.max(2).astype(np.int16)-image.min(2).astype(np.int16)
    # JPEG fringes of colored NC strokes can have chroma below 50 but remain
    # medium/light tinted ink. Remove such fringes only within two pixels of a
    # definite colored stroke; preserve digit cores and distant antialiasing.
    strong_colored = (chroma>=50) & (gray<230)
    near_color = cv2.dilate(strong_colored.astype(np.uint8),np.ones((5,5),np.uint8)) > 0
    color_fringe = near_color & (chroma>25) & (gray>=100) & (gray<230)
    black = (chroma<50) & (gray<190) & ~color_fringe
    colored = strong_colored | color_fringe
    cleaned = np.where(black,gray,255).astype(np.uint8)
    boxes,seen = [],{}
    for comp in doc.get("components",[]):
        if not isinstance(comp, dict) or comp.get("type") != "box": continue
        cid = comp.get("id")
        if not isinstance(cid, str): continue
        pins = doc.get("pins",{}).get(cid,[])
        evidence = doc.get("pin_evidence",{}).get(cid,[])
        for index,(point,ev) in enumerate(zip(pins,evidence)):
            if not isinstance(ev,dict): continue
            point = _finite_vector(point, 2)
            observed = _finite_vector(ev.get("point_img"), 2)
            anchor = _finite_vector(ev.get("anchor_img"), 2)
            if (point is None or observed is None or anchor is None
                    or not isinstance(ev.get("source"), str)
                    or ev.get("source") not in COLOR_SOURCES or ev.get("trust")!="trusted"
                    or ev.get("frame_confirmed") is not True
                    or math.dist(point,observed)>.001):
                continue
            edge = ev.get("edge")
            if edge not in ("left","right","top","bottom"): continue
            for source_index, rec in enumerate(reads):
                if not isinstance(rec, dict): continue
                if rec.get("number_only"): continue
                # Retain boxes with a numeric OCR alternative, including 0
                # failures; a new recognition is required before any output.
                alternatives = rec.get("reads", {})
                if not isinstance(alternatives, dict): continue
                if not any(isinstance(v,(list,tuple)) and len(v)>=1 and re.fullmatch(r"\d{1,3}",str(v[0]).strip())
                           for v in alternatives.values()):
                    continue
                raw = _rectangle(rec.get("bbox"))
                if raw is None: continue
                x0,y0,x1,y1 = raw
                glyph_height=max(1,y1-y0)
                if edge in ("left","right"):
                    # Horizontal pin digits sit just above their wire.
                    if not point[1]-max(22,glyph_height*1.5) <= y1 <= point[1]+3: continue
                    if abs((x0+x1)/2-anchor[0])>max(32,abs(point[0]-anchor[0])+12): continue
                    if edge=="left" and x0>=anchor[0]: continue
                    if edge=="right" and x1<=anchor[0]: continue
                    # Colored wires are already removed. Cutting at row-1
                    # truncated the last black digit row and made another pin
                    # produce a more complete version of the same glyph.
                    y1=y1+2
                    expanded=[x0-2,y0-2,x1+2,y1]
                else:
                    if min(abs(x0-point[0]),abs(x1-point[0]),abs((x0+x1)/2-point[0]))>max(8,glyph_height): continue
                    if abs((y0+y1)/2-anchor[1])>max(32,abs(point[1]-anchor[1])+12): continue
                    if edge=="top" and y0>=anchor[1]: continue
                    if edge=="bottom" and y1<=anchor[1]: continue
                    expanded=[x0-2,y0-2,x1+2,y1+2]
                clipped=_clip_external(expanded,edge,anchor,image.shape)
                if clipped is None: continue
                xa,ya = int(math.ceil(clipped[0])),int(math.ceil(clipped[1]))
                xb,yb = int(math.floor(clipped[2])),int(math.floor(clipped[3]))
                local_ink, removed_noise = _remove_numeric_specks(black[ya:yb,xa:xb])
                ys,xs=np.nonzero(local_ink)
                if len(xs)<3: continue
                noise_ys, noise_xs = np.nonzero(removed_noise)
                noise_points = [[xa+int(x), ya+int(y)] for y,x in zip(noise_ys,noise_xs)]
                # Half-open ink extent is the matching geometry. Padding is
                # only for OCR and must never make an 8 beat/lose to a 13.
                glyph = [xa+int(xs.min()), ya+int(ys.min()),
                         xa+int(xs.max())+1, ya+int(ys.max())+1]
                if _digit_row_owner(glyph, edge, pins, evidence) != index:
                    continue
                tight=[glyph[0]-2,glyph[1]-2,glyph[2]+2,glyph[3]+2]
                tight=_clip_external(tight,edge,anchor,image.shape)
                if tight is None: continue
                source_stats = {"bbox": raw, "read_index": source_index,
                                **_ink_stats(raw, black, colored)}
                key=(cid, index, *glyph)
                if key in seen:
                    prior = boxes[seen[key]]
                    if raw not in prior["original_bboxes"]:
                        prior["original_bboxes"].append(raw)
                        prior["source_boxes"].append(source_stats)
                    continue
                seen[key] = len(boxes)
                boxes.append({"bbox":[round(v,1) for v in tight],
                              "geometry_bbox":glyph,"glyph_bbox":glyph,
                              "component_id":cid,"pin_index":index,
                              "edge":edge,"anchor_img":list(anchor),
                              "entity_id":f"{cid}:{index}:" + ":".join(map(str,glyph)),
                              "original_bbox":raw,"original_bboxes":[raw],
                              "source_boxes":[source_stats],
                              "color_contamination_ratio":source_stats["color_contamination_ratio"],
                              "glyph_pixels":int(len(xs)),
                              "isolated_pixels_removed":int(len(noise_points)),
                              "noise_rule":"local_components_2x2_area4_with_substantial_body",
                              "_noise_pixels_img":noise_points,
                              "glyph_touches_search_boundary":bool(
                                  xs.min()==0 or ys.min()==0
                                  or xs.max()==xb-xa-1 or ys.max()==yb-ya-1),
                              "sampling_method":"low_chroma_dark_ink",
                              "frame_clipped":clipped != expanded})
    return boxes,cleaned


def read_color_numbers(doc, reads, engine, bgr, *, min_conf: float = 0.0):
    """Append role-limited candidates, never rewrite or synthesize a number.

    Call the existing recognizer on exact clipped windows: TextEngine.read's
    automatic padding would cross the confirmed frame and reintroduce circles.
    """
    boxes,gray=color_number_boxes(doc,bgr,reads)
    if not boxes: return []
    recognizer=engine.recognizer()
    if recognizer is None: return []
    values=recognizer.read_many(gray,[(r["bbox"],"pin_number") for r in boxes])
    # Each noisy sample gets its own image view. Clearing a tiny component in
    # one window must not erase a legitimate stroke in an overlapping window.
    # Keeping full image coordinates also preserves the recognizer's internal
    # padding and ink normalization around the exact original sampling area.
    for index, record in enumerate(boxes):
        noise_points = record.get("_noise_pixels_img", [])
        if not noise_points:
            continue
        local_gray = gray.copy()
        for x,y in noise_points:
            local_gray[y,x] = 255
        reread = recognizer.read_many(local_gray,[(record["bbox"],"pin_number")])
        if reread:
            values[index] = reread[0]
    return [{"bbox":r["bbox"],"geometry_bbox":r["geometry_bbox"],
             "glyph_bbox":r["glyph_bbox"],"reads":{"pin_number":(text,float(conf))},
             "number_only":True,"source":"color_external_number_crop",
             "color_crop":{k:v for k,v in r.items() if k!="bbox" and not k.startswith("_")}}
            for r,(text,conf) in zip(boxes,values)]
