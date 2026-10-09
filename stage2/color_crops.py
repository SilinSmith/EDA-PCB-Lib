"""Re-read external pin digits without bringing internal markers into crops."""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

COLOR_SOURCES = frozenset(("red_green_transition", "red_stub_endpoint",
                           "red_green_transition_uncertain"))


def _clip_external(box, edge, anchor, shape):
    x0,y0,x1,y1 = map(float,box)
    if edge == "left": x1 = min(x1,anchor[0])
    elif edge == "right": x0 = max(x0,anchor[0])
    elif edge == "top": y1 = min(y1,anchor[1])
    elif edge == "bottom": y0 = max(y0,anchor[1])
    h,w = shape[:2]
    x0,y0,x1,y1 = max(0,x0),max(0,y0),min(w,x1),min(h,y1)
    return [x0,y0,x1,y1] if x1-x0>=2 and y1-y0>=3 else None


def color_number_boxes(doc, reads, image):
    """Return external, ink-tight windows; existing OCR boxes supply glyph rows.

    No GT or expected numbers participate. Every window is constrained by an
    observed color pin and an actual frame side. Only low-chroma dark pixels
    are glyph evidence; blue labels and colored symbol/wire strokes are removed.
    """
    if image is None or image.ndim != 3:
        return [], None
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    chroma = image.max(2).astype(np.int16)-image.min(2).astype(np.int16)
    black = (chroma<50) & (gray<190)
    cleaned = np.where(black,gray,255).astype(np.uint8)
    boxes,seen = [],set()
    for comp in doc.get("components",[]):
        if comp.get("type") != "box": continue
        cid = comp["id"]
        pins = doc.get("pins",{}).get(cid,[])
        evidence = doc.get("pin_evidence",{}).get(cid,[])
        for index,(point,ev) in enumerate(zip(pins,evidence)):
            if not isinstance(ev,dict): continue
            observed=ev.get("point_img")
            anchor=ev.get("anchor_img")
            if (not isinstance(ev.get("source"),str)
                    or not isinstance(observed,(list,tuple)) or len(observed)!=2
                    or not isinstance(anchor,(list,tuple)) or len(anchor)!=2): continue
            try:
                valid=all(math.isfinite(float(v)) for v in list(point)+list(observed)+list(anchor))
            except (TypeError,ValueError,OverflowError):
                valid=False
            if (not valid or ev.get("source") not in COLOR_SOURCES or ev.get("trust")!="trusted"
                    or ev.get("frame_confirmed") is not True
                    or math.dist(point,observed)>.001):
                continue
            edge,anchor = ev.get("edge"),ev.get("anchor_img")
            if edge not in ("left","right","top","bottom") or not anchor: continue
            for rec in reads:
                if rec.get("number_only"): continue
                # Retain boxes with a numeric OCR alternative, including 0
                # failures; a new recognition is required before any output.
                if not any(isinstance(v,(list,tuple)) and len(v)>=1 and re.fullmatch(r"\d{1,3}",str(v[0]).strip())
                           for v in rec.get("reads",{}).values()):
                    continue
                raw = rec.get("bbox")
                if not raw or len(raw)!=4: continue
                x0,y0,x1,y1 = map(float,raw)
                glyph_height=max(1,y1-y0)
                if edge in ("left","right"):
                    # Horizontal pin digits sit just above their wire.
                    if not point[1]-max(22,glyph_height*1.5) <= y1 <= point[1]+3: continue
                    if abs((x0+x1)/2-anchor[0])>max(32,abs(point[0]-anchor[0])+12): continue
                    if edge=="left" and x0>=anchor[0]: continue
                    if edge=="right" and x1<=anchor[0]: continue
                    y1=min(y1+2,point[1]-1)
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
                ys,xs=np.nonzero(black[ya:yb,xa:xb])
                if len(xs)<3: continue
                tight=[xa+int(xs.min())-2,ya+int(ys.min())-2,
                       xa+int(xs.max())+3,ya+int(ys.max())+3]
                tight=_clip_external(tight,edge,anchor,image.shape)
                if tight is None: continue
                key=tuple(round(v,1) for v in tight)
                if key in seen: continue
                seen.add(key)
                boxes.append({"bbox":list(key),"component_id":cid,"pin_index":index,
                              "edge":edge,"anchor_img":list(anchor),"original_bbox":list(raw)})
    return boxes,cleaned


def read_color_numbers(doc, reads, engine, bgr):
    """Append role-limited candidates, never rewrite or synthesize a number.

    Call the existing recognizer on exact clipped windows: TextEngine.read's
    automatic padding would cross the confirmed frame and reintroduce circles.
    """
    boxes,gray=color_number_boxes(doc,reads,bgr)
    if not boxes: return []
    recognizer=engine.recognizer()
    if recognizer is None: return []
    values=recognizer.read_many(gray,[(r["bbox"],"pin_number") for r in boxes])
    return [{"bbox":r["bbox"],"reads":{"pin_number":(text,float(conf))},
             "number_only":True,"source":"color_external_number_crop",
             "color_crop":{k:v for k,v in r.items() if k!="bbox"}}
            for r,(text,conf) in zip(boxes,values)]
