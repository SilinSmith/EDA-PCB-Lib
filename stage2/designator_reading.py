"""Additional observations of compact IC captions, without string repair.

Compact observed words above or below an IC are considered. Two fixed
white-margin views contain the unchanged original caption pixels. Optional
PP-OCR absence returns no observations and never downloads a dependency.
"""
from __future__ import annotations

import math
import re
import hashlib

import cv2
import numpy as np

from .fragmented_text import _box, recover_fragmented_boxes
from .pin_name_ocr import get_pin_name_ocr
from .segmented_caption import read_segmented_caption


PADDINGS = (4, 6)
THRESHOLD = .90
_FORM = re.compile(r"^([A-Za-z]{1,4})(?:[0-9]{1,4}[A-Za-z]?|[0-9]{0,4}_[A-Za-z0-9]{1,4})$")


def _observed_designator(value, allowed_prefixes):
    if not isinstance(value, str):
        return None
    value = value.strip()
    found = _FORM.fullmatch(value)
    if not found or found.group(1).upper() not in allowed_prefixes:
        return None
    # Preserve separators and suffix letters exactly. No O/0, B/8 or
    # punctuation repair is performed by this independent recognizer.
    return value


def _small_caption_views(glyph, box, fragments, provider, raw_reads, allowed):
    """Use font-relative margins only for a complete measured small word."""
    if not fragments or glyph.size==0:return None
    ink=glyph<190
    if (ink[0].any() or ink[-1].any() or ink[:,0].any() or ink[:,-1].any()):return None
    ys,xs=np.nonzero(ink)
    if len(xs)<6:return None
    height=int(ys.max()-ys.min()+1)
    if not 5<=height<=8:return None
    paddings=[int(math.ceil(.2*height)),int(math.ceil(.4*height))]
    trials=[dict(provider.read_glyph(np.pad(glyph,p,constant_values=255)),padding=p) for p in paddings]
    values=[]
    for v in trials:
        text=_observed_designator(v.get('text'),allowed)
        try:certainty=float(v.get('confidence',0))
        except (ValueError,TypeError,OverflowError):return None
        if not text or not math.isfinite(certainty) or not THRESHOLD<=certainty<=1:return None
        values.append((text,certainty))
    if values[0][0]!=values[1][0]:return None
    for v in raw_reads:
        text=_observed_designator(v.get('text'),allowed)
        try:certainty=float(v.get('confidence',0))
        except (ValueError,TypeError,OverflowError):continue
        if text and math.isfinite(certainty) and THRESHOLD<=certainty<=1 and text!=values[0][0]:return None
    return dict(text=values[0][0],confidence=min(v[1] for v in values),views=trials,paddings=paddings,
                proof=dict(method='font_relative_margins_of_complete_observed_caption',core_height=height,
                           source_bbox=list(box),glyph_shape=list(glyph.shape),source_sha256=hashlib.sha256(glyph.tobytes()).hexdigest(),
                           natural_dark_margin=True,paddings=paddings))


def qualified_small_caption(metadata,text,confidence):
    proof=metadata.get('font_relative_caption');fragments=metadata.get('pixel_caption_fragments')
    if (not isinstance(proof,dict) or proof.get('method')!='font_relative_margins_of_complete_observed_caption'
            or not metadata.get('pixel_caption_recovery') or not fragments
            or proof.get('natural_dark_margin') is not True or proof.get('source_bbox')!=metadata.get('bbox')):return None
    h=proof.get('core_height');box=_box(metadata.get('bbox'))
    if isinstance(h,bool) or not isinstance(h,int) or not 5<=h<=8 or box is None:return None
    expected=[int(math.ceil(.2*h)),int(math.ceil(.4*h))]
    if metadata.get('paddings')!=expected or proof.get('paddings')!=expected:return None
    if proof.get('glyph_shape')!=[int(math.ceil(box[3])-math.floor(box[1])),int(math.ceil(box[2])-math.floor(box[0]))]:return None
    for fragment in fragments:
        b=_box(fragment)
        if b is None or not (box[0]<b[0] and box[1]<b[1] and b[2]<box[2] and b[3]<box[3]):return None
    views=metadata.get('raw_reads')
    if not isinstance(views,list) or len(views)!=2:return None
    certainties=[]
    for padding,v in zip(expected,views):
        try:c=float(v.get('confidence',0))
        except (ValueError,TypeError,OverflowError):return None
        if v.get('padding')!=padding or v.get('text')!=text or not math.isfinite(c) or not THRESHOLD<=c<=1:return None
        certainties.append(c)
    return text if confidence==min(certainties) else None


def caption_candidates(components, reads, *, include_ambiguous=False):
    """Unique compact outside captions near a rectangular IC, with owners."""
    found = {}
    for comp in components:
        if comp.get('type') not in ('box', 'bigic') or comp.get('id') is None:
            continue
        body = _box(comp.get('bbox_img', comp.get('bbox')))
        if body is None:
            continue
        for read in reads:
            b = _box(read.get('bbox'))
            if b is None:
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            if not (5 <= h <= 48 and 1.1 <= w / h <= 8):
                continue
            # Entire caption is outside; nearby pin labels within the body
            # must never be treated as component names.
            tolerance=1.5 if read.get('caption_ink_family')=='blue' else 0
            gap = body[1] - b[3] if b[3]<=body[1]+tolerance else b[1]-body[3]
            if -tolerance<=gap<0:gap=0
            if not (0 <= gap <= min(40, 2.5 * h)):
                continue
            center_x = (b[0] + b[2]) / 2
            if not body[0] - h <= center_x <= body[2] + h:
                continue
            found.setdefault(b, set()).add(comp['id'])
    return [(b, sorted(owner for owner in owners if owner is not None))
            for b, owners in sorted(found.items()) if len(owners) == 1 or include_ambiguous]


def _pixel_caption_boxes(bgr,components,reads,pin_evidence=None,*,ink_mask=None):
    """Join real outside glyph fragments missed by whole-word detection.

    An automatic text rectangle must corroborate the word; multiple real
    glyphs share a baseline. The lane never crosses the component body, and
    sides containing measured pins do not open this additional caption path.
    """
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY) if bgr.ndim==3 else bgr
    # Captions can be blue while the symbol is red. Connected-component
    # dimensions reject long lines; excluding chromatic ink loses blue U labels.
    dark=gray<190 if ink_mask is None else ink_mask
    text_boxes=[b for r in reads if (b:=_box(r.get('bbox'))) is not None]
    output=[]
    for comp in components:
        body=_box(comp.get('bbox_img',comp.get('bbox')))
        if comp.get('type')!='box' or comp.get('source')=='cv:bigic' or body is None:
            continue
        for side in ('top','bottom'):
            if any(e.get('edge')==side and e.get('trust')=='trusted' and e.get('frame_confirmed') is True
                   for e in (pin_evidence or {}).get(comp.get('id'),[]) if isinstance(e,dict)):
                continue
            margin=min(24,max(8,(body[2]-body[0])*.20))
            xa,xb=max(0,int(math.floor(body[0]-margin))),min(gray.shape[1],int(math.ceil(body[2]+margin)))
            ya,yb=(max(0,int(math.floor(body[1]))-40),max(0,int(math.floor(body[1])))) if side=='top' else (
                   min(gray.shape[0],int(math.ceil(body[3]))),min(gray.shape[0],int(math.ceil(body[3]))+42))
            if ink_mask is not None:
                if side=='top':yb=min(gray.shape[0],yb+2)
                else:ya=max(0,ya-2)
            if xb<=xa or yb<=ya:
                continue
            count,_,stats,_=cv2.connectedComponentsWithStats(dark[ya:yb,xa:xb].astype(np.uint8),8)
            pieces=[]
            for x,y,w,h,area in stats[1:]:
                if area>=2 and 2<=h<=32 and 1<=w<=48:
                    pieces.append([xa+int(x),ya+int(y),xa+int(x+w),ya+int(y+h)])
            parents=list(range(len(pieces)))
            def root(i):
                while parents[i]!=i:
                    parents[i]=parents[parents[i]];i=parents[i]
                return i
            for i,a in enumerate(pieces):
                for j,b in enumerate(pieces[:i]):
                    overlap=min(a[3],b[3])-max(a[1],b[1])
                    gap=max(a[0]-b[2],b[0]-a[2],0)
                    if overlap>=.35*min(a[3]-a[1],b[3]-b[1]) and gap<=max(2,min(10,.4*max(a[3]-a[1],b[3]-b[1]))):
                        parents[root(i)]=root(j)
            groups={}
            for i,piece in enumerate(pieces):groups.setdefault(root(i),[]).append(piece)
            for group in groups.values():
                if len(group)<2 and ink_mask is None:
                    continue
                box=[min(p[0] for p in group),min(p[1] for p in group),max(p[2] for p in group),max(p[3] for p in group)]
                w,h=box[2]-box[0],box[3]-box[1]
                if not (5<=h<=32 and 1.1<=w/h<=6):
                    continue
                text_supported=any(min(box[2],t[2])>max(box[0],t[0]) and min(box[3],t[3])>max(box[1],t[1]) for t in text_boxes)
                measured_edges={e.get('edge') for e in (pin_evidence or {}).get(comp.get('id'),[])
                                if isinstance(e,dict) and e.get('trust')=='trusted' and e.get('frame_confirmed') is True}
                if not text_supported and not (ink_mask is not None and {'left','right'}<=measured_edges):continue
                # One white pixel retains raster fringes without guessing any
                # missing glyph or accepting a component cut by the lane.
                padded=[max(xa,box[0]-1),max(ya,box[1]-1),min(xb,box[2]+1),min(yb,box[3]+1)]
                if box[0]==xa or box[1]==ya or box[2]==xb or box[3]==yb:
                    continue
                if ink_mask is not None:
                    if side=='top':padded[3]=min(padded[3],int(math.floor(body[1])))
                    else:padded[1]=max(padded[1],int(math.ceil(body[3])))
                record=dict(bbox=padded,caption_pixel_fragments=group)
                if ink_mask is not None:record['caption_ink_family']='blue'
                output.append(record)
    if ink_mask is None and bgr.ndim==3:
        channels=bgr.astype(np.int16)
        blue=(gray<190)&(channels[:,:,0]-channels[:,:,1]>=30)&(channels[:,:,0]-channels[:,:,2]>=30)
        if np.any(blue):
            output+=_pixel_caption_boxes(bgr,components,reads,pin_evidence,ink_mask=blue)
    unique={tuple(r['bbox']):r for r in output}
    return list(unique.values())


def central_caption_candidates(bgr,components,reads,pins,pin_evidence,allowed_prefixes):
    """Reread an existing compact identity word in a measured IC interior.

    Some symbols print their designator near the body center. Require current
    opposing black frame sides, an automatic literal identity observation and
    a unique central owner. Any word facing a measured pin's inward corridor
    is excluded, even when its spelling resembles an IC designator.
    """
    if not pins or not pin_evidence or bgr is None or bgr.ndim!=3:
        return []
    from . import structmatch
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    dark=(gray<190)&(bgr.max(2).astype(np.int16)-bgr.min(2).astype(np.int16)<30)
    found={}
    for comp in components:
        cid=comp.get('id');body=_box(comp.get('bbox_img',comp.get('bbox')))
        if (comp.get('type')!='box' or comp.get('source')=='cv:bigic' or body is None
                or body[2]-body[0]<60 or body[3]-body[1]<60):
            continue
        points=pins.get(cid) or []
        evidence=structmatch.normalize_name_evidence(body,points,pin_evidence.get(cid))
        if not evidence:
            continue
        sides=structmatch.confirmed_frame_sides(body,evidence)
        current=set()
        for edge,coordinate in sides.items():
            vertical=edge in ('left','right')
            a,b=(body[1],body[3]) if vertical else (body[0],body[2])
            start,stop=max(0,int(math.ceil(a))+4),int(math.floor(b))-3
            tangent_size=dark.shape[0 if vertical else 1]
            stop=min(tangent_size,stop)
            center=int(math.floor(coordinate+.5));normal_size=dark.shape[1 if vertical else 0]
            columns=range(max(0,center-1),min(normal_size,center+2))
            if stop-start<40:
                continue
            coverage=max((float(np.mean(dark[start:stop,x] if vertical else dark[x,start:stop]))
                          for x in columns),default=0)
            if coverage>=.65:
                current.add(edge)
        if not ({'left','right'}<=current or {'top','bottom'}<=current):
            continue
        width,height=body[2]-body[0],body[3]-body[1]
        for read in reads:
            box=_box(read.get('bbox'))
            if box is None:
                continue
            w,h=box[2]-box[0],box[3]-box[1]
            if not (5<=h<=48 and 1.1<=w/h<=6
                    and body[0]+.20*width<=box[0] and box[2]<=body[2]-.20*width
                    and body[1]+.20*height<=box[1] and box[3]<=body[3]-.20*height):
                continue
            supported=False
            for observed in (read.get('reads') or {}).values():
                try:
                    text,confidence=observed[0],float(observed[1])
                    supported=(_observed_designator(text,allowed_prefixes) is not None
                               and math.isfinite(confidence) and .4<=confidence<=1)
                except (TypeError,IndexError,ValueError,OverflowError):
                    continue
                if supported:break
            if not supported:
                continue
            candidate=dict(bbox=box,center=((box[0]+box[2])/2,(box[1]+box[3])/2))
            if any(e.get('trust')=='trusted' and e.get('frame_confirmed') is True
                   and structmatch._name_corridor_distance(candidate,body,points,index,evidence) is not None
                   for index,e in enumerate(evidence)):
                continue
            # Overlapping bodies cannot acquire a positive central identity.
            owners=[c.get('id') for c in components if c.get('type')=='box'
                    and (b:=_box(c.get('bbox_img',c.get('bbox')))) is not None
                    and b[0]<=box[0] and b[1]<=box[1] and b[2]>=box[2] and b[3]>=box[3]]
            if owners==[cid]:found.setdefault(box,set()).add(cid)
    return [(box,sorted(owners)) for box,owners in sorted(found.items()) if len(owners)==1]


def read_designator_captions(bgr, components, reads, *, provider=None,
                             mode='auto', allowed_prefixes=('U', 'USB', 'IC'), limit=32,
                             pin_evidence=None, pins=None):
    """Return role-only observations plus audit; do not select final identity.

    The caller appends only accepted observations before regular assignment.
    An injected provider makes tests deterministic without the dependency.
    """
    if limit <= 0:
        return [], []
    allowed = frozenset(x.upper() for x in allowed_prefixes)
    central=central_caption_candidates(bgr,components,reads,pins,pin_evidence,allowed)
    central_boxes={box for box,_owners in central}
    extra_boxes=_pixel_caption_boxes(bgr,components,reads,pin_evidence)
    old_candidates=caption_candidates(components,reads)
    # A small OCR fragment overlapping a complete measured word is the
    # reason for recovery. Only an old rectangle covering almost that entire
    # word suppresses a duplicate crop.
    def already_complete(extra, old):
        w = max(0, min(extra[2],old[2])-max(extra[0],old[0]))
        h = max(0, min(extra[3],old[3])-max(extra[1],old[1]))
        return w*h >= .80*(extra[2]-extra[0])*(extra[3]-extra[1])
    extra_boxes=[r for r in extra_boxes if r.get('caption_ink_family')=='blue'
                 or not any(already_complete(r['bbox'],b) for b,_ in old_candidates)]
    candidates = caption_candidates(components,list(reads)+extra_boxes)+central
    # A correctly shaped, strong EXISTING designator needs no additional
    # provider competing with it. Recover unread captions instead of changing
    # already reliable numeric identities through cross-model score races.
    trusted = set();strong_values={};literal_case_boxes=set();disputed_boxes=set()
    for read in reads:
        old = (read.get('reads') or {}).get('designator')
        if not isinstance(old, (tuple, list)) or len(old) < 2:
            continue
        try:
            confidence = float(old[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if (_observed_designator(old[0], allowed) is not None
                and math.isfinite(confidence) and THRESHOLD <= confidence <= 1):
            trusted.add(_box(read.get('bbox')))
            strong_values.setdefault(old[0],[]).append(_box(read.get('bbox')))
            if old[0] != old[0].upper():
                literal_case_boxes.add(_box(read.get('bbox')))
            for observed in (read.get('reads') or {}).values():
                try:
                    alternate=_observed_designator(observed[0],allowed)
                    certainty=float(observed[1])
                except (TypeError,ValueError,IndexError,OverflowError):
                    continue
                if alternate and alternate!=old[0] and math.isfinite(certainty) and .8<=certainty<=1:
                    disputed_boxes.add(_box(read.get('bbox')))
    duplicate_boxes={b for boxes in strong_values.values() for b in boxes
                     if any(other!=b and (min(other[2],b[2])<=max(other[0],b[0])
                                          or min(other[3],b[3])<=max(other[1],b[1])) for other in boxes)}
    # Overlapping detector boxes can make the owner ambiguous. A valid
    # identity remains held; a duplicated old identity can still be rejected
    # when both unchanged views read a prefix incompatible with every owner.
    ambiguous = [(b,owners) for b,owners in caption_candidates(
        components,list(reads)+extra_boxes,include_ambiguous=True)
        if len(owners)>1 and b in duplicate_boxes]
    candidates = candidates + ambiguous
    read_boxes = [_box(read.get('bbox')) for read in reads]
    restored = {b for b, _owners in candidates
                if recover_fragmented_boxes([old for old in read_boxes if old != b], [b])}
    # A first CRNN read of a restored word can be a confident wrong numeral.
    # Its observed fragmentation is independent evidence for rereading, so
    # shape/confidence alone must not suppress this recovery path.
    container_conflicts=set()
    by_id={c.get('id'):c for c in components}
    for b,owners in candidates:
        if len(owners)!=1:continue
        body=_box(by_id[owners[0]].get('bbox_img',by_id[owners[0]].get('bbox')))
        if any(c.get('id')!=owners[0] and c.get('source')=='cv:bigic'
               and (outer:=_box(c.get('bbox_img',c.get('bbox')))) is not None
               and outer[0]<=body[0] and outer[1]<=body[1] and outer[2]>=body[2] and outer[3]>=body[3]
               and outer[0]<=b[0] and outer[1]<=b[1] and outer[2]>=b[2] and outer[3]>=b[3] for c in components):
            container_conflicts.add(b)
    # A new pixel crop does not bypass protection of an old complete strong
    # word merely because its surrounding white margin differs by one pixel.
    protected_pixel_boxes=set()
    for r in extra_boxes:
        fragments=r.get('caption_pixel_fragments') or []
        if not fragments:continue
        core=[min(p[0] for p in fragments),min(p[1] for p in fragments),max(p[2] for p in fragments),max(p[3] for p in fragments)]
        for old in trusted-restored-duplicate_boxes-literal_case_boxes-disputed_boxes-container_conflicts-central_boxes:
            if old[0]<=core[0] and old[1]<=core[1] and old[2]>=core[2] and old[3]>=core[3]:
                protected_pixel_boxes.add(tuple(r['bbox']))
    candidates = [(b, owners) for b, owners in candidates if b not in protected_pixel_boxes and (b not in trusted
                  or b in restored or b in duplicate_boxes or b in literal_case_boxes or b in central_boxes
                  or b in disputed_boxes or b in container_conflicts)]
    # Apply the work budget after finding conflicts, not in left-to-right
    # detector order. Otherwise an early table of unrelated text can hide
    # the duplicated caption at the other end of a page.
    urgent=duplicate_boxes|disputed_boxes|container_conflicts|central_boxes|literal_case_boxes
    pixel_boxes={tuple(r['bbox']) for r in extra_boxes}
    literal_boxes=set()
    for r in reads:
        if any(isinstance(v,(tuple,list)) and len(v)>=2 and _observed_designator(v[0],allowed)
               for v in (r.get('reads') or {}).values()):literal_boxes.add(_box(r.get('bbox')))
    candidates=sorted(dict.fromkeys((b,tuple(owners)) for b,owners in candidates),
        key=lambda item:(0 if item[0] in urgent else 1 if item[0] in literal_boxes else
                         2 if item[0] in pixel_boxes else 3,item[0]))[:limit]
    candidates=[(b,list(owners)) for b,owners in candidates]
    if not candidates:
        return [], []
    if provider is None:
        provider = get_pin_name_ocr(mode)
    if provider is None:
        return [], []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    output, audit = [], []
    for box, owners in candidates:
        x0, y0, x1, y1 = (int(math.floor(box[0])), int(math.floor(box[1])),
                           int(math.ceil(box[2])), int(math.ceil(box[3])))
        if not (0 <= x0 < x1 <= gray.shape[1] and 0 <= y0 < y1 <= gray.shape[0]):
            continue
        glyph = gray[y0:y1, x0:x1].copy()
        meta = {'bbox': list(box), 'owner_candidates': owners,
                'source': 'ic_caption_ppocr', 'observation_role': 'designator',
                'complete_word_supported_by_fragments': box in restored,
                'pixel_caption_recovery': any(tuple(r['bbox'])==box for r in extra_boxes),
                'pixel_caption_fragments':next((r['caption_pixel_fragments'] for r in extra_boxes if tuple(r['bbox'])==box),[]),
                'measured_central_caption': box in central_boxes,
                'conflicting_existing_identity': box in duplicate_boxes,
                'literal_case_reread': box in literal_case_boxes,
                'conflicting_role_reads':box in disputed_boxes,
                'enclosing_container_conflict':box in container_conflicts,
                'rejection_only': len(owners)>1,
                'paddings': list(PADDINGS),
                'unchanged_source_pixels': True, 'accepted': False, 'raw_reads': []}
        values = []
        for padding in PADDINGS:
            result = provider.read_glyph(np.pad(glyph, padding, constant_values=255))
            view = dict(result, padding=padding)
            meta['raw_reads'].append(view)
            value = _observed_designator(result.get('text'), allowed)
            if len(owners)>1:
                value=None
            if value is None and box in duplicate_boxes:
                raw = result.get('text')
                # This observation can only invalidate the duplicated old
                # identity at this exact rectangle. Assignment still rejects
                # prefixes incompatible with the detected component type.
                if isinstance(raw,str) and re.fullmatch(r'[A-Za-z]{1,4}[0-9]{1,4}[A-Za-z]?',raw.strip()):
                    value=raw.strip()
            try:
                confidence = float(result.get('confidence', 0))
            except (TypeError, ValueError, OverflowError):
                confidence = float('nan')
            if value is None or not math.isfinite(confidence) or not THRESHOLD <= confidence <= 1:
                values.append(None)
            else:
                values.append((value, confidence))
        agreed = (values[0][0], min(v[1] for v in values)) if all(values) and values[0][0] == values[1][0] else None
        if agreed is None and len(owners)==1 and meta['pixel_caption_recovery']:
            retry=_small_caption_views(glyph,box,meta['pixel_caption_fragments'],provider,meta['raw_reads'],allowed)
            if retry is not None:
                meta['initial_margin_reads']=meta['raw_reads'];meta['raw_reads']=retry['views'];meta['paddings']=retry['paddings']
                meta['font_relative_caption']=retry['proof'];agreed=(retry['text'],retry['confidence'])
        if agreed is None and len(owners)==1 and len({v[0] for v in values if v is not None})<=1:
            retry=[dict(provider.read_glyph(np.pad(glyph,p,constant_values=255)),padding=p) for p in (2,4)]
            result=[]
            for v in retry:
                text=_observed_designator(v.get('text'),allowed)
                try:certainty=float(v.get('confidence',0))
                except (ValueError,TypeError,OverflowError):certainty=float('nan')
                result.append((text,certainty) if text and math.isfinite(certainty) and THRESHOLD<=certainty<=1 else None)
            if all(result) and result[0][0]==result[1][0]:
                conflict=any(v is not None and v[0]!=result[0][0] for v in values)
                if not conflict:
                    meta['initial_margin_reads']=meta['raw_reads'];meta['raw_reads']=retry;meta['paddings']=[2,4]
                    agreed=(result[0][0],min(v[1] for v in result))
        if agreed is None and len(owners) == 1:
            component = next(c for c in components if c.get('id') == owners[0])
            split = read_segmented_caption(gray, bgr, box, component.get('bbox_img', component.get('bbox')),
                                           meta['raw_reads'], provider, allowed)
            if split is not None:
                meta['segmented_caption'] = split
                agreed = (split['prefix']+split['suffix'], split['confidence'])
        if agreed is not None:
            value, confidence = agreed
            if len(owners)>1 and _observed_designator(value,allowed) is not None:
                audit.append(meta)
                continue
            meta['accepted'] = True
            meta['rejected_designator_prefix'] = _observed_designator(value,allowed) is None
            output.append({'bbox': list(box), 'reads': {'designator': (value, confidence)},
                           'source': 'ic_caption_ppocr', 'designator_only': True,
                           'designator_target': owners[0], 'designator_ocr': meta})
        audit.append(meta)
    return output, audit
