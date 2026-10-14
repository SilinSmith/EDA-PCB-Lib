"""Keep complete numeric source ink together with its measured attached wire."""
import copy
import numpy as np
from .structmatch import trusted_frame_pin_evidence


def intact_number_line_box(candidate, gray):
    if candidate.get('role')!='pin_number' or not candidate.get('flags'):
        return None
    e=candidate.get('cleanup_evidence') or {}
    stroke=e.get('stroke_probe') or {}
    evidence=e.get('physical_evidence') or {}
    target=candidate.get('pin_target') or {}
    if (evidence.get('edge')!=target.get('edge') or evidence.get('anchor_img')!=target.get('anchor_img')
            or not trusted_frame_pin_evidence(evidence,point=target.get('point_img'))):
        return None
    if (e.get('black_pixels_removed')!=0 or e.get('connected_ink_beyond_region_sides')!=[3]
        or e.get('boundary_touch_sides')!=[3] or e.get('unresolved_reason')
        or e.get('marker_cleanup_unresolved')
        or (e.get('physical_evidence') or {}).get('source') not in (None,'','frame_normal_stroke')):
        return None
    top,bottom=stroke.get('stroke_top'),stroke.get('stroke_bottom')
    if not isinstance(top,int) or not isinstance(bottom,int) or not 1<=bottom-top<=6:
        return None
    a,b,c,d=candidate['bbox'];region=candidate['region']
    if d!=top or not 0<=a<c<=gray.shape[1] or not 0<=b<d<bottom<=gray.shape[0]:
        return None
    lines=e.get('measured_line_evidence') or []
    measured=[]
    for y in range(top,bottom):
        matches=[line for line in lines if line.get('axis')=='horizontal' and line.get('row')==y
                 and line['span'][0]<=a and c<=line['span'][1]]
        if not matches:return None
        line=max(matches,key=lambda line:line['span'][1]-line['span'][0])
        xa,xb=line['span']
        if not 0<=xa<xb<=gray.shape[1] or np.mean(gray[y,xa:xb]<190)<.85:return None
        measured.append(line)
    xa=max(line['span'][0] for line in measured);xb=min(line['span'][1] for line in measured)
    end=bottom
    # Retain only a measured full-width raster fringe of this same wire.
    while end<gray.shape[0] and end-top<6 and np.mean(gray[end,xa:xb]<245)>=.90:
        end+=1
    target_row=candidate['pin_target']['anchor_img'][1]
    if any(abs(row-target_row)>1 and b<=row<end for row in e.get('same_side_rows',[])):
        return None
    if not region[0]<=a<c<=region[2]:return None
    result=copy.deepcopy(candidate)
    box=[a,b,c,end]
    result.update(bbox=box,geometry_bbox=list(box),glyph_bbox=list(box),flags=[],
                  region=[region[0],region[1],region[2],max(region[3],end)],
                  recognition_view='original_numeric_glyph_with_measured_attached_line')
    cleanup=result['cleanup_evidence']
    cleanup.update(prior_glyph_integrity=copy.deepcopy(e),integrity_unresolved=False,
                   known_line_exclusions=[],black_pixels_removed=0,
                   intact_attached_line={'stroke_rows':[top,bottom], 'included_end':end,
                    'measured_span':[xa,xb], 'no_source_pixel_deleted':True,
                    'source_bbox_before_inclusion':list(candidate['bbox'])})
    cleanup['physical_search_region']=[region[0],region[1],region[2],max(region[3],end)]
    return result
