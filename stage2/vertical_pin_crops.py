"""Find complete printed rows for measured vertical pins, before OCR.

Blank source rows set the caption bounds. A physical pin's pitch does not
truncate its letters. External words remain between measured neighboring
strokes and cannot cross another pin axis. No OCR string decides geometry.
"""
from __future__ import annotations

import cv2
import numpy as np

from .mono_pin_crops import (_clip, _regions, _name_word_region, _complete_group, _frame_inner_bounds,
                             _known_external_lines, _groups, mono_observations, COLOR_CONTACT_SOURCES)


def _caption_band(mask, region, anchor, pitch):
    if region is None:
        return None
    a,b,c,d = region
    bands = [(b+x,b+y) for x,y in _groups(mask[b:d,a:c].any(1),0) if y-x >= 3]
    if not bands:
        return None
    ranked = sorted(bands,key=lambda pair:abs((pair[0]+pair[1]-1)/2-anchor[1]))
    chosen = ranked[0]
    distance = abs((chosen[0]+chosen[1]-1)/2-anchor[1])
    if distance > max(3.,.55*pitch):
        return None
    if len(ranked)>1 and abs(abs((ranked[1][0]+ranked[1][1]-1)/2-anchor[1])-distance)<.5:
        return None
    return [a,max(b,chosen[0]-1),c,min(d,chosen[1]+1)]


def complete_vertical_pin_boxes(doc, image, candidates):
    if doc.get('pin_text_view_rotation') != 90:
        return candidates
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channels = image.astype(np.int16)
    blue, green, red = channels[:,:,0], channels[:,:,1], channels[:,:,2]
    chroma = channels.max(2)-channels.min(2)
    red_core = (red-green>=30)&(red-blue>=30)&(gray<190)
    wire_core = red_core | ((green-red>=30)&(green-blue>=30)&(gray<190))
    wire_light = (((red-green>=8)&(red-blue>=8))|((green-red>=8)&(green-blue>=8)))&(gray<245)
    mono = (gray<190)&(chroma<30)
    light = (gray<245)&(chroma<30)
    observations = list(mono_observations(doc))
    result = []
    for ob in observations:
        color = ob['evidence'].get('source') in COLOR_CONTACT_SOURCES
        mask = gray<190 if color else mono
        current_light = wire_light|light if color else light
        peers = [p for p in observations if p['component_id']==ob['component_id'] and p['edge']==ob['edge']]
        _,_,pitch,stroke = _regions(ob,peers,mask,current_light,stroke_mask=red_core if color else None)
        if stroke is None:
            continue
        row = ob['anchor_img'][1]
        frame = ob['frame']
        name_left,name_right=frame[0]+2,frame[2]-2
        if color:
            _,name_borders=_frame_inner_bounds(red_core,wire_light,ob)
            for border in name_borders:
                if border['side']==0:
                    name_left=border['persistent_band'][1]
                elif border['side']==2:
                    name_right=border['persistent_band'][0]
        low,high = frame[1]+2,frame[3]-2
        # Inspect the near half of the body, where this side's captions live.
        depth = min(32.,.30*(frame[2]-frame[0]))
        name_region = _clip([ob['anchor_img'][0]+2,low,ob['anchor_img'][0]+depth,high]
                            if ob['edge']=='left' else
                            [ob['anchor_img'][0]-depth,low,ob['anchor_img'][0]-2,high],gray.shape)
        band = _caption_band(mono if color else mask,name_region,ob['anchor_img'],pitch)
        if band:
            band[0],band[2] = ((max(frame[0]+2,ob['anchor_img'][0]+2),frame[2]-2)
                              if ob['edge']=='left' else
                              (frame[0]+2,min(frame[2]-2,ob['anchor_img'][0]-2)))
            band = _clip(band,gray.shape)
        prior = [p for p in peers if p['anchor_img'][1]<row-1]
        start = row-min(44.,1.15*pitch)
        if prior:
            previous = max(prior,key=lambda p:p['anchor_img'][1])
            start = previous['anchor_img'][1]+.5
        number_depth = min(96.,max(2.5*pitch,abs(ob['point_img'][0]-ob['anchor_img'][0])+pitch))
        extension = min(8.,.65*pitch) if color else -2
        number_region = _clip([ob['anchor_img'][0]-number_depth,start,ob['anchor_img'][0]+extension,row+.5]
                              if ob['edge']=='left' else
                              [ob['anchor_img'][0]-extension,start,ob['anchor_img'][0]+number_depth,row+.5],gray.shape)
        if color and band is not None and number_region is not None:
            # The complete interior caption supplies an independent inner
            # bound. A larger numeric window must not import its first letter
            # and mistake that clipped letter for an incomplete numeral.
            name_selected,_ = _name_word_region(mono,band,ob['anchor_img'],pitch)
            name_complete = _complete_group(mono,name_selected,ob,'pin_name',pitch,original_mask=mono)
            if name_complete is not None and not name_complete[1]['integrity_unresolved']:
                caption = name_complete[0]
                if ob['edge']=='left':
                    number_region[2]=min(number_region[2],caption[0]-1)
                else:
                    number_region[0]=max(number_region[0],caption[2]+1)
                number_region=_clip(number_region,gray.shape)
        if color and number_region is not None:
            intact_names=[c for c in candidates if c['role']=='pin_name' and not c['flags']
                          and c['pin_target']['component_id']==ob['component_id']
                          and c['pin_target']['pin_index']==ob['pin_index']]
            for name in intact_names:
                caption=name['bbox']
                if ob['edge']=='left':
                    number_region[2]=min(number_region[2],caption[0]-1)
                else:
                    number_region[0]=max(number_region[0],caption[2]+1)
            number_region=_clip(number_region,gray.shape)
        measured_number = None
        for role,region in (('pin_number',number_region),('pin_name',band)):
            if role=='pin_name':
                name_mask=mono.copy() if color else mask.copy()
                if measured_number is not None:
                    xa,ya,xb,yb=measured_number
                    name_mask[ya:yb,xa:xb]=False
                if color:
                    name_region=_clip([name_left,low,ob['anchor_img'][0]+depth,high]
                                      if ob['edge']=='left' else
                                      [ob['anchor_img'][0]-depth,low,name_right,high],gray.shape)
                region=_caption_band(name_mask,name_region,ob['anchor_img'],pitch)
                if region:
                    region[0],region[2]=((name_left,name_right)
                                        if ob['edge']=='left' else
                                        (name_left,name_right))
                    region=_clip(region,gray.shape)
            if region is None:
                continue
            full_region = list(region)
            known,lines = None,[]
            frame_text_pixels=[]
            local = name_mask if role=='pin_name' and measured_number is not None else mono if color else mask
            if role=='pin_number':
                known,lines = _known_external_lines(wire_core if color else mask,region,ob,pitch,stroke,
                                                     wire_light if color else light,gray)
                if color:
                    _, borders = _frame_inner_bounds(red_core,wire_light,ob)
                    own_side = 0 if ob['edge']=='left' else 2
                    measured = [e for e in borders if e['side']==own_side]
                    if measured:
                        xa,xb = measured[0]['persistent_band']
                        known[:,xa:xb] |= wire_light[:,xa:xb]
                        begin,end=measured[0]['tangent_span']
                        protected=np.zeros(mask.shape,bool)
                        for column in range(xa,xb):
                            baseline=np.median(channels[begin:end,column],axis=0)
                            delta=baseline-channels[:,column]
                            # Black label ink blended with a red frame reduces
                            # all three channels by the same amount. Crossing
                            # red/green wires change chroma instead and cannot
                            # claim this text protection. Keep original pixels.
                            dark=((delta.min(1)>=20)&(delta.max(1)-delta.min(1)<=4)
                                  &(gray[:,column]<190))
                            protected[:,column]=dark
                        known[protected]=False
                    else:
                        protected=np.zeros(mask.shape,bool)
                    local=(((gray<230)&(chroma<30))|protected)&~known
                else:
                    local = mask & ~known
            def external_word(box):
                return box[0]<ob['anchor_img'][0]-1 if ob['edge']=='left' else box[2]>ob['anchor_img'][0]+1
            selected,grouping = _name_word_region(local,region,ob['anchor_img'],pitch,include_enclosed_baseline=color,
                                                  word_filter=external_word if role=='pin_number' else None,
                                                  word_gap=(max(3,min(18,int(pitch*.52))) if role=='pin_number'
                                                            else max(2,int((region[3]-region[1])*.45))))
            complete = _complete_group(local,selected,ob,role,pitch,known_lines=known,
                                        original_mask=local if color else mask,stroke=stroke,line_evidence=lines)
            if complete is None:
                continue
            box,audit = complete
            if grouping.get('ambiguous_detached_small_ink_rows') or grouping.get('independent_ink_inside_selected_rectangle'):
                audit['integrity_unresolved'] = True
            a,b,c,d = box
            if role=='pin_number' and color:
                py,px=np.nonzero(protected[b:d,a:c])
                frame_text_pixels=[[a+int(x),b+int(y)] for y,x in zip(py,px)]
            yy,xx = np.nonzero(known[b:d,a:c]) if known is not None else ([],[])
            target = {k:ob[k] for k in ('component_id','pin_index','edge','point_img','anchor_img')}
            audit.update(method='complete_observed_text_band_of_vertical_pin',
                         known_line_exclusions=[[a+int(x),b+int(y)] for y,x in zip(yy,xx)],
                         measured_line_evidence=lines,black_pixels_removed=len(xx),
                         physical_search_region=full_region,physical_evidence=ob['evidence'],
                         text_row_selection=grouping,stroke_probe=stroke,
                         same_side_rows=[p['anchor_img'][1] for p in peers],pitch_pixels=pitch,
                         sampling_color_mode='original_grayscale_of_color_contact' if color else 'monochrome',
                         independent_of_ocr_text=True)
            if frame_text_pixels:
                audit['chromatic_frame_text_protection']=dict(
                    method='equal_rgb_darkening_against_measured_frame_band',
                    source_pixels=frame_text_pixels,source_pixels_unchanged=True,
                    minimum_channel_darkening=20,maximum_channel_difference=4)
            record = dict(bbox=box,geometry_bbox=box,glyph_bbox=box,region=full_region,role=role,
                           pin_target=target,flags=['glyph_integrity_unresolved'] if audit['integrity_unresolved'] else [],
                           cleanup_evidence=audit)
            if box[3]-box[1]>1.3*(box[2]-box[0]):
                record['recognition_rotations']=[0,-90]
            if role=='pin_number':
                crossing = box[2]>ob['anchor_img'][0] if ob['edge']=='left' else box[0]<ob['anchor_img'][0]
                if crossing:
                    if not color or not measured:
                        record['flags']=['glyph_integrity_unresolved']
                        audit['integrity_unresolved']=True
                    else:
                        audit['vertical_frame_crossing_number'] = dict(
                            method='complete_source_word_across_measured_chromatic_frame',
                            sampling_bbox=list(box),anchor_img=list(ob['anchor_img']),
                            point_img=list(ob['point_img']),edge=ob['edge'],
                            measured_frame_band=measured[0],maximum_inside_pixels=extension,
                            neutral_source_pixels_preserved=True)
                        if (not record['flags'] and qualified_crossing_vertical_number(
                                {'cleanup_evidence':audit},box,target)):
                            measured_number=box
            result.append(record)
    replacements = {(r['pin_target']['component_id'],r['pin_target']['pin_index'],r['role']):r for r in result}
    output = []
    for c in candidates:
        key = (c['pin_target']['component_id'],c['pin_target']['pin_index'],c['role'])
        new = replacements.pop(key,None)
        output.append(new if new is not None and not new['flags'] else c)
    return output + list(replacements.values())


def qualified_crossing_vertical_number(meta,rect,target):
    cleanup = meta.get('cleanup_evidence') or {}
    proof = cleanup.get('vertical_frame_crossing_number') or {}
    evidence = cleanup.get('physical_evidence') or {}
    if (proof.get('method')!='complete_source_word_across_measured_chromatic_frame'
            or proof.get('sampling_bbox')!=list(rect)
            or proof.get('anchor_img')!=target.get('anchor_img')
            or proof.get('point_img')!=target.get('point_img') or proof.get('edge')!=target.get('edge')
            or proof.get('neutral_source_pixels_preserved') is not True
            or evidence.get('source') not in COLOR_CONTACT_SOURCES
            or evidence.get('point_img')!=target.get('point_img')
            or evidence.get('anchor_img')!=target.get('anchor_img')
            or cleanup.get('integrity_unresolved')):
        return False
    edge = target['edge'];axis = 0 if edge in ('left','right') else 1
    # Rectangles are half-open samples; the far edge is not an ink pixel.
    inside = rect[axis+2]-1-target['anchor_img'][axis] if edge in ('left','top') else target['anchor_img'][axis]-rect[axis]
    extent = rect[axis+2]-rect[axis]
    maximum = proof.get('maximum_inside_pixels')
    band = proof.get('measured_frame_band') or {}
    measured_band = band.get('persistent_band')
    if not isinstance(measured_band,list) or len(measured_band)!=2:
        return False
    start,stop = measured_band
    if not (isinstance(start,int) and isinstance(stop,int) and 1<=stop-start<=12
            and start-8<=target['anchor_img'][axis]<=stop+8
            and band.get('side')=={'left':0,'top':1,'right':2,'bottom':3}[edge]):
        return False
    return (isinstance(maximum,(float,int)) and 0<inside<=maximum<=8 and inside<.70*extent
            and band.get('method')=='current_long_side_ink_band' and band.get('minimum_coverage')==.80)
