"""Measured hollow terminal markers attached to real monochrome IC sides."""
from __future__ import annotations
import math
import cv2
import numpy as np


def hollow_marker_bands(mask, box, edge, frame, *, ink_threshold=230):
    """Find an outside triangular hole with a connected, thin outward line.

    Labels, numbering and pitch are never consulted. A numeral hole alone is
    insufficient: its near end must touch the long frame, its contour must be
    triangular and a separate straight line must connect through the marker.
    Both a single triangle and opposing open triangles can qualify.
    """
    coord, confirmed, coverage = frame
    if ink_threshold not in (230,245):
        raise ValueError('Unsupported measured marker threshold')
    if edge not in (1,3) or not confirmed:
        return []
    sign = -1 if edge == 3 else 1
    center = int(math.floor(coord+.5))
    top, bottom = max(0,int(math.ceil(box[1]))+8), min(mask.shape[0],int(math.floor(box[3]))-7)
    depths = np.arange(0,81)
    columns = center+sign*depths
    columns = columns[(columns>=0)&(columns<mask.shape[1])]
    if bottom-top < 12 or len(columns)<24:
        return []
    strip = mask[top:bottom,columns].astype(np.uint8)
    contours, hierarchy = cv2.findContours(strip,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    bands = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] < 0:
            continue
        d,y,w,h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        vertices = cv2.approxPolyDP(contour,.08*cv2.arcLength(contour,True),True)[:,0,:]
        if (not 1<=d<=5 or not 5<=w<=28 or not 5<=h<=24
                or not .32<=area/(w*h)<=.70 or len(vertices)!=3):
            continue
        # A triangle has a base normal to the pin and a centered pointed end.
        ordered = sorted(vertices.tolist(),key=lambda p:p[0])
        base = ordered[:2] if abs(ordered[0][0]-ordered[1][0])<=2 else ordered[1:]
        tip = ordered[2] if base==ordered[:2] else ordered[0]
        if (abs(base[0][0]-base[1][0])>2 or abs(base[0][1]-base[1][1])<h*.60
                or abs(tip[1]-(base[0][1]+base[1][1])/2)>2):
            continue
        row = y+(h-1)/2
        a,b = max(0,int(math.floor(row))-1), min(strip.shape[0],int(math.ceil(row))+2)
        reach = min(strip.shape[1],3*w+16)
        starts = []
        for yy in range(a,b):
            on = strip[yy,d+w+2:reach].astype(bool)
            changes = np.diff(np.r_[False,on,False].astype(np.int8))
            for begin,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                if end-begin>=10:
                    starts.append((abs(yy-row),int(begin+d+w+2),int(end+d+w+2),yy))
        if not starts:
            continue
        _,start,stop,yy = min(starts)
        ya,yb = yy,yy+1
        for direction in (-1,1):
            q=yy+direction
            while a<=q<b and np.mean(strip[q,start:stop])>=.9:
                ya,yb=min(ya,q),max(yb,q+1)
                q+=direction
        if yb-ya>5:
            continue
        region = strip[max(0,y-2):min(strip.shape[0],y+h+2),:stop]
        _, labels = cv2.connectedComponents(region,8)
        offset=max(0,y-2)
        seeds=set(np.unique(labels[ya-offset:yb-offset,start:stop]).tolist())-{0}
        frame_ids=set(np.unique(labels[max(0,yy-offset-2):yy-offset+3,:2]).tolist())-{0}
        if not seeds.intersection(frame_ids):
            continue
        tangent=top+(ya+yb-1)/2
        # Exclusive bounds in the source coordinate system, independent of sign.
        xs=[int(min(columns[start:stop])),int(max(columns[start:stop]))+1]
        hole_box=[int(min(columns[d:d+w])),top+y,int(max(columns[d:d+w]))+1,top+y+h]
        polygon=[[int(center+sign*vx),int(top+vy)] for vx,vy in vertices]
        related=[]
        raster_contours=[]
        for j,other in enumerate(contours):
            if hierarchy[0][j][3]<0:
                continue
            dd,oy,ow,oh=cv2.boundingRect(other)
            approx=cv2.approxPolyDP(other,.08*cv2.arcLength(other,True),True)
            ratio=cv2.contourArea(other)/max(1,ow*oh)
            if (d<=dd and dd+ow<=start+2 and 5<=ow<=28 and 5<=oh<=24
                    and abs(oy+(oh-1)/2-yy)<=2 and .32<=ratio<=.70 and len(approx)==3):
                related.append((dd,oy,ow,oh))
                raster_contours.append([[int(center+sign*vx),int(top+vy)] for vx,vy in other[:,0,:]])
        envelope_top=min(q[1] for q in related)-3
        envelope_bottom=max(q[1]+q[3] for q in related)+3
        normal_end=min(strip.shape[1],start+8)
        envelope=[int(min(columns[:normal_end])),top+envelope_top,
                  int(max(columns[:normal_end]))+1,top+envelope_bottom]
        marker=dict(hole_bbox=hole_box,hole_vertices=polygon,hole_area=area,
                    hollow_edge_contours_img=raster_contours,
                    connected_marker_region=envelope,
                    outer_line_probe=[xs[0],top+ya,xs[1],top+yb],
                    outer_line_run=stop-start,outer_line_width=yb-ya,
                    connected_to_frame=True,near_frame_distance=d,
                    ink_threshold=ink_threshold)
        if any(abs(band["tangent_center"]-tangent)<=2 for band in bands):
            continue
        bands.append(dict(edge="left" if edge==3 else "right",edge_index=edge,
                          tangent_band=[top+ya,top+yb-1],tangent_center=tangent,
                          anchor_img=[float(coord),float(tangent)],frame_coverage=float(coverage),
                          marker=marker))
    return sorted(bands,key=lambda band:band["tangent_center"])


def valid_hollow_marker_evidence(rec):
    """Validate the recorded independent marker geometry, never a trust flag."""
    try:
        marker=rec["marker"]
        box=marker["hole_bbox"]; line=marker["outer_line_probe"]
        vertices=marker["hole_vertices"]; anchor=rec["anchor_img"]
        region=marker["connected_marker_region"]
        values=[*box,*line,*anchor,*region,*[v for p in vertices for v in p],
                marker["hole_area"],marker["outer_line_run"],marker["outer_line_width"],
                marker["near_frame_distance"],rec["frame_coverage"]]
        if (len(box)!=4 or len(line)!=4 or len(region)!=4 or len(anchor)!=2 or len(vertices)!=3
                or any(len(p)!=2 for p in vertices) or not all(math.isfinite(float(v)) for v in values)):
            return False
        w,h=box[2]-box[0],box[3]-box[1]
        sign=-1 if rec["edge"]=="left" else 1
        near=anchor[0]-box[2]+1 if sign<0 else box[0]-anchor[0]
        outside=(line[2]<box[0] if sign<0 else line[0]>box[2])
        return (rec.get("source")=="frame_hollow_marker"
                and rec.get("reason")=="frame_attached_hollow_marker"
                and rec.get("edge") in ("left","right")
                and marker.get("connected_to_frame") is True and marker.get("ink_threshold") in (230,245)
                and 5<=w<=28 and 5<=h<=24 and .32<=marker["hole_area"]/(w*h)<=.70
                and 1<=marker["near_frame_distance"]<=5 and abs(near-marker["near_frame_distance"])<=1
                and line[2]-line[0]>=10 and marker["outer_line_run"]==line[2]-line[0]
                and 1<=line[3]-line[1]<=5 and marker["outer_line_width"]==line[3]-line[1]
                and abs((line[1]+line[3]-1)/2-anchor[1])<=.6 and outside
                # A measured border center can lie halfway between raster
                # columns; the scanned strip starts at its rounded column.
                # Accept only that half-pixel discretization, never a marker
                # region detached from the actual frame or triangular hole.
                and region[0]-.5<=min(anchor[0],box[0]) and region[2]+.5>=max(anchor[0],box[2])
                and min(region[2],line[2])-max(region[0],line[0])>=5
                and region[1]<box[1] and region[3]>box[3]
                and .65<=rec["frame_coverage"]<=1)
    except (KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):
        return False
