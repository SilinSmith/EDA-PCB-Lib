import sys, os, json, glob, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import ptxt, pcore, config as C
tot = {True: [0,0], False: [0,0]}
for png in sorted(glob.glob(os.path.join(C.SYNTH_ROOT,'images','*.png'))):
    stem = os.path.splitext(os.path.basename(png))[0]
    labels = ptxt.synth_texts().get(stem)
    if not labels: continue
    bgr = ptxt.imread(png); H = bgr.shape[0]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,8)>0)
    for flip in (True, False):
        hit=0; nn=0
        for t in labels:
            b = [float(v) for v in t['bbox']]
            box = pcore.bbox_json_to_img(b,H) if flip else tuple(b)
            x1,y1,x2,y2 = [int(round(v)) for v in box]
            x1=max(0,x1); y1=max(0,y1); x2=min(ink.shape[1],x2); y2=min(ink.shape[0],y2)
            if x2<=x1 or y2<=y1: continue
            nn+=1
            if ink[y1:y2, x1:x2].mean() > 0.15: hit+=1
        tot[flip][0]+=hit; tot[flip][1]+=nn
for flip in (True,False):
    h,n = tot[flip]
    print('flip=%s  boxes with >15%% ink: %d/%d = %.3f' % (flip,h,n,h/max(1,n)))
