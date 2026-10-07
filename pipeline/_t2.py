import sys, json, cv2, numpy as np
sys.path.insert(0,'.')
from pipeline import pcore, validate_pins as VP
from collections import Counter
offs=[]; ncomp=0; ncomp_with=0; nleft=nright=ntop=nbot=0; inside=0; von_border=0
ppt=[]
for c in pcore.discover():
    if not c.gt_json: continue
    H = float(json.load(open(c.gt_json,encoding='utf-8')).get('components') and 0 or 0)
    doc = json.load(open(c.gt_json,encoding='utf-8'))
    im = cv2.imread(c.png, cv2.IMREAD_COLOR); H = im.shape[0]
    boxes = {k: pcore.bbox_json_to_img(v.get('bbox') or [0,0,0,0], H) for k,v in (doc.get('components') or {}).items()}
    # pins per comp
    for key, grp in (doc.get('pins') or {}).items():
        if key not in boxes: continue
        ncomp_with+=1
        x1,y1,x2,y2 = boxes[key]
        for pk,info in (grp or {}).items():
            pt=info.get('point') or {}
            x, y = float(pt.get('x',0)), H-float(pt.get('y',0))
            # offsets
            d={'L':x1-x,'R':x-x2,'T':y1-y,'B':y-y2}
            e=min(d,key=lambda k:abs(d[k]))
            offs.append((e, d[e]))
    ncomp+=1
print('cases', ncomp, 'comps with pins', ncomp_with, 'pins', len(offs))
for e in 'LRTB':
    v=[o for ee,o in offs if ee==e]
    v=np.array(v)
    print(e, 'n', len(v), 'median %.2f'%np.median(v), 'p10 %.2f p25 %.2f p75 %.2f p90 %.2f'%(np.percentile(v,10),np.percentile(v,25),np.percentile(v,75),np.percentile(v,90)))
allv=np.array([o for ee,o in offs])
print('overall median %.2f'%np.median(allv))
print('inside (offset<0):', int((allv<0).sum()), 'of', len(allv))
print('offset<=5:', int((allv<=5).sum()), '<=10:', int((allv<=10).sum()), '<=15:',int((allv<=15).sum()),'<=20:',int((allv<=20).sum()),'<=30:',int((allv<=30).sum()),'>30:',int((allv>30).sum()))
