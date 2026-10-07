import sys, json, cv2, numpy as np
sys.path.insert(0,'.')
from pipeline import pcore
c = pcore.find_case('0059')
doc = json.load(open(c.gt_json,encoding='utf-8'))
im = cv2.imread(c.png, cv2.IMREAD_COLOR); H,W = im.shape[:2]
print('case', c.case_id, 'png', im.shape)
print('root keys', list(doc.keys())[:20])
boxes = {k: v.get('bbox') for k,v in (doc.get('components') or {}).items()}
pins  = {k: v for k,v in (doc.get('pins') or {}).items()}
keys = list(boxes)[:8]
for k in keys:
    b = boxes[k]
    print('COMP', repr(k), 'bbox_json', b, 'bbox_img(naive)', b, 'bbox_img(yflip)', pcore.bbox_json_to_img(b,H))
    pl = pins.get(k, {})
    for pk, info in list(pl.items())[:4]:
        pt = info.get('point')
        print('   ', pk, 'point_json', pt, 'naive', (pt['x'],pt['y']), 'yflip', pcore.json_to_img((pt['x'],pt['y']),H), 'name', info.get('pinname'))
# how many comps are the y-flip vs naive better fit for the pin tips being on the bbox border?
stat_naive=[]; stat_flip=[]
for k, b in boxes.items():
    if not b: continue
    for pk, info in (pins.get(k) or {}).items():
        pt=info.get('point') or {}
        if pt.get('x') is None: continue
        ny = pt['y']; fy = H-pt['y']
        x1,y1,x2,y2 = b   # in the json frame
        d_naive = pcore.point_to_bbox_dist((pt['x'],ny), (x1,y1,x2,y2))
        d_flip  = pcore.point_to_bbox_dist((pt['x'],fy), (x1,H-y2,x2,H-y1))
        stat_naive.append(d_naive); stat_flip.append(d_flip)
import numpy as np
print('naive dist to json-bbox: median %.2f  <=5:%.3f' % (np.median(stat_naive), np.mean(np.array(stat_naive)<=5)))
print('flip  dist to json-bbox: median %.2f  <=5:%.3f' % (np.median(stat_flip), np.mean(np.array(stat_flip)<=5)))
