import sys, os, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import ptxt
from pipeline.validate_txt import iou
labels = ptxt.synth_texts()['synth_0002']
bgr = ptxt.imread(r'Y:\workspace\EDA\合成原理图demo\images\synth_0002.png'); H=bgr.shape[0]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
ink = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,8)
lines = ptxt.detect(bgr, ink=ink, close_k=2)
print('gt %d pred %d' % (len(labels), len(lines)))
# for each gt, best iou
best=[]
for t in labels:
    b=tuple(float(v) for v in t['bbox'])
    bi=0; bv=0
    for l in lines:
        v=iou(l.box,b)
        if v>bv: bv=v; bi=l
    best.append((bv,t,l if bv>0 else None))
best.sort(key=lambda x:-x[0])
print('--- best matches (top 6) ---')
for bv,t,l in best[:6]:
    print(' gt', t['content'], t['role'], [round(v,1) for v in t['bbox']], '-> iou %.2f' % bv, None if l is None else (l.n_glyphs, [round(v,1) for v in l.box]))
print('--- worst (bottom 6) ---')
for bv,t,l in best[-6:]:
    print(' gt', t['content'], t['role'], [round(v,1) for v in t['bbox']], '-> iou %.2f' % bv, None if l is None else (l.n_glyphs, [round(v,1) for v in l.box]))
import numpy as np
print('iou dist', np.percentile([b[0] for b in best],[10,25,50,75,90]).round(3))
