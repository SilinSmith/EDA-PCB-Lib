import sys, os, json, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import ptxt, pcore
c = pcore.find_case('0059')
print('case', c.case_id, 'ref', os.path.basename(c.ref_json or ''))
bgr = ptxt.imread(c.png); H=bgr.shape[0]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
ink = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,8)
for ck in (0,2):
  lines = ptxt.detect(bgr, ink=ink, close_k=ck)
  print('close_k=%d -> lines %d' % (ck, len(lines)))
ref = ptxt.ref_texts(c)
print('ref texts', len(ref))
gl = ptxt.glyph_components(bgr, ink=ink, close_k=0)
print('glyphs', len(gl), 'h pct', np.percentile([g.h for g in gl],[5,50,95]).round(1))
# sample ref boxes and check ink presence
ok=0
for box, content, role in ref[:400]:
    x1,y1,x2,y2=[int(round(v)) for v in box]
    x1=max(0,x1);y1=max(0,y1);x2=min(ink.shape[1],x2);y2=min(ink.shape[0],y2)
    if x2<=x1 or y2<=y1: continue
    if ink[y1:y2,x1:x2].mean()>0.15: ok+=1
print('ref boxes with ink: %d/%d' % (ok, min(400,len(ref))))
