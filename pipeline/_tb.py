import sys, os, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import ptxt
labels = ptxt.synth_texts()['synth_0002']
bgr = ptxt.imread(r'Y:\workspace\EDA\合成原理图demo\images\synth_0002.png'); H=bgr.shape[0]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
ink = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,8)
# pick a designator with a wide box (value text) and a designator
for want in ('U10','R2'):
    for t in labels:
        if t.get('content')==want and t.get('role')=='designator':
            b=[float(v) for v in t['bbox']]
            x1,y1,x2,y2=[int(v) for v in b]
            print('GT', want, b)
            sub=ink[y1-6:y2+6, x1-6:x2+6]
            for r in range(sub.shape[0]):
                print('   ', ''.join('#' if sub[r,c] else '.' for c in range(sub.shape[1])))
            # components in that crop
            n,lab,stats,_=cv2.connectedComponentsWithStats((sub>0).astype(np.uint8),8)
            print('    comps:', [(int(s[0]),int(s[1]),int(s[2]),int(s[3]),int(s[4])) for s in stats[1:]])
            break
