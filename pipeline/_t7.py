import sys, os, json, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import ptxt, pcore
bgr = ptxt.imread(r'Y:\workspace\EDA\合成原理图demo\images\synth_0002.png'); H=bgr.shape[0]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
ink = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,8)
print('ink frac %.4f' % (ink>0).mean())
# crop 300..500 x 100..145 and dump as ascii
sub = ink[100:150, 340:520]
for r in range(0, sub.shape[0], 2):
    print(''.join('#' if sub[r,c] else '.' for c in range(0, sub.shape[1], 1)))
