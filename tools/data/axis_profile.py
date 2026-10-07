# -*- coding: utf-8 -*-
"""Decisive y-axis test: correlate the image ink row/col profile with the
component-density profile derived from the GT JSON under each hypothesis."""
import json, os, glob, sys
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DS = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'

def load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def profile(d, H, W, flip):
    row = np.zeros(H); col = np.zeros(W)
    for c in d['components'].values():
        b = c.get('bbox')
        if not (b and len(b) == 4):
            continue
        x1, y1, x2, y2 = b
        if flip:
            r1, r2 = H - y2, H - y1
        else:
            r1, r2 = y1, y2
        r1 = max(0, int(np.floor(r1))); r2 = min(H, int(np.ceil(r2)) + 1)
        c1 = max(0, int(np.floor(x1))); c2 = min(W, int(np.ceil(x2)) + 1)
        row[r1:r2] += 1
        col[c1:c2] += 1
    return row, col

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / d) if d > 0 else 0.0

pairs = []
for sub in ['10GT', '10_GTcase', '200_train_cases']:
    for jp in sorted(glob.glob(os.path.join(DS, sub, '*', '*.json'))):
        pngs = glob.glob(os.path.join(os.path.dirname(jp), '*.png'))
        if pngs:
            pairs.append((pngs[0], jp))

res = {'up': [], 'down': []}
for png, jp in pairs:
    im = Image.open(png).convert('L')
    a = np.asarray(im)
    H, W = a.shape
    ink = (a < 200).astype(float)
    irow = ink.sum(axis=1); icol = ink.sum(axis=0)
    d = load(jp)
    for key, flip in (('up', True), ('down', False)):
        r, c = profile(d, H, W, flip)
        res[key].append((corr(irow, r) + corr(icol, c)) / 2.0)

for k in ('up', 'down'):
    v = np.array(res[k])
    print('%-5s mean corr = %.4f   median = %.4f   n=%d' % (k, v.mean(), np.median(v), len(v)))
print()
print('Interpretation: "up" = JSON is bottom-left origin (y_up = H - y_img);')
print('"down" = JSON shares the image top-left origin (y_json == y_img).')
