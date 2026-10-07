# -*- coding: utf-8 -*-
"""Decide the JSON y-axis direction by comparing the ink bbox of the PNG with the
union bbox of all annotated components."""
import json, os, sys
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

cases = [
    (os.path.join(DS, '10GT', '1', '1.png'), os.path.join(DS, '10GT', '1', '1.json')),
    (os.path.join(DS, '200_train_cases', '0001', '0001-KiCad.png'),
     os.path.join(DS, '200_train_cases', '0001', '0001-KiCad_target_named.json')),
    (os.path.join(DS, '10_GTcase', '0025-Altium Designer', '0025-Altium Designer.png'),
     os.path.join(DS, '10_GTcase', '0025-Altium Designer', '0025-Altium Designer_target.json')),
]

for png, js in cases:
    if not (os.path.exists(png) and os.path.exists(js)):
        print('missing', png, js); continue
    im = Image.open(png).convert('L')
    a = np.asarray(im)
    H, W = a.shape
    ink = a < 200
    rr = np.where(ink.any(axis=1))[0]
    cc = np.where(ink.any(axis=0))[0]
    ink_y0, ink_y1 = int(rr.min()), int(rr.max())
    ink_x0, ink_x1 = int(cc.min()), int(cc.max())

    d = load(js)
    ys, xs = [], []
    for c in d['components'].values():
        b = c.get('bbox')
        if b and len(b) == 4:
            xs += [b[0], b[2]]; ys += [b[1], b[3]]
    jx0, jx1, jy0, jy1 = min(xs), max(xs), min(ys), max(ys)

    print('=' * 70)
    print(os.path.basename(png), 'W=%d H=%d' % (W, H))
    print('  ink bbox (image px, top-left origin): x %d..%d  y %d..%d' % (ink_x0, ink_x1, ink_y0, ink_y1))
    print('  json union bbox:                      x %.0f..%.0f  y %.0f..%.0f' % (jx0, jx1, jy0, jy1))
    print('  -> as top-left (y unchanged) : y %.0f..%.0f  vs ink y %d..%d   |dy|=%s' %
          (jy0, jy1, ink_y0, ink_y1, (abs(jy0 - ink_y0) + abs(jy1 - ink_y1))))
    print('  -> as bottom-left (y=H-y)    : y %.0f..%.0f  vs ink y %d..%d   |dy|=%s' %
          (H - jy1, H - jy0, ink_y0, ink_y1, (abs((H - jy1) - ink_y0) + abs((H - jy0) - ink_y1))))
    print('  x check: json %.0f..%.0f vs ink %d..%d  |dx|=%s' %
          (jx0, jx1, ink_x0, ink_x1, (abs(jx0 - ink_x0) + abs(jx1 - ink_x1))))
    white_frac_top = float((~ink)[:20, :].mean())
    white_frac_bot = float((~ink)[-20:, :].mean())
    print('  margin: top 20px white=%.3f  bottom 20px white=%.3f' % (white_frac_top, white_frac_bot))
