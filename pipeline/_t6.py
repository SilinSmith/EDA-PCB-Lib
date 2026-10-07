import sys, os, json, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import ptxt, pcore
labels = ptxt.synth_texts()['synth_0002']
png = r'Y:\workspace\EDA\合成原理图demo\images\synth_0002.png'
bgr = ptxt.imread(png); H = bgr.shape[0]
print('image', bgr.shape)
gl = ptxt.glyph_components(bgr, close_k=2)
print('glyphs', len(gl))
print('glyph h  ', np.percentile([g.h for g in gl],[5,25,50,75,95]).round(1))
print('glyph w  ', np.percentile([g.w for g in gl],[5,25,50,75,95]).round(1))
ln = ptxt.group_lines(gl)
print('lines', len(ln))
print('line n_glyphs', np.percentile([l.n_glyphs for l in ln],[5,25,50,75,95]).round(1))
print('line w', np.percentile([l.box[2]-l.box[0] for l in ln],[5,25,50,75,95]).round(1))
print('line h', np.percentile([l.box[3]-l.box[1] for l in ln],[5,25,50,75,95]).round(1))
# a designator U1
for t in labels:
    if t.get('content')=='U1' and t.get('role')=='designator':
        gb = pcore.bbox_json_to_img([float(v) for v in t['bbox']], H); print('GT U1 designator', gb); break
for t in labels[:3]:
    print('GT', t.get('content'), t.get('role'), pcore.bbox_json_to_img([float(v) for v in t['bbox']],H))
print('--- predicted lines near U1 ---')
gbx=gb
for l in ln:
    d = max(0, max(gbx[0]-l.box[2], l.box[0]-gbx[2])) + max(0, max(gbx[1]-l.box[3], l.box[1]-gbx[3]))
    if d < 30: print('  near', tuple(round(v,1) for v in l.box), 'ng', l.n_glyphs)
