"""Upper bound on pin-tip recovery: what if the wire extraction were perfect?

The pipeline has to place pin tips geometrically, and ``validate_pins`` measured the first
attempt at that (skeleton endpoints + crossing points) at F1 0.087 -- unusable, because the
fine-stroke ink mask is full of text and symbol fragments whose endpoints look exactly like
pin tips.  This script isolates the *geometric* half of the problem: it feeds the ground-truth
net polylines (``edges`` in the competition JSON, 2 points each) as if wire extraction had been
perfect, binds their endpoints to component boxes, and scores the resulting tips against the
ground-truth pin tips at the same 5-px tolerance as ``metrics.pin_f1``.

    pin_tip_ceiling >> 0  -> the recovery rule is fine and the wire extractor is the bottleneck
    pin_tip_ceiling ~  0  -> the binding rule itself is wrong, however good the wires get
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, __file__.rsplit('\\', 2)[0])
    sys.stdout.reconfigure(encoding='utf-8')

from pipeline import pcore                       # noqa: E402


def gt_boxes(case):
    t = pcore.truth_of(case) if hasattr(pcore, 'truth_of') else None
    return t


def load(case):
    import json
    doc = json.load(open(case.gt_json, encoding='utf-8'))
    h = doc.get('height')
    if h is None:
        from PIL import Image
        with Image.open(case.png) as im:
            h = im.height
    return pcore.truth_from_doc(doc, int(round(float(h))), int(round(float(h))))


def wire_vertices(raw):
    """Every endpoint of every 2-point wire edge, in image coordinates."""
    pts = []
    for net in (raw.get('nets') or {}).values():
        for e in (net.get('edges') or {}).values():
            for p in e:
                pts.append((float(p['x']), float(p['y'])))
    return pts


def main(argv):
    n = 12
    reach = 14.0
    outer = False
    for a in argv:
        if a.startswith('--n='):
            n = int(a.split('=')[1])
        if a.startswith('--reach='):
            reach = float(a.split('=')[1])
        if a == '--outer':
            outer = True
    cases = [c for c in pcore.discover() if c.gt_json][:n]
    tot = Counter()
    for c in cases:
        import json
        raw = json.load(open(c.gt_json, encoding='utf-8'))
        h = float(raw.get('height') or 0)
        t = pcore.truth_from_doc(raw, int(h), int(h))
        verts = [(x, h - y) for x, y in wire_vertices(raw)]
        pins = list(t.pins.values())
        boxes = list(t.comps.values())
        box_arr = np.array([b.box for b in boxes], float)
        cx = (box_arr[:, 0] + box_arr[:, 2]) / 2.0
        cy = (box_arr[:, 1] + box_arr[:, 3]) / 2.0
        # assign every wire vertex to the nearest component centre, then keep, per component,
        # the vertices that sit on its own boundary band (|offset| <= reach from the bbox)
        cand = defaultdict(list)
        pv = np.array(verts, float)
        if len(pv) == 0:
            continue
        d2 = (pv[:, 0:1] - cx[None, :]) ** 2 + (pv[:, 1:2] - cy[None, :]) ** 2
        near = np.argmin(d2, axis=1)
        for i, (x, y) in enumerate(pv):
            b = boxes[int(near[i])]
            x1, y1, x2, y2 = b.box
            dx = max(x1 - x, x - x2, 0.0)
            dy = max(y1 - y, y - y2, 0.0)
            if np.hypot(dx, dy) <= reach:
                cand[b.key].append((x, y))
        if outer:
            # a pin tip is the wire vertex farthest from its component, not any vertex
            for k, pts in list(cand.items()):
                bx = boxes[[i for i, b in enumerate(boxes) if b.key == k][0]].box
                ccx, ccy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
                pts.sort(key=lambda p: -((p[0] - ccx) ** 2 + (p[1] - ccy) ** 2))
                keep = []
                for p in pts:
                    if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 5.0 for q in keep):
                        keep.append(p)
                cand[k] = keep
        for b in boxes:
            gt = [(p.x, p.y, p.suffix) for p in pins if p.comp == b.key]
            got = cand.get(b.key, [])
            used = set()
            for gx, gy, sfx in gt:
                best, bj = 1e9, -1
                for j, (px, py) in enumerate(got):
                    if j in used:
                        continue
                    dd = np.hypot(px - gx, py - gy)
                    if dd < best:
                        best, bj = dd, j
                if bj >= 0 and best <= 5.0:
                    used.add(bj)
                    tot['tp'] += 1
                else:
                    tot['fn'] += 1
            tot['fp'] += max(0, len(got) - len(used))
    p = tot['tp'] / max(1.0, tot['tp'] + tot['fp'])
    r = tot['tp'] / max(1.0, tot['tp'] + tot['fn'])
    print('cases=%d  wire-vertex pin tips: tp=%d fp=%d fn=%d  P=%.3f R=%.3f F1=%.3f'
          % (len(cases), tot['tp'], tot['fp'], tot['fn'], p, r, 2 * p * r / max(1e-9, p + r)))


if __name__ == '__main__':
    main(sys.argv[1:])
