"""A glyph gallery rendered from system fonts, instead of harvested from the corpus.

The harvested gallery has a measured generalisation problem: reading a vendor text box with the
evaluated case held out drops character accuracy to 0.493, while keeping that case in the gallery
gives 0.997 (``validate_read --break``).  In other words the nearest-neighbour matcher memorises
per-case rasterisation and does not transfer across vendors.

Font rendering removes that failure mode entirely: templates come from fonts, not from the
evaluation corpus, so the measured accuracy is an honest held-out number and every character of
the alphabet is available (the harvested gallery has 6002 templates but only 7 examples of ``Z``).

    $env:PYTHONIOENCODING='utf-8'; python -m pipeline.pglyph --build
    $env:PYTHONIOENCODING='utf-8'; python -m pipeline.pglyph --eval --n=8
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    sys.path.insert(0, __file__.rsplit('\\', 2)[0])
    sys.stdout.reconfigure(encoding='utf-8')

from pipeline import pread                      # noqa: E402

FONT_DIR = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
FONTS = ['consola.ttf', 'consolab.ttf', 'arial.ttf', 'arialbd.ttf', 'tahoma.ttf',
         'tahomabd.ttf', 'segoeui.ttf', 'seguisb.ttf', 'verdana.ttf', 'calibri.ttf',
         'trebuc.ttf', 'lucon.ttf', 'micross.ttf', 'times.ttf']
SIZES = [9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 26, 30]
CHARS = ('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
         '+-._/Ωµω%()[]')


def render(ch: str, font: ImageFont.FreeTypeFont) -> np.ndarray | None:
    img = Image.new('L', (96, 96), 0)
    d = ImageDraw.Draw(img)
    d.text((24, 24), ch, fill=255, font=font)
    a = np.asarray(img)
    ys, xs = np.nonzero(a > 96)
    if ys.size == 0:
        return None
    return (a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 96).astype(np.uint8)


def build(verbose: bool = True) -> pread.Gallery:
    g = pread.Gallery()
    per = {}
    for fn in FONTS:
        path = os.path.join(FONT_DIR, fn)
        if not os.path.exists(path):
            continue
        for size in SIZES:
            try:
                font = ImageFont.truetype(path, size)
            except Exception:
                continue
            for ch in CHARS:
                sub = render(ch, font)
                if sub is None or sub.shape[0] < 4 or sub.shape[1] < 2:
                    continue
                g.add(ch, sub, src='font:' + fn)
                per[ch] = per.get(ch, 0) + 1
    if verbose:
        print('font gallery: %d templates from %d fonts x %d sizes, %d chars'
              % (len(g), len(FONTS), len(SIZES), len(per)))
    return g


def build_synth(verbose: bool = True) -> pread.Gallery:
    """Templates harvested from the *synthetic* sheets, whose font the generator controls.

    ``合成原理图demo/texts.jsonl`` labels every synthetic text box with its exact string, so the
    character cuts are known rather than estimated -- and the sheets were never used to score the
    pipeline, so templates taken from them do not leak into the real-case measurements.
    """
    import glob
    import json
    from pipeline import config as C
    from pipeline import pskel, ptxt
    g = pread.Gallery()
    per = {}
    root = C.SYNTH_ROOT
    with open(os.path.join(root, 'texts.jsonl'), encoding='utf-8') as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    for row in rows:
        cid = row['case']
        hits = glob.glob(os.path.join(root, 'images', cid + '.*'))
        if not hits:
            continue
        bgr = ptxt.imread(hits[0])
        if bgr is None:
            continue
        ink = pskel.ink_mask(bgr, mode='adaptive', k=31, c=8)
        for t in row.get('texts') or []:
            content = str(t.get('content') or '')
            bb = t.get('bbox') or []
            if len(content) != 4 or len(bb) != 4:
                continue
            sub = pread.tight_crop(ink, tuple(float(v) for v in bb))
            if sub is None:
                continue
            cells = pread.split_chars(sub, n=len(content))
            if len(cells) != len(content):
                continue
            for ch, part in zip(content, cells):
                g.add(ch, part, 'synth:' + cid)
                per[ch] = per.get(ch, 0) + 1
    if verbose:
        print('synth gallery: %d templates, %d chars' % (len(g), len(per)))
    return g


def main(argv):
    if '--eval' in argv:
        from pipeline import validate_read as V
        n = 0
        for a in argv:
            if a.startswith('--n='):
                n = int(a.split('=')[1])
        cases = V.ref_cases(n)
        pres = [p for p in (V.prepare(c) for c in cases) if p]
        print('cases=%d boxes=%d' % (len(pres), sum(len(p['boxes']) for p in pres)))
        which = 'synth' if '--synth' in argv else 'font'
        g = build_synth() if which == 'synth' else build()
        print('== %s gallery' % which)
        per = V.breakdown(pres, g, use_cells=False, margin=0.0, min_score=0.0)
        for key in ('ALL', 'designator', 'value', 'short', 'long',
                    'len1', 'len2', 'len3', 'len4', 'len5'):
            if key in per:
                d = per[key]
                print('   %-11s n=%-4d exact=%.3f (%-4d)  char=%.3f'
                      % (key, d['n'], d['exact'] / max(1, d['n']), d['exact'],
                         d['cok'] / max(1, d['ctot'])))
        for tag, kw in (('cells m0 s45', dict(use_cells=True, margin=0.0, min_score=0.45)),
                        ('split m0 s45', dict(use_cells=False, margin=0.0, min_score=0.45)),
                        ('split m0 s00', dict(use_cells=False, margin=0.0, min_score=0.0))):
            V.report(tag, V.tally(pres, g, **kw))
        return
    g = build()
    print('saved ->', pread.save_gallery(g, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'glyph_gallery_font.pkl')))


if __name__ == '__main__':
    main(sys.argv[1:])
