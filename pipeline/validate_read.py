"""Measure the character reader on its own, leaving the evaluated case out of the gallery.

The 29 vendor-representation cases (``Dataset300我不知道干嘛用的先放置/*CVJsonStd.json``) carry a
``texts`` section with the exact string of every text box, which makes them the only labelled
source of glyphs in the corpus.  Two things follow:

* the gallery is *built* from those same boxes -- so a score computed with the evaluated case
  still in the gallery is optimistically biased.  Every number here is therefore produced with
  ``exclude=case_id`` (leave-one-case-out); each gallery template carries its source case.
* the vendor boxes are heavily padded (measured ink coverage inside a box is 0.217 on case
  0059), so the crop is tightened before reading, and the residual padding is why the
  aspect-ratio character count is unreliable.

Metrics, all per text box: ``exact`` (read string == content), ``abst`` (nothing read), ``char``
(position-wise accuracy over boxes whose reading has the content's length), ``cellcount`` (the
cell count agrees with the label -- an upper bound on what any segmenter can reach).

    $env:PYTHONIOENCODING='utf-8'; python -m pipeline.validate_read --n=8
    $env:PYTHONIOENCODING='utf-8'; python -m pipeline.validate_read --sweep
"""

from __future__ import annotations

import sys

if __name__ == '__main__':
    sys.path.insert(0, __file__.rsplit('\\', 2)[0])
    sys.stdout.reconfigure(encoding='utf-8')

from pipeline import pread, pcore, pskel, ptxt      # noqa: E402


def ref_cases(n: int = 0):
    out = sorted([c for c in pcore.discover() if c.ref_json], key=lambda c: c.case_id)
    return out[:n] if n else out


def prepare(case):
    """Read one sheet once: inks, ink components, and the labelled boxes with their content."""
    bgr = ptxt.imread(case.png)
    if bgr is None:
        return None
    inks = {m: pskel.ink_mask(bgr, mode=m, k=31, c=8) for m in ('adaptive', 'otsu')}
    glyphs = ptxt.glyph_components(bgr, ink=inks['adaptive'], close_k=0)
    boxes = [(content, box) for content, box, _ in pread.iter_gt_texts(case)]
    return dict(case=case, glyphs=glyphs, inks=inks, boxes=boxes)


def read_boxes(pre, g, margin=0.0, min_score=0.45, vote=0, use_cells=True,
               ink_mode='adaptive'):
    """Read every labelled text box of one prepared case -> ``(content, read, n_cells)``."""
    ink = pre['inks'][ink_mode]
    rows = []
    for content, box in pre['boxes']:
        sub = pread.tight_crop(ink, box)
        if sub is None:
            continue
        cells = pread.cells_from_glyphs(ink, box, pre['glyphs']) if use_cells else None
        if cells is not None and len(cells) != len(content):
            cells = None          # component count disagrees with the label: keep the even split
        if cells is None:
            cells = pread.split_chars(sub)
        text, _out, _sc = pread.read_glyphs(g, cells, margin=margin, min_score=min_score,
                                           exclude=pre['case'].case_id, vote=vote)
        rows.append((content, text, len(cells)))
    return rows


def tally(pres, g, **kw):
    n = exact = abst = 0
    char_ok = char_tot = ncell_ok = 0
    wrong = []
    for pre in pres:
        for content, text, ncell in read_boxes(pre, g, **kw):
            n += 1
            ncell_ok += (ncell == len(content))
            if text == content:
                exact += 1
            elif not text:
                abst += 1
            elif len(wrong) < 14:
                wrong.append('%s %r->%r' % (pre['case'].case_id, content, text))
            if text and len(text) == len(content):
                char_tot += len(content)
                char_ok += sum(1 for a, b in zip(text, content) if a == b)
    return dict(n=n, exact=exact, abst=abst, char_ok=char_ok, char_tot=char_tot,
                ncell_ok=ncell_ok, wrong=wrong)


def kind_of(content):
    """Which of the strings actually matter for the score (see ``metrics.component_text_ok``)."""
    import re
    if re.fullmatch(r'[A-Za-z]{1,4}\d{1,4}', content):
        return 'designator'
    if re.fullmatch(r'\d+(\.\d+)?\s*[a-zA-ZµΩ]*', content):
        return 'value'
    if len(content) <= 2:
        return 'short'
    return 'long'


def breakdown(pres, g, **kw):
    per = {}
    for pre in pres:
        for content, text, ncell in read_boxes(pre, g, **kw):
            for key in ('ALL', kind_of(content), 'len%d' % min(len(content), 8)):
                d = per.setdefault(key, dict(n=0, exact=0, cok=0, ctot=0))
                d['n'] += 1
                d['exact'] += (text == content)
                if text and len(text) == len(content):
                    d['ctot'] += len(content)
                    d['cok'] += sum(1 for a, b in zip(text, content) if a == b)
    return per


def gallery_from(pres, cut='comp'):
    """Build a gallery in memory with accurate character cuts.

    The saved gallery splits a vendor box evenly by aspect ratio, which mis-cuts whenever the
    pitch estimate is off -- and a mis-cut template poisons every later match.  Cutting with the
    detected ink components instead (only where their count equals the label's length) keeps the
    templates clean; boxes that are a single character are always safe to add.
    """
    g = pread.Gallery()
    for pre in pres:
        cid = pre['case'].case_id
        for content, box in pre['boxes']:
            cells = pread.cells_from_glyphs(pre['inks']['adaptive'], box, pre['glyphs'])
            if cut == 'comp' and cells is not None and len(cells) == len(content):
                for ch, part in zip(content, cells):
                    g.add(ch, part, cid)
            elif len(content) == 1:
                sub = pread.tight_crop(pre['inks']['adaptive'], box)
                if sub is not None:
                    g.add(content, sub, cid)
    return g


def report(tag, r):
    print('%-30s boxes=%-4d exact=%-4d %.3f  abst=%-4d %.3f  char=%.3f(%d/%d)  ccells=%.3f'
          % (tag, r['n'], r['exact'], r['exact'] / max(1, r['n']), r['abst'],
             r['abst'] / max(1, r['n']), r['char_ok'] / max(1, r['char_tot']),
             r['char_ok'], r['char_tot'], r['ncell_ok'] / max(1, r['n'])))


VARIANTS = [
    ('cells m0 s45', dict(use_cells=True, margin=0.0, min_score=0.45)),
    ('cells m0 s30', dict(use_cells=True, margin=0.0, min_score=0.30)),
    ('cells m0 s00', dict(use_cells=True, margin=0.0, min_score=0.0)),
    ('cells m02 s45', dict(use_cells=True, margin=0.02, min_score=0.45)),
    ('cells m04 s45', dict(use_cells=True, margin=0.04, min_score=0.45)),
    ('cells m0 s30 vote7', dict(use_cells=True, margin=0.0, min_score=0.30, vote=7)),
    ('cells m0 s00 vote7', dict(use_cells=True, margin=0.0, min_score=0.0, vote=7)),
    ('split m0 s45', dict(use_cells=False, margin=0.0, min_score=0.45)),
    ('split m0 s00', dict(use_cells=False, margin=0.0, min_score=0.0)),
    ('cells m0 s00 otsu', dict(use_cells=True, margin=0.0, min_score=0.0,
                               ink_mode='otsu')),
]


def main(argv):
    n = 0
    for a in argv:
        if a.startswith('--n='):
            n = int(a.split('=')[1])
    cases = ref_cases(n)
    print('cases=%d %s' % (len(cases), [c.case_id for c in cases]))
    pres = [p for p in (prepare(c) for c in cases) if p]
    print('prepared=%d  boxes=%d' % (len(pres), sum(len(p['boxes']) for p in pres)))
    g = pread.load_gallery()
    print('gallery=%d templates, source ids kept=%s'
          % (len(g), len(g.src) == len(g.chars) and any(g.src)))
    if '--sweep' in argv:
        for tag, kw in VARIANTS:
            report(tag, tally(pres, g, **kw))
        return
    if '--break' in argv:
        for gtag, gg in (('saved-even', g), ('comp-cut', gallery_from(pres))):
            print('== gallery %s: %d templates' % (gtag, len(gg)))
            per = breakdown(pres, gg, use_cells=False, margin=0.0, min_score=0.0)
            for key in ('ALL', 'designator', 'value', 'short', 'long',
                        'len1', 'len2', 'len3', 'len4', 'len5'):
                if key in per:
                    d = per[key]
                    print('   %-11s n=%-4d exact=%.3f (%-4d)  char=%.3f'
                          % (key, d['n'], d['exact'] / max(1, d['n']), d['exact'],
                             d['cok'] / max(1, d['ctot'])))
        return
    r = tally(pres, g, use_cells=True, margin=0.0, min_score=0.0)
    report('cells m0 s00', r)
    for w in r['wrong']:
        print('   wrong:', w)


if __name__ == '__main__':
    main(sys.argv[1:])
