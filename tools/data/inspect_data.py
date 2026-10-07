# -*- coding: utf-8 -*-
"""Inspect the schematic dataset structure: components / pins / nets."""
import json, os, glob, collections, sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DS = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'

def load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize(path, label):
    d = load(path)
    print('=' * 70)
    print(label, os.path.basename(path))
    print('top-level keys:', list(d.keys()))
    comps = d.get('components', {})
    pins = d.get('pins', {})
    nets = d.get('nets', {})
    print('n_components=%d n_pins_groups=%d n_nets=%d' % (len(comps), len(pins), len(nets)))

    # type histogram
    th = collections.Counter(c.get('type') for c in comps.values())
    print('types:', dict(th.most_common()))

    # one component example
    k = next(iter(comps))
    print('component example [%s]:' % k, json.dumps(comps[k], ensure_ascii=False))

    # pins
    if pins:
        pk = next(iter(pins))
        print('pins group example [%s]:' % pk, json.dumps(pins[pk], ensure_ascii=False)[:400])
    # net
    if nets:
        nk = next(iter(nets))
        print('net example [%s]:' % nk, json.dumps(nets[nk], ensure_ascii=False)[:600])

    # consistency: pin group keys vs components keys
    ck, pk_ = set(comps), set(pins)
    print('pins-only keys (not in components):', len(pk_ - ck), list(pk_ - ck)[:5])
    print('components without pins:', len(ck - pk_), list(ck - pk_)[:5])

    # coordinate ranges
    xs, ys = [], []
    for c in comps.values():
        b = c.get('bbox')
        if b and len(b) == 4:
            xs += [b[0], b[2]]; ys += [b[1], b[3]]
    if xs:
        print('bbox x range: %.1f .. %.1f ; y range: %.1f .. %.1f' % (min(xs), max(xs), min(ys), max(ys)))

    # nets edges stats
    ne = [len(n.get('edges', {})) for n in nets.values()]
    if ne:
        print('edges per net: min=%d max=%d mean=%.2f total=%d' % (min(ne), max(ne), sum(ne)/len(ne), sum(ne)))
    hg = [len(n.get('hyperGraph', '').strip('()').split(',')) if n.get('hyperGraph') else 0 for n in nets.values()]
    if hg:
        print('pins per net: min=%d max=%d mean=%.2f' % (min(hg), max(hg), sum(hg)/len(hg)))
    return d

if __name__ == '__main__':
    summarize(os.path.join(DS, '10GT', '1', '1.json'), '10GT case 1')
    # a training case
    g = glob.glob(os.path.join(DS, '200_train_cases', '0001', '*.json'))
    if g:
        summarize(g[0], 'train case 0001')
    g2 = glob.glob(os.path.join(DS, '10_GTcase', '0025*', '*.json'))
    if g2:
        summarize(g2[0], '10_GTcase 0025')
