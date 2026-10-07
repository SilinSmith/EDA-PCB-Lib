# -*- coding: utf-8 -*-
"""Global survey of the whole dataset: file layout, type vocabulary, pin key patterns."""
import json, os, glob, collections, sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DS = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'

def load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---- 1. file layout of each split ----
for sub in ['10GT', '10_GTcase', '200_train_cases']:
    root = os.path.join(DS, sub)
    if not os.path.isdir(root):
        print('MISSING', root); continue
    dirs = sorted(os.listdir(root))
    print('=' * 70)
    print('%s: %d entries' % (sub, len(dirs)))
    print('  first 3 dirs:', dirs[:3])
    print('  last 3 dirs :', dirs[-3:])
    # enumerate file-name suffix patterns
    pat = collections.Counter()
    n_dir = 0
    for d0 in dirs:
        p = os.path.join(root, d0)
        if not os.path.isdir(p):
            print('  (non-dir entry: %s)' % d0); continue
        n_dir += 1
        for f in os.listdir(p):
            base, ext = os.path.splitext(f)
            # normalise the leading case number out of the name
            pat[ext.lower()] += 1
    print('  file ext counts:', dict(pat))

# ---- 2. type vocabulary over ALL json ----
types = collections.Counter()
n_json = 0
pin_key_odd = collections.Counter()
net_key_style = collections.Counter()
comp_key_prefix = collections.Counter()
hg_sep = collections.Counter()
coord_float = collections.Counter()
for sub in ['10GT', '10_GTcase', '200_train_cases']:
    root = os.path.join(DS, sub)
    for jp in glob.glob(os.path.join(root, '*', '*.json')):
        try:
            d = load(jp)
        except Exception as e:
            print('LOAD FAIL', jp, e); continue
        n_json += 1
        for k, c in d.get('components', {}).items():
            types[c.get('type')] += 1
            comp_key_prefix['phi' if k.startswith('∅') else ('digit' if k[:1].isdigit() else 'alpha')] += 1
        for g in d.get('pins', {}).values():
            for pk in g:
                if not pk.startswith('pin_'):
                    pin_key_odd[pk] += 1
        for nk, n in d.get('nets', {}).items():
            net_key_style['numeric' if nk.startswith('net') else 'named'] += 1
            hg = n.get('hyperGraph', '')
            if hg and not (hg.startswith('(') and hg.endswith(')')):
                hg_sep['malformed'] += 1
        for c in d.get('components', {}).values():
            b = c.get('bbox') or []
            for v in b:
                coord_float['int' if float(v).is_integer() else 'float'] += 1

print('=' * 70)
print('total json files:', n_json)
print('--- type vocabulary (%d distinct) ---' % len(types))
for t, n in types.most_common():
    print('   %-18s %5d' % (t, n))
print('--- component key prefix:', dict(comp_key_prefix))
print('--- pin key not starting with pin_:', dict(pin_key_odd))
print('--- net key style:', dict(net_key_style))
print('--- malformed hyperGraph:', dict(hg_sep))
print('--- bbox number integrality:', dict(coord_float))
