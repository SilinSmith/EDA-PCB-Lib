import json, sys, glob, os, collections
sys.stdout.reconfigure(encoding='utf-8')

OUT = r'Y:\workspace\EDA\work\stage1_all2'
files = sorted(glob.glob(os.path.join(OUT, '*.json')))
print('n_files', len(files))
print('sample names', [os.path.basename(f) for f in files[:4]])

d = json.load(open(files[0], encoding='utf-8'))
print('\ntop keys', list(d.keys()))
print('case', d['case'], 'image', d['image'])
print('stats', d['stats'])
print('timing', d['timing'])
print('n components', len(d['components']))
c = d['components'][0]
print('\ncomponent[0] keys', list(c.keys()))
print('component[0]', json.dumps(c, ensure_ascii=False)[:600])
print('\nwires keys', list(d['wires'].keys()))
print('wires', json.dumps({k: v for k, v in d['wires'].items() if not isinstance(v, list)}, ensure_ascii=False))
print('wire_detail keys', list(d['wire_detail'].keys()),
      {k: len(v) for k, v in d['wire_detail'].items()})
print('\nattachments', len(d['attachments']))
if d['attachments']:
    print('attachment[0]', json.dumps(d['attachments'][0], ensure_ascii=False)[:600])
print('\nhypotheses keys', list(d['hypotheses'].keys()))
print('n slots', len(d['hypotheses'].get('slots', {})))
print('folds', d['hypotheses'].get('folds'))

# aggregate: how many components carry subtype / source / pending
src = collections.Counter(); prop = collections.Counter(); pend = 0
keyset = collections.Counter()
for f in files:
    dd = json.load(open(f, encoding='utf-8'))
    for cc in dd['components']:
        src[cc.get('source')] += 1
        prop[cc.get('proposal')] += 1
        pend += 1 if cc.get('pending') else 0
        keyset[cc.get('key')] += 1
print('\nsource', src.most_common())
print('pending', pend)
print('key values', keyset.most_common(5))
print('\nproposal top20', prop.most_common(20))
