import json, pathlib

d = json.loads(pathlib.Path(r'Y:\workspace\EDA\work\flow\_graphdata.json').read_text(encoding='utf-8'))
L = ['groups:']
for g in d['groups']:
    L.append('  %-6s %s  %s' % (g['id'], g.get('color'), g.get('title')))
L.append('')
L.append('nodes (%d):' % len(d['nodes']))
for n in d['nodes']:
    L.append('  %-13s %-5s %-5s w=%-4s xy=(%s,%s)  ins=%s  outs=%s' % (
        n['id'], n['group'], n['kind'], n.get('w'), n.get('x'), n.get('y'),
        [p['id'] for p in (n.get('ins') or [])], [p['id'] for p in (n.get('outs') or [])]))
L.append('')
L.append('links (%d):' % len(d['links']))
for l in d['links']:
    L.append('  %-28s -> %-28s %s' % (l['from'], l['to'], l.get('kind')))
pathlib.Path(r'Y:\workspace\EDA\work\flow\_wdump2.txt').write_text('\n'.join(L), encoding='utf-8')
print(len(L), 'lines')
