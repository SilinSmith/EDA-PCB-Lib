import json, pathlib

d = json.loads(pathlib.Path(r'Y:\workspace\EDA\work\flow\_graphdata.json').read_text(encoding='utf-8'))
L = []
for n in d['nodes']:
    L.append('#' * 70)
    L.append('## id=%s | group=%s | kind=%s | w=%s' % (n['id'], n['group'], n['kind'], n.get('w')))
    L.append('title: %s' % n.get('title'))
    L.append('sub  : %s' % n.get('sub'))
    L.append('file : %s' % n.get('file'))
    L.append('xy   : (%s,%s)' % (n.get('x'), n.get('y')))
    L.append('ins  : %s' % json.dumps(n.get('ins'), ensure_ascii=False))
    L.append('outs : %s' % json.dumps(n.get('outs'), ensure_ascii=False))
    L.append('params: %s' % json.dumps(n.get('params'), ensure_ascii=False))
    L.append('note:')
    L.append(n.get('note') or '(empty)')
    L.append('')
pathlib.Path(r'Y:\workspace\EDA\work\flow\_wfall.md').write_text('\n'.join(L), encoding='utf-8')
print('ok', len('\n'.join(L)))
