import json, pathlib

d = json.loads(pathlib.Path(r'Y:\workspace\EDA\work\flow\_graphdata.json').read_text(encoding='utf-8'))
tot = 0
for n in d['nodes']:
    nlen = len(n.get('note') or '')
    tot += nlen
    print('%-13s w=%-4s file=%-34s note=%5d  sub=%s' % (
        n['id'], n.get('w'), (n.get('file') or '')[:34], nlen, (n.get('sub') or '')[:60]))
print('total note chars', tot)
print('name', len(d['name']), 'note(top)', len(d['note']))
