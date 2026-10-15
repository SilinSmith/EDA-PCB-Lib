# -*- coding: utf-8 -*-
import io, json, re, sys
from collections import Counter

t = io.open(r'Y:\workspace\EDA\work\flow\workflow.html', encoding='utf-8').read()
m = re.search(r'application/json[^>]*>(.*?)</script>', t, re.S)
d = json.loads(m.group(1))

print('groups:', [(g['id'], g['title']) for g in d['groups']])
print()
for n in d['nodes']:
    if n['id'] in ('know', 'boxscan', 'ev_bridge', 'own'):
        print('%-10s kind=%-6s group=%-6s note=%5d chars bullets=%2d  %s'
              % (n['id'], n['kind'], n['group'], len(n.get('note', '')),
                 len(n.get('bullets', [])), n['title']))
print()
print(Counter(n['kind'] for n in d['nodes']))
print('kind=new:', [n['id'] for n in d['nodes'] if n['kind'] == 'new'])
print('kind=cand:', [n['id'] for n in d['nodes'] if n['kind'] == 'cand'])
