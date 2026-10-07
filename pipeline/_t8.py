import sys, os, json, collections
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import ptxt, pcore
d = ptxt.synth_texts()
print('cases', list(d.keys()))
for k,v in d.items(): print(' ', k, len(v), 'roles', collections.Counter(t.get('role') for t in v))
