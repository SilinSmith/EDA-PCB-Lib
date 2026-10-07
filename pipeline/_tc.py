import sys, os, glob, json, cv2, numpy as np, collections
sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0,'.')
from pipeline import pcore, config as C
hs=[]; ws=[]; roles=collections.Counter(); ncomp=0; hbyc=collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(C.REF_ROOT,'*','*CVJsonStd.json'))):
    d=json.load(open(p,encoding='utf-8'))
    ims=glob.glob(os.path.join(os.path.dirname(p),'*.png'))
    if not ims: continue
    im=pcore.__dict__  # not needed
    for tid,t in (d.get('texts') or {}).items():
        loc=t.get('location') or {}
        try: h=float(loc['y2'])-float(loc['y1']); w=float(loc['x2'])-float(loc['x1'])
        except Exception: continue
        if h<=0 or w<=0: continue
        hs.append(h); ws.append(w); roles[t.get('details') or '']+=1
hs=np.array(hs); ws=np.array(ws)
print('ref text boxes n=%d'%len(hs))
print('h  percentiles', np.percentile(hs,[5,25,50,75,95]).round(1), 'max', hs.max())
print('w  percentiles', np.percentile(ws,[5,25,50,75,95]).round(1), 'max', ws.max())
print('roles', dict(roles))
