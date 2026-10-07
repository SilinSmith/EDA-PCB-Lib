import sys, time, cv2, numpy as np
sys.path.insert(0,'.')
from pipeline import pcore, pskel, ppins, validate_pins as VP
cases = VP.pick(4)
ex = pskel.WireExtractor(use_colour=False, close_ink=True)
for c in cases:
    bgr = cv2.imread(c.png, cv2.IMREAD_COLOR)
    t=time.time(); thin, ink, ch = ex.foreground(bgr); t1=time.time()-t
    t=time.time(); skel = pskel.zhang_suen(thin); t2=time.time()-t
    t=time.time(); res = ex.extract(bgr); t3=time.time()-t
    t=time.time(); boxes = VP.gt_boxes(c)
    cands = ppins.recover_pins(res, [pcore.Comp(k,'',b) for k,b in boxes.items()]); t4=time.time()-t
    print('%s %s %s mask %.1fs thin %.1fs total %.1fs pins %.1fs npins %d nseg %d nskel %d' % (
        c.case_id, c.split, bgr.shape, t1, t2, t3, t4, len(cands), len(res.segments), int((res.skel>0).sum())))
