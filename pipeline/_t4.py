import sys, json, cv2, numpy as np
sys.path.insert(0,'.')
from pipeline import pcore, validate_skel as V
c = pcore.find_case('0059')
doc = json.load(open(c.gt_json,encoding='utf-8'))
im = cv2.imread(c.png, cv2.IMREAD_COLOR); H,W = im.shape[:2]
print('H',H,'W',W)
bx = [v.get('bbox') for v in (doc.get('components') or {}).values() if v.get('bbox')]
bx = np.array(bx)
print('bbox_json x %.0f..%.0f  y %.0f..%.0f' % (bx[:,0].min(), bx[:,2].max(), bx[:,1].min(), bx[:,3].max()))
print('bbox_img_yflip y %.0f..%.0f' % (H-bx[:,3].max(), H-bx[:,1].min()))
ref = V.ref_of(c)
sk = np.array(ref['skel']); pth = np.array([p for path in ref['polys'] for p in path])
print('ref skel x %.0f..%.0f y %.0f..%.0f' % (sk[:,0].min(), sk[:,0].max(), sk[:,1].min(), sk[:,1].max()))
print('ref path x %.0f..%.0f y %.0f..%.0f' % (pth[:,0].min(), pth[:,0].max(), pth[:,1].min(), pth[:,1].max()))
# ink bbox
ink = cv2.adaptiveThreshold(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,8)
ys,xs = np.nonzero(ink)
print('ink x %.0f..%.0f y %.0f..%.0f' % (xs.min(), xs.max(), ys.min(), ys.max()))
# reference coords in the json frame (flip)
rjy = H - sk[:,1]
print('ref skel in json frame y %.0f..%.0f' % (rjy.min(), rjy.max()))
