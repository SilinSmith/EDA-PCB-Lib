"""把 stage1 的产物画回原图，供人眼核验。用法：python stage1_draw.py 0050 0011 ..."""
import json, sys
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(r"Y:\workspace\EDA\work\stage1_all")
PRE = Path(r"Y:\workspace\EDA\work\preview")
PRE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import data as D

cases = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)}
# 颜色：YOLO 绿 / CV 品红 / 待裁决 橙 / 导线 蓝 / 附着点 黄
C_YOLO = (60, 200, 60)
C_CV = (220, 60, 220)
C_PEND = (0, 165, 255)
C_WIRE = (255, 120, 0)
C_ATT = (0, 255, 255)

tiles = []
for cid in sys.argv[1:]:
    f = OUT / f"{cid}.stage1.json"
    if not f.exists():
        print("缺", f)
        continue
    d = json.load(open(f, encoding="utf-8"))
    c = cases.get(cid)
    img = cv2.imread(str(c.png_path))
    H, W = img.shape[:2]
    maxside = 1500
    s = min(1.0, maxside / max(W, H))
    vis = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    for seg in (d.get("wire_detail") or {}).get("segments", []):
        pts = seg if isinstance(seg, list) else (seg.get("pts") or seg.get("points"))
        if not pts or len(pts) < 2:
            continue
        a = np.array([[int(p[0] * s), int(p[1] * s)] for p in pts[:2]], np.int32)
        cv2.polylines(vis, [a], False, C_WIRE, 2, cv2.LINE_AA)
    for x in d["components"]:
        b = x["bbox_img"]
        p1 = (int(b[0] * s), int(b[1] * s))
        p2 = (int(b[2] * s), int(b[3] * s))
        src = x["source"] or ""
        col = C_CV if src.startswith("cv:") else C_YOLO
        if x["pending"]:
            col = C_PEND
        cv2.rectangle(vis, p1, p2, col, 2)
    for a in d.get("attachments", []):
        for h in a.get("hits", [])[:1]:
            at = h["at"]
            while isinstance(at, (list, tuple)) and at and isinstance(at[0], (list, tuple)):
                at = at[-1]
            x0, y0 = float(at[0]), float(at[1])
            cv2.circle(vis, (int(x0 * s), int(y0 * s)), 4, C_ATT, -1)
    label = (f"{cid}  {W}x{H}  comp {len(d['components'])}  "
             f"pend {d['stats']['n_pending']}  wires {d['timing']['wires_s']:.1f}s")
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
    out = PRE / f"stage1_{cid}.png"
    cv2.imwrite(str(out), vis)
    print("写了", out, vis.shape)
    tiles.append(vis)

if len(tiles) > 1:
    w = max(t.shape[1] for t in tiles)
    rows = []
    for t in tiles:
        pad = cv2.copyMakeBorder(t, 0, 6, 0, w - t.shape[1],
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        rows.append(pad)
    sheet = np.vstack(rows)
    p = PRE / "stage1_sheet.png"
    cv2.imwrite(str(p), sheet)
    print("写了", p, sheet.shape)
print("图例：绿=YOLO  品红=CV专类  橙=待裁决  蓝=导线线段  黄点=附着点")
