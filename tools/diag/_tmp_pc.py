import json, sys
from pathlib import Path
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import data as D
from stage2.text import imread_unicode
from vlm import enrich as E
c = D.find_case("8")
doc = json.loads(Path(r"Y:\workspace\EDA\work\_fusion_out220no4\8.json").read_text(encoding="utf-8"))
bgr = imread_unicode(c.png_path)
k = max((doc.get("pins") or {}), key=lambda x: len(doc["pins"][x]))
bb = doc["components"][k]["bbox"]
pts = [(v["point"]["x"], v["point"]["y"]) for v in doc["pins"][k].values()]
ctr = ((bb[0]+bb[2])/2, (bb[1]+bb[3])/2)
win = E.adaptive_window(bb, 78.0, 58.0, 34.0, 762.0)
pitch = E._pin_pitch(pts); cp = min(win[2], max(6.0, 0.45*pitch))
print("key", k, "n", len(pts), "win", [round(v,1) for v in win], "pitch", round(pitch,1), "cp", round(cp,1))
for i,q in enumerate(pts[:6]):
    ux = ctr[0]-q[0]; uy = ctr[1]-q[1]
    a = E.crop_dir(bgr, q, ctr, win[0], win[1], win[2])
    b = E.crop_dir(bgr, q, ctr, win[0], win[1], win[2], cross_perp=cp)
    print(f"#{i} pt=({q[0]:.0f},{q[1]:.0f}) u=({ux:.0f},{uy:.0f}) old={a[0].shape[:2]} new={b[0].shape[:2]}")
