import json, os, sys
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from fusion import doc as DC
RUN = r"work\_submit_out220r36\runs"
tot = {"assigned_text":0, "order":0, "point":0, "n":0}
for case in ("0001","0002","0003","0004","0005"):
    s1 = json.load(open(os.path.join(RUN,"stage1","%s.stage1.json"%case), encoding="utf-8"))
    d  = json.load(open(os.path.join(RUN,"stage2","%s.json"%case), encoding="utf-8"))
    a  = json.load(open(os.path.join(RUN,"stage2","_assigned","%s.json"%case), encoding="utf-8"))
    H  = float(s1["image"]["h"])
    # 需要 stage1_doc 的 bbox_json 形式给 invert_keymap 用 -> 直接把 stage2 doc 当 base、
    # stage1 当 stage1_doc。这里只验证映射本身。
    cid2key = {}
    # 手工复刻 invert_keymap 的核心：用 assigned 的 desig_slots 不靠，直接用框重叠太绕；
    # 先用「stage1 元件框 ∩ doc 元件框」找同 cid。
    for c in (s1.get("components") or []):
        bb = c.get("bbox_json")
        if not bb: continue
        best=None
        for k, cv in (d.get("components") or {}).items():
            b2 = cv.get("bbox")
            if not b2: continue
            ox = min(bb[2],b2[2])-max(bb[0],b2[0]); oy = min(bb[3],b2[3])-max(bb[1],b2[1])
            if ox>0 and oy>0:
                ar = (ox*oy)/max(1.0,(bb[2]-bb[0])*(bb[3]-bb[1]))
                if best is None or ar>best[0]: best=(ar,k)
        if best and best[0]>0.3: cid2key[c["id"]]=best[1]
    for cid, rec in (a.get("comps") or {}).items():
        key = cid2key.get(cid)
        grp = (d.get("pins") or {}).get(key) if key else None
        if not grp: continue
        keys = list(grp.keys())
        pts = (s1.get("pins") or {}).get(cid) or []
        for k in range(len(keys) if len(keys)>=len(pts) else len(pts)):
            tot["n"] += 1
            # (i) assigned number text
            slot = (rec.get("pins") or {}).get(str(k)) or {}
            nt = ((slot.get("number") or {}).get("text") or "").strip()
            if nt and ("pin_%s"%nt) in grp: tot["assigned_text"] += 1
            # (ii) insertion order
            if k < len(keys): tot["order"] += 1
            # (iii) point match
            if k < len(pts):
                px, py = float(pts[k][0]), H-float(pts[k][1])
                for gk, gv in grp.items():
                    q = (gv.get("point") or {})
                    if abs(float(q.get("x",-1e9))-px)<1.0 and abs(float(q.get("y",-1e9))-py)<1.0:
                        tot["point"] += 1; break
print(tot)
if tot["n"]:
    for kk in ("assigned_text","order","point"):
        print("  %-14s 覆盖 %.3f" % (kk, tot[kk]/tot["n"]))
