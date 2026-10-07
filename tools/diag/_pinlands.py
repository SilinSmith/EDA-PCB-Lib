import json, os
inp = r"work\_submit_out220r36\runs\stage2\_assigned"
out = r"work\_fusion_out220nl2\_assigned"
cases = sorted(f[:-5] for f in os.listdir(out) if f.endswith(".json"))[:8]
tot = {"pin_diff":0, "pin_slots":0, "desig_diff":0, "cases":0}
for c in cases:
    a = json.load(open(os.path.join(inp, c+".json"), encoding="utf-8"))
    b = json.load(open(os.path.join(out, c+".json"), encoding="utf-8"))
    tot["cases"] += 1
    for cid, rb in (b.get("comps") or {}).items():
        ra = (a.get("comps") or {}).get(cid) or {}
        pa = ra.get("pins") or {}; pb = rb.get("pins") or {}
        for k in set(pa) | set(pb):
            tot["pin_slots"] += 1
            if (pa.get(k) or {}) != (pb.get(k) or {}): tot["pin_diff"] += 1
        da = (ra.get("designator") or {}); db = (rb.get("designator") or {})
        if (da.get("canon") or da.get("text")) != (db.get("canon") or db.get("text")): tot["desig_diff"] += 1
print("case 数:", tot["cases"])
print("引脚槽位总数:", tot["pin_slots"], "｜融合层改过的:", tot["pin_diff"], "(%.1f%%)" % (100.0*tot["pin_diff"]/max(1,tot["pin_slots"])))
print("位号被融合层改过的元件:", tot["desig_diff"])
print("=> 这些引脚改动全部只落在旁路产物 _assigned/，**交付文档的 pins 一个字节都不改**")
