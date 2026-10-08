import sys, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json
from schematic_parser.metrics import IC_TYPES, canon_type, normalise_doc
cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == "200_train_cases"]
st = collections.Counter(); lens = collections.Counter(); ex = []
for c in cs[:60]:
    gt = normalise_doc(load_json(c.json_path))
    box = {k for k, v in (gt.get("components") or {}).items()
           if canon_type((v or {}).get("type")) in IC_TYPES}
    for comp in box:
        for pk, g in ((gt.get("pins") or {}).get(comp) or {}).items():
            if not isinstance(g, dict): continue
            nm = str(g.get("pinname") or "")
            st["gt_pins"] += 1
            st["digit"] += nm.isdigit()
            st["empty"] += (nm.strip() == "")
            st["has_slash"] += ("/" in nm)
            lens[len(nm)] += 1
            if not nm.isdigit() and len(ex) < 15: ex.append((c.case_id, comp, pk, nm))
print("GT IC 引脚 %d 根：纯数字名 %d（%.1f%%）、空名 %d、含斜杠 %d" % (
    st["gt_pins"], st["digit"], 100.0*st["digit"]/max(1,st["gt_pins"]), st["empty"], st["has_slash"]))
print("名字长度分布（前 10）：", lens.most_common(10))
print("非数字名样本：", ex[:12])
