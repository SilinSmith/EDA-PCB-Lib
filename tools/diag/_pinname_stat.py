import json, sys, glob, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json
cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == "200_train_cases"]
st = collections.Counter()
for c in cs[:60]:
    p = r"Y:\workspace\EDA\work\_submit_out220r36\final_results\%s_Pred.json" % c.case_id
    try: d = json.load(open(p, encoding="utf-8"))
    except Exception: continue
    for comp, pins in (d.get("pins") or {}).items():
        for pk, v in (pins or {}).items():
            st["pins"] += 1
            nm = str((v or {}).get("pinname") or "")
            num = pk.replace("pin_", "").split("_")[0]
            st["name_is_num"] += (nm == num)
            st["name_digit"] += nm.isdigit()
            st["name_empty"] += (nm.strip() == "")
print(dict(st))
print("预测引脚总数 %d，其中 pinname 就是引脚号 %d（%.1f%%），纯数字 %d（%.1f%%）" % (
    st["pins"], st["name_is_num"], 100.0*st["name_is_num"]/max(1,st["pins"]),
    st["name_digit"], 100.0*st["name_digit"]/max(1,st["pins"])))
