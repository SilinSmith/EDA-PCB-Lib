import json, os
def load(p):
    return json.load(open(p, encoding="utf-8"))
base = r"work\_fusion_vlm"       # VLM 补强底稿
fus  = r"work\_fusion_out220nl2" # 融合产物（无 LLM 轮）
cases = sorted(f[:-5] for f in os.listdir(fus) if f.endswith(".json") and not f.startswith("_"))[:8]
print("%-10s %-12s %-12s %-12s" % ("case", "pins同?", "components同?", "nets同?"))
same_p = same_c = same_n = 0
for c in cases:
    a = load(os.path.join(base, c + ".json"))
    b = load(os.path.join(fus, c + ".json"))
    sp = a.get("pins") == b.get("pins")
    sc = a.get("components") == b.get("components")
    sn = a.get("nets") == b.get("nets")
    same_p += sp; same_c += sc; same_n += sn
    print("%-10s %-12s %-12s %-12s" % (c, sp, sc, sn))
print("---- 合计 %d 张：pins 相同 %d / components 相同 %d / nets 相同 %d" % (len(cases), same_p, same_c, same_n))
# 具体：融合到底动了什么
a = load(os.path.join(base, "0001.json")); b = load(os.path.join(fus, "0001.json"))
ka, kb = set(a["components"]), set(b["components"])
print("0001 元件键 新增:", sorted(kb-ka), "删除:", sorted(ka-kb))
for k in sorted(ka & kb):
    if a["components"][k] != b["components"][k]:
        print("   改:", k, a["components"][k], "->", b["components"][k])
print("0001 pins 键集相同:", set(a["pins"])==set(b["pins"]),
      "| pin 子键有差异的元件数:", sum(1 for k in a["pins"] if a["pins"].get(k) != b["pins"].get(k)))
