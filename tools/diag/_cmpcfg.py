import json, os
for d in ["_fusion_out220nl","_fusion_out220nl2","_fusion_out220nl3","_fusion_out220cf"]:
    p = os.path.join("work", d, "_fusion_report.json")
    if not os.path.exists(p):
        print("%-20s 报告缺失" % d); continue
    j = json.load(open(p, encoding="utf-8"))
    rows = j["cases"]
    r0 = rows[0]
    llm = sum(1 for r in rows if r.get("rounds"))
    cf = sum(1 for r in rows if (r.get("cropfix") or {}).get("n_fix"))
    print("%-20s 张=%3d wall=%ss | 有 LLM 轮的=%3d | cropfix 改动过的=%2d | 样例 sec=%s" % (
        d, len(rows), j.get("wall_s"), llm, cf, r0.get("sec")))
