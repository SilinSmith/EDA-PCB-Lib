import json, os
for p in [r"work\_smoke_pin\_fusion_report.json",
          r"work\_fusion_out220nl2\_fusion_report.json",
          r"work\_fusion_out220_old\_fusion_report.json"]:
    if not os.path.exists(p):
        print(p, "-> 不存在"); continue
    d = json.load(open(p, encoding="utf-8"))
    r = d["cases"][0]
    print("==", p)
    print("   case=%s sec=%s vlm_doc=%s rounds=%d cropfix=%s" % (
        r.get("case"), r.get("sec"), r.get("vlm_doc"), len(r.get("rounds") or []),
        json.dumps(r.get("cropfix"), ensure_ascii=False)))
    for rd in (r.get("rounds") or []):
        print("     R%d judge=%s audit=%s rescan=%s n_susp=%s" % (
            rd.get("round"), (rd.get("judge") or {}).get("n_q") if rd.get("judge") else None,
            (rd.get("audit") or {}).get("n"), rd.get("n_task"), rd.get("n_susp")))
