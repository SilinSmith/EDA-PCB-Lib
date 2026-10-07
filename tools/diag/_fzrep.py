"""一页看融合层的报告：逐 case 打印 cropfix / judge / audit / rescan / 分数。"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1
         else r"Y:\workspace\EDA\work\_fusion_out\_fusion_report.json")
d = json.loads(p.read_text(encoding="utf-8"))
cases = d.get("cases") if isinstance(d, dict) else d
if not isinstance(cases, list):
    cases = [cases]
print("case    slots cf=sus/fix/gain  judge=q/ans/new/rej/skip  "
      "audit=n/bad/sus/ok/add  rescan=plan/run  susp  sec   before→final")
for r in cases:
    cf = r.get("cropfix") or {}
    r0 = (r.get("rounds") or [{}])[0]
    j = r0.get("judge") or {}
    au = r0.get("audit") or {}
    cfs = "%s/%s/%s" % (cf.get("n_sus", 0), cf.get("n_fix", 0), cf.get("n_gain", 0))
    js = "%s/%s/%s/%s/%s" % (j.get("n_q", 0), j.get("n_answered", 0),
                             j.get("n_new", 0), j.get("n_reject", 0),
                             j.get("n_skip_ph", 0))
    aus = "%s/%s/%s/%s/%s" % (au.get("n", 0), au.get("n_bad", 0),
                              au.get("n_suspect", 0), au.get("n_ok", 0),
                              au.get("n_add", 0))
    rss = "%s/%s" % (r0.get("n_task", 0), r0.get("n_susp", 0))
    sb = (r.get("score_before") or {}).get("n_assigned")
    sf = (r.get("score_final") or {}).get("n_assigned")
    print("%-6s %5s  %-14s  %-22s  %-21s  %-12s  %4s  %5s  %s -> %s" % (
        r.get("case"), r.get("n_slots"), cfs, js, aus, rss,
        r0.get("n_susp", 0), r.get("sec"), sb, sf))
