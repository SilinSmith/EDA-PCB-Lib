import json, os, collections
p = r"work\_fusion_out220cf\_fusion_report.json"
if not os.path.exists(p):
    print("报告还没写（run.py 在跑完一批后才写）")
else:
    d = json.load(open(p, encoding="utf-8"))
    rows = d if isinstance(d, list) else (d.get("cases") or d.get("rows") or [])
    print("报告条数:", len(rows))
    keys = set()
    for r in rows: keys |= set(r.keys())
    print("字段:", sorted(keys))
    cf = collections.Counter()
    n_run = 0
    for r in rows:
        c = r.get("cropfix") or {}
        if isinstance(c, dict):
            if c.get("n_run") or c.get("n") or c.get("asked"):
                n_run += 1
            for k, v in c.items():
                if isinstance(v, (int, float)) and v:
                    cf[k] += v
    print("有 cropfix 活动的 case:", n_run)
    print("cropfix 汇总:", dict(cf))
    # 打印一条完整样例
    for r in rows:
        c = r.get("cropfix") or {}
        if isinstance(c, dict) and c:
            print("样例", r.get("case"), json.dumps(c, ensure_ascii=False)[:500]); break
