import json
d = json.load(open(r"work\_fusion_out220no4\_score.json", encoding="utf-8"))
cs = d["cases"]
cs.sort(key=lambda r: r["score"])
print("张数", len(cs), " 最低", round(cs[0]["score"],2), " 最高", round(cs[-1]["score"],2))
print("最低 12 张:", [(r["case_id"], round(r["score"],1)) for r in cs[:12]])
print("中位 6 张:", [(r["case_id"], round(r["score"],1)) for r in cs[len(cs)//2-3:len(cs)//2+3]])
print("最高 8 张:", [(r["case_id"], round(r["score"],1)) for r in cs[-8:]])
