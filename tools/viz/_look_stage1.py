import json, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
print("keys:", list(d))
c = d["components"]
print("n_comp", len(c))
print("types:", Counter(x["type"] for x in c).most_common(14))
for want in ("testpoint",):
    sub = [x for x in c if x["type"] == want]
    print(f"\n{want} n = {len(sub)}")
    for x in sub[:8]:
        print("  ", x["id"], x["proposal"], x["source"], "score", x["score"],
              "alts", x["n_alts"], "pending", x["pending"], x["bbox_img"])
print("\npending total:", sum(1 for x in c if x["pending"]))
print("timing:", d["timing"])
print("wires:", {k: v for k, v in d["wires"].items() if k != "polylines"})
print("stats:", d["stats"])
print("attach sample:", d["attachments"][:2] if d["attachments"] else None)
hy = d.get("hypotheses") or {}
print("hyps: slots", len(hy.get("slots", {})), "folds", hy.get("folds"))
