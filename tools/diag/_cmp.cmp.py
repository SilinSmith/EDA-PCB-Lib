import json, pathlib
B = pathlib.Path("work/_fast_base")
def load(n): return {r["case"]: r for r in json.loads((B/f"{n}.json").read_text(encoding="utf-8"))}
a, b = load("jsyn3"), load("recheck")
print("%-7s %-9s %-22s %-22s" % ("case", "split", "jsyn3 comp tp/fp/fn f1", "recheck comp tp/fp/fn f1"))
for c in a:
    if c not in b: continue
    ca, cb = a[c]["component"], b[c]["component"]
    ta = (ca["tp"], ca["fp"], ca["fn"], round(ca["f1"],4))
    tb = (cb["tp"], cb["fp"], cb["fn"], round(cb["f1"],4))
    mark = "  <<<" if ta != tb else ""
    print("%-7s %-9s %-22s %-22s%s" % (c, a[c]["split"], ta, tb, mark))
print()
print("0002 jsyn3 diag :", a["0002"]["diag"])
print("0002 recheck diag:", b["0002"]["diag"])
