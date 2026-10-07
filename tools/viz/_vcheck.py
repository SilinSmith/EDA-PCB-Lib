# -*- coding: utf-8 -*-
"""看 mkviz 产出的 <case>.json 里各数组的数量与样例（绘图自检用）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VIZ = ROOT / "work" / "viz"


def main():
    for cid in (sys.argv[1:] or ["0001"]):
        d = json.loads((VIZ / "cases" / ("%s.json" % cid)).read_text(encoding="utf-8"))
        print("\n==== %s  %sx%s scale=%s" % (cid, d.get("w"), d.get("h"), d.get("scale")))
        print("   score   = %.2f  分项 %s" % (
            float(d["score"].get("score") or 0),
            {k: round(float((d["score"].get(k) or {}).get("f1") or 0), 3)
             for k in ("component", "pin", "net_hypergraph", "net_line")}))
        for key in ("gt.comps", "ours.comps", "texts", "slots", "regions", "pinwins", "labels"):
            a, b = key.split(".") if "." in key else (key, None)
            v = d[a][b] if b else d[a]
            print("   %-12s n=%d" % (key, len(v or [])))
            for x in (v or [])[:2]:
                print("        %s" % json.dumps(x, ensure_ascii=False)[:300])
        print("   assets = %s" % d.get("assets"))


if __name__ == "__main__":
    main()
