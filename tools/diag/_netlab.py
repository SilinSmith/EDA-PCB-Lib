# -*- coding: utf-8 -*-
"""看我们的产物里网络标签到底有没有被用上。

`final_results/<case>_Pred.json` 的 `nets[*].label` 可能是 str 也可能是 dict，
`runs/stage2/_assigned/<case>.json` 的 `net_labels` 是 OCR 出来的候选框。

用法：
    python work\\tools\\diag\\_netlab.py --dir work\\_fast_out_wtol_def
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def lab_text(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        return str(v.get("text") or v.get("name") or v.get("label") or "").strip()
    return str(v).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "work", "_fast_out_wtol_def"))
    a = ap.parse_args()

    fdir = os.path.join(a.dir, "final_results")
    adir = os.path.join(a.dir, "runs", "stage2", "_assigned")
    cases = sorted(fn.split("_")[0] for fn in os.listdir(fdir) if fn.endswith("_Pred.json"))
    tot_n = tot_l = 0
    for case in cases:
        d = json.load(open(os.path.join(fdir, case + "_Pred.json"), encoding="utf-8"))
        nets = d.get("nets") or {}
        items = list(nets.items()) if isinstance(nets, dict) else list(enumerate(nets))
        labs = []
        for _k, n in items:
            t = lab_text((n or {}).get("label"))
            if t:
                labs.append(t)
        tot_n += len(items)
        tot_l += len(labs)
        cand = []
        p = os.path.join(adir, case + ".json")
        if os.path.exists(p):
            j = json.load(open(p, encoding="utf-8"))
            cand = [str(x.get("text")) for x in (j.get("net_labels") or [])]
        print("case %-4s 网络 %-3d 带 label %-6d | 标签 %s"
              % (case, len(items), len(labs), sorted(set(labs))[:14]))
        print("        _assigned.net_labels OCR 候选 (%d)：%s" % (len(cand), cand[:20]))
    print("\n合计：网络 %d · 带 label %d（%.1f%%）" % (tot_n, tot_l, 100.0 * tot_l / max(1, tot_n)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
