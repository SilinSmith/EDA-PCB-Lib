# -*- coding: utf-8 -*-
"""离线比较「VLM 值覆盖」的几种采纳策略：不重新调 VLM，只改 v18 的 value 再打分。

    python work\\tools\\diag\\_valpol.py --base work\\stage2_v17 --vlm work\\stage2_v18
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "work" / "tools" / "eval" / "stage2_eval.py"


def norm(s):
    return "" if s is None else str(s).strip()


def apply_policy(base_dir, vlm_dir, out_dir, policy):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(vlm_dir.glob("*.json")):
        bf = base_dir / f.name
        pb = json.loads(f.read_text(encoding="utf-8"))
        if bf.exists():
            pa = json.loads(bf.read_text(encoding="utf-8"))
            ca = pa.get("components") or {}
            for k, cv in (pb.get("components") or {}).items():
                if k not in ca:
                    continue
                va, vb = norm(cv.get("value")), norm((ca[k] or {}).get("value"))
                if va == vb:
                    continue
                same_ci = va.upper() == vb.upper()
                use_vb = {
                    "base": False,                       # 完全不用 VLM 的值
                    "all": True,                          # 一律用
                    "only_empty": (not vb),               # 只在管线读空时才用
                    "keep_ci": (same_ci or not vb),       # 只是大小写差异就保留管线
                    "fill_or_case": (not vb) or (same_ci and False),
                    "both_nonempty_prefer_vlm": True,
                }[policy]
                if not use_vb:
                    cv["value"] = ca[k].get("value")
        (out_dir / f.name).write_text(json.dumps(pb, ensure_ascii=False), encoding="utf-8")
    src = vlm_dir / "_assigned"
    if src.is_dir():
        dst = out_dir / "_assigned"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def score(d):
    r = subprocess.run([sys.executable, str(EVAL), str(d), "--quiet"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    first = r.stdout.strip().splitlines()
    return " | ".join(x.strip() for x in first[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--vlm", default=str(ROOT / "work" / "stage2_v18"))
    ap.add_argument("--policies", default="base,all,only_empty,keep_ci")
    a = ap.parse_args()
    base, vlm = Path(a.base), Path(a.vlm)
    for pol in [x.strip() for x in a.policies.split(",") if x.strip()]:
        out = ROOT / "work" / ("_vp_" + pol)
        if out.exists():
            shutil.rmtree(out)
        apply_policy(base, vlm, out, pol)
        print("[%-12s] %s" % (pol, score(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
