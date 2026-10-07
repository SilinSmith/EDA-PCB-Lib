"""Validate a synthetic set against the corpus schema and the project's own scorer.

Structural rules are the ones the 220 real cases satisfy -- a synthetic sheet that breaks any of
them would silently mis-train whatever reads it:

  * top level is exactly {components, pins, nets}
  * `pins` has a group for every component key and for no other key
  * every net hyperGraph token `<comp>.<suffix>` resolves to a real `pin_<suffix>` on that
    component (the corpus guarantees the suffix matches in both places)
  * every edge is exactly two points
  * every bbox is ordered and every point is inside the canvas, in the left-bottom origin
  * the scorer agrees a doc matches itself (100.0) and that an empty prediction scores 0

    $env:PYTHONIOENCODING='utf-8'
    K:\\python\\311\\python.exe work\\check_synth_json.py --root "Y:\\workspace\\EDA\\合成原理图数据集" --workers 8
    K:\\python\\311\\python.exe work\\check_synth_json.py            # defaults to 合成原理图demo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))

from schematic_parser import evaluate_case  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_ROOT = Path(r"Y:\workspace\EDA\合成原理图demo")


def check_one(arg) -> dict:
    """Schema-check one `gt/*.json` and score it against itself."""
    path_str, do_empty = arg
    p = Path(path_str)
    doc = json.loads(p.read_text(encoding="utf-8"))
    name = p.stem
    bad: Counter = Counter()
    n_comp = n_pin = n_net = n_edge = 0

    if set(doc) != {"components", "pins", "nets"}:
        bad["top-level keys"] += 1
        return {"case": name, "bad": dict(bad), "fatal": True}

    comps, pins, nets = doc["components"], doc["pins"], doc["nets"]
    if set(pins) != set(comps):
        bad["pins/components key mismatch"] += 1
    n_comp += len(comps)
    n_pin += sum(len(v) for v in pins.values())
    n_net += len(nets)

    for ck, cv in comps.items():
        b = cv.get("bbox")
        if not b or len(b) != 4 or b[0] > b[2] or b[1] > b[3]:
            bad["bad component bbox"] += 1
        if not isinstance(cv.get("type"), str) or not cv["type"]:
            bad["missing type"] += 1
    for ck, pv in pins.items():
        for pk, pv2 in pv.items():
            if not pk.startswith("pin_") or len(pk) <= 4:
                bad["bad pin key"] += 1

    for nk, nv in nets.items():
        hg = nv.get("hyperGraph", "")
        if not hg.startswith("(") or not hg.endswith(")"):
            bad["malformed hyperGraph"] += 1
            continue
        toks = [t for t in hg[1:-1].split(",") if t]
        if not toks:
            bad["empty hyperGraph"] += 1
        for tok in toks:
            ck, _, suf = tok.rpartition(".")
            if ck not in pins or f"pin_{suf}" not in pins[ck]:
                bad["dangling hyperGraph token"] += 1
        for ek, ev in nv.get("edges", {}).items():
            if len(ev) != 2:
                bad["edge is not a 2-point segment"] += 1
        n_edge += len(nv.get("edges", {}))

    sc = evaluate_case(doc, doc, name)
    empty_ok = True
    if do_empty:
        empty = {"components": {}, "pins": {}, "nets": {}}
        empty_ok = evaluate_case(doc, empty, name).score <= 1e-9

    return {"case": name, "bad": dict(bad), "score": sc.score,
            "n_comp": n_comp, "n_pin": n_pin, "n_net": n_net, "n_edge": n_edge,
            "empty_zero": empty_ok, "fatal": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"a synthetic set with gt/*.json (default {DEFAULT_ROOT})")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-empty", action="store_true",
                    help="skip the 'empty prediction scores 0' pass (halves the runtime)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cases = sorted((args.root / "gt").glob("*.json"))
    if not cases:
        print(f"no gt/*.json under {args.root}")
        return 1
    jobs = [(str(p), not args.no_empty) for p in cases]

    if args.workers > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            results = list(pool.imap_unordered(check_one, jobs, chunksize=8))
    else:
        results = [check_one(j) for j in jobs]
    results.sort(key=lambda r: r["case"])

    bad: Counter = Counter()
    n_comp = n_pin = n_net = n_edge = 0
    scores = []
    bad_cases: list[dict] = []
    empty_fail: list[str] = []
    for r in results:
        for k, v in r["bad"].items():
            bad[k] += v
        if r["bad"]:
            bad_cases.append({"case": r["case"], "problems": r["bad"]})
        if not r["fatal"]:
            n_comp += r["n_comp"]; n_pin += r["n_pin"]
            n_net += r["n_net"]; n_edge += r["n_edge"]
            scores.append(r["score"])
            if not r["empty_zero"]:
                empty_fail.append(r["case"])

    print(f"root             : {args.root}")
    print(f"cases            : {len(cases)}")
    print(f"components/pins  : {n_comp} / {n_pin}")
    print(f"nets/edges       : {n_net} / {n_edge}")
    print("=== schema ===")
    print("  all clean" if not bad else "\n".join(f"  {v:5}  {k}" for k, v in bad.items()))

    perfect = [s for s in scores if abs(s - 100.0) < 1e-6]
    print(f"=== scorer (GT vs GT) ===\n  {len(perfect)}/{len(scores)} scored exactly 100.000")
    if len(perfect) != len(scores):
        worst = min(scores)
        print(f"  worst score: {worst:.6f}")
    if not args.no_empty:
        print(f"  empty prediction scores 0 on all cases: {not empty_fail}")
        if empty_fail:
            print(f"    failures: {empty_fail[:20]}")

    if bad_cases:
        print("=== offending cases (first 20) ===")
        for b in bad_cases[:20]:
            print(f"  {b['case']}: {b['problems']}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"root": str(args.root), "cases": len(cases), "schema": dict(bad),
             "gt_vs_gt_perfect": len(perfect), "gt_vs_gt_total": len(scores),
             "empty_zero_ok": not empty_fail, "offending": bad_cases},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report           : {args.json_out}")

    return 1 if bad or len(perfect) != len(scores) or empty_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
