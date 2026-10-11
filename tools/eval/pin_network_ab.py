"""Compare saved predictions on all 10 official GT examples and historical 9.

This uses the repository's reconstructed scorer, not an official executable.
It only reads the explicitly supplied files; it never generates predictions,
invokes a model, or reuses a saved score report.

Example (from any working directory)::

    python pin_network_ab.py --gt-root /data/10GT \
        --prediction-dir before=/runs/before/final_results \
        --prediction-dir after=/runs/after/final_results --json comparison.json

Use --self-check without prediction directories to evaluate GT against itself.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schematic_parser import metrics  # noqa: E402
from schematic_parser.data import load_json  # noqa: E402


CASE_IDS = tuple(str(i) for i in range(1, 11))
HISTORICAL_IDS = tuple(cid for cid in CASE_IDS if cid != "6")
METRIC_FIELDS = ("component_f1", "pin_f1", "net_hypergraph_f1", "net_line_f1")


def _unique_file(candidates, label):
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise ValueError(f"Missing {label}: expected " + " or ".join(map(str, candidates)))
    if len(existing) != 1:
        raise ValueError(f"Ambiguous {label}: multiple exact files exist: "
                         + ", ".join(map(str, existing)))
    return existing[0]


def _prediction_arg(value):
    name, separator, directory = value.partition("=")
    if not separator or not name.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("Expected NAME=PATH for --prediction-dir")
    return name.strip(), Path(directory).expanduser().resolve()


def _summarise(scores, files):
    return {
        "aggregate_10": metrics.aggregate(scores),
        "aggregate_9_excluding_6": metrics.aggregate(
            [score for score in scores if score.case_id in HISTORICAL_IDS]),
        "cases": [dict(score.as_dict(), **files[score.case_id]) for score in scores],
    }


def _print_result(name, result):
    print(f"\n{name}")
    print("case    score   component      pin     netH     netL")
    for row in result["cases"]:
        print(f"{row['case_id']:>4} {row['score']:8.4f} "
              + " ".join(f"{row[field]:8.6f}" for field in METRIC_FIELDS))
    for label, key in (("all 10", "aggregate_10"),
                       ("historical 9 (exclude 6)", "aggregate_9_excluding_6")):
        summary = result[key]
        print(f"{label}: cases={summary['cases']} mean_score={summary['mean_score']:.4f} "
              + " ".join(f"{field}={summary[field]:.6f}" for field in METRIC_FIELDS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root", required=True, type=Path,
                        help="Official 10GT directory containing 1/1.json ... 10/10.json")
    parser.add_argument("--prediction-dir", action="append", default=[],
                        type=_prediction_arg, metavar="NAME=PATH",
                        help="Repeat for each final_results directory; exact 1_Pred.json "
                             "or 1.json names only, no numeric alias matching")
    parser.add_argument("--self-check", action="store_true",
                        help="Evaluate all 10 GT documents against themselves and require 100")
    parser.add_argument("--json", type=Path, dest="json_output",
                        help="Optional fresh JSON report path")
    args = parser.parse_args(argv)
    if not args.prediction_dir and not args.self_check:
        parser.error("Supply --prediction-dir NAME=PATH or --self-check")

    gt_root = args.gt_root.expanduser().resolve()
    try:
        if not gt_root.is_dir():
            raise ValueError(f"GT directory does not exist: {gt_root}")
        gt_files = {
            cid: _unique_file((gt_root / cid / f"{cid}.json", gt_root / f"{cid}.json"),
                              f"GT case {cid}")
            for cid in CASE_IDS
        }
        # Validate every input before scoring or writing a partial result.
        prediction_files = {}
        for name, directory in args.prediction_dir:
            if name in prediction_files:
                raise ValueError(f"Duplicate prediction name: {name}")
            if not directory.is_dir():
                raise ValueError(f"Prediction directory does not exist: {directory}")
            prediction_files[name] = {
                cid: _unique_file((directory / f"{cid}_Pred.json", directory / f"{cid}.json"),
                                  f"prediction {name}, case {cid}")
                for cid in CASE_IDS
            }

        gt_docs = {cid: load_json(str(path)) for cid, path in gt_files.items()}
        config = metrics.EvalConfig()
        report = {
            "evaluator": {
                "implementation": "schematic_parser.metrics",
                "status": "Repository reconstructed evaluator; not an official scorer",
                "matching_backend": "scipy_hungarian" if metrics._HAVE_SCIPY else "greedy_fallback",
                "config": asdict(config),
                "weights": dict(metrics.WEIGHTS),
                "aggregation": "Arithmetic mean of per-case scores and per-case F1",
            },
            "gt_root": str(gt_root),
            "case_ids_10": list(CASE_IDS),
            "historical_case_ids_9": list(HISTORICAL_IDS),
            "predictions": {},
        }
        if args.self_check:
            scores = [metrics.evaluate_case(gt_docs[cid], gt_docs[cid], case_id=cid, cfg=config)
                      for cid in CASE_IDS]
            failures = [score.case_id for score in scores
                        if not math.isclose(score.score, 100.0, abs_tol=1e-8)]
            if failures:
                raise ValueError("GT-vs-GT self-check failed for cases: " + ", ".join(failures))
            report["self_check"] = _summarise(scores, {
                cid: {"gt_file": str(gt_files[cid]), "prediction_file": str(gt_files[cid])}
                for cid in CASE_IDS})
        for name, files in prediction_files.items():
            scores = [metrics.evaluate_case(gt_docs[cid], load_json(str(files[cid])),
                                            case_id=cid, cfg=config) for cid in CASE_IDS]
            report["predictions"][name] = _summarise(scores, {
                cid: {"gt_file": str(gt_files[cid]), "prediction_file": str(files[cid])}
                for cid in CASE_IDS})
            report["predictions"][name]["directory"] = str(files[CASE_IDS[0]].parent)

        print("Repository reconstructed evaluator; not an official scorer.")
        print("Matching backend: " + report["evaluator"]["matching_backend"])
        if args.self_check:
            _print_result("GT-vs-GT self-check", report["self_check"])
        for name, result in report["predictions"].items():
            _print_result(name, result)
        if args.json_output:
            output = args.json_output.expanduser().resolve()
            # The report must not overwrite an input JSON document.
            all_inputs = set(gt_files.values())
            all_inputs.update(path for files in prediction_files.values() for path in files.values())
            if output in all_inputs:
                raise ValueError(f"Report path is an input document: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"\nJSON report: {output}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
