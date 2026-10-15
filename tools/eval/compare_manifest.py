"""Compare saved predictions against a fixed sample manifest.

This tool reads labels for evaluation only. It never produces predictions,
loads detector/OCR weights, or calls external services. The scorer is the
repository's reconstruction of the written specification, not an official
competition executable. SciPy Hungarian matching is mandatory for comparable
results. A fresh report path is required when --json is used.

Example::

    python tools/eval/compare_manifest.py --manifest samples.json --group dev \
        --prediction-dir before=/runs/before/final_results \
        --prediction-dir after=/runs/after/final_results --json comparison.json

The manifest must contain a samples list. Each sample needs unique case_id,
gt (absolute or relative to the manifest), and group (dev or verification).
Prediction files are exactly <case_id>_Pred.json inside each supplied folder.
Optional gt_sha256 fields are verified for selected samples. Unselected group
labels and predictions are not loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schematic_parser import metrics  # noqa: E402


METRIC_FIELDS = ("component_f1", "pin_f1", "net_hypergraph_f1", "net_line_f1")
COUNT_FIELDS = ("component", "pin", "net_hypergraph", "net_line")
GROUPS = ("dev", "verification")


def _prediction_arg(value):
    name, sep, directory = value.partition("=")
    if not sep or not name.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("Expected NAME=PATH for --prediction-dir")
    return name.strip(), Path(directory).expanduser().resolve()


def _read_doc(path: Path, label: str):
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    raw = path.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(doc, dict) or any(not isinstance(doc.get(k), dict) for k in ("components", "pins", "nets")):
        raise ValueError(f"Malformed {label}: expected components/pins/nets mappings in {path}")
    return doc, hashlib.sha256(raw).hexdigest()


def _load_samples(manifest_path: Path, group: str):
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid manifest JSON: {manifest_path}: {exc}") from exc
    samples = manifest.get("samples") if isinstance(manifest, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest samples must be a nonempty list")
    selected, seen = [], set()
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Each manifest sample must be a mapping")
        cid, gt, sample_group = row.get("case_id"), row.get("gt"), row.get("group")
        if not isinstance(cid, str) or not cid.strip() or cid != cid.strip() or cid in (".", "..") or any(ch in cid for ch in ("/", "\\", ":", "\x00")):
            raise ValueError(f"Unsafe or empty case_id: {cid!r}")
        if cid in seen:
            raise ValueError(f"Duplicate case_id in manifest: {cid}")
        seen.add(cid)
        if sample_group not in GROUPS:
            raise ValueError(f"Unknown group for {cid}: {sample_group!r}; expected dev or verification")
        if not isinstance(gt, str) or not gt.strip():
            raise ValueError(f"Missing gt path for {cid}")
        if group != "all" and sample_group != group:
            continue
        p = Path(gt).expanduser()
        if not p.is_absolute():
            p = manifest_path.parent / p
        selected.append(dict(row, gt_path=p.resolve()))
    if not selected:
        raise ValueError(f"No samples in requested group: {group}")
    return selected, hashlib.sha256(raw).hexdigest()


def _aggregate(scores):
    result = metrics.aggregate(scores)
    result["counts"] = {}
    for field in COUNT_FIELDS:
        tp = sum(getattr(score, field).tp for score in scores)
        fp = sum(getattr(score, field).fp for score in scores)
        fn = sum(getattr(score, field).fn for score in scores)
        result["counts"][field] = metrics.F1(tp, fp, fn).as_dict()
    return result


def _summary(scores, samples):
    groups = {}
    for group in GROUPS:
        ids = {s["case_id"] for s in samples if s["group"] == group}
        if ids:
            groups[group] = _aggregate([s for s in scores if s.case_id in ids])
    return {"aggregate": _aggregate(scores), "by_group": groups}


def _delta(before, after):
    keys = ("mean_score",) + METRIC_FIELDS if "mean_score" in before else ("score",) + METRIC_FIELDS
    out = {k: round(after[k] - before[k], 6) for k in keys}
    out["counts"] = {metric: {k: after["counts"][metric][k] - before["counts"][metric][k] for k in ("tp", "fp", "fn")} for metric in COUNT_FIELDS}
    return out


def compare(manifest_path, prediction_dirs, group="all", self_check=False, output=None,
            profile="latest"):
    """Evaluate selected files and optionally write one fresh report.

    Raises ValueError for missing files, duplicate arms, input/output aliases,
    malformed documents, manifest hash drift, or absent SciPy. No input is
    modified, and all selected files are preflighted before any evaluation.
    """
    config = metrics.EvalConfig.for_profile(profile)
    if not metrics._HAVE_SCIPY:
        raise ValueError("SciPy Hungarian matching is required; refusing the greedy fallback")
    import scipy
    manifest_path = Path(manifest_path).expanduser().resolve()
    if group not in ("all",) + GROUPS:
        raise ValueError(f"Unknown requested group: {group}")
    samples, manifest_hash = _load_samples(manifest_path, group)
    arms = []
    names = set()
    for name, directory in prediction_dirs:
        if not name or name in names:
            raise ValueError(f"Empty or duplicate prediction arm name: {name!r}")
        names.add(name)
        directory = Path(directory).expanduser().resolve()
        arms.append((name, directory))
    if not arms and not self_check:
        raise ValueError("Supply at least one --prediction-dir or --self-check")

    inputs = {manifest_path}
    for sample in samples:
        p = sample["gt_path"]
        inputs.add(p)
        if not p.is_file():
            raise ValueError(f"Missing GT for {sample['case_id']}: {p}")
        for name, directory in arms:
            p = directory / f"{sample['case_id']}_Pred.json"
            inputs.add(p.resolve())
            if not p.is_file():
                raise ValueError(f"Missing prediction for arm {name}, case {sample['case_id']}: {p}")
    if output is not None:
        output = Path(output).expanduser().resolve()
        if output in inputs:
            raise ValueError(f"Report cannot overwrite an input: {output}")
        if output.exists():
            raise ValueError(f"Report path already exists; use a fresh path: {output}")
        if not output.parent.is_dir():
            raise ValueError(f"Report parent directory does not exist: {output.parent}")

    docs, gt_hashes = {}, {}
    for sample in samples:
        cid = sample["case_id"]
        docs[cid], gt_hashes[cid] = _read_doc(sample["gt_path"], f"GT for {cid}")
        expected = sample.get("gt_sha256")
        if expected and gt_hashes[cid] != expected:
            raise ValueError(f"GT SHA256 mismatch for {cid}: {sample['gt_path']}")
    report = {
        "schema": "eda-pcb-manifest-comparison-v1",
        "evidence_boundary": "Saved predictions evaluated with the repository reconstructed scorer, not an official competition executable. Public data is not an unseen competition test set.",
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "selection": {"requested_group": group, "cases": len(samples), "groups": dict(Counter(s["group"] for s in samples))},
        "matching": {"backend": "scipy_hungarian", "scipy_version": scipy.__version__},
        "evaluator": {"profile": profile, "config": asdict(config), "weights": metrics.WEIGHTS},
        "predictions": {},
        "comparisons": {},
    }
    sample_by_id = {s["case_id"]: s for s in samples}
    if self_check:
        scores = [metrics.evaluate_case(docs[s["case_id"]], docs[s["case_id"]], s["case_id"], config) for s in samples]
        bad = [s.case_id for s in scores if abs(s.score - 100) > 1e-8]
        if bad:
            raise ValueError("GT self-check did not score 100: " + ", ".join(bad))
        report["self_check"] = dict(_summary(scores, samples), cases=[s.as_dict() for s in scores])
    for name, directory in arms:
        scores, cases = [], []
        for sample in samples:
            cid = sample["case_id"]
            pp = directory / f"{cid}_Pred.json"
            pred, pred_hash = _read_doc(pp, f"prediction for {name}/{cid}")
            score = metrics.evaluate_case(docs[cid], pred, cid, config)
            scores.append(score)
            cases.append(dict(score.as_dict(), group=sample["group"], gt=str(sample["gt_path"]), gt_sha256=gt_hashes[cid], prediction=str(pp), prediction_sha256=pred_hash))
        report["predictions"][name] = dict(_summary(scores, samples), directory=str(directory), cases=cases)
    if len(arms) > 1:
        baseline = arms[0][0]
        base = report["predictions"][baseline]
        base_cases = {r["case_id"]: r for r in base["cases"]}
        for name, _ in arms[1:]:
            candidate = report["predictions"][name]
            rows = []
            for row in candidate["cases"]:
                cid = row["case_id"]
                rows.append({"case_id": cid, "group": sample_by_id[cid]["group"], "baseline_score": base_cases[cid]["score"], "candidate_score": row["score"], "delta": _delta(base_cases[cid], row)})
            report["comparisons"][name] = {"baseline": baseline, "aggregate_delta": _delta(base["aggregate"], candidate["aggregate"]), "by_group_delta": {g: _delta(base["by_group"][g], candidate["by_group"][g]) for g in base["by_group"]}, "cases": rows}
    if output is not None:
        # Exclusive creation also protects against a file appearing after preflight.
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return report


def _print_report(report):
    print(f"matching={report['matching']['backend']} scipy={report['matching']['scipy_version']} requested_group={report['selection']['requested_group']} cases={report['selection']['cases']}")
    for name, result in report["predictions"].items():
        for group, summary in (("all evaluated", result["aggregate"]), *result["by_group"].items()):
            print(f"{name} {group}: n={summary['cases']} mean_score={summary['mean_score']:.4f} " + " ".join(f"{k}={summary[k]:.6f}" for k in METRIC_FIELDS))
    for name, comp in report["comparisons"].items():
        print(f"{name} versus {comp['baseline']}: mean_delta={comp['aggregate_delta']['mean_score']:+.6f}")
        print("case group before after delta component pin netH netL")
        for row in comp["cases"]:
            d = row["delta"]
            print(f"{row['case_id']} {row['group']} {row['baseline_score']:.4f} {row['candidate_score']:.4f} {d['score']:+.4f} " + " ".join(f"{d[k]:+.6f}" for k in METRIC_FIELDS))
    if "self_check" in report:
        print(f"GT self-check: n={report['self_check']['aggregate']['cases']} mean_score=100.0000")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--group", choices=("dev", "verification", "all"), default="all")
    parser.add_argument("--prediction-dir", action="append", type=_prediction_arg, default=[], metavar="NAME=PATH")
    parser.add_argument("--self-check", action="store_true", help="Require selected GT documents to score 100 against themselves")
    parser.add_argument("--profile", choices=metrics.RULE_PROFILES, default="latest")
    parser.add_argument("--json", "--output", dest="output", type=Path, help="Optional fresh JSON report path (never overwritten)")
    args = parser.parse_args(argv)
    try:
        report = compare(args.manifest, args.prediction_dir, args.group, args.self_check, args.output, args.profile)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
