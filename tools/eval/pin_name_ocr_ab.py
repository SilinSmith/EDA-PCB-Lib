"""Controlled pin-name OCR A/B with fresh local OCR and fixed upstream inputs.

Initial OCR, component/pin/wire geometry and GT remain frozen. Legacy local
number/text/lane crops are actually reread; off must reproduce the delivered
seventh-round JSON before on is evaluated. GT is evaluation only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
sys.path.insert(0, str(REPO))

from tools.eval.pin_lane_repair_ab import (
    aggregate, clean, counts, network_refs, output_correct_pairs, read, save, sha,
    FrozenInitialEngine,
)

BASELINE_COMMIT = "85b364f"


def input_hash_traces(value, path=""):
    """Keep actual input SHA metadata, irrespective of adapter field nesting."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            branch = f"{path}.{key}" if path else key
            if (isinstance(child, str) and "sha" in key.lower()
                    and any(word in branch.lower() for word in ("input", "tensor"))):
                assert len(child) == 64 and all(c in "0123456789abcdef" for c in child.lower())
                found.append({"path": branch, "sha256": child})
            found += input_hash_traces(child, branch)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found += input_hash_traces(child, f"{path}[{index}]")
    return found


def bound_slots(assigned, slot):
    return {(cid, str(index)): pin.get(slot)
            for cid, component in assigned.get("comps", {}).items()
            for index, pin in component.get("pins", {}).items() if pin.get(slot)}


def audit_name_records(records, min_conf=0.0):
    for record in records:
        assert set(record.get("reads", {})) <= {"pin_name"}, "new name OCR leaked another role"
        for text, conf in record.get("reads", {}).values():
            assert float(conf) >= min_conf, "new name OCR bypassed min_conf"
    return {"records": len(records), "role_limited": True,
            "min_conf_respected": True, "input_sha_traces": input_hash_traces(records)}


def duplicate_winner_checks(stage, assignments, index_maps):
    checks = []
    for comp in stage.get("components", []):
        if comp.get("type") != "box":
            continue
        cid = comp["id"]
        by_number = {}
        for index, pin in assignments["off"].get("comps", {}).get(cid, {}).get("pins", {}).items():
            number = (pin.get("number") or {}).get("text")
            if number:
                by_number.setdefault(number, []).append(str(index))
        for number, members in by_number.items():
            if len(members) <= 1:
                continue
            winners = {arm: sorted(index for index in members
                                   if index_maps[arm].get(cid, {}).get(index) == f"pin_{number}")
                       for arm in ("off", "on")}
            checks.append({"component_id": cid, "number": number, "observed_indices": members,
                           "winners": winners, "identical": winners["off"] == winners["on"]})
            assert winners["off"] == winners["on"], f"new name confidence changed duplicate-number winner: {cid}/{number}"
    return checks


def change_run_mode(run, args, enabled):
    if not hasattr(run, args.run_flag):
        raise AssertionError(f"stage2.run missing configured switch {args.run_flag}")
    old = getattr(run, args.run_flag)
    value = args.name_mode if enabled else "off"
    if isinstance(old, bool):
        value = enabled
    setattr(run, args.run_flag, value)
    os.environ[args.run_env] = args.name_mode if enabled else "off"


def integration_checks(args, engine, sources, expected, result):
    from stage2 import run
    reader = importlib.import_module("stage2.pin_name_reading")
    old_reader = reader.read_pin_names
    captured = []

    def capture_names(*positional, **keywords):
        records = old_reader(*positional, **keywords)
        captured.extend(copy.deepcopy(records))
        return records

    prior_pincol = run.PINCOL_WIN_TEXTS
    prior_flag = getattr(run, args.run_flag, None)
    prior_env = os.environ.get(args.run_env)
    checks = []
    requested = {c.strip() for c in args.process_cases.split(",") if c.strip()}
    options_requested = {c.strip() for c in args.process_option_cases.split(",") if c.strip()}
    try:
        reader.read_pin_names = capture_names
        run.PINCOL_WIN_TEXTS = False  # frozen initial OCR already includes captured windows
        for src in sources:
            case = src["case_id"]
            if case not in requested:
                continue
            initial = read(args.fixtures / "reads" / f"{case}.reads.json")
            for arm in ("off", "on"):
                captured.clear()
                change_run_mode(run, args, arm == "on")
                proxy = FrozenInitialEngine(engine, initial)
                target = args.out / "integration" / arm
                started = time.perf_counter()
                stats = run.process(case, args.stage1, target, proxy, texts="stage1", recut="off")
                got = read(target / f"{case}.json")
                equal = got == expected[(case, arm)]
                actual_index = read(target/f"{case}.pinindex.json")
                expected_index = read(args.out/arm/f"{case}.pinindex.json")
                assert actual_index == expected_index, f"{case}/{arm}: process pinindex differs from replay"
                audit = audit_name_records(captured)
                refs = network_refs(got)
                entry = {"case": case, "arm": arm, "identical_to_replay": equal,
                         "wall_seconds": time.perf_counter()-started, "statistics": stats,
                         "initial_read_calls": proxy.calls, "name_reads_audit": audit,
                         "network_references": refs}
                checks.append(entry)
                save(args.out / "integration/process-integration.json", checks)
                assert equal, f"real process differs from controlled {arm} replay in case {case}"
                assert refs["all_valid"] and len(proxy.calls) == 1
                if arm == "off":
                    assert not captured, "disabled name reader was invoked"
        options = (("pin_name_only", ["pin_name"], True, 0.0),
                   ("pin_number_only", ["pin_number"], True, 0.0),
                   ("custom_roles", ["designator", "pin_number"], True, .5),
                   ("custom_with_name", ["designator", "pin_name"], True, .5),
                   ("high_min_conf", None, True, .95),
                   ("no_ocr", None, False, 0.0))
        for src in sources:
            case = src["case_id"]
            if case not in options_requested:
                continue
            initial = read(args.fixtures / "reads" / f"{case}.reads.json")
            for label, roles, do_ocr, minimum in options:
                outputs, assigned, audits = {}, {}, {}
                for arm in ("off", "on"):
                    captured.clear()
                    change_run_mode(run, args, arm == "on")
                    proxy = FrozenInitialEngine(engine, initial)
                    target = args.out / "integration/options" / label / arm
                    stats = run.process(case, args.stage1, target, proxy, roles=roles,
                                        do_ocr=do_ocr, min_conf=minimum, texts="stage1", recut="off")
                    outputs[arm] = read(target / f"{case}.json")
                    assigned[arm] = read(target / "_assigned" / f"{case}.json")
                    audits[arm] = audit_name_records(captured, minimum)
                    assert len(proxy.calls) == (1 if do_ocr else 0)
                    assert network_refs(outputs[arm])["all_valid"]
                    if not do_ocr or (roles is not None and "pin_name" not in roles) or arm == "off":
                        assert not captured, f"{case}/{label}/{arm} invoked disabled new name OCR"
                disabled = {}
                for slot, role in (("name", "pin_name"), ("number", "pin_number")):
                    if not do_ocr or (roles is not None and role not in roles):
                        assert bound_slots(assigned["off"], slot) == bound_slots(assigned["on"], slot), \
                            f"{case}/{label} changed legacy binding fallback for disabled {role}"
                        disabled[role] = {"identical_to_off": True,
                                          "meaning": "OCR role selection only; legacy binder may borrow other allowed roles"}
                if not do_ocr or (roles is not None and "pin_name" not in roles):
                    assert outputs["off"] == outputs["on"], f"{case}/{label} changed final JSON without name OCR"
                checks.append({"case": case, "mode": label, "roles": roles,
                               "do_ocr": do_ocr, "min_conf": minimum, "passed": True,
                               "name_reads_audit": audits, "disabled_role_fallback": disabled})
                save(args.out / "integration/process-integration.json", checks)
        # A missing local backend must produce the unchanged fallback. The
        # adapter's imported getter can be replaced without deleting models.
        if hasattr(reader, "get_pin_name_ocr"):
            original_getter = reader.get_pin_name_ocr
            try:
                reader.get_pin_name_ocr = lambda mode="auto": None if mode == "auto" else original_getter(mode)
                for src in sources:
                    case = src["case_id"]
                    if case not in options_requested:
                        continue
                    initial = read(args.fixtures / "reads" / f"{case}.reads.json")
                    captured.clear()
                    # This is an AUTO fallback simulation. Explicit ON must
                    # raise when unavailable and is checked in the old runtime.
                    setattr(run, args.run_flag, "auto")
                    os.environ[args.run_env] = "auto"
                    proxy = FrozenInitialEngine(engine, initial)
                    target = args.out / "integration/fallback_missing_backend"
                    run.process(case, args.stage1, target, proxy, texts="stage1", recut="off")
                    got = read(target / f"{case}.json")
                    assert got == expected[(case, "off")], f"case {case}: missing backend changed fallback JSON"
                    assert not captured
                    checks.append({"case": case, "mode": "simulated_auto_missing_backend", "simulation": True,
                                   "identical_to_off": True})
            finally:
                reader.get_pin_name_ocr = original_getter
        else:
            result["fallback_check"] = "pending: reader does not expose get_pin_name_ocr; no missing-backend claim"
    finally:
        reader.read_pin_names = old_reader
        run.PINCOL_WIN_TEXTS = prior_pincol
        if prior_flag is not None:
            setattr(run, args.run_flag, prior_flag)
        if prior_env is None:
            os.environ.pop(args.run_env, None)
        else:
            os.environ[args.run_env] = prior_env
    result["process_integration"] = checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=WORK / "output/pin-name-direction-20261009")
    ap.add_argument("--stage1", type=Path, default=WORK / "output/pin-color-rule-20261009/stage1")
    ap.add_argument("--previous", type=Path, default=WORK / "output/pin-lane-repair-20261010/evaluation-final-complete")
    ap.add_argument("--physical-traces", type=Path, default=WORK / "output/pin-color-rule-20261009/root-causes/all_physical_pin_traces.json")
    ap.add_argument("--out", type=Path, default=WORK / "output/pin-ocr-next-20261010/evaluation")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--name-mode", default="auto")
    ap.add_argument("--run-flag", default="PIN_NAME_OCR")
    ap.add_argument("--run-env", default="STAGE2_PIN_NAME_OCR")
    ap.add_argument("--process-cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--process-option-cases", default="6,8")
    ap.add_argument("--skip-process", action="store_true")
    args = ap.parse_args()
    capture = read(args.fixtures / "capture-verification.json")
    sources_doc = read(args.fixtures / "source-verification.json")
    os.environ.update(capture["configuration"])
    os.environ["STAGE2_PIN_NAME_EDGE_MIDPOINT"] = "0"
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(4)
    from stage2 import assemble
    from stage2.text import TextEngine
    from stage2.color_crops import read_color_numbers
    from stage2.pin_text_crops import read_pin_texts
    from stage2.pin_lane_reading import read_pin_lanes
    from stage2.pin_name_reading import read_pin_names
    from stage2.pin_name_ocr import get_pin_name_ocr
    from tools.eval.color_pin_rule_ab import physical_side
    from schematic_parser import metrics
    binder = importlib.import_module("stage2.assign")
    assert sha(capture["ocr_weight"]) == capture["ocr_weight_sha256"]
    assert sha(sources_doc["type_spec_path"]) == sources_doc["type_spec_sha256"]
    production_files = sorted(str(p.relative_to(REPO)).replace("\\", "/")
                              for directory in ("stage2", "textrec") for p in (REPO/directory).glob("*.py"))
    production_files += ["schematic_parser/metrics.py", "schematic_parser/data.py"]
    source_hashes = {f: sha(REPO/f) for f in production_files}
    script_hash = sha(Path(__file__).resolve())
    provider = get_pin_name_ocr(args.name_mode)
    provider_metadata = dict(getattr(provider, "metadata", {}) or {})
    weight_files = {str(p): sha(p) for p in (REPO/"weights").rglob("*") if p.is_file()}
    # Include local PP artifacts even if the adapter keeps them outside weights.
    for directory in (REPO/"models", WORK/"models"):
        if directory.exists():
            weight_files.update({str(p): sha(p) for p in directory.rglob("*") if p.is_file()})
    provider_model = provider_metadata.get("model_path")
    if provider_model:
        weight_files[str(provider_model)] = sha(Path(provider_model))
        assert weight_files[str(provider_model)] == provider_metadata["model_sha256"]
    trace_map = {(r["case"], r["component"], r["pin"]): r for r in read(args.physical_traces)["rows"]}
    requested = {c.strip() for c in args.cases.split(",") if c.strip()}
    sources = [s for s in sources_doc["cases"] if s["case_id"] in requested]
    assert len(sources) == len(requested)
    previous_hashes = read(args.previous/"pin-lane-repair-ab.json")["input_hashes"]
    engine = TextEngine(capture["ocr_weight"], device="cpu", allow_missing=False)
    result = {"baseline_commit": BASELINE_COMMIT,
              "evidence_boundary": "Fixed observed stage1 and initial OCR; legacy and new local crops actually reread. Specified 10GT is a development set, not independent hidden testing or online scoring. GT is evaluation only.",
              "coordinate_free_definition": "Exact component key, pin_n and pinname; all coordinates ignored; micro P/R/F1.",
              "physical_binding_definition": "Evaluation-only fixed GT-to-observed physical pin trace, separate bound name/number correctness.",
              "source_hashes": source_hashes, "evaluation_script_sha256": script_hash,
              "weight_hashes": weight_files,
              "name_provider_metadata": provider_metadata,
              "ocr_checkpoint_sha256": capture["ocr_weight_sha256"],
              "physical_trace_sha256": sha(args.physical_traces),
              "type_spec": {"path": sources_doc["type_spec_path"], "sha256": sources_doc["type_spec_sha256"]},
              "configuration": dict(capture["configuration"], **{args.run_env: args.name_mode}),
              "all10_evaluated": requested == set(map(str, range(1, 11))),
              "cases": [], "physical_rows": [], "changes": []}
    expected, input_hashes = {}, {}
    scores, totals, colors = ({a: [] for a in ("off", "on")} for _ in range(3))
    evaluation_started = time.perf_counter()
    for src in sources:
        case = src["case_id"]
        stage_path = args.stage1/f"{case}.stage1.json"
        initial_path = args.fixtures/"reads"/f"{case}.reads.json"
        input_hashes[case] = {"stage1_sha256": sha(stage_path), "reads_sha256": sha(initial_path),
                              "image_sha256": sha(Path(src["image_path"])), "gt_sha256": sha(Path(src["gt_path"]))}
        assert input_hashes[case] == previous_hashes[case], f"seventh-round inputs drifted for case {case}"
        stage, initial = read(stage_path), read(initial_path)
        bgr = cv2.imdecode(np.fromfile(src["image_path"], np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None
        start = time.perf_counter()
        digits = read_color_numbers(stage, copy.deepcopy(initial), engine, bgr)
        old_texts = read_pin_texts(stage, copy.deepcopy(initial+digits), engine, bgr)
        lanes = read_pin_lanes(stage, copy.deepcopy(initial+digits+old_texts), engine, bgr, cleanup="light")
        old_reads = initial+digits+old_texts+lanes
        old_read_seconds = time.perf_counter()-start
        for folder, records in (("_baseline_digits", digits), ("_baseline_texts", old_texts), ("_baseline_lanes", lanes)):
            save(args.out/folder/f"{case}.json", records)
        # Recognition is completed before any GT label is read into this loop.
        started = time.perf_counter()
        names = read_pin_names(stage, copy.deepcopy(old_reads), bgr, mode=args.name_mode, min_conf=0.0)
        new_read_seconds = time.perf_counter()-started
        audit = audit_name_records(names)
        if names:
            assert audit["input_sha_traces"], f"case {case}: new name reads have no actual input SHA trace"
        save(args.out/"_new_name_reads"/f"{case}.json", names)
        gt = read(src["gt_path"])
        docs, assignments, arm_seconds, refs, index_maps = {}, {}, {}, {}, {}
        for arm, additions in (("off", []), ("on", names)):
            started = time.perf_counter()
            assigned = binder.assign(stage["components"], stage["pins"], copy.deepcopy(old_reads+additions),
                                     pincol=stage.get("pincol"), pin_evidence=stage.get("pin_evidence"))
            built = assemble.build_doc(stage, assigned, case)
            index_maps[arm] = json.loads(json.dumps(built.get("__pin_index", {})))
            doc = clean(built)
            arm_seconds[arm] = time.perf_counter()-started
            expected[(case, arm)] = docs[arm] = doc
            assignments[arm] = assigned
            if arm == "off":
                assert doc == read(args.previous/"full"/f"{case}.json"), f"fresh seventh-round baseline drift for case {case}"
            refs[arm] = network_refs(doc)
            assert refs[arm]["all_valid"], f"invalid net pin reference for {case}/{arm}"
            score = metrics.evaluate_case(gt, doc, case)
            scores[arm].append(score)
            counted = counts(gt, doc)
            totals[arm].append(counted)
            if case != "1":
                colors[arm].append(counted)
            save(args.out/arm/f"{case}.json", doc)
            save(args.out/arm/"_assigned"/f"{case}.json", assigned)
            save(args.out/arm/f"{case}.pinindex.json", index_maps[arm])
        duplicate_checks = duplicate_winner_checks(stage, assignments, index_maps)
        if case == "1":
            assert docs["off"] == docs["on"] and not names, "monochrome case 1 changed"
            result["monochrome_case1_unchanged"] = True
        height = stage["image"]["h"]
        for cid, comp in gt["components"].items():
            if comp.get("type") != "box":
                continue
            bb = comp["bbox"]
            box = [bb[0], height-bb[3], bb[2], height-bb[1]]
            for pin_key, gpin in gt["pins"].get(cid, {}).items():
                point = [gpin["point"]["x"], height-gpin["point"]["y"]]
                edge = physical_side(box, point)
                trace = trace_map.get((case, cid, pin_key))
                row = {"case": case, "component": cid, "pin": pin_key, "gt": gpin, "arms": {}}
                if trace:
                    ocid, pi = trace["cid"], trace["index"]
                    assert stage["pins"][ocid][pi] == trace["point_img"]
                    row.update(observed_component_id=ocid, observed_pin_index=pi)
                for arm in ("off", "on"):
                    pred = docs[arm]["pins"].get(cid, {}).get(pin_key)
                    correct, distance = False, None
                    if pred:
                        xy = [pred["point"]["x"], height-pred["point"]["y"]]
                        tangent = abs(xy[1]-point[1]) if edge in ("left", "right") else abs(xy[0]-point[0])
                        distance = math.dist(xy, point)
                        correct = pred["pinname"] == gpin["pinname"] and physical_side(box, xy) == edge and tangent <= 5
                    info = {"prediction": pred, "physical_name_number_row_correct": correct, "coordinate_distance": distance}
                    if trace:
                        bound = assignments[arm]["comps"][ocid]["pins"][pi]
                        name, number = (bound.get("name") or {}).get("text"), (bound.get("number") or {}).get("text")
                        info.update(bound_name=name, bound_number=number,
                                    bound_name_correct=name == gpin["pinname"],
                                    bound_number_correct=number == pin_key.removeprefix("pin_"))
                    row["arms"][arm] = info
                result["physical_rows"].append(row)
                if row["arms"]["off"] != row["arms"]["on"]:
                    result["changes"].append(row)
        old_pairs = output_correct_pairs(gt, docs["off"])
        result["cases"].append({"case": case, "baseline_equal_previous": True,
                                "new_name_reads": len(names), "new_name_reads_audit": audit,
                                "timing": {"fresh_legacy_local_seconds": old_read_seconds,
                                           "new_name_seconds": new_read_seconds, "assign_build_seconds": arm_seconds},
                                "scores": {a: scores[a][-1].as_dict() for a in ("off", "on")},
                                "coordinate_free": {a: totals[a][-1] for a in ("off", "on")},
                                "network_references": refs,
                                "duplicate_number_winners": duplicate_checks,
                                "output_regressions": [list(p) for p in sorted(old_pairs-output_correct_pairs(gt, docs["on"]))]})
        save(args.out/"evaluation-progress.json", result)
        print(f"case {case}: coordinate-free {totals['off'][-1]['correct']}/{totals['off'][-1]['predicted']} -> "
              f"{totals['on'][-1]['correct']}/{totals['on'][-1]['predicted']}; score {scores['off'][-1].score:.4f} -> {scores['on'][-1].score:.4f}", flush=True)
    color_rows = [r for r in result["physical_rows"] if r["case"] != "1"]
    assert all("observed_component_id" in row for row in color_rows)
    result["input_hashes"] = input_hashes
    result["coordinate_free"] = {a: {"all10": aggregate(totals[a]), "color2to10": aggregate(colors[a])} for a in ("off", "on")}
    result["scores"] = {a: metrics.aggregate(scores[a]) for a in ("off", "on")}
    result["color_binding_diagnostics"] = {a: {
        "total": len(color_rows), "name_correct": sum(r["arms"][a]["bound_name_correct"] for r in color_rows),
        "number_correct": sum(r["arms"][a]["bound_number_correct"] for r in color_rows),
        "both_correct": sum(r["arms"][a]["bound_name_correct"] and r["arms"][a]["bound_number_correct"] for r in color_rows)} for a in ("off", "on")}
    result["regressions"] = {"physical_complete": [r for r in color_rows if r["arms"]["off"]["physical_name_number_row_correct"] and not r["arms"]["on"]["physical_name_number_row_correct"]],
                             "name": [r for r in color_rows if r["arms"]["off"]["bound_name_correct"] and not r["arms"]["on"]["bound_name_correct"]],
                             "number": [r for r in color_rows if r["arms"]["off"]["bound_number_correct"] and not r["arms"]["on"]["bound_number_correct"]]}
    result["regressions"]["coordinate_free_complete"] = [
        {"case": case["case"], "component": pair[0], "pin": pair[1]}
        for case in result["cases"] for pair in case["output_regressions"]]
    if not args.skip_process:
        integration_checks(args, engine, sources, expected, result)
    else:
        result["process_integration"] = "skipped; replay-only evidence"
    assert source_hashes == {f: sha(REPO/f) for f in production_files}, "production source changed during evaluation"
    assert script_hash == sha(Path(__file__).resolve()), "evaluation script changed during evaluation"
    assert weight_files == {p: sha(Path(p)) for p in weight_files}, "OCR artifact changed during evaluation"
    assert sha(sources_doc["type_spec_path"]) == sources_doc["type_spec_sha256"]
    for src in sources:
        case = src["case_id"]
        assert sha(args.stage1/f"{case}.stage1.json") == input_hashes[case]["stage1_sha256"]
        assert sha(args.fixtures/"reads"/f"{case}.reads.json") == input_hashes[case]["reads_sha256"]
        assert sha(Path(src["gt_path"])) == input_hashes[case]["gt_sha256"]
        assert sha(Path(src["image_path"])) == input_hashes[case]["image_sha256"]
    result["inputs_sources_weights_unchanged"] = True
    result["total_seconds"] = time.perf_counter()-evaluation_started
    save(args.out/"pin-name-ocr-ab.json", result)
    print(json.dumps({k: result[k] for k in ("coordinate_free", "color_binding_diagnostics", "scores")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
