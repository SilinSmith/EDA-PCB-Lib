"""Controlled numeral repair A/B against an isolated eighth-round snapshot.

Initial OCR and observed stage1 geometry remain frozen. Every legacy local
crop and new repair crop is recognized with real model weights. GT enters
only after recognition and is never passed to production code.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
sys.path.insert(0, str(REPO))

from tools.eval.pin_lane_repair_ab import (
    aggregate, clean, counts, network_refs, output_correct_pairs, read, save,
    sha, FrozenInitialEngine,
)
from tools.eval.pin_name_ocr_ab import input_hash_traces, bound_slots

BASELINE_COMMIT = "4685555b2f2dd37bd5795c088c74ef5dc8b2b938"
OPTIONS = (
    ("pin_name_only", ["pin_name"], True, 0.0),
    ("pin_number_only", ["pin_number"], True, 0.0),
    ("custom_roles", ["designator", "pin_number"], True, .5),
    ("custom_without_number", ["designator", "pin_name"], True, .5),
    ("high_min_conf", None, True, .95),
    ("no_ocr", None, False, 0.0),
)


def requested_ids(value):
    return {item.strip() for item in value.split(",") if item.strip()}


def case_inputs(args, src):
    case = src["case_id"]
    return {
        "stage1_sha256": sha(args.stage1 / f"{case}.stage1.json"),
        "reads_sha256": sha(args.fixtures / "reads" / f"{case}.reads.json"),
        "image_sha256": sha(src["image_path"]),
        "gt_sha256": sha(src["gt_path"]),
    }


def snapshot_hashes(args):
    return {str(p.relative_to(args.baseline_source)).replace("\\", "/"): sha(p)
            for p in args.baseline_source.rglob("*.py")}


def assert_snapshot(args):
    verification = read(args.baseline_source.parent / "baseline-source-verification.json")
    assert verification["commit"] == BASELINE_COMMIT
    assert snapshot_hashes(args) == verification["files"], "eighth-round source snapshot changed"
    return verification


def configure(args):
    capture = read(args.fixtures / "capture-verification.json")
    sources_doc = read(args.fixtures / "source-verification.json")
    os.environ.update(capture["configuration"])
    os.environ.update(STAGE2_PIN_NAME_EDGE_MIDPOINT="0", STAGE2_PIN_NAME_OCR=args.name_mode,
                      STAGE2_PIN_NUMBER_REPAIR="0")
    assert sha(capture["ocr_weight"]) == capture["ocr_weight_sha256"]
    assert sha(sources_doc["type_spec_path"]) == sources_doc["type_spec_sha256"]
    sources = [s for s in sources_doc["cases"] if s["case_id"] in requested_ids(args.cases)]
    assert len(sources) == len(requested_ids(args.cases)), "unknown case IDs"
    previous_inputs = read(args.previous / "pin-name-ocr-ab.json")["input_hashes"]
    for src in sources:
        assert case_inputs(args, src) == previous_inputs[src["case_id"]], "eighth-round input drift"
    return capture, sources_doc, sources


def image(src):
    import cv2
    import numpy as np
    bgr = cv2.imdecode(np.fromfile(src["image_path"], np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    return bgr


def build(stage, records, case, binder, assemble):
    assigned = binder.assign(stage["components"], stage["pins"], copy.deepcopy(records),
                             pincol=stage.get("pincol"), pin_evidence=stage.get("pin_evidence"))
    built = assemble.build_doc(stage, assigned, case)
    index = json.loads(json.dumps(built.get("__pin_index", {})))
    return clean(built), assigned, index


def save_arm(args, case, arm, doc, assigned, index):
    save(args.out / arm / f"{case}.json", doc)
    save(args.out / arm / "_assigned" / f"{case}.json", assigned)
    save(args.out / arm / f"{case}.pinindex.json", index)


def assert_baseline_modules(args):
    origins = {}
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] not in ("stage2", "textrec") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(args.baseline_source.resolve()), f"snapshot leaked current module: {name}: {path}"
        origins[name] = str(path)
    return origins


def baseline_worker(args):
    """Fresh child process; all stage2/textrec imports come from the snapshot."""
    assert not any(n.split(".")[0] in ("stage2", "textrec") for n in sys.modules)
    sys.path.insert(0, str(args.baseline_source))
    capture, _, sources = configure(args)
    snapshot = assert_snapshot(args)
    import torch
    torch.set_num_threads(4)
    from stage2 import assemble, run
    from stage2.text import TextEngine
    from stage2.color_crops import read_color_numbers
    from stage2.pin_text_crops import read_pin_texts
    from stage2.pin_lane_reading import read_pin_lanes
    from stage2.pin_name_reading import read_pin_names
    binder = importlib.import_module("stage2.assign")
    engine = TextEngine(capture["ocr_weight"], device="cpu", allow_missing=False)
    run.PINCOL_WIN_TEXTS = False
    run.PIN_NAME_OCR = args.name_mode
    rows, process_checks = [], []
    started = time.perf_counter()
    for src in sources:
        case = src["case_id"]
        stage = read(args.stage1 / f"{case}.stage1.json")
        initial = read(args.fixtures / "reads" / f"{case}.reads.json")
        bgr = image(src)
        start = time.perf_counter()
        digits = read_color_numbers(stage, copy.deepcopy(initial), engine, bgr)
        texts = read_pin_texts(stage, copy.deepcopy(initial + digits), engine, bgr)
        lanes = read_pin_lanes(stage, copy.deepcopy(initial + digits + texts), engine, bgr, cleanup="light")
        legacy = initial + digits + texts + lanes
        names = read_pin_names(stage, copy.deepcopy(legacy), bgr, mode=args.name_mode, min_conf=0.0)
        for folder, records in (("_baseline_digits", digits), ("_baseline_texts", texts),
                                ("_baseline_lanes", lanes), ("_baseline_names", names),
                                ("_baseline_reads", legacy)):
            save(args.out / folder / f"{case}.json", records)
        doc, assigned, index = build(stage, legacy + names, case, binder, assemble)
        assert doc == read(args.previous / "on" / f"{case}.json"), f"fresh eighth-round baseline drift: {case}"
        assert index == read(args.previous / "on" / f"{case}.pinindex.json"), f"baseline index drift: {case}"
        save_arm(args, case, "off", doc, assigned, index)
        refs = network_refs(doc)
        assert refs["all_valid"]
        rows.append({"case": case, "baseline_equal_previous": True, "fresh_local_seconds": time.perf_counter() - start,
                     "digit_reads": len(digits), "text_reads": len(texts), "lane_reads": len(lanes),
                     "name_reads": len(names), "input_hashes": case_inputs(args, src), "network_references": refs})
        if not args.skip_process and case in requested_ids(args.process_cases):
            proxy = FrozenInitialEngine(engine, initial)
            target = args.out / "baseline-integration" / "standard"
            stats = run.process(case, args.stage1, target, proxy, texts="stage1", recut="off")
            assert read(target / f"{case}.json") == doc, f"snapshot process/replay drift: {case}"
            assert read(target / f"{case}.pinindex.json") == index
            assert len(proxy.calls) == 1
            process_checks.append({"case": case, "mode": "standard", "identical_to_replay": True,
                                   "statistics": stats, "initial_read_calls": proxy.calls})
        if not args.skip_process and case in requested_ids(args.process_option_cases):
            for label, roles, do_ocr, minimum in OPTIONS:
                proxy = FrozenInitialEngine(engine, initial)
                target = args.out / "baseline-integration" / "options" / label
                stats = run.process(case, args.stage1, target, proxy, roles=roles, do_ocr=do_ocr,
                                    min_conf=minimum, texts="stage1", recut="off")
                assert len(proxy.calls) == (1 if do_ocr else 0)
                assert network_refs(read(target / f"{case}.json"))["all_valid"]
                process_checks.append({"case": case, "mode": label, "statistics": stats,
                                       "initial_read_calls": proxy.calls})
        save(args.out / "baseline-progress.json", rows)
        print(f"snapshot case {case}: fresh eighth-round JSON and index reproduced", flush=True)
    result = {"baseline_commit": BASELINE_COMMIT, "snapshot_hashes": snapshot["files"], "cases": rows,
              "process_integration": process_checks, "module_origins": assert_baseline_modules(args),
              "total_seconds": time.perf_counter() - started, "all_local_crops_fresh": True}
    assert_snapshot(args)
    for src, row in zip(sources, rows):
        assert case_inputs(args, src) == row["input_hashes"]
    save(args.out / "baseline-verification.json", result)


def dispatch_baseline(args):
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--baseline-worker"]
    for name in ("fixtures", "stage1", "previous", "baseline_source", "out", "cases", "name_mode",
                 "process_cases", "process_option_cases"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.skip_process:
        command.append("--skip-process")
    subprocess.run(command, cwd=REPO, check=True)


def audit_number_records(records, minimum=0.0):
    for record in records:
        assert set(record.get("reads", {})) <= {"pin_number"}, "new numeral OCR leaked another role"
        assert record.get("pin_target"), "new numeral read lacks a physical target"
        assert input_hash_traces(record), "new numeral read lacks its actual model input SHA"
        for text, confidence in record.get("reads", {}).values():
            assert str(text).isdigit(), "new numeral read is not an observed digit sequence"
            assert float(confidence) >= minimum, "new numeral read bypassed min_conf"
    traces = input_hash_traces(records)
    if records:
        assert traces, "new numeral reads lack actual model input SHA"
    reasons = Counter(record.get("pin_number_ocr", {}).get("held_reason") for record in records
                      if not record.get("reads"))
    return {"records": len(records), "accepted_records": sum(bool(r.get("reads")) for r in records),
            "held_reasons": dict(reasons), "role_limited": True, "min_conf_respected": True,
            "input_sha_traces": traces}


def duplicate_number_audit(stage, assigned, indexes):
    """Record physical winners; genuine numeral repairs may change winners."""
    result = []
    for component in stage.get("components", []):
        if component.get("type") != "box":
            continue
        cid = component["id"]
        numbers = {}
        for arm in ("off", "on"):
            for index, pin in assigned[arm].get("comps", {}).get(cid, {}).get("pins", {}).items():
                number = (pin.get("number") or {}).get("text")
                if number:
                    numbers.setdefault(number, {}).setdefault(arm, []).append(str(index))
        for number, candidates in numbers.items():
            if not any(len(group) > 1 for group in candidates.values()):
                continue
            winners = {arm: sorted(index for index, key in indexes[arm].get(cid, {}).items()
                                   if key == f"pin_{number}") for arm in ("off", "on")}
            result.append({"component_id": cid, "number": number, "physical_candidates": candidates,
                           "winners": winners, "winner_changed": winners["off"] != winners["on"]})
    return result


def integration_checks(args, engine, sources, expected, result):
    from stage2 import run
    reader = importlib.import_module("stage2.pin_number_reading")
    original = reader.read_pin_numbers
    captured = []
    called = 0

    def capture(*positional, **keywords):
        nonlocal called
        called += 1
        records = original(*positional, **keywords)
        captured.extend(copy.deepcopy(records))
        return records

    old_flag, old_pincol = run.PIN_NUMBER_REPAIR, run.PINCOL_WIN_TEXTS
    old_env = os.environ.get("STAGE2_PIN_NUMBER_REPAIR")
    checks = []
    try:
        reader.read_pin_numbers = capture
        run.PINCOL_WIN_TEXTS = False
        for src in sources:
            case = src["case_id"]
            initial = read(args.fixtures / "reads" / f"{case}.reads.json")
            if case in requested_ids(args.process_cases):
                for arm in ("off", "on"):
                    captured.clear()
                    called = 0
                    run.PIN_NUMBER_REPAIR = arm == "on"
                    os.environ["STAGE2_PIN_NUMBER_REPAIR"] = "1" if arm == "on" else "0"
                    proxy = FrozenInitialEngine(engine, initial)
                    target = args.out / "integration" / arm
                    started = time.perf_counter()
                    stats = run.process(case, args.stage1, target, proxy, texts="stage1", recut="off")
                    got = read(target / f"{case}.json")
                    assert got == expected[(case, arm)], f"real process differs from replay: {case}/{arm}"
                    assert read(target / f"{case}.pinindex.json") == read(args.out / arm / f"{case}.pinindex.json")
                    assert len(proxy.calls) == 1 and network_refs(got)["all_valid"]
                    if arm == "off":
                        assert called == 0 and not captured, "disabled new numeral reader invoked"
                    elif case != "1":
                        assert json.loads(json.dumps(captured)) == read(args.out / "_new_number_reads" / f"{case}.json"), \
                            f"real process numeral observations differ from replay: {case}"
                    checks.append({"case": case, "arm": arm, "identical_to_replay": True,
                                   "wall_seconds": time.perf_counter() - started, "statistics": stats,
                                   "number_reader_calls": called,
                                   "initial_read_calls": proxy.calls, "number_reads_audit": audit_number_records(captured)})
                    save(args.out / "integration/process-integration.json", checks)
            if case not in requested_ids(args.process_option_cases):
                continue
            for label, roles, do_ocr, minimum in OPTIONS:
                outputs, assignments, audits = {}, {}, {}
                for arm in ("off", "on"):
                    captured.clear()
                    called = 0
                    run.PIN_NUMBER_REPAIR = arm == "on"
                    os.environ["STAGE2_PIN_NUMBER_REPAIR"] = "1" if arm == "on" else "0"
                    proxy = FrozenInitialEngine(engine, initial)
                    target = args.out / "integration/options" / label / arm
                    run.process(case, args.stage1, target, proxy, roles=roles, do_ocr=do_ocr,
                                min_conf=minimum, texts="stage1", recut="off")
                    outputs[arm] = read(target / f"{case}.json")
                    assignments[arm] = read(target / "_assigned" / f"{case}.json")
                    audits[arm] = dict(audit_number_records(captured, minimum), number_reader_calls=called)
                    assert len(proxy.calls) == (1 if do_ocr else 0)
                    assert network_refs(outputs[arm])["all_valid"]
                    if arm == "off":
                        baseline = args.out / "baseline-integration/options" / label
                        assert outputs[arm] == read(baseline / f"{case}.json"), \
                            f"current off differs from isolated snapshot options: {case}/{label}"
                        assert read(target / f"{case}.pinindex.json") == read(baseline / f"{case}.pinindex.json")
                    if arm == "off" or not do_ocr or (roles is not None and "pin_number" not in roles):
                        assert called == 0 and not captured, f"new numeral OCR invoked while disabled: {case}/{label}/{arm}"
                disabled = not do_ocr or (roles is not None and "pin_number" not in roles)
                if disabled:
                    assert outputs["off"] == outputs["on"], f"JSON changed without numeral OCR: {case}/{label}"
                    assert bound_slots(assignments["off"], "number") == bound_slots(assignments["on"], "number")
                checks.append({"case": case, "mode": label, "roles": roles, "do_ocr": do_ocr,
                               "min_conf": minimum, "passed": True, "number_reads_audit": audits,
                               "snapshot_off_identical": True, "disabled_json_identical": True if disabled else None,
                               "roles_meaning": "OCR observation selection; legacy binder may borrow other permitted roles"})
                save(args.out / "integration/process-integration.json", checks)
    finally:
        reader.read_pin_numbers = original
        run.PIN_NUMBER_REPAIR, run.PINCOL_WIN_TEXTS = old_flag, old_pincol
        if old_env is None:
            os.environ.pop("STAGE2_PIN_NUMBER_REPAIR", None)
        else:
            os.environ["STAGE2_PIN_NUMBER_REPAIR"] = old_env
    result["process_integration"] = checks


def evaluate(args):
    capture, sources_doc, sources = configure(args)
    snapshot = assert_snapshot(args)
    if not args.reuse_baseline:
        dispatch_baseline(args)
    baseline = read(args.out / "baseline-verification.json")
    assert baseline["snapshot_hashes"] == snapshot["files"]
    assert {row["case"] for row in baseline["cases"]} == {s["case_id"] for s in sources}
    for src in sources:
        row = next(row for row in baseline["cases"] if row["case"] == src["case_id"])
        assert row["input_hashes"] == case_inputs(args, src)
    import torch
    torch.set_num_threads(4)
    from stage2 import assemble
    from stage2.text import TextEngine
    from stage2.pin_name_reading import read_pin_names
    from stage2.pin_number_reading import read_pin_numbers
    from stage2.pin_name_ocr import get_pin_name_ocr
    from tools.eval.color_pin_rule_ab import physical_side
    from schematic_parser import metrics
    binder = importlib.import_module("stage2.assign")
    files = sorted(str(p.relative_to(REPO)).replace("\\", "/")
                   for directory in ("stage2", "textrec") for p in (REPO / directory).glob("*.py"))
    files += ["schematic_parser/metrics.py", "schematic_parser/data.py"]
    hashes = {f: sha(REPO / f) for f in files}
    script_hash = sha(Path(__file__).resolve())
    provider_metadata = dict(getattr(get_pin_name_ocr(args.name_mode), "metadata", {}) or {})
    weight_paths = [p for folder in (REPO / "weights", REPO / "models", WORK / "models")
                    if folder.exists() for p in folder.rglob("*") if p.is_file()]
    if provider_metadata.get("model_path"):
        weight_paths.append(Path(provider_metadata["model_path"]))
    weight_hashes = {str(p): sha(p) for p in weight_paths}
    engine = TextEngine(capture["ocr_weight"], device="cpu", allow_missing=False)
    trace_map = {(r["case"], r["component"], r["pin"]): r for r in read(args.physical_traces)["rows"]}
    result = {"baseline_commit": BASELINE_COMMIT,
              "evidence_boundary": "Specified 10GT development set; fixed observed stage1 and initial OCR. All legacy and new local crops freshly recognized. Not fresh upstream detection, hidden validation, or official online scoring. GT is evaluation only.",
              "coordinate_free_definition": "Exact component key, pin_n and pinname; all coordinates ignored; micro P/R/F1.",
              "physical_binding_definition": "Evaluation-only GT-to-observed pin trace; name and numeral correctness checked independently.",
              "recognition_order": "legacy color/text/lane -> new numeral repair -> PP name OCR",
              "source_hashes": hashes, "baseline_snapshot_hashes": snapshot["files"],
              "baseline_verification": baseline, "evaluation_script_sha256": script_hash,
              "weight_hashes": weight_hashes, "name_provider_metadata": provider_metadata,
              "physical_trace_sha256": sha(args.physical_traces), "ocr_checkpoint_sha256": capture["ocr_weight_sha256"],
              "type_spec": {"path": sources_doc["type_spec_path"], "sha256": sources_doc["type_spec_sha256"]},
              "configuration": dict(capture["configuration"], STAGE2_PIN_NUMBER_REPAIR="1", STAGE2_PIN_NAME_OCR=args.name_mode),
              "all10_evaluated": requested_ids(args.cases) == set(map(str, range(1, 11))),
              "cases": [], "physical_rows": [], "changes": []}
    expected, inputs = {}, {}
    scores, totals, colors = ({arm: [] for arm in ("off", "on")} for _ in range(3))
    started = time.perf_counter()
    for src in sources:
        case = src["case_id"]
        inputs[case] = case_inputs(args, src)
        stage = read(args.stage1 / f"{case}.stage1.json")
        legacy = read(args.out / "_baseline_reads" / f"{case}.json")
        off_names = read(args.out / "_baseline_names" / f"{case}.json")
        bgr = image(src)
        current_off, current_assigned, current_index = build(stage, legacy + off_names, case, binder, assemble)
        assert current_off == read(args.out / "off" / f"{case}.json"), f"production off changed eighth-round baseline: {case}"
        assert current_index == read(args.out / "off" / f"{case}.pinindex.json")
        start = time.perf_counter()
        digits = read_pin_numbers(stage, copy.deepcopy(legacy), engine, bgr, min_conf=0.0)
        number_seconds = time.perf_counter() - start
        audit = audit_number_records(digits)
        save(args.out / "_new_number_reads" / f"{case}.json", digits)
        start = time.perf_counter()
        on_names = read_pin_names(stage, copy.deepcopy(legacy + digits), bgr, mode=args.name_mode, min_conf=0.0)
        name_seconds = time.perf_counter() - start
        save(args.out / "_on_name_reads" / f"{case}.json", on_names)
        # GT is loaded only after both actual recognition arms are complete.
        gt = read(src["gt_path"])
        docs, assignments, indexes, refs = {}, {}, {}, {}
        docs["off"], assignments["off"], indexes["off"] = current_off, current_assigned, current_index
        docs["on"], assignments["on"], indexes["on"] = build(stage, legacy + digits + on_names, case, binder, assemble)
        for arm in ("off", "on"):
            expected[(case, arm)] = docs[arm]
            refs[arm] = network_refs(docs[arm])
            assert refs[arm]["all_valid"]
            scores[arm].append(metrics.evaluate_case(gt, docs[arm], case))
            totals[arm].append(counts(gt, docs[arm]))
            if case != "1":
                colors[arm].append(totals[arm][-1])
            if arm == "on":
                save_arm(args, case, arm, docs[arm], assignments[arm], indexes[arm])
        if case == "1":
            assert docs["off"] == docs["on"] and not digits, "monochrome case 1 changed"
            result["monochrome_case1_unchanged"] = True
        height = stage["image"]["h"]
        for cid, component in gt["components"].items():
            if component.get("type") != "box":
                continue
            bb = component["bbox"]
            box = [bb[0], height - bb[3], bb[2], height - bb[1]]
            for pin_key, gpin in gt["pins"].get(cid, {}).items():
                point = [gpin["point"]["x"], height - gpin["point"]["y"]]
                edge = physical_side(box, point)
                trace = trace_map.get((case, cid, pin_key))
                row = {"case": case, "component": cid, "pin": pin_key, "gt": gpin, "arms": {}}
                if trace:
                    ocid, index = trace["cid"], trace["index"]
                    assert stage["pins"][ocid][index] == trace["point_img"], "physical trace drift"
                    row.update(observed_component_id=ocid, observed_pin_index=index)
                for arm in ("off", "on"):
                    pred = docs[arm]["pins"].get(cid, {}).get(pin_key)
                    correct, distance = False, None
                    if pred:
                        import math
                        xy = [pred["point"]["x"], height - pred["point"]["y"]]
                        tangent = abs(xy[1] - point[1]) if edge in ("left", "right") else abs(xy[0] - point[0])
                        distance = math.dist(xy, point)
                        correct = pred["pinname"] == gpin["pinname"] and physical_side(box, xy) == edge and tangent <= 5
                    info = {"prediction": pred, "physical_name_number_row_correct": correct, "coordinate_distance": distance}
                    if trace:
                        bound = assignments[arm]["comps"][ocid]["pins"][index]
                        name, number = (bound.get("name") or {}).get("text"), (bound.get("number") or {}).get("text")
                        info.update(bound_name=name, bound_number=number, bound_name_correct=name == gpin["pinname"],
                                    bound_number_correct=number == pin_key.removeprefix("pin_"))
                    row["arms"][arm] = info
                result["physical_rows"].append(row)
                if row["arms"]["off"] != row["arms"]["on"]:
                    result["changes"].append(row)
        old_pairs = output_correct_pairs(gt, docs["off"])
        result["cases"].append({"case": case, "baseline_equal_previous": True, "current_off_equal_snapshot": True,
                                "new_number_reads": len(digits), "new_number_reads_audit": audit,
                                "timing": {"new_number_seconds": number_seconds, "on_name_seconds": name_seconds},
                                "scores": {a: scores[a][-1].as_dict() for a in ("off", "on")},
                                "coordinate_free": {a: totals[a][-1] for a in ("off", "on")}, "network_references": refs,
                                "duplicate_number_winners": duplicate_number_audit(stage, assignments, indexes),
                                "output_regressions": [list(pair) for pair in sorted(old_pairs - output_correct_pairs(gt, docs["on"]))]})
        save(args.out / "evaluation-progress.json", result)
        print(f"case {case}: coordinate-free {totals['off'][-1]['correct']}/{totals['off'][-1]['predicted']} -> "
              f"{totals['on'][-1]['correct']}/{totals['on'][-1]['predicted']}; score {scores['off'][-1].score:.4f} -> {scores['on'][-1].score:.4f}", flush=True)
    color_rows = [row for row in result["physical_rows"] if row["case"] != "1"]
    assert all("observed_component_id" in row for row in color_rows)
    result["input_hashes"] = inputs
    result["coordinate_free"] = {a: {"all10": aggregate(totals[a]), "color2to10": aggregate(colors[a])} for a in ("off", "on")}
    result["scores"] = {a: metrics.aggregate(scores[a]) for a in ("off", "on")}
    result["color_binding_diagnostics"] = {a: {
        "total": len(color_rows), "name_correct": sum(row["arms"][a]["bound_name_correct"] for row in color_rows),
        "number_correct": sum(row["arms"][a]["bound_number_correct"] for row in color_rows),
        "both_correct": sum(row["arms"][a]["bound_name_correct"] and row["arms"][a]["bound_number_correct"] for row in color_rows)
    } for a in ("off", "on")}
    result["number_observation_diagnostics"] = {
        "attempted": sum(case["new_number_reads_audit"]["records"] for case in result["cases"]),
        "accepted_modifications": sum(case["new_number_reads_audit"]["accepted_records"] for case in result["cases"]),
        "held_reasons": dict(sum((Counter(case["new_number_reads_audit"]["held_reasons"])
                                  for case in result["cases"]), Counter())),
    }
    result["regressions"] = {
        "physical_complete": [row for row in color_rows if row["arms"]["off"]["physical_name_number_row_correct"] and not row["arms"]["on"]["physical_name_number_row_correct"]],
        "name": [row for row in color_rows if row["arms"]["off"]["bound_name_correct"] and not row["arms"]["on"]["bound_name_correct"]],
        "number": [row for row in color_rows if row["arms"]["off"]["bound_number_correct"] and not row["arms"]["on"]["bound_number_correct"]],
        "coordinate_free_complete": [{"case": case["case"], "component": pair[0], "pin": pair[1]}
                                     for case in result["cases"] for pair in case["output_regressions"]],
    }
    if not args.skip_process:
        integration_checks(args, engine, sources, expected, result)
    else:
        result["process_integration"] = "skipped; replay-only evidence"
    assert hashes == {f: sha(REPO / f) for f in files}, "production sources changed during evaluation"
    assert script_hash == sha(Path(__file__).resolve()), "evaluation script changed during evaluation"
    assert weight_hashes == {path: sha(path) for path in weight_hashes}, "OCR weights changed during evaluation"
    assert sha(args.physical_traces) == result["physical_trace_sha256"]
    assert sha(sources_doc["type_spec_path"]) == sources_doc["type_spec_sha256"]
    assert_snapshot(args)
    for src in sources:
        assert case_inputs(args, src) == inputs[src["case_id"]]
    result["inputs_sources_weights_unchanged"] = True
    result["total_seconds"] = time.perf_counter() - started
    save(args.out / "pin-number-ocr-ab.json", result)
    print(json.dumps({k: result[k] for k in ("coordinate_free", "color_binding_diagnostics", "scores")}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=WORK / "output/pin-name-direction-20261009")
    parser.add_argument("--stage1", type=Path, default=WORK / "output/pin-color-rule-20261009/stage1")
    parser.add_argument("--previous", type=Path, default=WORK / "output/pin-ocr-next-20261010/evaluation-final")
    parser.add_argument("--baseline-source", type=Path, default=WORK / "output/pin-number-next-20261010/baseline_source")
    parser.add_argument("--physical-traces", type=Path, default=WORK / "output/pin-color-rule-20261009/root-causes/all_physical_pin_traces.json")
    parser.add_argument("--out", type=Path, default=WORK / "output/pin-number-next-20261010/evaluation")
    parser.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--name-mode", default="auto")
    parser.add_argument("--process-cases", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--process-option-cases", default="6,8")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--baseline-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-only", action="store_true", help="prepare isolated fresh baseline without loading production repair")
    parser.add_argument("--reuse-baseline", action="store_true", help="reuse separately verified snapshot observations, never reread cached new repair crops")
    args = parser.parse_args()
    if args.baseline_worker:
        baseline_worker(args)
    elif args.baseline_only:
        dispatch_baseline(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
