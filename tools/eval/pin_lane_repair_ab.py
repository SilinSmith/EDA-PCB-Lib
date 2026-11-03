"""Evaluate observed pin lane OCR without letting GT guide recognition.

The detected components, pins, wires and initial OCR are frozen. Local crops
are recognized afresh with the same checkpoint. The snapshot baseline must
reproduce the sixth-round delivered JSON before any comparison is accepted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
sys.path.insert(0, str(REPO))
BASELINE_COMMIT = "ddb94e83d1d2a50cb6e97cd23975216917a2ce92"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(doc):
    doc = copy.deepcopy(doc)
    doc.pop("__pin_index", None)
    return doc


def counts(gt, pred):
    gc = {k for k, v in gt["components"].items() if v.get("type") == "box"}
    pc = {k for k, v in pred["components"].items() if v.get("type") == "box"}
    g = {(c, k): v for c, ps in gt["pins"].items() if c in gc for k, v in ps.items()}
    p = {(c, k): v for c, ps in pred["pins"].items() if c in pc for k, v in ps.items()}
    common = set(g) & set(p)
    correct = sum(g[k]["pinname"] == p[k]["pinname"] for k in common)
    return {"gt": len(g), "predicted": len(p), "correct": correct,
            "matching_number": len(common), "name_mismatch": len(common) - correct,
            "missing_number": len(set(g) - set(p)), "extra_number": len(set(p) - set(g))}


def aggregate(rows):
    if not rows:
        return {}
    out = {k: sum(r[k] for r in rows) for k in rows[0]}
    out.update(precision=out["correct"] / out["predicted"] if out["predicted"] else 0,
               recall=out["correct"] / out["gt"] if out["gt"] else 0,
               f1=2 * out["correct"] / (out["predicted"] + out["gt"])
               if out["predicted"] + out["gt"] else 0)
    return out


def role_reads(records, role):
    """Filter a shared actual recognition batch, never re-recognize an arm."""
    out = []
    for record in records:
        if role not in record.get("reads", {}):
            continue
        item = copy.deepcopy(record)
        item["reads"] = {role: item["reads"][role]}
        item["name_only"] = role == "pin_name"
        item["number_only"] = role == "pin_number"
        out.append(item)
    return out


def network_refs(doc):
    from schematic_parser.data import parse_hypergraph, pin_key_to_number
    pins = {(cid, pin_key_to_number(key)) for cid, group in doc.get("pins", {}).items()
            for key in group}
    checked, bad = 0, []
    for netid, net in doc.get("nets", {}).items():
        for cid, number in parse_hypergraph(net.get("hyperGraph", "")):
            checked += 1
            if cid not in doc.get("components", {}) or (cid, number) not in pins:
                bad.append({"net": netid, "component": cid, "number": number})
    return {"references_checked": checked, "invalid": bad, "all_valid": not bad}


def output_correct_pairs(gt, pred):
    return {(cid, key) for cid, group in gt.get("pins", {}).items()
            if gt.get("components", {}).get(cid, {}).get("type") == "box"
            for key, pin in group.items()
            if pred.get("pins", {}).get(cid, {}).get(key, {}).get("pinname") == pin["pinname"]}


class FrozenInitialEngine:
    """Freeze only the initial detector OCR; every local crop uses real weights."""
    def __init__(self, engine, initial):
        self.engine = engine
        self.initial = initial
        self.calls = []

    def read(self, image, boxes, roles=None):
        if self.calls:
            raise AssertionError("unexpected extra TextEngine.read call in fixed-input process")
        records = copy.deepcopy(self.initial)
        if roles is not None:
            wanted = set(roles)
            for record in records:
                record["reads"] = {k: v for k, v in record.get("reads", {}).items() if k in wanted}
        self.calls.append({"boxes": len(boxes), "roles": roles, "returned_reads": len(records)})
        return records

    def recognizer(self):
        return self.engine.recognizer()


def process_checks(args, engine, sources, arms, expected, result):
    from stage2 import run
    from stage2 import pin_lane_reading
    # The roles argument selects OCR observations. The legacy binder may
    # borrow a numeric designator read for a pin number; it is not an output
    # role prohibition. Compare the actual snapshot on identical options,
    # while separately auditing every new lane observation for role leakage.
    baseline_run = load("stage2._lane_baseline_run", args.baseline_source / "stage2/run.py")
    for filename in ("assign.py", "assemble.py", "color_crops.py", "pin_text_crops.py", "text.py"):
        assert sha(REPO / "stage2" / filename) == sha(args.baseline_source / "stage2" / filename), \
            f"snapshot option dependencies differ: {filename}"
    baseline_run.PINCOL_WIN_TEXTS = False
    prior_lane_reader = pin_lane_reading.read_pin_lanes
    captured_lane_reads = []

    def capture_lane_reads(*positional, **keywords):
        records = prior_lane_reader(*positional, **keywords)
        captured_lane_reads.extend(copy.deepcopy(records))
        return records

    def bound_slots(assigned, slot):
        return {(cid, str(index)): pin.get(slot)
                for cid, component in assigned["comps"].items()
                for index, pin in component.get("pins", {}).items() if pin.get(slot)}
    requested = {c.strip() for c in args.process_cases.split(",") if c.strip()}
    wanted = requested & {s["case_id"] for s in sources}
    if not wanted:
        return
    prior_env = {key: os.environ.get(key) for key in ("", "")}
    prior_run_flag = getattr(run, "PIN_LANE_CROP", None)
    prior_run_cleanup = getattr(run, "PIN_LANE_CLEANUP", None)
    prior_pincol = run.PINCOL_WIN_TEXTS
    run.PINCOL_WIN_TEXTS = False  # initial frozen reads already include captured windows
    checks = []
    try:
        pin_lane_reading.read_pin_lanes = capture_lane_reads
        for source in sources:
            case = source["case_id"]
            if case not in wanted:
                continue
            initial = read(args.fixtures / "reads" / f"{case}.reads.json")
            for arm in ("binding_only", "full"):
                enabled = arm == "full"
                os.environ[""] = "1" if enabled else "0"
                os.environ["STAGE2_PIN_LANE_CLEANUP"] = args.default_cleanup
                if hasattr(run, "PIN_LANE_CROP"):
                    run.PIN_LANE_CROP = enabled
                if hasattr(run, "PIN_LANE_CLEANUP"):
                    run.PIN_LANE_CLEANUP = args.default_cleanup
                proxy = FrozenInitialEngine(engine, initial)
                target = args.out / "integration" / arm
                t0 = time.perf_counter()
                stats = run.process(case, args.stage1, target, proxy, texts="stage1", recut="off")
                got = read(target / f"{case}.json")
                equal = got == expected[(case, arm)]
                checks.append({"case": case, "arm": arm, "identical_to_replay": equal,
                               "wall_seconds": time.perf_counter() - t0,
                               "initial_read_calls": proxy.calls, "statistics": stats,
                               "network_references": network_refs(got)})
                if not equal:
                    save(args.out / "integration" / "process-integration.json", checks)
                assert equal, f"real process differs from controlled {arm} replay in case {case}"
                assert len(proxy.calls) == 1
        option_cases = {c.strip() for c in args.process_option_cases.split(",") if c.strip()}
        options = (
            ("pin_name_only", ["pin_name"], True, 0.0),
            ("pin_number_only", ["pin_number"], True, 0.0),
            ("custom_roles", ["designator", "pin_name"], True, .5),
            ("high_min_conf", None, True, .95),
            ("no_ocr", None, False, 0.0),
        )
        os.environ[""] = "1"
        if hasattr(run, "PIN_LANE_CROP"):
            run.PIN_LANE_CROP = True
        for source in sources:
            case = source["case_id"]
            if case not in option_cases:
                continue
            initial = read(args.fixtures / "reads" / f"{case}.reads.json")
            for mode, roles, do_ocr, minimum in options:
                old_proxy = FrozenInitialEngine(engine, initial)
                old_target = args.out / "integration" / "options_baseline" / mode
                baseline_stats = baseline_run.process(
                    case, args.stage1, old_target, old_proxy, roles=roles,
                    do_ocr=do_ocr, min_conf=minimum, texts="stage1", recut="off")
                baseline_assigned = read(old_target / "_assigned" / f"{case}.json")
                assert len(old_proxy.calls) == (1 if do_ocr else 0)
                proxy = FrozenInitialEngine(engine, initial)
                target = args.out / "integration" / "options" / mode
                captured_lane_reads.clear()
                t0 = time.perf_counter()
                stats = run.process(case, args.stage1, target, proxy, roles=roles,
                                    do_ocr=do_ocr, min_conf=minimum, texts="stage1", recut="off")
                got = read(target / f"{case}.json")
                assigned = read(target / "_assigned" / f"{case}.json")
                wanted_roles = set(roles) if roles is not None else {"pin_name", "pin_number"}
                for record in captured_lane_reads:
                    assert set(record.get("reads", {})) <= wanted_roles, \
                        f"{case}/{mode} lane OCR leaked a disabled role"
                    for _text, confidence in record.get("reads", {}).values():
                        assert float(confidence) >= minimum, f"{case}/{mode} lane OCR bypassed minimum confidence"
                if not do_ocr:
                    assert not captured_lane_reads, f"{case}/{mode} called lane OCR while OCR was disabled"
                disabled_role_checks = {}
                for slot, role in (("name", "pin_name"), ("number", "pin_number")):
                    if not do_ocr or (roles is not None and role not in roles):
                        old_bound = bound_slots(baseline_assigned, slot)
                        new_bound = bound_slots(assigned, slot)
                        assert new_bound == old_bound, f"{case}/{mode} changed legacy fallback for disabled {role}"
                        disabled_role_checks[role] = {
                            "baseline_bound_count": len(old_bound), "new_bound_count": len(new_bound),
                            "identical_to_snapshot": True,
                            "meaning": "OCR role selection; unchanged binder may borrow other allowed roles"}
                for component in assigned["comps"].values():
                    for pin in component.get("pins", {}).values():
                        for slot, role in (("name", "pin_name"), ("number", "pin_number")):
                            bound = pin.get(slot)
                            if not do_ocr:
                                assert not bound, f"{case}/{mode} bound OCR text while OCR was disabled"
                            if bound and minimum:
                                assert float(bound["conf"]) >= minimum, f"{case}/{mode} bypassed minimum confidence"
                refs = network_refs(got)
                assert refs["all_valid"], f"invalid option-mode net references {case}/{mode}"
                assert len(proxy.calls) == (1 if do_ocr else 0)
                checks.append({"case": case, "mode": mode, "roles": roles,
                               "do_ocr": do_ocr, "min_conf": minimum,
                               "completed": True, "wall_seconds": time.perf_counter() - t0,
                               "initial_read_calls": proxy.calls, "statistics": stats,
                               "baseline_statistics": baseline_stats,
                               "lane_reads_role_audit": {"records": len(captured_lane_reads),
                                                         "allowed_roles_only": True,
                                                         "minimum_confidence_respected": True},
                               "disabled_role_fallback": disabled_role_checks,
                               "network_references": refs})
    finally:
        pin_lane_reading.read_pin_lanes = prior_lane_reader
        run.PINCOL_WIN_TEXTS = prior_pincol
        if prior_run_flag is not None:
            run.PIN_LANE_CROP = prior_run_flag
        if prior_run_cleanup is not None:
            run.PIN_LANE_CLEANUP = prior_run_cleanup
        for key, value in prior_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result["process_integration"] = checks
    save(args.out / "integration" / "process-integration.json", checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=WORK / "output/pin-name-direction-20261009")
    ap.add_argument("--stage1", type=Path, default=WORK / "output/pin-color-rule-20261009/stage1")
    ap.add_argument("--previous", type=Path, default=WORK / "output/pin-text-repair-20261010")
    ap.add_argument("--baseline-source", type=Path, default=WORK / "output/pin-lane-repair-20261010/baseline_source")
    ap.add_argument("--physical-traces", type=Path,
                    default=WORK / "output/pin-color-rule-20261009/root-causes/all_physical_pin_traces.json")
    ap.add_argument("--out", type=Path, default=WORK / "output/pin-lane-repair-20261010/evaluation")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--cleanups", default="light,raw", help="one shared read batch per cleanup policy")
    ap.add_argument("--default-cleanup", default="light")
    ap.add_argument("--baseline-crops", choices=("fresh", "cached"), default="fresh")
    ap.add_argument("--process-cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--process-option-cases", default="6", help="check roles, min-conf and no-OCR compatibility")
    ap.add_argument("--skip-process", action="store_true", help="development replay only; no entry-point claim")
    args = ap.parse_args()
    capture = read(args.fixtures / "capture-verification.json")
    source = read(args.fixtures / "source-verification.json")
    os.environ.update(capture["configuration"])
    os.environ["STAGE2_PIN_NAME_EDGE_MIDPOINT"] = "0"
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(4)
    from stage2 import assemble
    from stage2.text import TextEngine
    from stage2.pin_lane_reading import read_pin_lanes
    from tools.eval.color_pin_rule_ab import physical_side
    from schematic_parser import metrics
    active = importlib.import_module("stage2.assign")
    old_sm = load("stage2._lane_baseline_structmatch", args.baseline_source / "stage2/structmatch.py")
    old_assign = load("stage2._lane_baseline_assign", args.baseline_source / "stage2/assign.py")
    old_assign.structmatch = old_sm
    old_assemble = load("stage2._lane_baseline_assemble", args.baseline_source / "stage2/assemble.py")
    old_digits = load("stage2._lane_baseline_color_crops", args.baseline_source / "stage2/color_crops.py")
    old_text = load("stage2._lane_baseline_pin_text", args.baseline_source / "stage2/pin_text_crops.py")
    assert sha(capture["ocr_weight"]) == capture["ocr_weight_sha256"]
    assert sha(source["type_spec_path"]) == source["type_spec_sha256"]
    engine = TextEngine(capture["ocr_weight"], device="cpu", allow_missing=False)
    policies = list(dict.fromkeys(c.strip() for c in args.cleanups.split(",") if c.strip()))
    assert args.default_cleanup in policies, "default cleanup must be included in --cleanups"
    arms = ["baseline", "binding_only", "new_names", "new_numbers", "full"]
    arms += [f"full_{p}" for p in policies if p != args.default_cleanup]
    files = sorted(str(p.relative_to(REPO)).replace("\\", "/")
                   for directory in ("stage2", "textrec") for p in (REPO / directory).glob("*.py"))
    files += ["schematic_parser/metrics.py", "schematic_parser/data.py"]
    hashes = {f: sha(REPO / f) for f in files}
    snapshot_hashes = {str(p.relative_to(args.baseline_source)).replace("\\", "/"): sha(p)
                       for p in args.baseline_source.rglob("*.py")}
    result = {"baseline_commit": BASELINE_COMMIT,
              "scope": "specified 10GT; chip box pins; color diagnostics exclude monochrome case 1",
              "evidence_boundary": "Fixed observed stage1 component/pin/wire geometry and initial OCR; local crops freshly recognized with the same checkpoint. GT is evaluation only. Not fresh upstream detection, an independent hidden test, or official online scoring.",
              "coordinate_free_definition": "Exact component key, pin_n and pinname; coordinates ignored; micro counts.",
              "physical_binding_definition": "Evaluation-only fixed GT-to-observed-pin trace, independently checking bound name and number.",
              "candidate_arm_definition": "new_names/new_numbers filter one shared full lane read batch. They isolate candidate insertion, not independent OCR or direction inference; new numbers may share observed direction votes from new names.",
              "source_hashes": hashes, "baseline_snapshot_hashes": snapshot_hashes,
              "ocr_checkpoint_sha256": capture["ocr_weight_sha256"],
              "configuration": capture["configuration"], "cleanup_policies": policies,
              "baseline_crop_mode": args.baseline_crops, "cases": [], "physical_rows": [], "changes": []}
    traces = read(args.physical_traces)["rows"]
    trace_map = {(r["case"], r["component"], r["pin"]): r for r in traces}
    result["physical_trace_sha256"] = sha(args.physical_traces)
    capture_cases = {r["case_id"]: r for r in capture["cases"]}
    requested = {c.strip() for c in args.cases.split(",") if c.strip()}
    sources = [s for s in source["cases"] if s["case_id"] in requested]
    assert len(sources) == len(requested), "unknown case IDs"
    result["evaluated_case_ids"] = [s["case_id"] for s in sources]
    result["all10_evaluated"] = requested == {str(i) for i in range(1, 11)}
    scores = {a: [] for a in arms}
    all_counts = {a: [] for a in arms}
    color_counts = {a: [] for a in arms}
    expected, input_hashes = {}, {}
    evaluation_started = time.perf_counter()
    for src in sources:
        case = src["case_id"]
        stage_path = args.stage1 / f"{case}.stage1.json"
        initial_path = args.fixtures / "reads" / f"{case}.reads.json"
        assert sha(src["gt_path"]) == src["gt_sha256"]
        assert sha(src["image_path"]) == src["image_sha256"]
        assert sha(initial_path) == capture_cases[case]["reads_sha256"]
        input_hashes[case] = {"stage1_sha256": sha(stage_path), "reads_sha256": sha(initial_path),
                              "image_sha256": src["image_sha256"], "gt_sha256": src["gt_sha256"]}
        stage = read(stage_path)
        initial = read(initial_path)
        bgr = cv2.imdecode(np.fromfile(src["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None
        t0 = time.perf_counter()
        if args.baseline_crops == "fresh":
            digits = old_digits.read_color_numbers(stage, initial, engine, bgr)
            texts = old_text.read_pin_texts(stage, list(initial) + digits, engine, bgr)
        else:
            digits = read(args.previous / "_new_digits" / f"{case}.json")
            texts = read(args.previous / "_pin_text_reads" / f"{case}.json")
        baseline_reads = list(initial) + digits + texts
        baseline_crop_seconds = time.perf_counter() - t0
        save(args.out / "_baseline_digits" / f"{case}.json", digits)
        save(args.out / "_baseline_texts" / f"{case}.json", texts)
        lane_reads, lane_seconds = {}, {}
        for policy in policies:
            t0 = time.perf_counter()
            lane_reads[policy] = read_pin_lanes(stage, copy.deepcopy(baseline_reads), engine, bgr,
                                               include_names=True, include_numbers=True, cleanup=policy)
            lane_seconds[policy] = time.perf_counter() - t0
            save(args.out / "_lane_reads" / policy / f"{case}.json", lane_reads[policy])
        new = lane_reads[args.default_cleanup]
        new_by_arm = {"baseline": [], "binding_only": [], "new_names": role_reads(new, "pin_name"),
                      "new_numbers": role_reads(new, "pin_number"), "full": new}
        new_by_arm.update({f"full_{p}": lane_reads[p] for p in policies if p != args.default_cleanup})
        docs, assignments, arm_seconds, net_checks = {}, {}, {}, {}
        # Recognition has completed before loading GT labels; none are passed into production calls.
        gt = read(src["gt_path"])
        for arm in arms:
            binder, builder = (old_assign, old_assemble) if arm == "baseline" else (active, assemble)
            t0 = time.perf_counter()
            assigned = binder.assign(stage["components"], stage["pins"],
                                     copy.deepcopy(baseline_reads + new_by_arm[arm]),
                                     pincol=stage.get("pincol"), pin_evidence=stage.get("pin_evidence"))
            doc = clean(builder.build_doc(stage, assigned, case))
            arm_seconds[arm] = time.perf_counter() - t0
            docs[arm], assignments[arm] = doc, assigned
            expected[(case, arm)] = doc
            if arm == "baseline":
                assert doc == read(args.previous / "full" / f"{case}.json"), f"sixth-round baseline drift {case}"
            net_checks[arm] = network_refs(doc)
            assert net_checks[arm]["all_valid"], f"invalid net pin reference in {case}/{arm}"
            score = metrics.evaluate_case(gt, doc, case)
            scores[arm].append(score)
            cnt = counts(gt, doc)
            all_counts[arm].append(cnt)
            if case != "1":
                color_counts[arm].append(cnt)
            save(args.out / arm / f"{case}.json", doc)
            save(args.out / arm / "_assigned" / f"{case}.json", assigned)
        if case == "1":
            for arm in arms:
                assert docs["baseline"] == docs[arm], f"monochrome case changed in {arm}"
            assert all(not values for values in lane_reads.values()), "monochrome case opened colored lane"
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
                    observed_cid, index = trace["cid"], trace["index"]
                    assert stage["pins"][observed_cid][index] == trace["point_img"], "physical trace drift"
                    row.update(observed_component_id=observed_cid, observed_pin_index=index)
                for arm in arms:
                    pred = docs[arm]["pins"].get(cid, {}).get(pin_key)
                    correct, distance = False, None
                    if pred:
                        xy = [pred["point"]["x"], height - pred["point"]["y"]]
                        tangent = abs(xy[1] - point[1]) if edge in ("left", "right") else abs(xy[0] - point[0])
                        distance = math.dist(xy, point)
                        correct = pred["pinname"] == gpin["pinname"] and physical_side(box, xy) == edge and tangent <= 5
                    info = {"prediction": pred, "physical_name_number_row_correct": correct,
                            "coordinate_distance": distance}
                    if trace:
                        bound = assignments[arm]["comps"][observed_cid]["pins"][index]
                        name = (bound.get("name") or {}).get("text")
                        number = (bound.get("number") or {}).get("text")
                        info.update(bound_name=name, bound_number=number,
                                    bound_name_correct=name == gpin["pinname"],
                                    bound_number_correct=number == pin_key.removeprefix("pin_"))
                    row["arms"][arm] = info
                result["physical_rows"].append(row)
                if row["arms"]["baseline"]["prediction"] != row["arms"]["full"]["prediction"]:
                    result["changes"].append(row)
        old_pairs = output_correct_pairs(gt, docs["baseline"])
        result["cases"].append({"case": case, "baseline_equal_previous": True,
                                "baseline_digit_reads": len(digits), "baseline_text_reads": len(texts),
                                "new_lane_reads": {p: len(v) for p, v in lane_reads.items()},
                                "new_name_reads": len(new_by_arm["new_names"]),
                                "new_number_reads": len(new_by_arm["new_numbers"]),
                                "timing": {"baseline_crop_seconds": baseline_crop_seconds,
                                           "shared_lane_read_seconds": lane_seconds,
                                           "assign_build_seconds": arm_seconds},
                                "scores": {a: scores[a][-1].as_dict() for a in arms},
                                "coordinate_free": {a: all_counts[a][-1] for a in arms},
                                "network_references": net_checks,
                                "output_regressions": {a: [list(pair) for pair in sorted(old_pairs - output_correct_pairs(gt, docs[a]))]
                                                       for a in arms}})
        print(f"case {case}: coordinate-free {all_counts['baseline'][-1]['correct']}/{all_counts['baseline'][-1]['predicted']} -> "
              f"{all_counts['full'][-1]['correct']}/{all_counts['full'][-1]['predicted']}; "
              f"local score {scores['baseline'][-1].score:.4f} -> {scores['full'][-1].score:.4f}", flush=True)
    color = [r for r in result["physical_rows"] if r["case"] != "1"]
    assert all("observed_component_id" in row for row in color), "incomplete color physical trace mapping"
    result["input_hashes"] = input_hashes
    result["coordinate_free"] = {a: {"all10": aggregate(all_counts[a]), "color2to10": aggregate(color_counts[a])} for a in arms}
    result["scores"] = {a: metrics.aggregate(scores[a]) for a in arms}
    result["color_binding_diagnostics"] = {a: {
        "name_correct": sum(r["arms"][a]["bound_name_correct"] for r in color),
        "number_correct": sum(r["arms"][a]["bound_number_correct"] for r in color),
        "both_correct": sum(r["arms"][a]["bound_name_correct"] and r["arms"][a]["bound_number_correct"] for r in color)
    } for a in arms}
    result["color_physical_correct"] = {a: sum(r["arms"][a]["physical_name_number_row_correct"] for r in color) for a in arms}
    result["regressions"] = {a: [r for r in color if r["arms"]["baseline"]["physical_name_number_row_correct"]
                                and not r["arms"][a]["physical_name_number_row_correct"]] for a in arms}
    result["coordinate_conflicts"] = [r for r in color if r["arms"]["full"]["physical_name_number_row_correct"]
                                      and r["arms"]["full"]["coordinate_distance"] > 5]
    if not args.skip_process:
        process_checks(args, engine, sources, arms, expected, result)
    else:
        result["process_integration"] = "skipped; replay evidence only"
    result["total_seconds"] = time.perf_counter() - evaluation_started
    assert hashes == {f: sha(REPO / f) for f in files}, "production sources changed during evaluation"
    for src in sources:
        case = src["case_id"]
        assert sha(args.stage1 / f"{case}.stage1.json") == input_hashes[case]["stage1_sha256"]
        assert sha(args.fixtures / "reads" / f"{case}.reads.json") == input_hashes[case]["reads_sha256"]
        assert sha(src["gt_path"]) == src["gt_sha256"] and sha(src["image_path"]) == src["image_sha256"]
    result["inputs_and_sources_unchanged"] = True
    save(args.out / "pin-lane-repair-ab.json", result)
    print(json.dumps({k: result[k] for k in ("coordinate_free", "color_binding_diagnostics", "scores")}, ensure_ascii=False, indent=2))
    print("physical full regressions", len(result["regressions"]["full"]), flush=True)


if __name__ == "__main__":
    main()
