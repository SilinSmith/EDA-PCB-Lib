"""Replay frozen geometry/OCR while testing measured color terminals.

The original GT files remain unchanged. Local GT scoring and the pin
number/name/physical-row diagnostic are reported separately: the latter ignores
normal-axis GT disagreement and is not an official score.
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
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_document(document):
    result = copy.deepcopy(document)
    result.pop("__pin_index", None)
    return result


def physical_side(box, point):
    """A normal terminal projects onto a finite body side, including corners.

    The closest infinite line is wrong for a long pin near a box corner: a
    left terminal can be closer to the top line while still projecting to the
    left side. This helper is only an audit and never changes a prediction.
    """
    x0,y0,x1,y1 = box
    x,y = point
    if y0 <= y <= y1:
        if x < x0:
            return "left"
        if x > x1:
            return "right"
    if x0 <= x <= x1:
        if y < y0:
            return "top"
        if y > y1:
            return "bottom"
    anchors = {"top":(min(x1,max(x0,x)),y0), "right":(x1,min(y1,max(y0,y))),
               "bottom":(min(x1,max(x0,x)),y1),"left":(x0,min(y1,max(y0,y)))}
    return min(anchors,key=lambda e:math.dist(point,anchors[e]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--color-number-ocr", action="store_true",
                        help="Also re-read external digit crops with the captured OCR checkpoint")
    args = parser.parse_args()
    capture = read(args.fixtures / "capture-verification.json")
    source = read(args.fixtures / "source-verification.json")
    os.environ.update(capture["configuration"])
    import cv2
    import numpy as np
    from stage1 import pincolor, pipeline, pin_evidence, nets
    from stage2 import assemble
    from schematic_parser import metrics
    from schematic_parser.data import load_json
    active_assign = importlib.import_module("stage2.assign")
    engine = None
    if args.color_number_ocr:
        import torch
        torch.set_num_threads(4)
        from stage2.text import TextEngine
        from stage2.color_crops import read_color_numbers
        assert sha(capture["ocr_weight"]) == capture["ocr_weight_sha256"]
        engine = TextEngine(capture["ocr_weight"],device="cpu",allow_missing=False)
    evaluated_files = ("stage1/pincolor.py","stage1/pipeline.py","stage1/nets.py",
                       "stage2/assign.py","stage2/structmatch.py","stage2/assemble.py",
                       "stage2/color_crops.py","stage2/run.py")
    source_hashes = {rel:sha(REPO/rel) for rel in evaluated_files}
    old_sm = load_module("stage2._color_baseline_structmatch", args.baseline_source / "structmatch.py")
    old_assign = load_module("stage2._color_baseline_assign", args.baseline_source / "assign.py")
    old_assign.structmatch = old_sm
    old_assemble = load_module("stage2._color_baseline_assemble", args.baseline_source / "assemble.py")
    assert sha(source["type_spec_path"]) == source["type_spec_sha256"]
    score_rows = {arm: [] for arm in ("baseline", "binding_only", "color_rule", "color_legacy_binding")}
    if engine is not None:
        score_rows["color_number_ocr"] = []
    result = {"scope": "specified official10 only", "baseline_commit": "653158ae",
              "gt_files_modified": False,
              "evidence_boundary": "fixed detected components, wire segments and OCR reads; rerun color terminals, binding, assembly and net attachment; not a fresh detector/OCR run",
              "coordinate_rule": "connected color pins use measured red/green transitions; floating pins use measured red endpoints",
              "cases": [], "physical_row_rows": [], "coordinate_conflicts": []}
    if engine is not None:
        result["evidence_boundary"] += "; extra external digit OCR is freshly run with the same captured checkpoint"
        result["ocr_checkpoint_sha256"] = capture["ocr_weight_sha256"]

    for rec in source["cases"]:
        case = rec["case_id"]
        frozen = next(c for c in capture["cases"] if c["case_id"] == case)
        stpath, rdpath = args.fixtures/"stage1"/f"{case}.stage1.json", args.fixtures/"reads"/f"{case}.reads.json"
        assert sha(rec["image_path"]) == rec["image_sha256"]
        assert sha(rec["gt_path"]) == rec["gt_sha256"]
        assert sha(stpath) == frozen["stage1_sha256"] and sha(rdpath) == frozen["reads_sha256"]
        old_stage, reads = read(stpath), read(rdpath)
        bgr = cv2.imdecode(np.fromfile(rec["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read source image: {rec['image_path']}")
        color_records = pincolor.extract_color_pins(bgr, old_stage["components"])
        stage = copy.deepcopy(old_stage)
        cp, synth, overlays, connected, ledger = pipeline.apply_color_pins(
            stage["components"], stage["pins"], {}, color_records)
        if ledger:
            stage["pins"] = {cid: [[round(float(v),1) for v in p] for p in pl] for cid,pl in cp.items()}
            ink = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 170).astype(np.uint8)
            gray_evidence = pin_evidence.build_pin_evidence(stage["components"], cp, ink, stage["texts"])
            stage["pin_evidence"] = pipeline.overlay_color_evidence(gray_evidence, overlays)
            report = {}
            stage["nets"], groups = nets.build_nets(cp, stage["wire_detail"]["segments"], ink=ink,
                pin_synth=synth, pin_connected=connected, components=stage["components"],
                guard_attachments=pipeline.PIN_ATTACHMENT_GUARD, report=report)
            stage["wire_detail"]["groups"] = groups
            stage["pin_join"] = report
            stage["pin_color"] = ledger
        save(args.out/"stage1"/f"{case}.stage1.json", stage)
        arms = {"baseline": (old_stage,old_assign,old_assemble),
                "binding_only": (old_stage,active_assign,assemble),
                "color_rule": (stage,active_assign,assemble),
                "color_legacy_binding": (stage,old_assign,old_assemble)}
        extra = read_color_numbers(stage,reads,engine,bgr) if engine is not None else []
        if engine is not None:
            arms["color_number_ocr"] = (stage,active_assign,assemble)
            save(args.out/"color_number_ocr"/"_extra_reads"/f"{case}.json",extra)
        docs = {}
        for arm,(geometry,binder,assembler) in arms.items():
            input_reads = list(reads) + list(extra) if arm == "color_number_ocr" else reads
            assigned = binder.assign(geometry["components"], geometry["pins"], copy.deepcopy(input_reads),
                pincol=geometry.get("pincol"), pin_evidence=geometry.get("pin_evidence"))
            doc = clean_document(assembler.build_doc(geometry,assigned,case))
            docs[arm] = doc
            if arm == "baseline":
                assert doc == read(args.baseline_predictions/f"{case}.json"), f"Baseline replay drift: {case}"
            assert doc["components"] == docs["baseline"]["components"]
            save(args.out/arm/f"{case}.json",doc)
            save(args.out/arm/"_assigned"/f"{case}.json",assigned)
            score_rows[arm].append(metrics.evaluate_case(load_json(rec["gt_path"]),doc,case))
        gt = load_json(rec["gt_path"])
        heights = stage["image"]["h"]
        # Physical-row audit uses the original GT component and pin label.
        # No coordinates or labels are written back into predictions or GT.
        for key,component in gt["components"].items():
            if component.get("type") != "box":
                continue
            bb = component["bbox"]
            box = [bb[0],heights-bb[3],bb[2],heights-bb[1]]
            for pk,gpin in gt.get("pins",{}).get(key,{}).items():
                gp = [gpin["point"]["x"],heights-gpin["point"]["y"]]
                edge = physical_side(box,gp)
                row = {"case":case,"component":key,"pin":pk,"gt":gpin,"edge":edge,"arms":{}}
                for arm,doc in docs.items():
                    pp = doc.get("pins",{}).get(key,{}).get(pk)
                    valid=False
                    if pp:
                        p = [pp["point"]["x"],heights-pp["point"]["y"]]
                        axis_gap = abs(p[1]-gp[1]) if edge in ("left","right") else abs(p[0]-gp[0])
                        normal_gap = abs(p[0]-gp[0]) if edge in ("left","right") else abs(p[1]-gp[1])
                        distance = math.dist(p,gp)
                        valid = pp["pinname"] == gpin["pinname"] and physical_side(box,p)==edge and axis_gap<=5
                        info = {"correct_name_number_row":valid,"prediction":pp,"axis_gap":axis_gap,
                                "normal_gap":normal_gap,"distance":distance,
                                "strict_correct":pp["pinname"]==gpin["pinname"] and distance<=5}
                    else:
                        info = {"correct_name_number_row":False,"prediction":None,"strict_correct":False}
                    row["arms"][arm]=info
                result["physical_row_rows"].append(row)
                if case != "1" and row["arms"]["color_rule"]["correct_name_number_row"] and row["arms"]["color_rule"].get("distance",0)>5:
                    result["coordinate_conflicts"].append(row)
        result["cases"].append({"case":case,"colored_components":len(ledger),
            "extra_number_reads":len(extra),
            "measured_color_pins":sum(len(v) for v in color_records.values()),
            "scores":{arm:score_rows[arm][-1].as_dict() for arm in arms}})
        print(f"case {case}: measured={sum(len(v) for v in color_records.values())}, local pinTP={score_rows['baseline'][-1].pin.tp}->{score_rows['color_rule'][-1].pin.tp}",flush=True)

    result["scores"] = {arm:dict(metrics.aggregate(values),pin_counts={k:sum(getattr(s.pin,k) for s in values) for k in ("tp","fp","fn")}) for arm,values in score_rows.items()}
    color_rows=[r for r in result["physical_row_rows"] if r["case"]!='1']
    result["color_physical_rows"]={arm:sum(r["arms"][arm]["correct_name_number_row"] for r in color_rows) for arm in score_rows}
    result["color_original_strict_tp_regressions"]=[r for r in color_rows if r["arms"]["baseline"]["strict_correct"] and not r["arms"]["color_rule"]["correct_name_number_row"]]
    if engine is not None:
        result["color_ocr_original_strict_tp_regressions"]=[r for r in color_rows if r["arms"]["baseline"]["strict_correct"] and not r["arms"]["color_number_ocr"]["correct_name_number_row"]]
        result["color_ocr_coordinate_conflicts"]=[r for r in color_rows if r["arms"]["color_number_ocr"]["correct_name_number_row"] and not r["arms"]["color_number_ocr"]["strict_correct"]]
    assert source_hashes == {rel:sha(REPO/rel) for rel in evaluated_files}, "Source changed during replay"
    result["source_hashes"]=source_hashes
    save(args.out/"color-rule-ab.json",result)
    print(json.dumps({k:result[k] for k in ("scores","color_physical_rows")},ensure_ascii=False,indent=2))
    print("original color strict TP row/name regressions:",len(result["color_original_strict_tp_regressions"]))
    if engine is not None:
        print("with recrop original color strict TP row/name regressions:",len(result["color_ocr_original_strict_tp_regressions"]))


if __name__ == "__main__":
    main()
