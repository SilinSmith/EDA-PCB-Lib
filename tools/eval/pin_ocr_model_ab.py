"""Compare two recognizers on the same observed pin-lane images.

No GT enters crop extraction, input preprocessing, orientation or recognition.
The trace file is joined only after all model calls for final diagnostic scores.
Run in an isolated environment with rapidocr_onnxruntime and the repo deps.
This does not modify weights or production recognition/assembly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage2.isolated_ocr import prepare_isolated_sample
from stage2.pin_lane_crops import pin_lane_boxes
from textrec.dataset import crop_gray
from textrec.infer import Recognizer


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"D:\codex-work\output\pin-name-direction-20261009\source-verification.json"))
    parser.add_argument("--stage1", type=Path, default=Path(r"D:\codex-work\output\pin-color-rule-20261009\stage1"))
    parser.add_argument("--previous", type=Path, default=Path(r"D:\codex-work\output\pin-lane-repair-20261010\evaluation-final-complete"))
    parser.add_argument("--trace", type=Path, default=Path(r"D:\codex-work\output\pin-color-rule-20261009\root-causes\all_physical_pin_traces.json"))
    parser.add_argument("--old-report", type=Path, default=ROOT/"docs/reports/第七轮引脚裁片实测.json")
    parser.add_argument("--out", type=Path, default=Path(r"D:\codex-work\output\pin-ocr-next-20261010\model-ab"))
    parser.add_argument("--include-numbers", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = load(args.source)
    start = time.perf_counter()
    rec = Recognizer.load(ROOT/"weights/textrec_best.pt",device="cpu")
    crnn_load = time.perf_counter()-start
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
    import onnxruntime
    pkg = Path(rapidocr_onnxruntime.__file__).parent
    model_path = pkg/"models/ch_PP-OCRv4_rec_infer.onnx"
    config = {"model_path":str(model_path),"rec_batch_num":1,"rec_img_shape":[3,48,320],
              "intra_op_num_threads":4,"inter_op_num_threads":1,"use_cuda":False,"use_dml":False}
    start = time.perf_counter()
    alternate = TextRecognizer(config)
    rapid_load = time.perf_counter()-start
    ppchars = alternate.postprocess_op.character
    inputs, rows = [], []
    input_hashes = {}
    role_filter = {"pin_name", "pin_number"} if args.include_numbers else {"pin_name"}
    for item in source["cases"]:
        case = item["case_id"]
        if case == "1":
            continue
        doc = load(args.stage1/f"{case}.stage1.json")
        image = cv2.imdecode(np.fromfile(item["image_path"],dtype=np.uint8),cv2.IMREAD_COLOR)
        raw = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        candidates, cleaned = pin_lane_boxes(doc,image)
        lane = load(args.previous/"_lane_reads/light"/f"{case}.json")
        legacy = load(args.previous/"_baseline_texts"/f"{case}.json")
        lane_map = {(r["pin_target"]["component_id"],r["pin_target"]["pin_index"],r["pin_lane_crop"]["role"]):r for r in lane}
        directions, votes = {}, defaultdict(Counter)
        for observation in legacy + lane:
            target = observation.get("pin_target",{})
            meta = observation.get("pin_text_crop") or observation.get("pin_lane_crop") or {}
            rotation = meta.get("rotation")
            reading = observation.get("reads",{}).get("pin_name")
            if rotation in (-90,90) and reading and reading[1]>=.6:
                directions[(target.get("component_id"),target.get("pin_index"))] = rotation
                votes[(target.get("component_id"),target.get("edge"))][rotation] += float(reading[1])
        input_hashes[case] = {"image":file_hash(item["image_path"]),"stage1":file_hash(args.stage1/f"{case}.stage1.json")}
        for candidate in candidates:
            role = candidate["role"]
            if role not in role_filter:
                continue
            target = candidate["pin_target"]
            identity = (target["component_id"],target["pin_index"])
            rotation = 0
            if candidate.get("vertical"):
                rotation = directions.get(identity)
                if rotation is None:
                    side_votes = votes[(target["component_id"],target["edge"])]
                    if side_votes:
                        rotation = side_votes.most_common(1)[0][0]
                if rotation is None:
                    # Read both directions for diagnostics; no GT-based choice.
                    rotations = (-90,90)
                else:
                    rotations = (rotation,)
            else:
                rotations = (0,)
            saved = lane_map.get(identity+(role,))
            view = saved["pin_lane_crop"]["view"] if saved else ("light" if role=="pin_number" else "raw")
            x0,y0,x1,y1 = map(int,candidate["bbox"])
            gray = (cleaned if view=="light" else raw)[y0:y1,x0:x1].copy()
            for rotation in rotations:
                canvas,bbox,sampling = prepare_isolated_sample({"gray":gray,"bbox":[0,0,gray.shape[1],gray.shape[0]],
                    "role":role,"rotation":rotation,"padding":3 if role=="pin_name" else 2})
                normalized = np.asarray(crop_gray(Image.fromarray(canvas),bbox,rec.img_h,rec.max_w))
                stem = f"{case}-{target['component_id']}-p{target['pin_index']}-{role}-r{rotation}"
                image_path = args.out/f"{stem}.png"
                Image.fromarray(normalized).save(image_path)
                rows.append({"case":case,"component_id":target["component_id"],"pin_index":target["pin_index"],
                    "edge":target["edge"],"role":role,"rotation":rotation,"orientation_source":"horizontal" if not candidate.get("vertical") else "observed_name_direction" if len(rotations)==1 else "unresolved_both_directions",
                    "view":view,"bbox":candidate["bbox"],"flags":candidate.get("flags",[]),"sampling":sampling,
                    "model_input_path":str(image_path)})
                inputs.append((canvas,bbox,normalized))
    # All images and direction decisions exist before any label is loaded.
    for row,(canvas,bbox,normalized) in zip(rows,inputs):
        start = time.perf_counter()
        text,confidence = rec.read_many(canvas,[(bbox,row["role"])])[0]
        row["crnn"] = {"text":text,"confidence":float(confidence),"seconds":time.perf_counter()-start}
        color = cv2.cvtColor(normalized,cv2.COLOR_GRAY2BGR)
        start = time.perf_counter()
        reading,internal = alternate(color)
        row["ppocr_v4"] = {"text":str(reading[0][0]),"confidence":float(reading[0][1]),"seconds":time.perf_counter()-start,"forward_seconds":internal}
    labelmap = {(t["case"],t["cid"],t["index"]):t for t in load(args.trace)["rows"]}
    oldreport = load(args.old_report)
    olderrors = {(r["case"],r["component"],r["pin"]):r for r in oldreport["remaining_errors"]}
    for row in rows:
        label = labelmap.get((row["case"],row["component_id"],row["pin_index"]))
        if label:
            row["evaluation_component"] = label["component"]
            row["evaluation_pin"] = label["pin"]
            expected = label["gt"]["pinname"] if row["role"]=="pin_name" else label["pin"].removeprefix("pin_")
            row["evaluation_label"] = expected
            key = (row["case"],label["component"],label["pin"])
            field = "bound_name_correct" if row["role"]=="pin_name" else "bound_number_correct"
            row["previous_bound_correct"] = olderrors[key]["current"][field] if key in olderrors else True
    summary = {}
    for role in sorted(role_filter):
        selected = [r for r in rows if r["role"]==role and "evaluation_label" in r and r["orientation_source"]!="unresolved_both_directions"]
        summary[role] = {"count":len(selected),"crnn_exact":sum(r["crnn"]["text"]==r["evaluation_label"] for r in selected),
            "ppocr_v4_exact":sum(r["ppocr_v4"]["text"]==r["evaluation_label"] for r in selected),
            "previous_bound_correct_count":sum(r["previous_bound_correct"] for r in selected),
            "ppocr_v4_regressions_vs_previous_bound":[r for r in selected if r["previous_bound_correct"] and r["ppocr_v4"]["text"]!=r["evaluation_label"]],
            "crnn_regressions_vs_previous_bound":[r for r in selected if r["previous_bound_correct"] and r["crnn"]["text"]!=r["evaluation_label"]],
            "ppocr_v4_recoveries_vs_crnn":[r for r in selected if r["crnn"]["text"]!=r["evaluation_label"] and r["ppocr_v4"]["text"]==r["evaluation_label"]],
            "ppocr_v4_regressions_vs_crnn":[r for r in selected if r["crnn"]["text"]==r["evaluation_label"] and r["ppocr_v4"]["text"]!=r["evaluation_label"]]}
    names = [r for r in rows if r["role"]=="pin_name"]
    parentheses = [r for r in names if "(" in r.get("evaluation_label","") or ")" in r.get("evaluation_label","")]
    report = {"evidence_boundary":"10GT development diagnostics on complete geometry-generated local crops. Both models receive the same isolated 32px image; PP-OCR performs its own required 48px recognition normalization. GT joined only after recognition. This is not production pipeline accuracy or unseen validation.",
        "source_paths":{"source":str(args.source),"stage1":str(args.stage1),"previous":str(args.previous),"trace":str(args.trace)},
        "model_provenance":{"crnn":{"path":str(ROOT/"weights/textrec_best.pt"),"sha256":file_hash(ROOT/"weights/textrec_best.pt")},
            "ppocr_v4":{"path":str(model_path),"sha256":file_hash(model_path),"package":"rapidocr_onnxruntime==1.4.4","onnxruntime_version":onnxruntime.__version__,"upstream":"https://github.com/RapidAI/RapidOCR","config":config,
                "detector_used":False,"angle_classifier_used":False,"providers":alternate.session.session.get_providers(),
                "character_support":{s:all(c in ppchars for c in s) for s in ("abcdefghijklmnopqrstuvwxyz","ABCDEFGHIJKLMNOPQRSTUVWXYZ","0123456789","_","(",")","[]","+-./%")},
                "dictionary_count":len(ppchars),"dictionary_source":"ONNX model metadata character list"}},
        "dependency_versions":{name:importlib.metadata.version(name) for name in ("rapidocr-onnxruntime","onnxruntime","numpy","opencv-python","torch","Pillow")},
        "input_hashes":input_hashes,"summary":summary,"parentheses_rows":parentheses,
        "timing":{"crnn_load_seconds":crnn_load,"ppocr_v4_load_seconds":rapid_load,"crnn_crop_total":sum(r["crnn"]["seconds"] for r in rows),"ppocr_v4_crop_total":sum(r["ppocr_v4"]["seconds"] for r in rows),"crop_count":len(rows)},
        "rows":rows}
    path = args.out/"pin-ocr-model-ab.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    (args.out/"ppocr-dictionary.json").write_text(json.dumps(list(ppchars),ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"report":str(path),"summary":{k:{name:value for name,value in val.items() if not isinstance(value,list)} for k,val in summary.items()},"timing":report["timing"]},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
