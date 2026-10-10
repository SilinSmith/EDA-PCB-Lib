"""Frozen recognizer comparison on visually reviewed source text candidates.

Uses reviewed orientation for BOTH models; evaluates recognition, not automatic
pin detection/ownership/direction. Historical checkpoint training overlap unknown.
This script never chooses preprocessing or thresholds from review labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from textrec.dataset import crop_gray
from textrec.infer import Recognizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and all(r["role"] == "pin_name" for r in rows)
    assert all(r["verified_status"] in ("verified_accept", "verified_corrected") for r in rows)
    torch.set_num_threads(4)
    checkpoint = ROOT / "weights/textrec_best.pt"
    rec = Recognizer.load(checkpoint, device="cpu")
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
    model_path = Path(rapidocr_onnxruntime.__file__).parent / "models/ch_PP-OCRv4_rec_infer.onnx"
    config = {"model_path": str(model_path), "rec_batch_num": 1, "rec_img_shape": [3, 48, 320],
              "intra_op_num_threads": 4, "inter_op_num_threads": 1, "use_cuda": False, "use_dml": False}
    alternate = TextRecognizer(config)
    hashes = {str(args.manifest): sha(args.manifest), str(checkpoint): sha(checkpoint), str(model_path): sha(model_path)}
    observations = []
    for row in rows:
        crop_path = Path(row["crop_path"])
        hashes[str(crop_path)] = sha(crop_path)
        gray = np.asarray(Image.open(crop_path).convert("L"))
        canvas, bbox, sampling = prepare_isolated_sample({"gray": gray,
            "bbox": [0, 0, gray.shape[1], gray.shape[0]], "role": "pin_name",
            "rotation": row["review_rotation_degrees_ccw"], "padding": 3}, img_h=rec.img_h, max_w=rec.max_w)
        normalized = np.asarray(crop_gray(Image.fromarray(canvas), bbox, rec.img_h, rec.max_w))
        stem = f"{row['case']}-{row['text_id']}"
        input_path = args.out / f"{stem}-input32.png"
        Image.fromarray(normalized).save(input_path)
        start = time.perf_counter()
        old_text, old_conf = rec.read_many(canvas, [(bbox, "pin_name")])[0]
        old_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reading, internal = alternate(cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR))
        pp_seconds = time.perf_counter() - start
        pp_text, pp_conf = str(reading[0][0]), float(reading[0][1])
        # Frozen BEFORE viewing any reviewed-candidate model outputs.
        eligible = bool(pp_text) and pp_conf >= .90
        used_text = pp_text if eligible else old_text
        observations.append({"case": row["case"], "text_id": row["text_id"], "family_group": row["family_group"],
            "source_image": row["image"], "source_json": row["source_json"], "bbox": row["bbox"],
            "original_annotation": row["original_text"], "reviewed_label": row["text"], "review_status": row["verified_status"],
            "reviewed_rotation_degrees_ccw": row["review_rotation_degrees_ccw"], "sampling": sampling,
            "model_input_path": str(input_path),
            "crnn": {"text": str(old_text), "confidence": float(old_conf), "seconds": old_seconds},
            "ppocr_v4": {"text": pp_text, "confidence": pp_conf, "seconds": pp_seconds, "forward_seconds": internal},
            "frozen_gate": {"threshold": .90, "ppocr_accepted": eligible, "text": used_text,
                            "source": "ppocr_v4" if eligible else "crnn"}})
    for row in observations:
        label = row["reviewed_label"]
        for arm in ("crnn", "ppocr_v4", "frozen_gate"):
            row[arm]["exact"] = row[arm]["text"] == label
    summary = {"count": len(observations), "case_count": len({r["case"] for r in observations}),
        "family_count": len({r["family_group"] for r in observations}),
        "correct": {arm: sum(r[arm]["exact"] for r in observations) for arm in ("crnn", "ppocr_v4", "frozen_gate")},
        "gate_ppocr_accept_count": sum(r["frozen_gate"]["ppocr_accepted"] for r in observations),
        "gate_recoveries_vs_crnn": sum(not r["crnn"]["exact"] and r["frozen_gate"]["exact"] for r in observations),
        "gate_regressions_vs_crnn": sum(r["crnn"]["exact"] and not r["frozen_gate"]["exact"] for r in observations),
        "seconds": {arm: sum(r[arm]["seconds"] for r in observations) for arm in ("crnn", "ppocr_v4")}}
    by_case = []
    for case in sorted({r["case"] for r in observations}):
        selected = [r for r in observations if r["case"] == case]
        by_case.append({"case": case, "count": len(selected),
                        "correct": {arm: sum(r[arm]["exact"] for r in selected) for arm in ("crnn", "ppocr_v4", "frozen_gate")}})
    report = {"evidence_boundary": "Assistant visually reviewed 75 names from 10 Dataset300 source image groups, excluding GT model/layout clues. Labels and manual directions are assessment references, not automatic production inference. No retraining or threshold tuning. Existing checkpoint training-image manifest absent, so historical overlap unknown; no hidden-test claim.",
        "input_policy": {"view": "original grayscale annotation crop", "padding": 3,
                         "normalized_height": rec.img_h, "max_width": rec.max_w,
                         "directions": "assistant visual review; identical for both models",
                         "ppocr_internal_height": 48, "old_vs_new_pixels": "same isolated 32px image; PP applies required internal recognition normalization",
                         "threshold_frozen_on_development_before_independent_outputs": .90},
        "models": {"crnn": {"path": str(checkpoint), "sha256": sha(checkpoint)},
                   "ppocr_v4": {"path": str(model_path), "sha256": sha(model_path), "config": config}},
        "manifest_sha256": sha(args.manifest), "summary": summary, "by_case": by_case,
        "source_hashes": hashes, "all_inputs_and_weights_unchanged": all(sha(p) == h for p, h in hashes.items()),
        "dependency_versions": {name: importlib.metadata.version(name) for name in
                                ("rapidocr-onnxruntime", "onnxruntime", "numpy", "opencv-python", "torch", "Pillow")},
        "rows": observations}
    (args.out / "independent-name-model-ab.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "by_case": by_case, "all_inputs_and_weights_unchanged": report["all_inputs_and_weights_unchanged"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
