"""Trace observed pin-name inputs without changing recognition or GT files.

GT names below are evaluation labels only. Every geometry comes from the
recorded detector/binder or the production geometry-only lane generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(value):
    data = value.read_bytes() if isinstance(value, Path) else np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(data).hexdigest()


def crop(array, box):
    xa, ya = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    xb, yb = min(array.shape[1], math.ceil(box[2])), min(array.shape[0], math.ceil(box[3]))
    return array[ya:yb, xa:xb].copy()


class CaptureRecognizer:
    def __init__(self, original, out, prefix):
        self.original, self.out, self.prefix = original, out, prefix
        self.img_h, self.max_w = original.img_h, original.max_w
        self.calls = []

    def read_many(self, image, items):
        from textrec.dataset import crop_gray
        from textrec.infer import _as_pil
        items = list(items)
        gray = _as_pil(image)
        arrays = [np.asarray(crop_gray(gray, box, self.img_h, self.max_w)) for box, role in items]
        results = self.original.read_many(image, items)
        batch_width = max((a.shape[1] for a in arrays), default=0)
        records = []
        for index, ((box, role), a, result) in enumerate(zip(items, arrays, results)):
            filename = self.out / "inputs" / f"{self.prefix}-call{len(self.calls):03}-{index:03}-{role}.png"
            padded = np.pad(a, ((0, 0), (0, batch_width-a.shape[1])), constant_values=255)
            if role == "pin_name":
                filename.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(a).save(filename)
            records.append({"bbox": list(box), "role": role,
                            "text": result[0], "confidence": float(result[1]),
                            "shape": list(a.shape), "model_input_sha256": sha(a),
                            "batch_width": batch_width, "batch_padded_sha256": sha(padded),
                            "effective_steps": max(1, a.shape[1] // 8),
                            "input_png": str(filename) if role == "pin_name" else None})
        self.calls.append(records)
        return results


def isolated(rec, gray, box, rotation, out, prefix, padding=3):
    from stage2.isolated_ocr import prepare_isolated_sample
    from textrec.dataset import crop_gray
    sample = {"gray": gray, "bbox": box, "role": "pin_name", "rotation": rotation, "padding": padding}
    canvas, read_box, metadata = prepare_isolated_sample(sample, img_h=rec.img_h, max_w=rec.max_w)
    a = np.asarray(crop_gray(Image.fromarray(canvas), read_box, rec.img_h, rec.max_w))
    assert sha(a) == metadata["model_input_sha256"]
    text, confidence = rec.read_many(canvas, [(read_box, "pin_name")])[0]
    input_path = out / "inputs" / f"{prefix}.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(input_path)
    return {"text": text, "confidence": float(confidence), "sampling": metadata, "input_png": str(input_path)}


def fit_panel(value, w, h, *, resample=Image.Resampling.NEAREST):
    img = value if isinstance(value, Image.Image) else Image.fromarray(value)
    img = img.convert("RGB")
    factor = min(w / img.width, h / img.height, 8)
    return img.resize((max(1, round(img.width*factor)), max(1, round(img.height*factor))), resample)


def contact_sheet(rows, images, out):
    font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 16)
    for page_index in range(0, len(rows), 8):
        page = Image.new("RGB", (1600, 8*194 + 42), "white")
        draw = ImageDraw.Draw(page)
        draw.text((10, 7), "Observed context | actual old input | lane raw input | lane light input  (nearest enlargement)", font=font, fill="black")
        subset = rows[page_index:page_index+8]
        for ri, row in enumerate(subset):
            yy = 42 + ri*194
            status = "ERR" if not row["bound_name_correct"] else "OK"
            draw.text((10, yy), f'{status} {row["case"]}/{row["component"]}/{row["pin"]}  GT={row["gt_name"]}  bound={row["bound_name"]}', font=font, fill="black")
            draw.text((10, yy+23), f'old={row.get("legacy_replay_text")}  raw={row.get("raw_lane", {}).get("text")}  light={row.get("light_lane", {}).get("text")}  source={row.get("bound_source")}', font=font, fill="black")
            panels = images[row["id"]]
            for col, key in enumerate(("context", "old_input", "raw_input", "light_input")):
                if panels.get(key) is None:
                    continue
                fitted = fit_panel(panels[key], 375, 125)
                page.paste(fitted, (col*400+12, yy+53))
            draw.line((0, yy+191, 1600, yy+191), fill=(210, 210, 210))
        path = out / f"name-input-contact-{page_index//8+1:02}.png"
        page.crop((0, 0, 1600, len(subset)*194+42)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=WORK / "output/pin-ocr-next-20261010/diagnosis")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from stage2.text import TextEngine
    from stage2.pin_text_crops import pin_text_boxes, _padded_crop
    from stage2.pin_lane_crops import pin_lane_boxes
    from textrec.infer import Recognizer
    root = WORK / "output/pin-lane-repair-20261010/evaluation-final-complete"
    fixtures = WORK / "output/pin-name-direction-20261009"
    sources = read(fixtures / "source-verification.json")
    evaluation = read(root / "pin-lane-repair-ab.json")
    weight = REPO / "weights/textrec_best.pt"
    rec = Recognizer.load(weight, device="cpu")
    all_rows, images = [], {}
    source_hashes = {}
    replay_mismatches = []
    for src in sources["cases"]:
        case = src["case_id"]
        if case == "1":
            continue
        bgr = cv2.imdecode(np.fromfile(src["image_path"], np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        stage_path = WORK / "output/pin-color-rule-20261009/stage1" / f"{case}.stage1.json"
        stage = read(stage_path)
        initial = read(fixtures / "reads" / f"{case}.reads.json")
        old_texts = read(root / "_baseline_texts" / f"{case}.json")
        old_digits = read(root / "_baseline_digits" / f"{case}.json")
        lanes = read(root / "_lane_reads/light" / f"{case}.json")
        assigned = read(root / "full/_assigned" / f"{case}.json")
        previous_reads = initial + old_digits + old_texts
        candidates, cleaned = pin_lane_boxes(stage, bgr, reads=previous_reads)
        candidate_by_pin = {(c["pin_target"]["component_id"], c["pin_target"]["pin_index"]): c
                            for c in candidates if c["role"] == "pin_name"}
        cap = CaptureRecognizer(rec, args.out, f"case{case}-old")
        engine = TextEngine(weight, device="cpu", allow_missing=False)
        engine._rec, engine._tried = cap, True
        reconstructed = engine.read(bgr, [r["bbox"] for r in initial])
        flat_captures = [v for call in cap.calls for v in call]
        captures = [v for v in flat_captures if v["role"] == "pin_name"]
        captures_by_role = {(i, v["role"]): v for i in range(len(initial)) for v in flat_captures[5*i:5*i+5]}
        assert len(captures) == len(initial)
        for i, (old, new) in enumerate(zip(initial, reconstructed)):
            if old["reads"].get("pin_name")[0] != new["reads"].get("pin_name")[0]:
                replay_mismatches.append({"case": case, "read_index": i,
                                          "recorded": old["reads"].get("pin_name"),
                                          "reconstructed": new["reads"].get("pin_name")})
        # Reconstruct the legacy atlas including all directions and roles.
        text_boxes, old_gray = pin_text_boxes(stage, bgr)
        turns, turn_keys = [], []
        for index, box in enumerate(text_boxes):
            core = _padded_crop(old_gray, box["bbox"])
            for rotation in (90, -90):
                turns.append(np.rot90(core, rotation//90).copy())
                turn_keys.append((index, rotation))
        atlas_captures = {}
        if turns:
            atlas = np.full((sum(c.shape[0]+4 for c in turns), max(c.shape[1] for c in turns)), 255, np.uint8)
            items, yy = [], 0
            for turned, (index, rotation) in zip(turns, turn_keys):
                h, w = turned.shape
                atlas[yy:yy+h, :w] = turned
                items.append(([0, yy, w, yy+h], text_boxes[index]["role"]))
                yy += h+4
            atlas_cap = CaptureRecognizer(rec, args.out, f"case{case}-atlas")
            atlas_cap.read_many(atlas, items)
            atlas_captures = {(tuple(text_boxes[index]["bbox"]), rotation): captured
                              for (index, rotation), captured in zip(turn_keys, atlas_cap.calls[0])}
        case_rows = [r for r in evaluation["physical_rows"] if r["case"] == case]
        for trace in case_rows:
            cid, pi = trace["observed_component_id"], trace["observed_pin_index"]
            bound = assigned["comps"][cid]["pins"][str(pi)].get("name")
            row_id = f'{case}-{trace["component"]}-{trace["pin"]}'
            row = {"id": row_id, "case": case, "component": trace["component"], "pin": trace["pin"],
                   "observed_component_id": cid, "observed_pin_index": pi,
                   "gt_name": trace["gt"]["pinname"], "gt_used_only_as_evaluation_label": True,
                   "bound_name": bound["text"] if bound else None,
                   "bound_name_correct": trace["arms"]["full"]["bound_name_correct"],
                   "bound_number": trace["arms"]["full"]["bound_number"],
                   "bound_number_correct": trace["arms"]["full"]["bound_number_correct"],
                   "bound_record": bound}
            panel = {"old_input": None, "raw_input": None, "light_input": None}
            matches = [(label, i, rr) for label, group in (("initial", initial), ("old_digit", old_digits), ("old_text", old_texts), ("lane", lanes))
                       for i, rr in enumerate(group) if bound and rr.get("bbox") == bound["box"]
                       and any(v[0] == bound["text"] and abs(float(v[1])-bound["conf"]) < 1e-6
                               for v in rr.get("reads", {}).values())]
            row["matching_source_records"] = [{"source_group": label, "index": i, "record": rr} for label, i, rr in matches]
            if matches:
                label, index, source = matches[-1]
                row["bound_source"] = label
                if label == "initial":
                    chosen_role = next(role for role, v in source["reads"].items()
                                       if v[0] == bound["text"] and abs(float(v[1])-bound["conf"]) < 1e-6)
                    row["bound_ocr_role"] = chosen_role
                    capture = captures_by_role[(index, chosen_role)]
                    if capture["input_png"] is None:
                        from textrec.dataset import crop_gray
                        actual = crop_gray(Image.fromarray(gray), capture["bbox"], rec.img_h, rec.max_w)
                        path = args.out / "inputs" / f"{row_id}-selected-{chosen_role}.png"
                        actual.save(path)
                        capture["input_png"] = str(path)
                        assert sha(np.asarray(actual)) == capture["model_input_sha256"]
                elif label == "old_text":
                    rotation = source["pin_text_crop"]["rotation"]
                    capture = atlas_captures[(tuple(source["bbox"]), rotation)]
                elif label == "lane":
                    md = source["pin_lane_crop"]
                    array = gray if md["view"] == "raw" else cleaned
                    capture = isolated(rec, crop(array, source["bbox"]), [0, 0, source["bbox"][2]-source["bbox"][0], source["bbox"][3]-source["bbox"][1]], md["rotation"], args.out, row_id+"-current")
                    row["lane_model_input_hash_matches_recorded"] = capture["sampling"]["model_input_sha256"] == md["sampling"]["model_input_sha256"]
                else:
                    capture = None
                if capture:
                    row["legacy_reconstruction"] = capture
                    row["legacy_replay_text"] = capture["text"]
                    panel["old_input"] = Image.open(capture["input_png"]).copy()
            candidate = candidate_by_pin.get((cid, pi))
            if candidate:
                row["candidate"] = candidate
                rotation = next((rr.get("pin_text_crop", {}).get("rotation") for rr in old_texts
                                 if rr.get("name_only") and rr.get("pin_target", {}).get("component_id") == cid
                                 and rr.get("pin_target", {}).get("pin_index") == pi), None)
                if not candidate["vertical"]:
                    rotation = 0
                elif rotation is None:
                    rotation = 90
                row["rotation_source"] = "legacy_observed_orientation" if rotation else "horizontal_geometry"
                box = candidate["bbox"]
                for view, array in (("raw", gray), ("light", cleaned)):
                    core = crop(array, box)
                    value = isolated(rec, core, [0, 0, core.shape[1], core.shape[0]], rotation, args.out, row_id+"-"+view)
                    value["correct"] = value["text"] == row["gt_name"]
                    row[view+"_lane"] = value
                    panel[view+"_input"] = Image.open(value["input_png"]).copy()
                xy = [max(0, box[0]-6), max(0, box[1]-6), min(bgr.shape[1], box[2]+6), min(bgr.shape[0], box[3]+6)]
            elif bound:
                xy = [bound["box"][0]-6, bound["box"][1]-6, bound["box"][2]+6, bound["box"][3]+6]
            else:
                point = stage["pins"][cid][pi]
                xy = [point[0]-30, point[1]-30, point[0]+30, point[1]+30]
            context = Image.fromarray(cv2.cvtColor(crop(bgr, xy), cv2.COLOR_BGR2RGB))
            panel["context"] = context
            context_path = args.out / "context" / f"{row_id}.png"
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context.save(context_path)
            row["observed_context_bbox"] = xy
            row["context_png"] = str(context_path)
            images[row_id], all_rows = panel, all_rows + [row]
        source_hashes[case] = {"image": sha(Path(src["image_path"])), "gt": sha(Path(src["gt_path"])), "stage1": sha(stage_path)}
        print(f'case {case}: {len(case_rows)} physical name rows traced', flush=True)
    selected = [r for r in all_rows if not r["bound_name_correct"]]
    controls = []
    for case in map(str, range(2, 11)):
        controls += [r for r in all_rows if r["case"] == case and r["bound_name_correct"]][:2]
    contact_sheet(selected + controls, images, args.out)
    summary = {"total_color_physical_names": len(all_rows), "incorrect_bound_names": len(selected),
               "raw_lane_correct_on_wrong": sum(r.get("raw_lane", {}).get("correct", False) for r in selected),
               "light_lane_correct_on_wrong": sum(r.get("light_lane", {}).get("correct", False) for r in selected),
               "raw_lane_correct_all": sum(r.get("raw_lane", {}).get("correct", False) for r in all_rows),
               "light_lane_correct_all": sum(r.get("light_lane", {}).get("correct", False) for r in all_rows),
               "initial_name_text_replay_mismatches": len(replay_mismatches)}
    report = {"evidence_boundary": "Observed bboxes only; same original checkpoint; GT names evaluation only. Legacy initial batch reconstructed with original full role order/chunk; atlas reconstructed with all original boxes/rotations. No production source or weight changes. Lane raw/light reads are diagnostic alternatives and do not change binding.",
              "weight_sha256": sha(weight), "source_hashes": source_hashes, "summary": summary,
              "initial_name_replay_mismatches": replay_mismatches, "rows": all_rows}
    save(args.out / "name-input-diagnosis.json", report)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
