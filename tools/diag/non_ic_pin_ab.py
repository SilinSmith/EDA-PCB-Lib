"""Colour-pin inference and image verification; no GT annotations accepted.

Inference opens images and frozen predictions. Verification checks published
coordinates against those image measurements and checks IC output isolation.
These checks establish provenance and consistency, not independent accuracy.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage1 import nets
from stage1.passive_pins import AXIAL_TYPES, SINGLE_TYPES, refine_pins


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def infer(stage1_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    changes = {}
    for path in sorted(stage1_dir.glob("*.stage1.json")):
        doc = read(path)
        bgr = cv2.imdecode(np.fromfile(doc["image"]["path"], np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(doc["image"]["path"])
        before = copy.deepcopy(doc["pins"])
        refined, report = refine_pins(doc["components"], doc["pins"], bgr)
        for c in doc["components"]:
            if c["type"] == "box":
                assert refined.get(c["id"]) == before.get(c["id"])
        synth = {}
        connected = {}
        # IC metadata are kept aligned with the untouched IC point sequence.
        for cid, records in doc.get("pin_evidence", {}).items():
            if records:
                synth[cid] = [int(r.get("synthetic", False)) for r in records]
                connected[cid] = [r.get("connected") for r in records]
        segs = doc["wire_detail"]["segments"]
        groups = doc["wire_detail"]["groups"]
        original_groups = np.asarray(groups, dtype=int)
        ink = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 180
        net_report = {}
        # Grouping is deliberately frozen: the experiment only changes which
        # measured terminal attaches to an existing wire group.
        with patch.object(nets.labels_mod, "wire_groups", return_value=original_groups):
            new_nets, _ = nets.build_nets(refined, segs, ink=ink,
                pin_synth={k: v for k, v in synth.items() if k not in report},
                pin_connected={k: v for k, v in connected.items() if k not in report},
                components=doc["components"],
                report=net_report)
            old_nets, _ = nets.build_nets(before, segs, ink=ink,
                pin_synth=synth, pin_connected=connected, components=doc["components"],
                report={})
        assert old_nets == doc["nets"], (path.name, "Frozen baseline does not reproduce")
        doc["pins"], doc["nets"], doc["pin_join"] = refined, new_nets, net_report
        doc["passive_pin_report"] = report
        (out_dir / path.name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        changes[doc["case"]] = report
    (out_dir.parent / "terminal-changes.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def hashes(paths):
    return {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}


def finalize(stage1_dir, initial_dir, out_dir, cases, mono):
    """Replay stage2 with frozen initial OCR; later selective OCR still runs."""
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", EDA_GPU_CAP="0",
        STAGE2_PINNUM_CONSENSUS="1", STAGE2_BOX_PIN_DUPLICATE_BEST="1",
        STAGE2_PIN_TRUST_BIND="1", STAGE2_PIN_NAME_EDGE_MIDPOINT="0",
        STAGE2_PIN_NAME_OCR="on", STAGE2_MONO_PIN_OCR=mono)
    from stage2 import run
    from stage2.text import TextEngine
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = TextEngine(device="cpu", allow_missing=False)
    stats = []
    for case in cases:
        calls = []
        def frozen_read(image, boxes, roles=None):
            calls.append(True)
            assert len(calls) == 1, "Unexpected repeated initial OCR"
            records = read(initial_dir / f"{case}.json")
            if roles is not None:
                wanted = set(roles)
                for record in records:
                    record["reads"] = {k: v for k, v in record["reads"].items() if k in wanted}
            return records
        engine.read = frozen_read
        result = run.process(case, stage1_dir, out_dir, engine, texts="both", recut="off")
        assert len(calls) == 1
        stats.append(result)
        print(f"{case}: {result['t_total']:.2f}s", flush=True)
    origins = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
               if name.startswith("stage2") and getattr(module, "__file__", None)}
    root = Path(__file__).resolve().parents[2]
    assert all(Path(path).is_relative_to(root) for path in origins.values())
    save(out_dir / "inference-verification.json", dict(source=str(root), mono=mono,
        stats=stats, module_origins=origins, initial_capture=False, gt_used_in_inference=False))


def smoke(images, out_dir):
    """Normal entry, actual YOLO/CV/wires, no labels or detector replacements."""
    from stage1 import pipeline
    root = Path(__file__).resolve().parents[2]
    weights = root / "weights" / "pcb43_v2_from_best.pt"
    summaries = []
    for image in images:
        case = SimpleNamespace(case_id=image.stem, png_path=image)
        docs = {}
        for variant, enabled in (("baseline", False), ("candidate", True)):
            with patch.object(pipeline, "NON_IC_PINS_ON", enabled):
                doc = pipeline.run_case(case, str(weights), imgsz=1600, device="cpu")
            assert not doc["errors"], (case.case_id, doc["errors"])
            save(out_dir / variant / f"{case.case_id}.stage1.json", doc)
            docs[variant] = doc
            print(case.case_id, variant, doc["timing"]["total_s"], flush=True)
        b, c = docs["baseline"], docs["candidate"]
        assert b["components"] == c["components"]
        assert b["wire_detail"] == c["wire_detail"]
        for comp in b["components"]:
            if comp["type"] == "box":
                cid = comp["id"]
                assert b["pins"].get(cid) == c["pins"].get(cid)
                assert b["pin_evidence"].get(cid) == c["pin_evidence"].get(cid)
        summaries.append(dict(case=case.case_id, image_sha256=hashes([image])[str(image)],
            components=len(c["components"]), changed_non_ic=len(c["passive_pin_report"]),
            ic_pins_and_evidence_unchanged=True, errors=[],
            timing={v: d["timing"] for v, d in docs.items()}))
    save(out_dir / "verification.json", dict(gt_used_in_inference=False,
        normal_entry=True, detector="YOLO+CV", device="cpu", imgsz=1600, cases=summaries))


def image_check(stage1_dir, baseline_stage1, baseline_dir, candidate_dir, out_dir):
    """Check published coordinates against image measurements, without any GT."""
    totals, types, cases, failures = Counter(), {}, [], []
    def output_key(doc, comp):
        box = [round(float(x), 2) for x in comp["bbox_json"]]
        options = [k for k, c in doc["components"].items()
                   if c["type"] == comp["type"] and c["bbox"] == box]
        return options[0] if len(options) == 1 else None
    for path in sorted(stage1_dir.glob("*.stage1.json")):
        st = read(path)
        case = st["case"]
        old = read(baseline_stage1 / path.name)
        baseline = read(baseline_dir / f"{case}.json")
        candidate = read(candidate_dir / f"{case}.json")
        index = read(candidate_dir / f"{case}.pinindex.json")
        assert baseline["components"] == candidate["components"]
        for key, c in baseline["components"].items():
            if c["type"] == "box":
                assert baseline["pins"].get(key) == candidate["pins"].get(key)
        stats = Counter()
        h = st["image"]["h"]
        for comp in st["components"]:
            cid, typ = comp["id"], comp["type"]
            if typ == "box":
                assert old["pins"].get(cid) == st["pins"].get(cid)
            report = st.get("passive_pin_report", {}).get(cid)
            if not report:
                continue
            key = output_key(candidate, comp)
            type_stats = types.setdefault(typ, Counter())
            for counter in (totals, stats, type_stats):
                counter["confirmed_components"] += 1
                counter["old_candidates"] += len(report["before"])
                counter["measured_terminals"] += len(report["after"])
            for i, point in enumerate(st["pins"][cid]):
                pk = index.get(cid, {}).get(str(i))
                final = candidate["pins"].get(key, {}).get(pk)
                expected = np.array([point[0], h-point[1]], float)
                if final is None:
                    failures.append(dict(case=case, cid=cid, index=i, reason="not_published"))
                    continue
                got = np.array([final["point"]["x"], final["point"]["y"]], float)
                if np.linalg.norm(got-expected) > 0.02:
                    failures.append(dict(case=case, cid=cid, index=i,
                                         reason="coordinate_overwritten", expected=expected.tolist(), got=got.tolist()))
                else:
                    for counter in (totals, stats, type_stats):
                        counter["published_at_measured_boundary"] += 1
        if not st.get("passive_pin_report"):
            assert old["pins"] == st["pins"], "No colour evidence must leave all candidates unchanged"
        cases.append(dict(case=case, counts=dict(stats), ic_unchanged=True,
                          no_colour_changes=not bool(st.get("passive_pin_report"))))
    result = dict(gt_read=False, coordinate_reference="image_red_green_boundary",
                  counts=dict(totals), by_type={t: dict(v) for t,v in types.items()},
                  cases=cases, failures=failures,
                  note="Consistency/provenance check, not independently annotated accuracy. Ambiguous and unconfirmed components use the old path.")
    save(out_dir / "image-check.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert not failures, "Measured coordinates were omitted or overwritten"


def main():
    p = argparse.ArgumentParser()
    modes = p.add_subparsers(dest="mode", required=True)
    sp = modes.add_parser("infer")
    sp.add_argument("--stage1", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp = modes.add_parser("finalize")
    for name in ("stage1", "initial", "out"):
        sp.add_argument(f"--{name}", type=Path, required=True)
    sp.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    sp.add_argument("--mono", choices=("on", "off", "auto"), default="on")
    sp = modes.add_parser("smoke")
    sp.add_argument("--images", type=Path, nargs="+", required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp = modes.add_parser("image-check")
    for name in ("stage1", "baseline-stage1", "baseline", "candidate", "out"):
        sp.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "infer":
        infer(a.stage1, a.out / "stage1")
    elif a.mode == "finalize":
        finalize(a.stage1, a.initial, a.out, [s.strip() for s in a.cases.split(",")], a.mono)
    elif a.mode == "smoke":
        smoke(a.images, a.out)
    elif a.mode == "image-check":
        image_check(a.stage1, a.baseline_stage1, a.baseline, a.candidate, a.out)


if __name__ == "__main__":
    main()
