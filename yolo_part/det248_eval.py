"""Evaluate any YOLO weight on the det248 held-out val split (same data.yaml, imgsz 1024).

Usage:
    python eda-pcb/yolo_part/det248_eval.py <weights.pt> <tag>
Writes eda-pcb/yolo_part/runs/det248_eval/<tag>.json and prints overall metrics.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part")
DATA = ROOT / "training_data2" / "data.yaml"


def main() -> None:
    weights = sys.argv[1]
    tag = sys.argv[2]
    out_dir = ROOT / "runs" / "det248_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    m = model.val(data=str(DATA), split="val", imgsz=1024, batch=8, device=0, verbose=False)

    res = {
        "weights": weights,
        "data": str(DATA),
        "imgsz": 1024,
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
