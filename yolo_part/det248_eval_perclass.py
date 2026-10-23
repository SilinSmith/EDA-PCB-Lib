"""Per-class val comparison of old pcb43_v2 vs new det248 on the same det248 held-out split.

Usage: python eda-pcb/yolo_part/det248_eval_perclass.py
Writes eda-pcb/yolo_part/runs/det248_eval/perclass.json
"""
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part")
DATA = ROOT / "training_data2" / "data.yaml"
OUT = ROOT / "runs" / "det248_eval" / "perclass.json"

WEIGHTS = {
    "old_pcb43_v2": Path(r"Y:\workspace\EDA\eda-pcb\weights\pcb43_v2_from_best.pt"),
    "new_det248": ROOT / "weights" / "det248_best.pt",
}


def main() -> None:
    table = {}
    for tag, w in WEIGHTS.items():
        m = YOLO(str(w)).val(data=str(DATA), split="val", imgsz=1024, batch=8, device=0, verbose=False)
        names = m.names
        table[tag] = {
            names[int(i)]: {"mAP50": float(m.box.ap50[k]), "mAP50_95": float(m.box.ap[k])}
            for k, i in enumerate(m.box.ap_class_index)
        }
        table[tag]["__all__"] = {
            "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map),
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")

    classes = sorted(set(table["old_pcb43_v2"]) | set(table["new_det248"]))
    print(f"{'class':20s} {'old50':>7s} {'new50':>7s} {'old50-95':>9s} {'new50-95':>9s}")
    for c in classes:
        o = table["old_pcb43_v2"].get(c, {})
        n = table["new_det248"].get(c, {})
        print(
            f"{c:20s} {o.get('mAP50', float('nan')):7.3f} {n.get('mAP50', float('nan')):7.3f} "
            f"{o.get('mAP50_95', float('nan')):9.3f} {n.get('mAP50_95', float('nan')):9.3f}"
        )


if __name__ == "__main__":
    main()
