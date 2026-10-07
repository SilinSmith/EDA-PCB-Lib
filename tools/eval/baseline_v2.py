"""Score the previous model on the NEW yardstick, so a fine-tuned run can still be compared.

`训练集_v2` moved the 10 held-out `10GT` sheets out of `val`, so the old `mAP50 0.6049` was
measured on 91 val samples and the new run will be measured on 77. Comparing those two directly
would credit the new run for a set change. Re-scoring the old checkpoint here gives the baseline
on exactly the same 77 val samples, and also gives the first ever number on the 10GT test set.

    $env:PYTHONIOENCODING='utf-8'
    K:\\python\\311\\python.exe work\\baseline_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_trainset_v2 import ASCII_LINK, DEFAULT_OUT  # noqa: E402

OLD = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect\pcb43_yolo11m_im1024-2\weights\best.pt")
PROJECT = Path(r"Y:\workspace\EDA\work\recon\baseline_v2")


def main() -> int:
    from ultralytics import YOLO

    if not OLD.is_file():
        raise SystemExit(f"missing old checkpoint: {OLD}")

    # `data.yaml` prefers an ASCII junction; make sure it exists so the path never has to carry
    # Chinese characters into the trainer.
    if not ASCII_LINK.exists():
        import subprocess

        subprocess.run(["cmd", "/c", "mklink", "/J", str(ASCII_LINK), str(DEFAULT_OUT)], check=True)

    root = ASCII_LINK.as_posix()
    for name, val in (("data.yaml", "images/val"), ("data_test.yaml", "images/test")):
        p = DEFAULT_OUT / name
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines = [l for l in lines if not l.startswith(("path:", "train:", "val:"))]
        nc = next(l for l in lines if l.startswith("nc:"))
        names = lines[lines.index(nc) + 1:]
        p.write_text(f"path: {root}\ntrain: images/train\nval: {val}\n{nc}\n"
                     + "\n".join(names) + "\n", encoding="utf-8")

    out = {}
    for name, split in (("data.yaml", "val77"), ("data_test.yaml", "test10gt")):
        model = YOLO(str(OLD))
        r = model.val(data=str(DEFAULT_OUT / name), imgsz=1024, batch=8, device=0,
                      project=str(PROJECT), name=split, verbose=False, plots=False)
        sub = "test" if split == "test10gt" else "val"
        n = sum(1 for p in (DEFAULT_OUT / "images" / sub).iterdir() if p.is_file())
        out[split] = {
            "mAP50": round(float(r.box.map50), 5),
            "mAP50-95": round(float(r.box.map), 5),
            "precision": round(float(r.box.mp), 5),
            "recall": round(float(r.box.mr), 5),
            "n_images": n,
        }
        print(f"{split:9} mAP50 {out[split]['mAP50']:.4f}  mAP50-95 {out[split]['mAP50-95']:.4f}"
              f"  P {out[split]['precision']:.4f}  R {out[split]['recall']:.4f}"
              f"  ({out[split]['n_images']} images)")
        # per-class, so a drop on the two classes that only the synthetic set teaches shows up
        names = model.names
        per = {}
        for c, i in enumerate(r.box.ap_class_index):
            per[names[int(i)]] = round(float(r.box.maps[c]), 5)
        out[split]["per_class_50_95"] = per

    (PROJECT / "baseline.json").parent.mkdir(parents=True, exist_ok=True)
    (PROJECT / "baseline.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"\nwrote {PROJECT / 'baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
