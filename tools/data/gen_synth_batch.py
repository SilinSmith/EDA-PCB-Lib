"""Generate a large synthetic schematic set (default 3000) into `合成原理图数据集`.

Why not `synth/make_demo.py`
----------------------------
It montages every generated image at the end.  At 3000 sheets that is a needless gigabyte of
PIL work, so this script calls `synth.scene.generate` + `synth.emit.emit/write` directly.

Parallelism
-----------
Each worker owns a private staging root `_parts/w<k>/` and writes through the *canonical*
`synth.emit.write`, so the on-disk format cannot drift from the demo set.  The parent then
`os.replace`s every file into the final `images/ gt/ labels_comp/ labels_text/` tree (a rename
inside one volume: 12000 metadata ops, a couple of seconds) and concatenates the per-worker
`texts.jsonl` in case order.  Nothing is ever copied, and two workers never touch one file.

Reproducibility
---------------
Master seed (default 20261102) -> `random.Random(master)` -> N per-case seeds consumed in index
order.  Worker count and chunking cannot change any case's seed, so the same master seed always
yields the same N sheets.

    $env:PYTHONIOENCODING='utf-8'
    K:\python\311\python.exe work\\gen_synth_batch.py --n 3000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(EDA / "eda-pcb"))

from synth import emit as emit_mod  # noqa: E402
from synth.scene import GenConfig, generate  # noqa: E402

DEFAULT_OUT = EDA / "合成原理图数据集"
REAL_CLASSES = EDA / "PCB训练集" / "classes.txt"

# Cycled, not random: every palette has to be equally represented, and `datasheet` (pure
# greyscale) is the stress test for any scheme that segments by colour.
PALETTE_CYCLE = ["kicad", "altium", "jlc", "datasheet"]

MASTER_SEED = 20261102

SUBS = ("images", "gt", "labels_comp", "labels_text")

_CLASS_ID: dict[str, int] = {}


def load_classes() -> list[str]:
    """`PCB训练集/classes.txt` is the single source of truth for the 43 label ids."""
    names = [l.strip() for l in REAL_CLASSES.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not names:
        raise SystemExit(f"empty class list: {REAL_CLASSES}")
    return names


def case_seeds(n: int, master: int) -> list[int]:
    rng = random.Random(master)
    return [rng.randrange(1 << 30) for _ in range(n)]


def _init(class_names: list[str]) -> None:
    global _CLASS_ID
    _CLASS_ID = {n: i for i, n in enumerate(class_names)}


def _run_chunk(job) -> dict:
    """Generate cases [start, end) into this worker's private staging root."""
    start, end, stage_s, seeds = job
    stage = Path(stage_s)

    types: Counter = Counter()
    roles: Counter = Counter()
    dropped = 0
    rows = []
    t0 = time.time()
    for i in range(start, end):
        palette = PALETTE_CYCLE[i % len(PALETTE_CYCLE)]
        name = f"synth_{i + 1:04d}"
        cfg = GenConfig(palette=palette, seed=seeds[i])  # width/height None -> CANVAS_SIZES
        scene = generate(cfg)
        em = emit_mod.emit(scene, name, _CLASS_ID)
        emit_mod.write(em, stage, save_image=True)

        types.update(c.ctype for c in scene.comps)
        roles.update(t.role for t in scene.texts if t.content)
        dropped += len(scene.comps) - len(em.comp_labels)
        rows.append({
            "case": name, "palette": palette, "size": [scene.w, scene.h], "seed": seeds[i],
            "components": len(scene.comps), "nets": len(scene.nets),
            "edges": sum(len(n.edges) for n in scene.nets),
            "texts": len(em.texts),
            "pins_on_nets": sum(len(n.pins) for n in scene.nets),
        })
        done = i - start + 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  [part {start:04d}-{end:04d}] {done}/{end - start} "
                  f"({el / done:.2f} s/sheet, eta {(end - i - 1) * el / done:.0f}s)", flush=True)
    return {"start": start, "rows": rows, "types": dict(types), "roles": dict(roles),
            "dropped": dropped, "seconds": time.time() - t0}


def merge(out: Path, parts: list[Path]) -> None:
    """Rename every staged artefact into the final tree, then rebuild texts.jsonl."""
    for sub in SUBS:
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for stage in parts:
        for sub in SUBS:
            src_dir = stage / sub
            if not src_dir.is_dir():
                continue
            for f in src_dir.iterdir():
                dst = out / sub / f.name
                if dst.exists():
                    dst.unlink()
                os.replace(f, dst)
                n += 1
    with (out / "texts.jsonl").open("w", encoding="utf-8") as dst:
        for stage in parts:
            p = stage / "texts.jsonl"
            if p.exists():
                dst.write(p.read_text(encoding="utf-8"))
    for stage in parts:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"merged {n} files into {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--master-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--force", action="store_true", help="delete an existing output tree first")
    args = ap.parse_args()

    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} exists and is not empty; pass --force to replace it")
        shutil.rmtree(out)
    parts_root = out / "_parts"
    for sub in SUBS + ("_parts",):
        (out / sub).mkdir(parents=True, exist_ok=True)

    class_names = load_classes()
    seeds = case_seeds(args.n, args.master_seed)
    print(f"classes      : {len(class_names)} (from {REAL_CLASSES})")
    print(f"master seed  : {args.master_seed}")
    print(f"target       : {args.n} sheets -> {out}")
    print("palette plan : " + ", ".join(
        f"{p}={sum(1 for i in range(args.n) if PALETTE_CYCLE[i % 4] == p)}" for p in PALETTE_CYCLE))

    # Balanced chunks so every worker gets an equal slice of the palette cycle.
    workers = max(1, args.workers)
    bounds = [(args.n * k) // workers for k in range(workers + 1)]
    stages = [parts_root / f"w{k:02d}" for k in range(workers) if bounds[k + 1] > bounds[k]]
    jobs = [(bounds[k], bounds[k + 1], str(stages[j]), seeds)
            for j, k in enumerate(k for k in range(workers) if bounds[k + 1] > bounds[k])]
    for s in stages:
        for sub in SUBS:
            (s / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if len(jobs) == 1:
        results = [_run_chunk(jobs[0])]
    else:
        with Pool(len(jobs), initializer=_init, initargs=(class_names,)) as pool:
            results = list(pool.imap_unordered(_run_chunk, jobs))
    gen_elapsed = time.time() - t0
    merge(out, stages)
    elapsed = time.time() - t0

    rows: list[dict] = []
    types: Counter = Counter()
    roles: Counter = Counter()
    dropped = 0
    for r in results:
        rows.extend(r["rows"])
        types.update(r["types"])
        roles.update(r["roles"])
        dropped += r["dropped"]
    rows.sort(key=lambda r: r["case"])

    emit_mod.write_class_lists(out, class_names)
    for f in ("data_comp.yaml", "data_text.yaml"):
        t = (out / f).read_text(encoding="utf-8").replace("train: images\n", "train: images/\n")
        (out / f).write_text(t, encoding="utf-8")

    n_comp = sum(r["components"] for r in rows)
    n_net = sum(r["nets"] for r in rows)
    n_text = sum(r["texts"] for r in rows)
    palettes = Counter(r["palette"] for r in rows)
    sizes = Counter(tuple(r["size"]) for r in rows)
    summary = {
        "n": len(rows), "master_seed": args.master_seed,
        "seed_derivation": "seeds = [random.Random(master_seed).randrange(1<<30) for _ in range(n)]",
        "workers": len(jobs), "generation_seconds": round(gen_elapsed, 2),
        "total_seconds": round(elapsed, 2),
        "cpu_seconds_per_sheet": round(sum(r["seconds"] for r in results) / max(1, len(rows)), 3),
        "wall_seconds_per_sheet": round(gen_elapsed / max(1, len(rows)), 3),
        "classes": class_names,
        "palette_counts": dict(palettes),
        "canvas_counts": {f"{w}x{h}": c for (w, h), c in sizes.most_common()},
        "class_hist_components": dict(types),
        "text_role_hist": dict(roles),
        "components": n_comp, "nets": n_net, "texts": n_text,
        "edges": sum(r["edges"] for r in rows),
        "pins_on_nets": sum(r["pins_on_nets"] for r in rows),
        "dropped_by_normalisation": dropped,
        "cases": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    print("\n=== generated ===")
    print(f"sheets            : {len(rows)}   in {gen_elapsed / 60:.2f} min "
          f"({gen_elapsed / max(1, len(rows)):.3f} s/sheet wall, {len(jobs)} workers)")
    print(f"palettes          : {dict(palettes)}")
    print(f"canvases (top 6)  : " + "  ".join(f"{w}x{h}:{c}" for (w, h), c in sizes.most_common(6)))
    print(f"components        : {n_comp}  ({n_comp / max(1, len(rows)):.1f}/sheet)")
    print(f"nets / edges      : {n_net} / {summary['edges']}")
    print(f"texts             : {n_text}  ({n_text / max(1, len(rows)):.1f}/sheet)")
    print(f"component types   : {len(types)} of {len(class_names)} classes present")
    print(f"text roles        : " + "  ".join(f"{k}:{v}" for k, v in roles.most_common()))
    print(f"normalisation loss: {dropped} component boxes")
    print(f"out               : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
