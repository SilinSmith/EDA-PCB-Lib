#!/usr/bin/env python3
"""构建 training_data2：官方 200 页（沿用原 180/20 划分）+ 新补充 48 页（全进 train）。

- 类别表与 training_data/classes.txt 逐字节一致（序号不能变）
- 新集按 png 的 md5 去重，只取 case_* 前缀（SCH_* 丢弃）
- 新集坐标：像素单位、原点左下、Y 向上 → y_img = H - y_label，x 不变、无缩放
- 与官方 200 页交叉查重，避免新图混进 val 的分母

用法:
    python det248_prepare.py            # 构建 training_data2
    python det248_prepare.py --overlay 5  # 额外渲染 5 张检查图到 runs/det248/preview
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OLD_DATA = HERE / "training_data"
NEW_ROOT = Path(r"Y:\workspace\EDA\新补充训练集2")
OUT_DATA = HERE / "training_data2"
# 不能放在 runs/det248 里：该目录必须不存在，ultralytics 才会原样用它而不会另起 det248-2
PREVIEW = HERE / "runs" / "det248_preview"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_classes() -> list[str]:
    text = (OLD_DATA / "classes.txt").read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def collect_new_pairs() -> list[tuple[str, Path, Path]]:
    """返回 (输出 stem, png, json)；按内容 md5 去重，dataset/ 优先。"""
    cands: list[Path] = []
    ds = NEW_ROOT / "dataset"
    cands += [ds / f for f in sorted(p.name for p in ds.glob("*.png"))]
    base = NEW_ROOT / "新补充训练集(1)"
    for d in sorted(base.iterdir()):
        if d.is_dir():
            cands += [d / f for f in sorted(p.name for p in d.glob("*.png"))]

    seen: dict[str, str] = {}
    picked: list[tuple[str, Path, Path]] = []
    skipped_sch = 0
    for png in cands:
        case = png.stem
        if not case.startswith("case_"):
            skipped_sch += 1
            continue
        jso = png.with_suffix(".json")
        if not jso.is_file():
            print(f"WARN 缺 json，跳过: {png}")
            continue
        digest = md5(png)
        if digest in seen:
            continue
        seen[digest] = str(png)
        # 同名不同图会撞名（dataset\case_0034 与 NN\34\case_0034 内容不同），加来源标记
        tag = "ds" if png.parent == ds else "n" + png.parent.name
        picked.append((f"{case}__{tag}", png, jso))
    print(f"新集候选 {len(cands)} 张，去重后 case_* {len(picked)} 张，丢弃 SCH_* {skipped_sch} 张")
    return picked


def yolo_lines(jso: Path, W: int, H: int, class2id: dict[str, int]):
    data = json.loads(jso.read_text(encoding="utf-8"))
    lines, dropped = [], Counter()
    for _, comp in data.get("components", {}).items():
        t = comp.get("type")
        bbox = comp.get("bbox")
        if not t or not bbox or len(bbox) != 4:
            continue
        if t not in class2id:
            dropped[t] += 1
            continue
        x1, y1_low, x2, y2_low = (float(v) for v in bbox)
        # 原点左下、Y 向上 → 原点左上、Y 向下
        y1_up, y2_up = H - y2_low, H - y1_low
        x1, x2 = min(x1, x2), max(x1, x2)
        y1_up, y2_up = min(y1_up, y2_up), max(y1_up, y2_up)
        xc = (x1 + x2) / 2.0 / W
        yc = (y1_up + y2_up) / 2.0 / H
        w = (x2 - x1) / W
        h = (y2_up - y1_up) / H
        xc, yc = max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc))
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))
        lines.append(f"{class2id[t]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, dropped


def copy_old(class_names: list[str]) -> dict[str, int]:
    counts = {}
    for split in ("train", "val"):
        (OUT_DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
        imgs = sorted((OLD_DATA / "images" / split).glob("*.png"))
        for img in imgs:
            lab = OLD_DATA / "labels" / split / (img.stem + ".txt")
            if not lab.is_file():
                raise SystemExit(f"官方标签缺失: {lab}")
            shutil.copy2(img, OUT_DATA / "images" / split / img.name)
            shutil.copy2(lab, OUT_DATA / "labels" / split / lab.name)
        counts[split] = len(imgs)
    return counts


def overlay_check(picks: list[tuple[str, Path, Path]], class2id: dict[str, int], n: int) -> None:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for stem, png, jso in picks[:n]:
        data = json.loads(jso.read_text(encoding="utf-8"))
        with Image.open(png) as im:
            W, H = im.size
            canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for _, comp in data.get("components", {}).items():
            if comp.get("type") not in class2id:
                continue
            bbox = comp.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1_low, x2, y2_low = (float(v) for v in bbox)
            draw.rectangle([x1, H - y2_low, x2, H - y1_low], outline=(255, 0, 0), width=3)
        out = PREVIEW / f"{stem}_boxes.png"
        canvas.save(out)
        print(f"预览: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", type=int, default=0, help="渲染 N 张带框预览图到 runs/det248/preview")
    args = ap.parse_args()

    class_names = load_classes()
    class2id = {c: i for i, c in enumerate(class_names)}
    print(f"类别表: {len(class_names)} 类（沿用 training_data/classes.txt）")
    if len(class_names) != 41:
        raise SystemExit(f"类别数异常: {len(class_names)}")

    if OUT_DATA.exists():
        print(f"清空旧输出: {OUT_DATA}")
        shutil.rmtree(OUT_DATA)
    OUT_DATA.mkdir(parents=True)

    # 1) 官方 200 页：原样复制，划分不变
    old_counts = copy_old(class_names)
    print(f"官方 200 页: train={old_counts['train']} val={old_counts['val']}")

    # 2) 新补充 48 页：全进 train
    picks = collect_new_pairs()
    old_hashes = {md5(p) for p in (OLD_DATA / "images" / "train").glob("*.png")}
    old_hashes |= {md5(p) for p in (OLD_DATA / "images" / "val").glob("*.png")}
    cross_dup = [stem for stem, png, _ in picks if md5(png) in old_hashes]

    dropped_total: Counter = Counter()
    new_boxes = 0
    for stem, png, jso in picks:
        with Image.open(png) as im:
            W, H = im.size
        lines, dropped = yolo_lines(jso, W, H, class2id)
        dropped_total += dropped
        new_boxes += len(lines)
        shutil.copy2(png, OUT_DATA / "images" / "train" / f"{stem}.png")
        (OUT_DATA / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"新补充 48 页全部进 train: 框 {new_boxes} 个；丢弃类别 {dict(dropped_total) or '（无）'}")
    if cross_dup:
        print(f"注意: {len(cross_dup)} 张新图与官方 200 页内容重复: {cross_dup}")
    else:
        print("与官方 200 页无重复图像")

    # 3) classes.txt（逐字节复制）+ data.yaml
    shutil.copy2(OLD_DATA / "classes.txt", OUT_DATA / "classes.txt")
    import yaml

    (OUT_DATA / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(OUT_DATA),
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": class_names,
            },
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    stats = {
        "old_train": old_counts["train"],
        "old_val": old_counts["val"],
        "new_train": len(picks),
        "new_boxes": new_boxes,
        "dropped_types": dict(dropped_total),
        "cross_dup_with_official": cross_dup,
    }
    (OUT_DATA / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    tr = len(list((OUT_DATA / "images" / "train").glob("*.png")))
    va = len(list((OUT_DATA / "images" / "val").glob("*.png")))
    print(f"完成: train={tr} val={va} 合计 {tr + va} 页 → {OUT_DATA}")

    if args.overlay:
        overlay_check(picks, class2id, args.overlay)
        # 官方 val 也导出一张做对照
        vimg = sorted((OLD_DATA / "images" / "val").glob("*.png"))[0]
        PREVIEW.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vimg, PREVIEW / f"official_val_{vimg.stem}.png")
        shutil.copy2(OLD_DATA / "labels" / "val" / (vimg.stem + ".txt"), PREVIEW / f"official_val_{vimg.stem}.txt")
        print(f"官方 val 原图+标签: {PREVIEW / ('official_val_' + vimg.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
