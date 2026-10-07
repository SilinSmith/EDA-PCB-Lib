#!/usr/bin/env python3
"""
预处理脚本：将赛题训练数据（200_train_cases）转换为 YOLO 格式（左上角坐标系）。
输出目录结构：
  training_data/
    images/train/   # 180 张 PNG
    images/val/     # 20 张 PNG
    labels/train/   # 180 个 TXT
    labels/val/     # 20 个 TXT
    data.yaml
    classes.txt

用法：
    python prepare_yolo_dataset.py \
        --input_dir "Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)\200_train_cases" \
        --output_dir "Y:\workspace\EDA\eda-pcb\yolo_part\training_data" \
        --val_split 0.1
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Convert race dataset to YOLO format")
    parser.add_argument("--input_dir", required=True, help="Path to 200_train_cases folder")
    parser.add_argument("--output_dir", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def collect_classes(input_dir):
    """扫描所有json，收集所有出现的type，返回有序类别列表"""
    classes_set = set()
    input_path = Path(input_dir)
    for case_dir in sorted(input_path.iterdir()):
        if not case_dir.is_dir():
            continue
        # 查找包含 _target 的 .json 文件
        json_file = None
        for f in case_dir.iterdir():
            if "_target" in f.name and f.suffix == ".json":
                json_file = f
                break
        if json_file is None:
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for comp_key, comp_val in data.get("components", {}).items():
            t = comp_val.get("type")
            if t and isinstance(t, str):
                classes_set.add(t)
    return sorted(classes_set)


def process_case(case_dir, output_dir, class2id, mode="train"):
    """
    处理单个case：读取图片和json，翻转坐标，写入YOLO标签文件。
    返回 (img_rel_path, label_rel_path) 或 None（若出错）
    """
    png_path = None
    json_path = None
    for f in case_dir.iterdir():
        if f.suffix.lower() == ".png":
            png_path = f
        elif "_target" in f.name and f.suffix == ".json":
            json_path = f
    if png_path is None or json_path is None:
        print(f"Warning: {case_dir} missing png or json, skip.")
        return None

    # 读取图片获取高度
    try:
        with Image.open(png_path) as img:
            W, H = img.size
    except Exception as e:
        print(f"Error reading image {png_path}: {e}, skip.")
        return None

    # 读取json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建YOLO标签内容
    lines = []
    for comp_key, comp_val in data.get("components", {}).items():
        t = comp_val.get("type")
        bbox = comp_val.get("bbox")
        if not t or not bbox or len(bbox) != 4:
            continue
        if t not in class2id:
            continue

        # bbox: [x1, y1, x2, y2] 左下角坐标系
        x1, y1_lower, x2, y2_lower = bbox
        # 翻转y: 左上角坐标系
        y1_upper = H - y2_lower
        y2_upper = H - y1_lower
        # 确保 x1<x2, y1<y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1_upper > y2_upper:
            y1_upper, y2_upper = y2_upper, y1_upper

        # 归一化中心坐标和宽高
        x_center = (x1 + x2) / 2.0 / W
        y_center = (y1_upper + y2_upper) / 2.0 / H
        width = (x2 - x1) / W
        height = (y2_upper - y1_upper) / H

        # 裁剪到[0,1]防止越界
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        lines.append(f"{class2id[t]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 构造输出路径
    stem = png_path.stem  # 例如 "0001-KiCad"
    img_out_dir = output_dir / "images" / mode
    label_out_dir = output_dir / "labels" / mode
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片
    dst_img = img_out_dir / f"{stem}.png"
    shutil.copy2(str(png_path), str(dst_img))

    # 写入标签
    dst_label = label_out_dir / f"{stem}.txt"
    dst_label.write_text("\n".join(lines), encoding="utf-8")

    return (f"images/{mode}/{stem}.png", f"labels/{mode}/{stem}.txt")


def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input_dir {input_dir} does not exist.")
        sys.exit(1)

    # Step 1: 收集所有类别
    print("Scanning all cases to collect classes...")
    classes = collect_classes(input_dir)
    if not classes:
        print("No classes found. Check your input data.")
        sys.exit(1)
    print(f"Found {len(classes)} classes: {classes}")

    class2id = {c: i for i, c in enumerate(classes)}
    # 保存类别文件
    classes_file = output_dir / "classes.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    classes_file.write_text("\n".join(classes), encoding="utf-8")

    # Step 2: 获取所有case目录
    case_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Total cases: {len(case_dirs)}")

    # 随机打乱并划分
    random.shuffle(case_dirs)
    num_val = max(1, int(len(case_dirs) * args.val_split))
    val_dirs = case_dirs[:num_val]
    train_dirs = case_dirs[num_val:]
    print(f"Train cases: {len(train_dirs)}, Val cases: {len(val_dirs)}")

    # Step 3: 处理训练集
    print("Processing training set...")
    for case_dir in train_dirs:
        process_case(case_dir, output_dir, class2id, mode="train")

    # Step 4: 处理验证集
    print("Processing validation set...")
    for case_dir in val_dirs:
        process_case(case_dir, output_dir, class2id, mode="val")

    # Step 5: 生成 data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes
    }
    yaml_path = output_dir / "data.yaml"
    import yaml
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"data.yaml written to {yaml_path}")

    print("Done! Dataset prepared at:", output_dir)


if __name__ == "__main__":
    main()