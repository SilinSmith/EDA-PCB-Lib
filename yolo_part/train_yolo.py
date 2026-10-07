#!/usr/bin/env python3
"""
训练 YOLO 模型（元件检测）
用法：python train_yolo.py
训练完成后，最佳模型会被复制为 weights/best_component.pt
"""

from ultralytics import YOLO
import shutil
from pathlib import Path

# ---------- 配置参数（你可以根据需要修改） ----------
MODEL_PATH = "yolo11s.pt"              # 预训练权重
DATA_YAML = "training_data/data.yaml"  # 数据集配置
EPOCHS = 100                           # 训练轮数
BATCH = 16                             # 批次大小（根据显存调整）
IMGSZ = 640                            # 输入图片尺寸
DEVICE = 0                             # GPU 编号（0 表示第一块显卡）
PROJECT = "weights"                    # 保存项目目录
NAME = "component_detection"           # 实验名称
# ---------------------------------------------------

def main():
    print("正在加载模型...")
    model = YOLO(MODEL_PATH)

    print("开始训练...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        verbose=True
    )

    # 训练完成后，最佳模型位于 PROJECT/NAME/weights/best.pt
    best_model_path = Path(PROJECT) / NAME / "weights" / "best.pt"
    target_path = Path(PROJECT) / "best_component.pt"

    if best_model_path.exists():
        shutil.copy2(str(best_model_path), str(target_path))
        print(f"最佳模型已复制到: {target_path}")
    else:
        print("警告：未找到 best.pt，请检查训练是否成功。")

    print("训练结束！")

if __name__ == "__main__":
    main()