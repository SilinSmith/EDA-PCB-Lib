from ultralytics import YOLO
import shutil
from pathlib import Path

MODEL_PATH = "yolo11l.pt"
DATA_YAML  = "training_data/data.yaml"
EPOCHS     = 120
BATCH      = 8
IMGSZ      = 1536
DEVICE     = 0
PROJECT    = "weights"
NAME       = "component_yolo11l_im1536"
MOSAIC     = 0.7
CLOSE_MOSAIC = 15
LR0        = 0.0005
WARMUP     = 3
PATIENCE   = 50

def main():
    model = YOLO(MODEL_PATH)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        amp=True,
        rect=False,
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        optimizer="auto",
        lr0=LR0,
        warmup_epochs=WARMUP,
        cos_lr=True,
        patience=PATIENCE,
        workers=2,
        # 漏检/难分类：适度加大分类权重，别一次加太猛
        cls=1.5,          # 默认0.5，漏检且类别混淆可1.0~1.5
        box=10,          # 默认7.5，框偏位可试8~10
        dfl=2.5,          # 默认1.5，小目标可试1.8~2.0
        hsv_h=0.2, hsv_s=0.2, hsv_v=0.2,
        fliplr=0.0, flipud=0.0, degrees=0.0,
        copy_paste=0.5,   # 小元件/小目标增强，提召回
        # 不要写 fl_gamma / fl_alpha，高层不支持
    )

    best = Path(PROJECT)/NAME/"weights"/"best.pt"
    if best.exists():
        shutil.copy2(best, Path(PROJECT)/"best_component_11l.pt")
        print("已保存 weights/best_component_11l.pt")

    # 低阈值看真实召回
    model.val(data=DATA_YAML, conf=0.1, iou=0.6, imgsz=IMGSZ, device=DEVICE)

if __name__ == "__main__":
    main()