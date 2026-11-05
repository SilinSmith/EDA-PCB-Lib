"""在 PCB训练集（43 类 / 763 张）上训练原理图元件检测器。

与 train_yolo11l.py 的关系
-------------------------
沿用同一套超参哲学（cls/box/dfl 加权、mosaic 0.7、cos_lr、低置信度复验），
三处刻意不同：

1. **模型默认 yolo11m.pt。**
    原先 weights/ 里的命名是个陷阱，已经被整理过了：
       yolo11m.pt / yolo11l.pt   = 官方 COCO 权重（nc=80，date 2024-09-25，train_args.data=coco.yaml）
       yolo11s.pt / yolo26n.pt   = 官方 COCO 权重（nc=80）
       yolo11m_41cls_e120_im1280.pt / yolo11l_41cls_e120_im1536.pt
                                 = **团队自己在 training_data 上训出来的成品**
                                   （nc=41，类名就是这批元件类，train_args.data='training_data\data.yaml'）
       pre2-yolo11*.pt / pre3-yolo11*.pt / pre4-yolo11*.pt = 同上，也是团队成品（nc=41）
    整理前 `yolo11m.pt` / `yolo11l.pt` 这两个 Ultralytics 官方文件名是被团队成品覆盖掉的，
    谁写 YOLO('yolo11m.pt') 想拿 COCO，拿到的其实是已在那 200 张图上训过 120 轮的模型。
    从那种权重继续训，本次 val 里的图纸它早就见过，mAP 只能反映记忆而不是泛化
    （实测同为 1 epoch/5% 数据：污染起点 mAP50 0.758，干净起点 8.6e-05，差四个数量级）。
    脚本会用 check_checkpoint() 自动识别并警告；若指到 *_41cls_*.pt / pre2* / pre3* / pre4*，
    它会打印 `[提示]` 告诉你那是团队成品，但仍允许你继续（有时就是要微调）。
    架构仍选 11m：团队 11 组实验里 11m 每组都赢 11l
      component_yolo11m_im1536  mAP50 0.5761  mAP50-95 0.2937
      component_yolo11l_im1536-2 mAP50 0.5583  mAP50-95 0.3026
    `--model yolo11l.pt` 可以随时换回 11l。

2. **imgsz 默认 1024，不是 1536。**
   数据集里 550/763 张是 1024×1024 的原生切片，imgsz=1024 时它们**原样进网络**，
   没有任何重采样 —— 这才是切片的意义。拉到 1536 反而要先上采样 1.5×。

3. **加了 preflight 检查与 `--smoke`。**
   训 200 轮之前先确认 data.yaml 的 nc/路径/图-标配对没问题，避免跑几小时才发现路径写错。

用法
----
    python train_pcb43.py                            # 就这一条，直接开训（200 epoch / 11m / 1024）
    python train_pcb43.py --smoke                    # 1 分钟冒烟，先验证管线通不通
    python train_pcb43.py --model yolo11l.pt --imgsz 1280 --batch 4
    python train_pcb43.py --resume                   # 断点续训

用哪个 python 都行：本机 `python`（`K:\python\311`）与项目 venv（`Y:\workspace\EDA\eda-pcb\eda_yolo`）
都装了完整依赖，直接跑都行。若换到没装 ultralytics 的解释器上，脚本会自动改用 venv 重启自己。

显存参考（RTX 4080 16GB）：
    imgsz 1024 + 11m  → batch 8~12 稳，16 可能 OOM
    imgsz 1280 + 11m  → batch 4~6
    imgsz 1536 + 11m  → batch 2~4
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
RUNS = HERE / "runs" / "detect"
VENV_PY = HERE.parent / "eda_yolo" / "Scripts" / "python.exe"

# 中文目录名对训练器是个没必要的故障点，优先用指向它的 ASCII junction
YAML_CANDIDATES = [
    Path(r"Y:\workspace\EDA\trainset_v3\data.yaml"),
    Path(r"Y:\workspace\EDA\训练集_v3\data.yaml"),
    Path(r"Y:\workspace\EDA\trainset_v2\data.yaml"),
    Path(r"Y:\workspace\EDA\训练集_v2\data.yaml"),
    Path(r"Y:\workspace\EDA\PCB_trainset\data.yaml"),
    Path(r"Y:\workspace\EDA\PCB训练集\data.yaml"),
    # 重建版（tools/dev/build_trainset_v4.py 产物）——排在最后，免得悄悄改变默认基线。
    # v4 = 43 类（与 v3 同类名同语义，只清掉了 6 个重复框）；v4_merged = 42 类
    # （pin + testpoint 合成 tp，给「检测阶段合并、绑定阶段再分类」用，需要 --nc 42）。
    Path(r"Y:\workspace\EDA\work\trainset_v4\data.yaml"),
    Path(r"Y:\workspace\EDA\work\trainset_v4_merged\data.yaml"),
]
EXPECTED_NC = 43
# 允许的 nc 例外：重建版把 pin/testpoint 合成 tp 之后就是 42 类。
# 放在这里而不是「随便放行」，是因为 nc 变了而没人注意，跑一天的训练就白跑了。
NC_EXCEPTIONS = {42: "pin+testpoint 已合并成 tp（见 build_trainset_v4.py 的 merge_classes）"}


def ensure_runtime() -> None:
    """让 `python train_pcb43.py` 在任何解释器上都能直接起跑。

    本机 `python`（K:\\python\\311）已经装了完整依赖，所以这一层平时不会触发。
    但换成没装 ultralytics 的解释器时，缺依赖就自动用项目 venv 把自己重启一次，
    于是永远只需要 `python train_pcb43.py` 一条命令。
    """
    import importlib.util
    import os

    if importlib.util.find_spec("ultralytics") is not None:
        return
    if os.environ.get("PCB43_REEXEC") == "1":  # 重启过还是缺 -> 真没有，别来回 exec
        raise SystemExit(f"venv 解释器里也没有 ultralytics：{sys.executable}")
    if VENV_PY.is_file():
        print(f"[bootstrap] {sys.executable}\n            缺 ultralytics -> 改用 {VENV_PY}\n")
        sys.stdout.flush()
        os.environ["PCB43_REEXEC"] = "1"
        os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(
        "当前解释器没有 ultralytics，也没找到项目 venv：\n"
        f"  {VENV_PY}\n"
        "先建 venv 装依赖，或显式用带依赖的解释器运行本脚本。"
    )


def check_checkpoint(model_ref: str, names: list) -> None:
    """拦住「拿已经在同一批数据上训过的权重，跑出一个漂亮 mAP 然后当成基线」。

    整理后的 weights/ 命名约定：
        yolo11m.pt / yolo11l.pt / yolo11s.pt / yolo26n.pt
            = 官方 COCO 权重（nc=80，80 个英文类名）—— 干净起点
        yolo11m_41cls_e120_im1280.pt / yolo11l_41cls_e120_im1536.pt
        pre2-yolo11*.pt / pre3-yolo11*.pt / pre4-yolo11*.pt
            = 团队自己在 training_data 上训出来的成品（nc=41，类名就是这批元件类）
    指到第二组时，本次 val 里的图纸它早就见过 —— mAP 只能说明它记得住。
    """
    import torch

    p = Path(model_ref)
    if not p.is_file():
        return
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"[权重检查] 读不了 {p.name}（{type(e).__name__}），跳过\n")
        return
    ck_names = getattr(ck.get("model"), "names", None) if isinstance(ck, dict) else None
    if isinstance(ck_names, dict):
        ck_names = [ck_names[i] for i in sorted(ck_names)]
    if not ck_names:
        print(f"[权重检查] {p.name}: 读不到类名，按未知处理\n")
        return

    overlap = sorted(set(ck_names) & set(names))
    if not overlap:
        print(f"[权重检查] {p.name}: {len(ck_names)} 个类名，与 43 类元件集无重名 -> 干净起点 ✓")
        print("=" * 68)
        return

    ta = ck.get("train_args") or {} if isinstance(ck, dict) else {}
    same_nc = len(ck_names) == len(names)
    if same_nc and ta and any(k in str(ta.get("data")) for k in ("PCB训练集", "PCB_trainset", "训练集_v2", "trainset_v2", "训练集_v3", "trainset_v3")):
        # 这是「拿自己上一轮的 43 类成品继续训练」——本次想要的做法，不是事故。
        print("!" * 68)
        print(f"[继续训练] {p.name} 是同类名的 {len(names)} 类成品（data={ta.get('data')} "
              f"epochs={ta.get('epochs')} imgsz={ta.get('imgsz')}）")
        print("    类数与类名完全对齐，ultralytics 会整套加载检测头，不做任何 remap —— 这是最快的起点。")
        print("    但两件事必须清楚：")
        print("      1) 它的训练图与本次同源，本次 val 它已经见过。")
        print("         所以「接着训」的 mAP 只能说明它没被训坏，不能证明泛化变好。")
        print("         要一个诚实的数字，用 --model yolo11m.pt 从头训一轮做对照。")
        print("      2) 学习率要降（0.0005 -> 0.0002 上下），epoch 也可以砍，")
        print("         否则会把它已经学会的符号形状洗掉。")
        print("!" * 68)
        return

    print("!" * 68)
    print(f"[权重污染警告] {p.name} 已经认识本数据集的 {len(overlap)}/{len(names)} 个类名：")
    print(f"    {overlap[:12]}{' ...' if len(overlap) > 12 else ''}")
    if ta:
        print(f"    来历: data={ta.get('data')}  epochs={ta.get('epochs')}  imgsz={ta.get('imgsz')}")
    print("    它的训练图与本次数据同源，其中一部分正是本次 val。")
    print("    从它接着训出的 mAP 不是泛化能力，是记忆 —— 不能当基线。")
    if ta and "training_data" in str(ta.get("data")):
        print("    这是团队在 training_data 上训出来的成品，不是官方预训练权重。")
    print("    要真基线请用官方 COCO 权重：  --model yolo11m.pt")
    print("!" * 68)


def resolve_data_yaml(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"--data 指向的 data.yaml 不存在：{p}")
        return p
    for p in YAML_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit("找不到 data.yaml，检查 训练集_v2 / PCB训练集 是否还在：\n  "
                     + "\n  ".join(map(str, YAML_CANDIDATES)))


def resolve_model(name: str) -> str:
    """裸模型名先去 weights/ 找本地副本，找不到再交给 ultralytics 自己下载。"""
    p = Path(name)
    if p.is_file():
        return str(p)
    local = WEIGHTS / name
    if local.is_file():
        return str(local)
    return name


def preflight(yaml_path: Path, expected_nc: int | None = None) -> str:
    import yaml

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = cfg.get("names")
    nc = cfg.get("nc")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    names = list(names or [])
    if nc != len(names):
        raise SystemExit(f"data.yaml 自相矛盾：nc={nc} 但 names 有 {len(names)} 项")
    want = expected_nc or EXPECTED_NC
    if nc != want:
        detail = NC_EXCEPTIONS.get(nc)
        if detail and not expected_nc:
            print(f"注意      : nc={nc}（不是默认 {EXPECTED_NC}）—— {detail}")
        else:
            raise SystemExit(f"nc={nc}，期望 {want}。classes.txt 变了？"
                             + (f"\n（{nc} 是已知的例外：{detail}）" if detail else ""))

    root = Path(cfg.get("path") or yaml_path.parent)
    if not root.is_dir():
        raise SystemExit(f"data.yaml 里的 path 不存在：{root}\n（复制/移动过 PCB训练集？同步改 path）")

    counts = {}
    for key in ("train", "val"):
        rel = Path(cfg[key])
        d = rel if rel.is_absolute() else root / rel
        if not d.is_dir():
            raise SystemExit(f"{key} 目录不存在：{d}")
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        # 标签在 images/<split> 的同级 labels/<split>
        ldir = d.parent.parent / "labels" / d.name
        lbls = sorted(ldir.iterdir()) if ldir.is_dir() else []
        if len(imgs) != len(lbls):
            raise SystemExit(f"{key}: 图 {len(imgs)} 张、标签 {len(lbls)} 个，数量不匹配")
        counts[key] = len(imgs)

    print("=" * 68)
    print(f"data.yaml : {yaml_path}")
    print(f"数据集根   : {root}")
    print(f"类别       : nc={nc}")
    print(f"图像       : train={counts['train']}  val={counts['val']}  合计 {sum(counts.values())}")

    # 长尾一览：让训练前就知道哪些类几乎没样本
    hist = [0] * nc
    for split in ("train", "val"):
        rel = Path(cfg[split])
        d = rel if rel.is_absolute() else root / rel
        ldir = d.parent.parent / "labels" / d.name
        if not ldir.is_dir():
            continue
        for f in ldir.iterdir():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    hist[int(line.split()[0])] += 1
    order = sorted(range(nc), key=lambda i: -hist[i])
    head = "  ".join(f"{names[i]}={hist[i]}" for i in order[:6])
    tail = "  ".join(f"{names[i]}={hist[i]}" for i in order[-8:] if hist[i] < 20)
    print(f"最多 6 类  : {head}")
    print(f"少于 20 框 : {tail or '（无）'}")
    print("=" * 68)
    return str(yaml_path), names


def main() -> int:
    ap = argparse.ArgumentParser(description="PCB训练集 43 类元件检测训练", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11m.pt", help="权重名或路径；默认 yolo11m.pt（官方 COCO 干净起点，见文件头说明）")
    ap.add_argument("--data", default=None, help="data.yaml 路径；默认取 训练集_v3（找不到再回退 训练集_v2 / PCB训练集），"
                                                "最后才轮到重建版 work\\trainset_v4")
    ap.add_argument("--nc", type=int, default=None,
                    help="覆盖类别数检查（默认要求 43）。给 42 就是重建版的 pin+testpoint 合并集")
    ap.add_argument("--imgsz", type=int, default=1024, help="默认 1024，与切片原生尺寸一致")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=12, help="显存不够就减半；-1 = 自动估算")
    ap.add_argument("--device", default="0", help="'0' / '0,1' / 'cpu'")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr0", type=float, default=0.0005,
                    help="只有 --optimizer 不是 auto 时才生效；auto 会自己定 lr 并忽略本参数")
    ap.add_argument("--optimizer", default="auto",
                    help="auto(默认,自带 lr 搜索,会忽略 --lr0) / SGD / AdamW / ...；"
                         "想精确控制 lr 就写 --optimizer AdamW --lr0 0.0002")
    ap.add_argument("--mosaic", type=float, default=0.7)
    ap.add_argument("--box", type=float, default=10.0, help="框回归损失权重；默认 10.0 = 上一轮 pcb43_v2_from_best 的值")
    ap.add_argument("--cls", type=float, default=1.5, help="分类损失权重；默认 1.5 = 上一轮的值")
    ap.add_argument("--dfl", type=float, default=2.5, help="DFL 损失权重；默认 2.5 = 上一轮的值，小目标可再高")
    ap.add_argument("--close-mosaic", type=int, default=15)
    ap.add_argument("--copy-paste", type=float, default=0.3, help="小目标增强；0 关闭")
    ap.add_argument("--project", default=str(RUNS))
    ap.add_argument("--name", default=None, help="默认 pcb43_<model>_im<imgsz>")
    ap.add_argument("--cache", nargs="?", const="ram", default=False,
                    help="--cache 或 --cache ram 塞内存，--cache disk 写盘；3624 张 @1024 约 11GB，"
                         "内存不够就用 disk（第一次慢，之后快）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-period", type=int, default=-1,
                    help="每 N 个 epoch 存一次 checkpoint（-1 只存 last/best）；按我们的指标选模型时用 1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch / 5%% 数据 / imgsz 640，只验证管线通不通")
    args = ap.parse_args()

    ensure_runtime()
    from ultralytics import YOLO

    yaml_path, class_names = preflight(resolve_data_yaml(args.data), args.nc)
    model_ref = resolve_model(args.model)
    check_checkpoint(model_ref, class_names)
    stem = Path(model_ref).stem

    if args.smoke:
        args.epochs, args.imgsz, args.batch = 1, 640, 4
        args.name = (args.name or f"smoke_{stem}") + "_smoke"
        print(">>> 冒烟模式：1 epoch / 5% 数据 / imgsz 640")

    name = args.name or f"pcb43_{stem}_im{args.imgsz}"
    print(f"模型      : {model_ref}\n输出      : {Path(args.project) / name}")

    def train(batch: int):
        model = YOLO(model_ref)
        model.train(
            data=yaml_path,
            epochs=args.epochs,
            batch=batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=name,
            resume=args.resume,
            save_period=args.save_period,
            amp=True,
            rect=False,
            mosaic=args.mosaic,
            close_mosaic=args.close_mosaic,
            optimizer=args.optimizer,
            lr0=args.lr0,
            warmup_epochs=3,
            cos_lr=True,
            patience=args.patience,
            workers=args.workers,
            cache=args.cache,
            seed=args.seed,
            plots=True,
            # 长尾 + 难分类：适度加大分类权重，别一次加太猛。
            # 默认值取 pcb43_v2_from_best 那一轮实际用的 10.0/1.5/2.5，
            # 这样「续训」和上一轮是同一个目标函数，分数差才好归因。
            cls=args.cls,     # 上一轮 1.5（ultralytics 8.4 默认 1.0）
            box=args.box,     # 上一轮 10.0（默认 8.0，框偏位可 8~10）
            dfl=args.dfl,     # 上一轮 2.5（默认 2.0，小目标可 1.8~2.5）
            hsv_h=0.2, hsv_s=0.2, hsv_v=0.2,
            # 原理图不会被左右镜像，翻转是纯噪声
            fliplr=0.0, flipud=0.0, degrees=0.0,
            copy_paste=args.copy_paste,
            fraction=0.05 if args.smoke else 1.0,
        )
        return model

    # 显存不够就自动减半重试：跑一晚上的训练不该死在一次 OOM 上
    batch = args.batch
    model = None
    for attempt in range(4):
        try:
            model = train(batch)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch == -1 or attempt == 3:
                raise
            import torch

            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"\n[OOM] 显存不足，batch 降到 {batch} 重试（第 {attempt + 2} 次）\n")

    # 目录已存在时 ultralytics 会把输出自动改成 `name-2` / `name-3`，而这里的 `name` 不会变。
    # 必须从 trainer 回读真正落盘的目录，否则「复制 best.pt」和「低阈值复验」都会指向
    # **上一次**训练留下的旧权重——安静地给出一个属于旧模型的分数。
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(args.project) / name))
    if save_dir.name != name:
        print(f"注意      : 目录已存在，ultralytics 实际写入 {save_dir.name}/（不是 {name}/）")
    name = save_dir.name

    best = save_dir / "weights" / "best.pt"
    if best.is_file():
        dst = WEIGHTS / f"best_component_{name}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"已保存 {dst}")

    print("\n>>> 低阈值复验（看真实召回，conf=0.1）")
    # 一定要带 project/name，否则 ultralytics 会往「当前工作目录」下写 runs/detect/val*，
    # 在哪个目录敲命令就在那里留一堆垃圾，跟文件头「不依赖 CWD」的承诺也不符。
    YOLO(str(best) if best.is_file() else model_ref).val(
        data=yaml_path, conf=0.1, iou=0.6, imgsz=args.imgsz, device=args.device,
        project=args.project, name=f"{name}_conf010",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
