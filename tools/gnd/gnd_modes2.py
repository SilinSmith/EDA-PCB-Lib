"""GT 框把符号切掉了 —— 那把裁剪放大再聚类一次，看那 10 个模式是不是「切出来的假象」。

对照：
  tight  = 只裁 GT 框（之前 gnd_modes.py 的做法）
  margin = 以框中心为中心，外扩到 1.9 倍宽高、每边至少 MINPAD 像素

两种裁法都归一到 32x32 零均值单位方差，同样 k=10 聚类，比较
模式大小分布 / 类内紧致度 / 接触表。
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import discover, load_json          # noqa: E402

DATA_ROOT = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)"
SPLITS = ("10GT", "10_GTcase", "200_train_cases")
OUT = Path(r"Y:\workspace\EDA\work\preview")
K = 10
MINPAD = 8.0
FACTOR = 1.9
TILE = 32
SEED = 0


def norm_patch(gray, box, mode):
    H, W = gray.shape
    x1, y1, x2, y2 = box
    if mode == "margin":
        cw, ch = x2 - x1, y2 - y1
        hw = max(cw * FACTOR / 2, cw / 2 + MINPAD)
        hh = max(ch * FACTOR / 2, ch / 2 + MINPAD)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, x2 = int(cx - hw), int(cx + hw)
        y1, y2 = int(cy - hh), int(cy + hh)
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W, int(x2)); y2 = min(H, int(y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    p = np.asarray(Image.fromarray(gray[y1:y2, x1:x2]).resize((TILE, TILE), Image.LANCZOS),
                   dtype=np.float32)
    p = (p - p.mean()) / (p.std() + 1e-6)
    return p


def kmeans(X, k, iters=60, seed=SEED):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = (X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2 * X @ C.T
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    d = (X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2 * X @ C.T
    inertia = float(d[np.arange(len(X)), lab].mean())
    return lab, C, inertia


def main():
    grays = {}
    recs = []
    for case in discover(DATA_ROOT, SPLITS):
        try:
            doc = load_json(case.json_path)
            gray = np.asarray(Image.open(str(case.png_path)).convert("L"), dtype=np.uint8)
        except Exception:
            continue
        H = gray.shape[0]
        got = 0
        for key, comp in (doc.get("components") or {}).items():
            if comp.get("type") != "gnd":
                continue
            x1, y1, x2, y2 = comp["bbox"]
            box = (x1, H - y2, x2, H - y1)
            if (box[2] - box[0]) < 3 or (box[3] - box[1]) < 3:
                continue
            recs.append((case.case_id, box, gray))
            got += 1
        if got:
            grays[case.case_id] = got
    print(f"gnd instances {len(recs)}")

    for mode in ("tight", "margin"):
        X, keep = [], []
        for cid, box, gray in recs:
            p = norm_patch(gray, box, mode)
            if p is None:
                continue
            X.append(p.reshape(-1)); keep.append((cid, box))
        X = np.asarray(X, dtype=np.float32)
        lab, C, inertia = kmeans(X, K)
        sizes = Counter(lab.tolist())
        # 类内紧致度：每个模式内样本到质心的平均距离
        tight = float(np.mean([np.linalg.norm(X[i] - C[lab[i]]) for i in range(len(X))]))
        print(f"\n[{mode}] n={len(X)}  inertia(mean sq dist)={inertia:.1f}  "
              f"类内平均欧氏距={tight:.2f}")
        print("  模式大小:", {j: sizes.get(j, 0) for j in range(K)})

        # 接触表：每行一个模式
        CW = CH = 44; PAD = 3; PER = 12
        sheet = Image.new("RGB", (PER * (CW + PAD) + PAD, K * (CH + 18 + PAD) + PAD), (14, 17, 23))
        d = ImageDraw.Draw(sheet)
        for j in range(K):
            idx = [i for i in np.nonzero(lab == j)[0]]
            if not idx:
                continue
            # 取离质心最近的若干 + 最远的若干，看清模式内部一致性
            dist = np.array([np.linalg.norm(X[i] - C[j]) for i in idx])
            order = np.argsort(dist)
            pick = [idx[o] for o in order[:PER]]
            y = PAD + j * (CH + 18 + PAD)
            d.text((PAD, y), f"mode {j} (n={sizes.get(j,0)})", fill=(200, 210, 225))
            for t, i in enumerate(pick):
                cid, box = keep[i]
                gray = dict((c, g) for c, b, g in recs if c == cid).get(cid)
                if gray is None:
                    continue
                H = gray.shape[0]
                x1, y1, x2, y2 = box
                if mode == "margin":
                    cw, ch = x2 - x1, y2 - y1
                    hw = max(cw * FACTOR / 2, cw / 2 + MINPAD); hh = max(ch * FACTOR / 2, ch / 2 + MINPAD)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    x1, x2 = int(cx - hw), int(cx + hw); y1, y2 = int(cy - hh), int(cy + hh)
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(gray.shape[1], int(x2)); y2 = min(H, int(y2))
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                im = Image.fromarray(gray[y1:y2, x1:x2]).convert("RGB").resize((CW, CH), Image.LANCZOS)
                sheet.paste(im, (PAD + t * (CW + PAD), y + 12))
        p = OUT / f"gnd_modes_{mode}.png"
        sheet.save(p)
        print(f"  wrote {p} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
