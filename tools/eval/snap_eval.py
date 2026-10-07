"""把「框尺寸错了」和「框位置错了」分开量。

官方口径：bbox 只要求坐标误差 <=20px，不做 IoU 匹配。而 GT 框对符号类是固定图章
（r 8x20 / v 10x10 / c 16x20 / gnd 20x19 / pin 20x20 / testpoint 20x20）。
=> 只要「类别 + 中心」对，把预测框换成以预测中心为中心的规范图章，就该判对。

本脚本对同一个模型跑两种口径：
  A. baseline : 原框 vs GT，IoU>=0.5
  B. snapped  : 把预测框换成「以预测中心为中心的规范图章」，角点差 <=20px
"""
import argparse, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")

TOL = 20.0
IOU_MIN = 0.5


def load_gt(label_path, W, H):
    out = []
    p = Path(label_path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
        x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
        out.append((c, np.array([x1, y1, x2, y2])))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def corner_ok(a, b, tol=TOL):
    return max(abs(a[i] - b[i]) for i in range(4)) <= tol


def greedy(preds, gts, ok_fn):
    """preds/gts: list of (cls, box). 返回 (tp, fp, fn, pairs)"""
    cand = []
    for i, (pc, pb) in enumerate(preds):
        for j, (gc, gb) in enumerate(gts):
            if pc != gc:
                continue
            o = ok_fn(pb, gb)
            if o:
                cand.append((float(o), i, j))
    cand.sort(key=lambda t: -t[0])
    up, ug, pairs = set(), set(), []
    for o, i, j in cand:
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); pairs.append((i, j))
    tp = len(pairs)
    return tp, len(preds) - tp, len(gts) - tp, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"Y:\workspace\EDA\trainset_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--canon-split", default="train",
                    help="图章尺寸从哪个 split 统计（默认 train，避免用 val 泄漏）")
    ap.add_argument("--canon-json-root",
                    default=r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)",
                    help="从官方 JSON 直接读图章尺寸（推荐；传空串则改用标签统计）")
    ap.add_argument("--weights", default=r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect\pcb43_v2_from_best\weights\best.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO
    names = None
    dy = Path(a.root) / "data.yaml"
    if dy.exists():
        import re
        txt = dy.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
        if m:
            names = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    if names is None:
        names = (Path(a.root) / "classes.txt").read_text().split()
    print(f"classes {len(names)}")

    # 规范图章：对 GT 尺寸取值 <=2 的类，用 (max(w), max(h)) 的方框
    img_dir = Path(a.root) / "images" / a.split
    lab_dir = Path(a.root) / "labels" / a.split
    sizes = defaultdict(Counter)
    if a.canon_json_root:
        # 图章是官方标注格式的常数，直接从官方 JSON 读，不从（被合成图污染的）train 标签统计
        import json
        ncase = 0
        for jp in Path(a.canon_json_root).rglob("*.json"):
            try:
                doc = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            comps = doc.get("components")
            if not isinstance(comps, dict):
                continue
            ncase += 1
            for _, c in comps.items():
                t, b = c.get("type"), c.get("bbox")
                if not t or not isinstance(b, list) or len(b) != 4:
                    continue
                if t not in names:
                    continue
                sizes[names.index(t)][(round(abs(b[2] - b[0])), round(abs(b[3] - b[1])))] += 1
        print(f"图章尺寸取自官方 JSON：{a.canon_json_root}（{ncase} 个 case）")
    else:
        canon_img = Path(a.root) / "images" / a.canon_split
        canon_lab = Path(a.root) / "labels" / a.canon_split
        for lp in sorted(canon_lab.glob("*.txt")):
            ip = canon_img / (lp.stem + ".png")
            if not ip.exists():
                ip = canon_img / (lp.stem + ".jpg")
            if not ip.exists():
                continue
            W, H = Image.open(ip).size
            for c, b in load_gt(lp, W, H):
                sizes[c][(round(b[2] - b[0]), round(b[3] - b[1]))] += 1
        print(f"图章尺寸取自 {a.canon_split} 标签（{len(list(canon_lab.glob('*.txt')))} 张）")
    canon = {}
    nsnap = {}
    for c, cnt in sizes.items():
        # 真图章 = 尺寸取值在 ±2px 容差下只有 1~2 种（两种=横竖两朝向）
        coarse = Counter()
        for (w, h), k in cnt.items():
            coarse[(round(w / 2) * 2, round(h / 2) * 2)] += k
        nsnap[c] = len(coarse)
        if len(coarse) <= 2:
            w = max(s[0] for s in cnt); h = max(s[1] for s in cnt)
            canon[c] = max(w, h)
    print("规范图章（方框边长）:", {names[c]: v for c, v in sorted(canon.items())})
    print("非图章类（保持预测框原样）:", [names[c] for c in sorted(nsnap) if c not in canon])

    model = YOLO(a.weights)
    stat = {k: Counter() for k in ("base", "corner", "snap")}
    per = defaultdict(lambda: {k: Counter() for k in ("base", "corner", "snap")})
    files = sorted(lab_dir.glob("*.txt"))
    for n, lp in enumerate(files, 1):
        ip = img_dir / (lp.stem + ".png")
        if not ip.exists():
            ip = img_dir / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        im = Image.open(ip)
        W, H = im.size
        gts = load_gt(lp, W, H)
        r = model.predict(str(ip), imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        preds, snapped = [], []
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            preds.append((int(cls), box.astype(float)))
            side = canon.get(int(cls))
            if side is None:
                snapped.append((int(cls), box.astype(float)))
            else:
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                snapped.append((int(cls), np.array([cx-side/2, cy-side/2, cx+side/2, cy+side/2])))
        # 三臂：base=YOLO 口径(IoU>=0.5, 原框)
        #      corner=官方口径(角点差<=20px, 仍用原框)  <- 只看判据变化的收益
        #      snap  =官方口径 + 贴规范图章                 <- 只看贴图章的收益
        arms = (("base", preds, lambda p, g: iou(p, g) >= IOU_MIN),
                ("corner", preds, lambda p, g: corner_ok(p, g)),
                ("snap", snapped, lambda p, g: corner_ok(p, g)))
        for tag, pl, fn in arms:
            t, f, nn, pairs = greedy(pl, gts, fn)
            stat[tag].update(tp=t, fp=f, fn=nn)
            for i, j in pairs:
                per[pl[i][0]][tag].update(tp=1)
            hit = {i for i, _ in pairs}
            for i, (c, _) in enumerate(pl):
                if i not in hit:
                    per[c][tag].update(fp=1)
            gh = {j for _, j in pairs}
            for j, (c, _) in enumerate(gts):
                if j not in gh:
                    per[c][tag].update(fn=1)
        if n % 20 == 0:
            print(f"  {n}/{len(files)}")

    def pr(c):
        p_ = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0
        r_ = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0
        f = 2*p_*r_/(p_+r_) if p_+r_ else 0
        return p_, r_, f

    for tag in ("base", "corner", "snap"):
        p_, r_, f = pr(stat[tag])
        print(f"[{tag:>6}] micro  P {p_:.3f}  R {r_:.3f}  F1 {f:.3f}   "
              f"tp {stat[tag]['tp']}  fp {stat[tag]['fp']}  fn {stat[tag]['fn']}")
    print(f"\n{'class':<16}{'n_gt':>6} |{'base(IoU)':>22} |{'corner(原框)':>22} |{'snap(图章)':>22}")
    print(f"{'':<16}{'':>6} |{'P':>7}{'R':>7}{'F1':>8} |{'P':>7}{'R':>7}{'F1':>8} |{'P':>7}{'R':>7}{'F1':>8}")
    rows = []
    for c in range(len(names)):
        b = per[c]["base"]; k = per[c]["corner"]; s = per[c]["snap"]
        if b["tp"] + b["fp"] + b["fn"] == 0 and s["tp"] + s["fp"] + s["fn"] == 0:
            continue
        pb, rb, fb = pr(b); pk, rk, fk = pr(k); ps, rs, fs = pr(s)
        rows.append((b["tp"] + b["fn"], names[c], (pb, rb, fb), (pk, rk, fk), (ps, rs, fs)))
    for n_gt, nm, (pb, rb, fb), (pk, rk, fk), (ps, rs, fs) in sorted(rows, reverse=True):
        flag = "  <<<图章" if fs - fk > 0.10 else ("  <<<判据" if fk - fb > 0.10 else "")
        print(f"{nm:<16}{n_gt:>6} | {pb:>7.3f}{rb:>7.3f}{fb:>8.3f} | "
              f"{pk:>7.3f}{rk:>7.3f}{fk:>8.3f} | {ps:>7.3f}{rs:>7.3f}{fs:>8.3f}{flag}")


if __name__ == "__main__":
    main()
