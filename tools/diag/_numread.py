"""引脚号（pin_number）识别精度的直接基准 + **按框宽分桶** + 裁剪外扩网格扫描。

为什么要这个：第 6d 节修掉 `FAR` 之后，`pin_number` 的候选池已经装满了（case 1 的
`U1`：21 → 26），可 `pin_f1` **一分没动** —— 因为 **CRNN 把 3 位号读成 2 位**
（框宽 34~35px 只读出 `13`/`14`/`15`，GT 是 `113`/`114`/`115`），和位号那边
`C160→C10` / `C161→C101` / `C155→C15` 是同一个病。所以这次**必须按框宽分桶看**，
不能再只看整体 exact（整体的 79% 会被 1~2 位号撑起来，把 3 位号的 0% 淹掉）。

`TextEngine.read` 的链路上有**两次外扩**：
  ① `stage2\\text.py` 的 `pad_box(box)`（默认 `PAD_MIN=2.0` / `PAD_MIN_Y=3.0` / `PAD_FRAC_Y=0.20`）
  ② `textrec\\dataset.py:143 crop_gray(img, bbox)` 自己按 `PAD_X_FRAC=0.12` / `PAD_Y_FRAC=0.25`
     再扩一次（它的 docstring 写着「训练与推理必须共用这一个函数」）
⇒ 推理侧的取景比训练侧大一圈，字形被压小。本工具就是为了量这件事。

用法：
    python work\\tools\\diag\\_numread.py --dir work\\_fast_out_ownbb
    python work\\tools\\diag\\_numread.py --only 1 --show 40
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "eda-pcb" / "stage2"))

import text as T                                            # noqa: E402

DSROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
GT9 = ("1", "2", "3", "4", "5", "7", "8", "9", "10")
IC_TYPES = ("box", "boxic", "ic")


def bucket(w: float) -> str:
    if w < 20:
        return "1字(<20px)"
    if w < 30:
        return "2字(20-30px)"
    return "3字+(>=30px)"


def gt_of(case):
    for p in DSROOT.rglob(case + ".json"):
        if p.parent.name == case:
            return json.load(open(p, encoding="utf-8"))
    return None


def collect(dirp: Path, only=None, maxdist=12.0):
    """-> [(case, cid, gt_number, box_img, base_text, w, h, d)]"""
    out = []
    for case in (only and [only] or GT9):
        g = gt_of(case)
        if g is None:
            print("  !! 找不到 GT", case)
            continue
        s1 = json.load(open(dirp / "runs" / "stage1" / (case + ".stage1.json"), encoding="utf-8"))
        H = float(s1["image"]["h"])
        ctype = {c["id"]: str(c.get("type") or "").lower() for c in s1["components"]}
        try:
            a = json.load(open(dirp / "runs" / "stage2" / "_assigned" / (case + ".json"),
                              encoding="utf-8"))
        except FileNotFoundError:
            print("  !! 没有 _assigned，先跑一次 run.py：", case)
            continue
        gt_pins = []                      # (label, number, x, y)  —— JSON 空间
        for ck, cv_ in (g.get("pins") or {}).items():
            cvt = str(((g.get("components") or {}).get(ck) or {}).get("type") or "").lower()
            if cvt not in IC_TYPES:
                continue
            for pk, pv in (cv_ or {}).items():
                num = pk.split("_", 1)[1] if "_" in pk else pk
                pt = pv.get("point") or {}
                if pt:
                    gt_pins.append((ck, num, float(pt["x"]), float(pt["y"])))
        for cid, rec in (a.get("comps") or {}).items():
            if ctype.get(cid) not in IC_TYPES:
                continue
            pins = rec.get("pins") or {}
            pts = s1["pins"].get(cid) or []
            for k, pv in pins.items():
                num = pv.get("number") or {}
                if not num.get("text") or not num.get("box"):
                    continue
                try:
                    ix = int(k)
                except (TypeError, ValueError):
                    continue
                if ix >= len(pts):
                    continue
                px, py = float(pts[ix][0]), float(pts[ix][1])
                jy = H - py
                best, bd = None, 1e9
                for (ck, gnum, gx, gy) in gt_pins:
                    d = ((gx - px) ** 2 + (jy - gy) ** 2) ** 0.5
                    if d < bd:
                        best, bd = (ck, gnum), d
                if best is None or bd > maxdist:
                    continue
                b = [float(v) for v in num["box"]]
                # 归因：这个**文字框**离哪个 GT 脚点最近？
                # 若它比「我们给它配的那个脚」更贴近**别的**脚，那读错其实是**归属错**
                # （把邻居的号绑到了这个脚上），不是识别错。这两种错要分开修。
                bcx, bcy = (b[0] + b[2]) / 2.0, H - (b[1] + b[3]) / 2.0
                own, od = best[1], 1e9
                for (ck, gnum, gx, gy) in gt_pins:
                    d2 = ((gx - bcx) ** 2 + (bcy - gy) ** 2) ** 0.5
                    if d2 < od:
                        own, od = gnum, d2
                out.append((case, cid, best[1], tuple(b), str(num["text"]),
                            b[2] - b[0], b[3] - b[1], bd, own, od))
    return out


def _dump(samples, dirp: Path, out: Path, scale=6):
    """把「基线读错的」样本裁片拼成一张竖排图（每行一个，左边是 crop_gray 前的原框）。"""
    bad = [s for s in samples if s[4] != s[2]]
    if not bad:
        print("没有错例")
        return
    tiles, labels = [], []
    cache = {}
    for (case, cid, gnum, box, base, w, h, d) in bad:
        if case not in cache:
            s1 = json.load(open(dirp / "runs" / "stage1" / (case + ".stage1.json"), encoding="utf-8"))
            img = cv2.imdecode(np.fromfile(str(Path(s1["image"]["path"])), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            cache[case] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cache[case]
        # 上下各留 0.6h，看有没有把邻居的字吃进来
        x0 = max(0, int(box[0] - 0.3 * h))
        x1 = min(g.shape[1], int(box[2] + 0.3 * h))
        y0 = max(0, int(box[1] - 0.6 * h))
        y1 = min(g.shape[0], int(box[3] + 0.6 * h))
        sub = g[y0:y1, x0:x1]
        sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
        # 在原框位置画红线
        cv2.rectangle(sub, (int(box[0]) - x0, int(box[1]) - y0),
                      (int(box[2]) - x0, int(box[3]) - y0), (0, 0, 255), 1)
        tiles.append(cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
        labels.append("%s/%s GT=%s 读=%s w=%d" % (case, cid, gnum, base, w))
    W = max(t.shape[1] for t in tiles)
    H = sum(t.shape[0] for t in tiles) + 4 * len(tiles)
    canvas = np.full((H, W + 300, 3), 255, np.uint8)
    y = 0
    for t, lb in zip(tiles, labels):
        canvas[y:y + t.shape[0], 300:300 + t.shape[1]] = t
        cv2.putText(canvas, lb, (4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1,
                    cv2.LINE_AA)
        y += t.shape[0] + 4
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", canvas)[1].tofile(str(out))
    print("错例图 -> %s（%d 个）" % (out, len(tiles)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out_ownbb"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--dump", default=None, help="把错例裁片拼成一张图存到这个路径")
    args = ap.parse_args()

    samples = collect(Path(args.dir), args.only)
    if not samples:
        print("没有样本")
        return
    print("引脚号样本 %d 个（官方 9 张里能对上 GT 点的 pin_number 读数）" % len(samples))
    print("  按 case:", dict(Counter(s[0] for s in samples)))

    by = defaultdict(list)
    for s in samples:
        by[bucket(s[5])].append(s)
    print()
    print("=== 基线（产物里已经写下的读数）按框宽分桶 ===")
    print("%-16s %-7s %-6s %s" % ("桶", "exact", "n", "错例"))
    for bk in ("1字(<20px)", "2字(20-30px)", "3字+(>=30px)"):
        ss = by.get(bk) or []
        if not ss:
            continue
        ok = [s for s in ss if s[4] == s[2]]
        bad = ["%s/%s>%s(w%d)" % (s[0], s[2], s[4], s[5]) for s in ss if s[4] != s[2]]
        print("%-16s %-7.4f %-6d %s" % (bk, len(ok) / len(ss), len(ss), " ".join(bad[:args.show])))
    ok = sum(1 for s in samples if s[4] == s[2])
    print("%-16s %-7.4f %-6d" % ("整体", ok / len(samples), len(samples)))

    print()
    print("=== 归因：错例里「文字框其实属于别的脚」（归属错）vs「框对、字读错」（识别错）===")
    wrong = [s for s in samples if s[4] != s[2]]
    own_err = [s for s in wrong if s[8] != s[2]]
    rec_err = [s for s in wrong if s[8] == s[2]]
    print("错例 %d = 归属错 %d（%.1f%%） + 识别错 %d（%.1f%%）"
          % (len(wrong), len(own_err), 100 * len(own_err) / max(1, len(wrong)),
             len(rec_err), 100 * len(rec_err) / max(1, len(wrong))))
    print("  归属错（框离 GT 脚点最近的那个号 != 应该配的号）:")
    print("   ", " ".join("%s/%s应=%s读=%s框属=%s" % (s[0], s[1], s[2], s[4], s[8])
                          for s in own_err[:args.show]))
    print("  识别错（框的位置是对的，就是字读错）:")
    print("   ", " ".join("%s/%s GT=%s读=%s(w%d)" % (s[0], s[1], s[2], s[4], s[5])
                          for s in rec_err[:args.show]))
    print()
    print("=== 识别错按「读成什么」统计 ===")
    print("  ", dict(Counter("%s>%s" % (s[2], s[4]) for s in rec_err)))

    if args.dump:
        _dump(samples, Path(args.dir), Path(args.dump))

    eng = T.TextEngine()
    gray = {}
    for case in sorted({s[0] for s in samples}):
        s1 = json.load(open(Path(args.dir) / "runs" / "stage1" / (case + ".stage1.json"),
                            encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(str(Path(s1["image"]["path"])), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        gray[case] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (名字, pad_min_x, pad_frac_x, pad_min_y, pad_frac_y)   None = 完全不 pad_box
    grid = [
        ("RAW(不pad_box)", None, None, None, None),
        ("cur(2,0,3,.20)", 2.0, 0.0, 3.0, 0.20),
        ("(0,0,0,0)", 0.0, 0.0, 0.0, 0.0),
        ("(2,0,0,0)", 2.0, 0.0, 0.0, 0.0),
        ("(2,0,2,0)", 2.0, 0.0, 2.0, 0.0),
        ("(2,0,3,0)", 2.0, 0.0, 3.0, 0.0),
        ("(3,0,3,0)", 3.0, 0.0, 3.0, 0.0),
        ("(4,0,4,0)", 4.0, 0.0, 4.0, 0.0),
        ("(0,.12,0,.25)", 0.0, 0.12, 0.0, 0.25),
        ("(2,.12,2,.25)", 2.0, 0.12, 2.0, 0.25),
        ("(1,0,1,0)", 1.0, 0.0, 1.0, 0.0),
        ("(0,0,-1,0)", 0.0, 0.0, -1.0, 0.0),      # 负值 = 内缩（裁紧一点）
        ("(-1,0,-1,0)", -1.0, 0.0, -1.0, 0.0),
        ("(0,0,-2,0)", 0.0, 0.0, -2.0, 0.0),
    ]
    idx = {}
    for i, s in enumerate(samples):
        idx.setdefault(s[0], []).append(i)

    print()
    print("=== 外扩网格（按桶） ===")
    hdr = "%-16s %-9s %-9s %-9s %-9s %s" % ("pad", "整体", "1字", "2字", "3字+",
                                             "3字+ 的错例")
    print(hdr)
    for name, mx, fx, my, fy in grid:
        got = [None] * len(samples)
        for case, ids in idx.items():
            if mx is None:
                sub = [(samples[i][3], "pin_number") for i in ids]
            else:
                sub = [(T.pad_box(samples[i][3], mx, fx, pad_min_y=my, pad_frac_y=fy),
                        "pin_number") for i in ids]
            res = eng.recognizer().read_many(gray[case], sub)
            for i, (txt, conf) in zip(ids, res):
                got[i] = txt
        per = {}
        for bk in ("1字(<20px)", "2字(20-30px)", "3字+(>=30px)"):
            per[bk] = [i for i, s in enumerate(samples) if bucket(s[5]) == bk]
        def acc(ids):
            return sum(1 for i in ids if got[i] == samples[i][2]) / len(ids) if ids else 0.0
        bad3 = ["%s>%s(w%d)" % (samples[i][2], got[i], samples[i][5])
                for i in per["3字+(>=30px)"] if got[i] != samples[i][2]]
        print("%-16s %-9.4f %-9.4f %-9.4f %-9.4f %s"
              % (name, acc(range(len(samples))), acc(per["1字(<20px)"]),
                 acc(per["2字(20-30px)"]), acc(per["3字+(>=30px)"]),
                 " ".join(bad3[:args.show])))


    print()
    print("=== 描边剔除：只留低饱和暗墨（黑字）再读 ===")
    print("动机：case 8/10 的错例全是「框里除了数字还有一条**绿色的 no-connect 叉/导线**」")
    print("      （`10/c0001 p16 GT=4 读=K4`：X 被读成 K）。绿线/暗红框/蓝位号都是**有色**的，")
    print("      引脚号是**黑字**，所以按颜色就能把它切掉。")
    color = {}
    for case in sorted({s[0] for s in samples}):
        s1 = json.load(open(Path(args.dir) / "runs" / "stage1" / (case + ".stage1.json"),
                            encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(str(Path(s1["image"]["path"])), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        color[case] = img
    for thr_n, thr_d in ((60, 160), (50, 140), (80, 170)):
        tight, rule = {}, {}
        for case, ids in idx.items():
            img = color[case]
            boxes = []
            for i in ids:
                x1, y1, x2, y2 = [int(round(v)) for v in samples[i][3]]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
                sub = img[y1:y2, x1:x2].astype(np.int16)
                mx = sub.max(2); mn = sub.min(2)
                g = sub.mean(2)
                m = ((mx - mn) < thr_n) & (g < thr_d)
                cols = m.sum(0)
                on = np.nonzero(cols > 0)[0]
                if len(on) == 0:
                    boxes.append(samples[i][3])
                    continue
                a0 = max(0, int(on[0]) - 1); a1 = min(sub.shape[1], int(on[-1]) + 2)
                rows = m[:, a0:a1].sum(1)
                ron = np.nonzero(rows > 0)[0]
                b0 = max(0, int(ron[0]) - 1) if len(ron) else 0
                b1 = min(sub.shape[0], int(ron[-1]) + 2) if len(ron) else sub.shape[0]
                boxes.append(T.pad_box([float(x1 + a0), float(y1 + b0),
                                        float(x1 + a1), float(y1 + b1)],
                                       2.0, 0.0, pad_min_y=3.0, pad_frac_y=0.0))
            res = eng.recognizer().read_many(gray[case], [(bb, "pin_number") for bb in boxes])
            for i, (txt, conf) in zip(ids, res):
                tight[i] = txt
        for i in range(len(samples)):
            t = tight.get(i, "")
            b = samples[i][4]
            rule[i] = t if (t and t.isdigit()) else b
        def acc(g):
            return sum(1 for i in range(len(samples)) if g[i] == samples[i][2]) / len(samples)
        def accb(bk):
            ids = [i for i in range(len(samples)) if bucket(samples[i][5]) == bk]
            return sum(1 for i in ids if g_[i] == samples[i][2]) / len(ids) if ids else 0.0
        g_ = tight
        s_t = "tight(thr %d/%d) 整体 %.4f | 1字 %.4f 2字 %.4f 3字+ %.4f" % (
            thr_n, thr_d, acc(g_), accb("1字(<20px)"), accb("2字(20-30px)"), accb("3字+(>=30px)"))
        g_ = rule
        s_r = "rule=取纯数字 整体 %.4f | 1字 %.4f 2字 %.4f 3字+ %.4f" % (
            acc(g_), accb("1字(<20px)"), accb("2字(20-30px)"), accb("3字+(>=30px)"))
        print(" ", s_t)
        print(" ", s_r)
        if thr_n == 60:
            print("     tight 的错例:", " ".join(
                "%s/%s GT=%s tight=%s" % (samples[i][0], samples[i][1], samples[i][2], tight[i])
                for i in range(len(samples)) if tight[i] != samples[i][2])[:600])


if __name__ == "__main__":
    main()
