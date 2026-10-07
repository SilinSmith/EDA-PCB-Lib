"""把 stage1 导线提取的错例挑成「典型示例集」，用于人工分析到底哪里出了问题。

和之前的 `fn_fp_official.py`（元件级漏检/误检）对应，这一份是**导线级**的：
  FP 示例 —— 提取出来但离 GT 很远的线段，按它贴在什么上面分类：comp / text / frame / other
  FN 示例 —— GT 有、我们没提出来（或提偏了）的走线，按原因分类：
      no_ink       GT 那条边走线上本来就没墨（GT 与图不一致，或我们抽样口径不对）
      in_text      GT 边走线穿过一个文字框（被我们涂掉了）
      in_comp      GT 边走线穿过一个已检出元件框（被我们遮挡了）
      thin_broken  线上有墨但很细/被切断（骨架没连上）
      other        其他

每个示例渲染成一个窗口裁剪：原图 + 我们提取的段（绿）+ GT 段（蓝）+ 被讨论的那条（红）。

用法:
  python work\\wire_cases.py [stage1产出目录] [--per-cat N] [--win 240]
"""
import glob
import json
import os
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "eval"))
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))

from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs, sample_polyline  # noqa: E402

OUT_DIR = r"Y:\workspace\EDA\work\preview\wirecases"
TOL = 5.0
INK = 170


def src_of(path):
    p = os.path.basename(path).lower().replace(" ", "")
    return ("kicad" if "kicad" in p else "altium" if "altium" in p
            else "datasheet" if "datasheet" in p else "jlc" if "jlc" in p else "other")


def near_box(px, py, boxes, pad):
    return any(b[0] - pad <= px <= b[2] + pad and b[1] - pad <= py <= b[3] + pad for b in boxes)


def classify_fp(mx, my, W, H, cboxes, tboxes):
    if near_box(mx, my, cboxes, 3.0):
        return "comp"
    if near_box(mx, my, tboxes, 3.0):
        return "text"
    if (mx < 0.03 * W or mx > 0.97 * W or my < 0.03 * H or my > 0.97 * H
            or (mx > 0.60 * W and my > 0.62 * H)):
        return "frame"
    return "other"


def classify_fn(p, ink, textmask, tboxes, cboxes):
    """p = [x1,y1,x2,y2] GT 段（图像坐标）。返回 (原因, 采样点墨覆盖率, 采样点到最近提取段的最差距离)。

    注意：**不要用「靠近文字框」判断 in_text** —— 一张图上平均 567 个文字框，
    1px 容差下几乎任何位置都算「靠近文字」。必须用真正被涂掉的那张掩膜。
    """
    pts = sample_polyline(np.array(p[:2], float), np.array(p[2:], float), 20)
    H, W = ink.shape
    n_ink = n_txt = n_comp = 0
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        if ink[yi, xi]:
            n_ink += 1
        if textmask is not None and textmask[yi, xi]:
            n_txt += 1
        if any(b[0] - 1 <= x <= b[2] + 1 and b[1] - 1 <= y <= b[3] + 1 for b in cboxes):
            n_comp += 1
    N = len(pts)
    cov = n_ink / N
    if cov < 0.55:
        cause = "no_ink"
    elif n_txt / N > 0.25:
        cause = "in_text"
    elif n_comp / N > 0.25:
        cause = "in_comp"
    elif cov < 0.95:
        cause = "thin_broken"
    else:
        cause = "other"
    return cause, cov


def _gt_img_segs(gt, H):
    """GT 的 net edges -> 线段列表 [[x1,y1],[x2,y2]]（图像坐标，y 翻转），可直接喂 seg_arrays。"""
    out = []
    for net in gt["nets"].values():
        for pts in (net.get("edges") or {}).values():
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                out.append([[float(pts[i]["x"]), H - float(pts[i]["y"])],
                            [float(pts[i + 1]["x"]), H - float(pts[i + 1]["y"])]])
    return out


def draw_win(img, cx, cy, win, segs_green, segs_blue, seg_red, title):
    H, W = img.shape[:2]
    x0 = int(max(0, min(W - win, cx - win // 2)))
    y0 = int(max(0, min(H - win, cy - win // 2)))
    x1, y1 = min(W, x0 + win), min(H, y0 + win)
    crop = img[y0:y1, x0:x1].copy()
    if crop.shape[0] < win or crop.shape[1] < win:                 # 补足尺寸便于拼版
        pad = np.full((win, win, 3), 255, np.uint8)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad

    def shift(seg):
        return (int(round(seg[0] - x0)), int(round(seg[1] - y0)),
                int(round(seg[2] - x0)), int(round(seg[3] - y0)))

    for s in segs_blue:
        a = shift(s)
        if -win < a[0] < 2 * win or -win < a[2] < 2 * win:
            cv2.line(crop, a[:2], a[2:], (255, 150, 0), 2, cv2.LINE_AA)
    for s in segs_green:
        a = shift(s)
        if -win < a[0] < 2 * win or -win < a[2] < 2 * win:
            cv2.line(crop, a[:2], a[2:], (0, 190, 0), 1, cv2.LINE_AA)
    a = shift(seg_red)
    cv2.line(crop, a[:2], a[2:], (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(crop, (0, 0), (win - 1, win - 1), (170, 170, 170), 1)
    cv2.rectangle(crop, (0, 0), (win - 1, 22), (255, 255, 255), -1)
    cv2.putText(crop, title[:46], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    return crop


def sheet(cards, cols, path):
    if not cards:
        return
    rows = int(np.ceil(len(cards) / cols))
    h, w = cards[0].shape[:2]
    out = np.full((rows * h, cols * w, 3), 245, np.uint8)
    for i, c in enumerate(cards):
        r, cc = divmod(i, cols)
        out[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = c
    cv2.imwrite(path, out)
    print(f"  -> {path}  ({out.shape[1]}x{out.shape[0]}, {len(cards)} 格)")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else r"Y:\workspace\EDA\work\stage1_all9"
    per_cat = int(sys.argv[sys.argv.index("--per-cat") + 1]) if "--per-cat" in sys.argv else 8
    win = int(sys.argv[sys.argv.index("--win") + 1]) if "--win" in sys.argv else 240
    os.makedirs(OUT_DIR, exist_ok=True)

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    fps = {"comp": [], "text": [], "frame": [], "other": []}
    fns = {"no_ink": [], "in_text": [], "in_comp": [], "thin_broken": [], "other": []}
    seen_case = {"FP": {}, "FN": {}}

    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        cid = d["case"]
        case = cases.get(cid)
        if case is None:
            for k, v in cases.items():
                if k.lstrip("0") == cid.lstrip("0"):
                    case, cid = v, k
                    break
        if case is None:
            continue
        H = float(d["image"]["h"]); W = float(d["image"]["w"])
        img = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < INK).astype(np.uint8)

        gt = json.load(open(case.json_path, encoding="utf-8"))
        gsegs = _gt_img_segs(gt, H)
        GA = seg_arrays(gsegs)
        PA = seg_arrays(d["wire_detail"]["segments"])
        if PA.shape[0] == 0:
            continue

        # --- 预测段到 GT 的最差采样距离 ---
        pp = np.concatenate([sample_polyline(PA[i, :2], PA[i, 2:], 16) for i in range(PA.shape[0])])
        po = np.concatenate([np.full(16, i) for i in range(PA.shape[0])])
        pdist = dist_pts_to_segs(pp, GA) if GA.shape[0] else np.full(pp.shape[0], 1e9)
        pworst = np.zeros(PA.shape[0])
        np.maximum.at(pworst, po, pdist)

        # --- GT 边是否被覆盖（任一预测段全部采样点都在 tol 内即算覆盖）---
        gworst = np.full(GA.shape[0], 1e9)
        if PA.shape[0] and GA.shape[0]:
            gp = np.concatenate([sample_polyline(GA[i, :2], GA[i, 2:], 12) for i in range(GA.shape[0])])
            go = np.concatenate([np.full(12, i) for i in range(GA.shape[0])])
            gd = dist_pts_to_segs(gp, PA)
            gw = np.full(GA.shape[0], 0.0)
            np.maximum.at(gw, go, gd)
            gworst = gw

        cboxes = [c["bbox_img"] for c in d.get("components", [])]
        tboxes = [t["bbox_img"] for t in d.get("texts", [])]
        try:
            from cvparts.textbox import text_mask as _tmask
            textmask = _tmask(ink.shape, tboxes, pad=1.0).astype(np.uint8)
        except Exception:                                          # noqa: BLE001
            textmask = None
        src = src_of(d["image"]["path"])

        for i in range(PA.shape[0]):
            if pworst[i] <= TOL:
                continue
            mx = float((PA[i, 0] + PA[i, 2]) / 2); my = float((PA[i, 1] + PA[i, 3]) / 2)
            L = float(np.hypot(PA[i, 2] - PA[i, 0], PA[i, 3] - PA[i, 1]))
            if L < 6:
                continue
            cat = classify_fp(mx, my, W, H, cboxes, tboxes)
            fps[cat].append(dict(case=cid, src=src, seg=PA[i].tolist(), dist=float(pworst[i]),
                                 L=L, mx=mx, my=my, score=L * min(float(pworst[i]), 400.0)))

        for i in range(GA.shape[0]):
            if gworst[i] <= TOL:
                continue
            L = float(np.hypot(GA[i, 2] - GA[i, 0], GA[i, 3] - GA[i, 1]))
            if L < 8:
                continue
            cause, cov = classify_fn(GA[i].tolist(), ink, textmask, tboxes, cboxes)
            fns[cause].append(dict(case=cid, src=src, seg=GA[i].tolist(), L=L,
                                   mx=float((GA[i, 0] + GA[i, 2]) / 2),
                                   my=float((GA[i, 1] + GA[i, 3]) / 2),
                                   cov=cov, d=float(gworst[i]), score=L))

    # ---- 选例：按 score 降序，但同 case+同 src 最多贡献 2 个，保证多样性 ----
    def pick(pool, n):
        pool = sorted(pool, key=lambda r: -r["score"])
        out, used = [], {}
        for r in pool:
            k = (r["case"], r["src"])
            if used.get(k, 0) >= 2:
                continue
            used[k] = used.get(k, 0) + 1
            out.append(r)
            if len(out) >= n:
                break
        return out

    print(f"stage1 产物 {len(files)} 份   tol={TOL}px")
    print("\n=== FP 候选（离 GT > tol 的提取段）===")
    for k, v in fps.items():
        tot = sum(r["L"] for r in v)
        print(f"  {k:8s} {len(v):7d} 段   总长 {tot:9.0f}px")
    print("\n=== FN 候选（没被覆盖的 GT 边）===")
    print(f'  {"原因":12s} {"条数":>7s} {"总长px":>9s} {"到最近提取段距离":>10s}  {"<=15px":>7s} {"<=40px":>7s}')
    for k, v in fns.items():
        if not v:
            print(f"  {k:12s} {0:7d}")
            continue
        ds = np.array([r["d"] for r in v])
        Ls = sum(r["L"] for r in v)
        print(f'  {k:12s} {len(v):7d} {Ls:9.0f} {np.percentile(ds,50):10.1f}  '
              f'{(ds<=15).mean():7.3f} {(ds<=40).mean():7.3f}')
    allfn = [r for v in fns.values() for r in v]
    if allfn:
        ds = np.array([r["d"] for r in allfn])
        print(f'\n  全部 {len(allfn)} 条未覆盖 GT 边，到最近提取段的最差距离：'
              f'p10 {np.percentile(ds,10):.1f}  p50 {np.percentile(ds,50):.1f}  '
              f'p90 {np.percentile(ds,90):.1f}')
        print(f'  其中 <=15px（只是没对准）：{int((ds<=15).sum())} 条 = {(ds<=15).mean():.1%}')
        print(f'  >40px（那段彻底没提出来）：{int((ds>40).sum())} 条 = {(ds>40).mean():.1%}')

    picks = {k: pick(v, per_cat) for k, v in fps.items()}
    picksn = {k: pick(v, per_cat) for k, v in fns.items()}

    # ---- 渲染 ----
    for cat, rows in picks.items():
        if not rows:
            continue
        cards = []
        for r in rows:
            case = cases.get(r["case"]) or next(v for k, v in cases.items()
                                                if k.lstrip("0") == r["case"].lstrip("0"))
            img = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
            H = img.shape[0]
            d = json.load(open(os.path.join(out_dir, f"{r['case']}.stage1.json"), encoding="utf-8"))
            gt = json.load(open(case.json_path, encoding="utf-8"))
            gs = list(seg_arrays(_gt_img_segs(gt, H)))
            ps = list(seg_arrays(d["wire_detail"]["segments"]))
            t = f"FP:{cat} {r['case']} {r['src']} L{r['L']:.0f} d{r['dist']:.0f}"
            cards.append(draw_win(img, r["mx"], r["my"], win, ps, gs, np.array(r["seg"]), t))
        sheet(cards, 4, os.path.join(OUT_DIR, f"fp_{cat}.png"))

    for cause, rows in picksn.items():
        if not rows:
            continue
        cards = []
        for r in rows:
            case = cases.get(r["case"]) or next(v for k, v in cases.items()
                                                if k.lstrip("0") == r["case"].lstrip("0"))
            img = cv2.imread(case.png_path, cv2.IMREAD_COLOR)
            H = img.shape[0]
            d = json.load(open(os.path.join(out_dir, f"{r['case']}.stage1.json"), encoding="utf-8"))
            ps = list(seg_arrays(d["wire_detail"]["segments"]))
            t = f"FN:{cause} {r['case']} {r['src']} L{r['L']:.0f}"
            cards.append(draw_win(img, r["mx"], r["my"], win, ps, [], np.array(r["seg"]), t))
        sheet(cards, 4, os.path.join(OUT_DIR, f"fn_{cause}.png"))

    print(f"\n产物目录 {OUT_DIR}")
    print("红=被讨论的那条（FP 提取段 / FN 的 GT 边）  绿=我们提取的所有段  蓝=GT 所有段")


if __name__ == "__main__":
    main()
