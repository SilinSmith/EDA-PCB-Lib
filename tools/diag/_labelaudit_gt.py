"""全类别 GT 错标体检 —— 扫描 + 逐框指标 + CSV。

只读数据集（不写 eda-pcb 下任何文件）。产出只有两个：
    Y:\\workspace\\EDA\\work\\_labelaudit.csv      每个可疑框一行
    Y:\\workspace\\EDA\\work\\_labelaudit.md       报告

墨迹阈值 **200**（不是 128）：
    本数据集符号笔画是**浅灰**——灰度直方图在 151/152 有独立尖峰（151: 101956 px,
    152: 136864 px），另有近白底色 243~253。用 <128 只捞到深色导线/文字，
    会把大量正常符号框误判成空框。直方图命令见报告。可用 LABELAUDIT_INK 覆盖。

口径沿用 work/_gndbox.py（读法、y 翻转、外扩 25px）：
    img_y1, img_y2 = sorted((H - y2, H - y1))
    ink   = 框内 (gray < INK_THR) 占比
    ink25 = 外扩 25px 后框内 (gray < INK_THR) 占比

新增量（本脚本）：
    ink_same : 与外扩窗内"与框内墨迹同一连通域"的墨迹占框面积比例
    off_cc   : 上述同一连通域墨迹的质心 -> 框心距离(px)
               （不用外扩窗全部墨迹的质心：那会混进旁边导线/文字，
                 质心只反映"旁边有什么"而不是"框偏了多少"）
    span_min/span_max : 框内墨迹包围盒占框比例（"贴满整框" vs "只占一角"）
    txt      : 框面积被 cvparts/textbox.py 检出的文字行框覆盖的比例
    dup      : 同 case 同 type 的另一框与它 IoU>=0.5 或中心距<=3px

用法：
    python work\\tools\\diag\\_labelaudit_gt.py            # 扫描并写 CSV
    python work\\tools\\diag\\_labelaudit_gt.py --json     # 逐框记录打到 stdout
    python work\\tools\\diag\\_labelaudit_gt.py --crops     # 抽可疑框裁图到 %TEMP% 供肉眼复核
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

EDA = Path(r"Y:\workspace\EDA\eda-pcb")
WORK = Path(r"Y:\workspace\EDA\work")
OUT_CSV = WORK / "_labelaudit.csv"

sys.path.insert(0, str(EDA))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from schematic_parser import data as S  # noqa: E402

import cv2  # noqa: E402

sys.path.insert(0, str(EDA / "cvparts"))
import textbox  # noqa: E402

SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other"]
GROW = 25                                  # 外扩像素（沿用 _gndbox.py）
INK_THR = int(os.environ.get("LABELAUDIT_INK", "200"))   # 见文件头说明
EMPTY_INK = 0.02   # ink <= 2% 记 empty
EMPTY25 = 0.005    # 外扩 25px 后仍 <= 0.5% 记 empty25（彻底空框）
OFF_HI = 20.0      # 同域墨迹质心偏移过大
TXT_HI = 0.50      # 框面积 >50% 被文字行框覆盖
DUP_IOU = 0.5
DUP_DIST = 3.0
CSV_COLS = ["case", "split", "key", "type", "w", "h", "ink", "ink25",
            "off_cc", "off_raw", "hollow", "hard", "reason"]


def source_of(p) -> str:
    """图源从 png 文件名匹配；其余 unknown（沿用 _gndbox.py）。"""
    n = Path(p).name
    for t in SRC_ORDER:
        if t in n:
            return t
    return "unknown"


def box_px(box, h):
    """GT bbox（左下原点）-> 图像像素坐标（左上原点）。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    iy1, iy2 = sorted((h - y2, h - y1))
    x1, x2 = sorted((x1, x2))
    return x1, iy1, x2, iy2


def _txtmask(bgr):
    """文字行框掩膜 + 框列表 + 合并前的字形颗粒。

    复用 cvparts/textbox.py。**注意**：该检测器在 ~20px 尺度上无法区分
    "文字" 与 "符号轮廓"——二极管三角形会被整块当成文字行（ASCII 复核见报告）。
    因此额外返回合并前的字形颗粒 `_glyphs`：真文字由**多个**小颗粒组成，
    单个符号轮廓只是**一两个**大颗粒。用颗粒数把两者分开。
    """
    boxes = textbox.detect_boxes(bgr)
    m = np.zeros(bgr.shape[:2], np.uint8)
    for b in boxes:
        cv2.rectangle(m, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 255, -1)
    # textbox.detect_boxes 的内部前置步骤，复刻出来拿"合并前"的颗粒
    g = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = np.where(g < textbox.INK, 255, 0).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    glyphs = textbox._glyphs(cv2.bitwise_and(ink, cv2.bitwise_not(textbox._long_lines(ink))))
    return m, boxes, glyphs


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def scan(verbose: bool = False):
    """返回 (recs, meta)。recs 为逐框 dict 列表。"""
    cases = S.discover()
    recs = []
    meta = {"ink_thr": INK_THR, "text_ok": 0, "pin_n": 0, "pin_hit20": 0,
            "cases": len(cases), "text_boxes": 0, "sizes": {}}
    t0 = time.perf_counter()
    for ci, c in enumerate(cases):
        if not c.json_path:
            continue
        try:
            gray = np.asarray(Image.open(c.png_path).convert("L"))
        except Exception as e:  # noqa: BLE001
            print("skip", c.case_id, e)
            continue
        h, w = gray.shape
        meta["sizes"][c.key] = (w, h)
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        tm, tboxes, glyphs = _txtmask(bgr)
        # 字形颗粒中心（判断框内"是不是由多个小颗粒组成的真文字"）
        if glyphs:
            gar = np.asarray(glyphs, dtype=np.float64)
            gcx = (gar[:, 0] + gar[:, 2]) / 2.0
            gcy = (gar[:, 1] + gar[:, 3]) / 2.0
            gwd = gar[:, 2] - gar[:, 0]
        else:
            gcx = gcy = gwd = np.zeros(0)
        # 引脚点（图像坐标）：独立于像素的几何一致性检查
        doc = S.load_json(c.json_path)
        pinxy = defaultdict(list)
        for _k, gp in ((doc or {}).get("pins") or {}).items():
            for p in (gp or {}).values():
                pt = (p or {}).get("point") or {}
                if pt:
                    pinxy[_k].append((float(pt.get("x", 0)), h - float(pt.get("y", 0))))
        meta["text_ok"] += 1
        meta["text_boxes"] += len(tboxes)
        comps = (doc or {}).get("components") or {}
        src = source_of(c.png_path)
        # 文字检测自检锚点：GT 引脚名（排除 P_<n> 占位名）附近 20px 是否有文字框
        arr = np.asarray(tboxes, dtype=np.float64) if tboxes else np.zeros((0, 4))
        for _k, gp in ((doc or {}).get("pins") or {}).items():
            comp = comps.get(_k)
            if not comp or str(comp.get("type") or "").lower() != "box":
                continue
            for p in (gp or {}).values():
                nm = str((p or {}).get("pinname") or "")
                if not nm or (nm.startswith("P_") and nm[2:].isdigit()):
                    continue
                pt = (p or {}).get("point") or {}
                px, py = float(pt.get("x", 0)), h - float(pt.get("y", 0))
                meta["pin_n"] += 1
                if arr.shape[0]:
                    dx = np.maximum(np.maximum(arr[:, 0] - px, px - arr[:, 2]), 0.0)
                    dy = np.maximum(np.maximum(arr[:, 1] - py, py - arr[:, 3]), 0.0)
                    if float(np.hypot(dx, dy).min()) <= 20.0:
                        meta["pin_hit20"] += 1
        cur = []
        for key, v in comps.items():
            typ = str(v.get("type") or "").lower()
            bb = v.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = box_px(bb, h)
            # 框越界（客观几何缺陷，与像素无关）
            oob = bool(x1 < 0 or y1 < 0 or x2 > w or y2 > h)
            wholly_oob = bool(x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h)
            a, b = max(0, int(y1)), min(h, int(round(y2)))
            cc, dd = max(0, int(x1)), min(w, int(round(x2)))
            sub = gray[a:b, cc:dd]
            if sub.size == 0:
                continue
            m = sub < INK_THR
            ink = float(m.mean())
            ga, gb = max(0, int(y1) - GROW), min(h, int(round(y2)) + GROW)
            gc, gd = max(0, int(x1) - GROW), min(w, int(round(x2)) + GROW)
            gw = gray[ga:gb, gc:gd]
            gmask = (gw < INK_THR).astype(np.uint8)
            ink25 = float(gmask.mean())
            # 同一连通域过滤：只保留与框内有交集的连通域
            n_lab, lab = cv2.connectedComponents(gmask, connectivity=8)
            oa, ob = a - ga, b - ga
            oc, od = cc - gc, dd - gc
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            off_cc = float("nan")
            ink_same = 0.0
            dom_in = float("nan")
            if n_lab > 1 and ob > oa and od > oc:
                ids = np.unique(lab[oa:ob, oc:od])
                ids = ids[ids != 0]
                if ids.size:
                    sel = np.isin(lab, ids)
                    ink_same = float(sel.sum()) / box_area
                    # 同域墨迹里**落在框内**的比例：框外导线很长时 off_cc 会被拉大，
                    # 但 dom_in 仍然高 ⇒ 框是对的。dom_in 低才是"框不在这个域上"。
                    tot = float(sel.sum())
                    dom_in = float(sel[oa:ob, oc:od].sum()) / tot if tot else float("nan")
                    ys, xs = np.nonzero(sel)
                    cy = float(ys.mean()) + ga
                    cx = float(xs.mean()) + gc
                    off_cc = float(np.hypot(cx - (x1 + x2) / 2.0,
                                            cy - (y1 + y2) / 2.0))
            if gmask.any():
                ys, xs = np.nonzero(gmask)
                off = float(np.hypot(float(xs.mean()) + gc - (x1 + x2) / 2.0,
                                     float(ys.mean()) + ga - (y1 + y2) / 2.0))
            else:
                off = float("nan")
            bw, bh = float(x2 - x1), float(y2 - y1)
            if m.any():
                iys, ixs = np.nonzero(m)
                span_w = (ixs.max() - ixs.min() + 1) / max(1.0, bw)
                span_h = (iys.max() - iys.min() + 1) / max(1.0, bh)
            else:
                span_w = span_h = 0.0
            # 「框内没墨」有两种截然不同的原因，必须分开（肉眼复核 r/R1、r/R5 得到）：
            #   a) 真错标 —— 框落在空白处，周围也没有东西。
            #   b) 框只框住了**符号的空心内部** —— 电阻画成空心矩形，GT 框只套住里面的白区。
            #      位置没错，但框明显偏小、没把符号轮廓包进来（对 YOLO 仍是缺陷标签）。
            # 判据：从外扩窗边缘对"非墨迹"做 4-连通泛洪，框心白区若不与边缘连通
            #       ⇒ 框被符号轮廓包住 ⇒ 是 (b)。
            enclosed = False
            if ink <= EMPTY_INK and gmask.size:
                _nf, flab = cv2.connectedComponents(
                    (gmask == 0).astype(np.uint8), connectivity=4)
                border = set(np.unique(np.concatenate(
                    [flab[0, :], flab[-1, :], flab[:, 0], flab[:, -1]])).tolist())
                border.discard(0)
                cyw, cxw = (oa + ob) // 2, (oc + od) // 2
                if 0 <= cyw < flab.shape[0] and 0 <= cxw < flab.shape[1]:
                    l0 = int(flab[cyw, cxw])
                    enclosed = bool(l0 != 0 and l0 not in border)
            tsub = tm[a:b, cc:dd]
            txt = float((tsub > 0).mean()) if tsub.size else 0.0
            # 框内有几个"字形颗粒"（已排除长直线）——真文字是多个小颗粒
            # 但 GND 符号是 3 条**竖着叠**的横线，也会给出 3 个颗粒 ⇒ 必须再看排列：
            # 真文字是"同一行、横向排开"，符号笔画是"同 x、纵向叠放"。
            nrow, xrow = 0, 0.0
            if gcx.size:
                inside = (gcx >= x1) & (gcx < x2) & (gcy >= y1) & (gcy < y2)
                nglyph = int(inside.sum())
                gw_max = float(gwd[inside].max()) if inside.any() else 0.0
                ix, iy = gcx[inside], gcy[inside]
                for yy in iy:                      # 以每个颗粒的 y 为中心开一条 ±3px 横带
                    k = np.abs(iy - yy) <= 3.0
                    if int(k.sum()) > nrow:
                        nrow = int(k.sum())
                        xrow = float(ix[k].max() - ix[k].min())
            else:
                nglyph, gw_max = 0, 0.0
            # 引脚点到框的距离（0 = 落在框内）
            pd = [float(np.hypot(max(x1 - px, 0.0, px - x2),
                                 max(y1 - py, 0.0, py - y2)))
                  for px, py in pinxy.get(str(key), [])]
            pin_min = min(pd) if pd else float("nan")
            pin_max = max(pd) if pd else float("nan")
            cur.append(dict(
                case=c.case_id, split=c.split, key=str(key), type=typ,
                w=bw, h=bh, ink=ink, ink25=ink25, ink_same=ink_same, dom_in=dom_in,
                off=off, off_cc=off_cc, nglyph=nglyph, gw_max=gw_max,
                nrow=nrow, xrow=xrow, enclosed=enclosed,
                oob=oob, wholly_oob=wholly_oob,
                n_pin=len(pd), pin_min=pin_min, pin_max=pin_max,
                span_min=min(span_w, span_h), span_max=max(span_w, span_h),
                txt=txt, src=src, box=(x1, y1, x2, y2),
                cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, dup=False))
        # 同 case 同 type 的重复框
        byty = defaultdict(list)
        for r in cur:
            byty[r["type"]].append(r)
        for _t, rs in byty.items():
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    a1, a2 = rs[i], rs[j]
                    d = float(np.hypot(a1["cx"] - a2["cx"], a1["cy"] - a2["cy"]))
                    if _iou(a1["box"], a2["box"]) >= DUP_IOU or d <= DUP_DIST:
                        a1["dup"] = a2["dup"] = True
        recs.extend(cur)
        if verbose and (ci + 1) % 50 == 0:
            print(f"  {ci+1}/{len(cases)}  {time.perf_counter()-t0:.1f}s")
    print(f"scanned {meta['text_ok']} cases, {len(recs)} boxes, "
          f"{time.perf_counter()-t0:.1f}s (ink<{INK_THR}); text boxes "
          f"{meta['text_boxes']}; pin-name cov@20px {meta['pin_hit20']}/"
          f"{meta['pin_n']} = {meta['pin_hit20']/max(1,meta['pin_n']):.3f}")
    return recs, meta


def modal_size(rs):
    cnt = Counter((round(r["w"]), round(r["h"])) for r in rs)
    (mw, mh), n = cnt.most_common(1)[0]
    return (float(mw), float(mh)), n / len(rs), len(cnt)


def on_text(r) -> bool:
    """框是否**真的**压在一行文字上。

    单纯 `txt>=0.5` 不可信：textbox 在 ~20px 尺度上会把符号轮廓（二极管三角形、
    电容极板、GND 符号）整块当成"文字行"。ASCII 逐像素复核见报告 m00093 段。
    因此再加一道**排列**闸门：框内至少 3 个字形颗粒落在**同一条横带**（±3px）里
    且横向铺开 >=10px——真文字行的形态；符号笔画是纵向叠放或离散的，过不了。
    """
    return (r["txt"] >= TXT_HI and r["nrow"] >= 3 and r["xrow"] >= 10.0)


def reasons_of(r, mode=None) -> str:
    """可疑标签，分号连接。mode: 该 type 的众数尺寸 (w,h)。"""
    out = []
    if r.get("wholly_oob"):
        out.append("oob!")               # 整框落在图像外
    elif r.get("oob"):
        out.append("oob")                # 部分越界
    if r["ink25"] <= EMPTY25:
        out.append("empty25")            # 外扩 25px 内一点墨都没有
    elif r["ink"] <= EMPTY_INK:
        out.append("hollow" if r.get("enclosed") else "empty")
    if on_text(r):
        out.append("text%.2f" % r["txt"])
    if r["off_cc"] == r["off_cc"] and r["off_cc"] > OFF_HI:
        out.append("off%.0f" % r["off_cc"])
    if r["dup"]:
        out.append("dup")
    if 0 < r["span_max"] < 0.34 and max(r["ink"], r["ink_same"]) > EMPTY_INK:
        out.append("stub")
    if mode is not None:
        mw, mh = mode
        if abs(r["w"] - mw) > 0.6 or abs(r["h"] - mh) > 0.6:
            out.append("oddtpl")
    return ";".join(out)


def build_rows(recs):
    per_type = defaultdict(list)
    for r in recs:
        per_type[r["type"]].append(r)
    modes = {t: modal_size(rs)[0] for t, rs in per_type.items() if len(rs) >= 5}
    rows = []
    for r in recs:
        rs = reasons_of(r, modes.get(r["type"]))
        if not rs:
            continue
        r["reason"] = rs
        f = lambda v: "" if v != v else "%.2f" % v          # noqa: E731
        hard = any(t.split(".")[0] in ("empty", "hollow", "dup", "stub", "oob", "oob!")
                   for t in rs.split(";"))
        rows.append([r["case"], r["split"], r["key"], r["type"],
                     "%.0f" % r["w"], "%.0f" % r["h"],
                     "%.4f" % r["ink"], "%.4f" % r["ink25"],
                     f(r["off_cc"]), f(r["off"]), "1" if r.get("enclosed") else "0",
                     "1" if hard else "0", rs])
    return rows


def _cache_path():
    return Path(os.environ.get("TEMP", ".")) / "labelaudit_gt.pkl"


def _save_cache(recs, meta):
    import pickle
    with open(_cache_path(), "wb") as fh:
        pickle.dump({"recs": recs, "meta": meta}, fh)


def load_or_scan():
    """优先读 %TEMP% 缓存，省重复扫描；不存在则扫。"""
    import pickle
    p = _cache_path()
    if "--fresh" not in sys.argv and p.exists():
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        if d.get("meta", {}).get("ink_thr") == INK_THR:
            print(f"cache hit {p}  ink<{INK_THR}  n={len(d['recs'])}")
            return d["recs"], d["meta"]
    recs, meta = scan(verbose=True)
    _save_cache(recs, meta)
    return recs, meta


def make_crops(recs, n_per_type: int = 3):
    """把可疑框裁图（框+外扩 25px，放大 3 倍）写到 %TEMP%\\labelaudit_crops\\ 供肉眼复核。"""
    outd = Path(os.environ.get("TEMP", ".")) / "labelaudit_crops"
    outd.mkdir(parents=True, exist_ok=True)
    by_type = defaultdict(list)
    for r in recs:
        if reasons_of(r):
            by_type[r["type"]].append(r)
    cmap = {(c.split, c.case_id): c for c in S.discover()}
    made = 0
    for t, rs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        rs = sorted(rs, key=lambda r: (-(r["ink25"] <= EMPTY25), r["ink"], -r["txt"]))
        for r in rs[:n_per_type]:
            c = cmap.get((r["split"], r["case"]))
            if c is None:
                continue
            gray = np.asarray(Image.open(c.png_path).convert("L"))
            x1, y1, x2, y2 = r["box"]
            a = max(0, int(y1) - GROW - 6); b = min(gray.shape[0], int(y2) + GROW + 6)
            cc = max(0, int(x1) - GROW - 6); dd = min(gray.shape[1], int(x2) + GROW + 6)
            crop = gray[a:b, cc:dd].copy()
            if crop.size == 0:
                continue
            g = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(g, (int(x1) - cc, int(y1) - a), (int(x2) - cc, int(y2) - a),
                          (0, 0, 255), 1)
            g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            nm = "%s_%s_%s_ink%.3f.png" % (r["split"], r["case"], t, r["ink"])
            cv2.imwrite(str(outd / nm), g)
            made += 1
    print(f"crops -> {outd}  ({made} files)")


def main() -> None:
    if "--json" in sys.argv:
        recs, meta = load_or_scan()
        for r in recs:
            r.pop("box", None)
            if r["off_cc"] != r["off_cc"]:
                r["off_cc"] = None
        json.dump({"recs": recs, "meta": meta}, sys.stdout, ensure_ascii=False)
        return
    if "--crops" in sys.argv:
        recs, _m = load_or_scan()
        make_crops(recs)
        return
    recs, _meta = load_or_scan()
    n = write_csv(recs)
    print(f"wrote {OUT_CSV}  rows={n}  (of {len(recs)} boxes)")


def write_csv(recs) -> int:
    """把可疑框写成 OUT_CSV，返回行数。"""
    rows = build_rows(recs)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(CSV_COLS)
        wtr.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    main()
