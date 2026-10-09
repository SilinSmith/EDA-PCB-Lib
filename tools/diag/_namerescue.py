# -*- coding: utf-8 -*-
"""**引脚名救援**可行性：IC 引脚「号读对了、名字没读到」的那些桩，名字到底还在不在图上？

背景（`_pinledger.py` 官方 9 张）：`pin_f1` 只有 box(IC) 算分，而缺号 51 里 **23 条
（45%）是「号已经读出来了，但 `assemble.py` 因为 `STAGE2_BOX_PIN_NEED_NAME` 没名字
整条不发」**。关掉那条闸门是负结果（保护精度），所以唯一出路是**把名字读出来**。

本探针只回答一件事：这 23 个（以及所有「号读对、名缺失」的）引脚，名字能不能救回来。
对每个这样的引脚，在**不依赖任何新模型**的前提下试三种取证：

  ``det``   图里已有的文字检出框里，离该脚点 ≤R px 且不是号框的那些，直接读
  ``strip`` IC 体内部、贴边对齐该脚的一条带里，取暗墨紧致框再读（自造裁片）
  ``mir``   把「号框」沿最近的 IC 边镜像进体内再读（名通常印在号的对面）

同时报告「体内这个位置**根本没有墨**」的比例（= 图上确实没印名字，救不回来）。

    cd Y:\\workspace\\EDA\\eda-pcb
    python -X utf8 tools\\diag\\_namerescue.py --run Y:\\workspace\\EDA\\work\\_fast_out_reg9 --n 10
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

import cv2                                                         # noqa: E402
from cvparts import textseg                                        # noqa: E402
from stage2 import libpins                                         # noqa: E402
from stage2.text import TextEngine, imread_unicode                 # noqa: E402
from schematic_parser import metrics as M                          # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402

H_CFG = dict(ink_mode="max", min_dim=2, min_area=3, line_len=15,
             resid_min_h=4, resid_min_dim=1)
R_NEAR = 45.0          # 「附近的文字框」半径（与 _pinledger 的 NEAR_PX 对齐）
DEPTH = 46.0           # 「体内带」往内多深
INK_GRAY = 170         # 名字是黑字
INK_SAT = 70


def _pt(d):
    if not isinstance(d, dict):
        return None
    try:
        return float(d["x"]), float(d["y"])
    except Exception:
        return None


def _ctr(box):
    if not box or len(box) < 4:
        return None
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def _norm(s):
    return "".join(ch for ch in str(s or "").upper() if ch.isalnum())


def _find(run: Path, cid: str, kind: str):
    sub = "stage1" if kind == "stage1" else "stage2"
    name = ("%s.stage1.json" % cid) if kind == "stage1" else None
    cands = []
    if kind == "stage1":
        cands = [run / "runs" / sub / name, run / sub / name]
    else:
        cands = [run / "runs" / sub / "_assigned" / (cid + ".json"),
                 run / sub / "_assigned" / (cid + ".json")]
    for p in cands:
        if p.exists():
            return load_json(p)
    return None


def _find_pred(run: Path, cid: str):
    for p in (run / "final_results" / ("%s_Pred.json" % cid),
              run / "final_results" / ("%s.json" % cid)):
        if p.exists():
            return load_json(p)
    return None


def tight_ink(bgr, box, gray_max=INK_GRAY, sat_max=INK_SAT, pad=2.0):
    """框内「暗且低饱和」墨迹的紧致外接框（找不到返回 None）。"""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    mx, mn, g = sub.max(2), sub.min(2), sub.mean(2)
    m = ((mx - mn) < sat_max) & (g < gray_max)
    if int(m.sum()) < 6:
        return None
    on = np.nonzero(m.sum(0) > 0)[0]
    oy = np.nonzero(m[:, int(on[0]):int(on[-1]) + 1].sum(1) > 0)[0]
    if len(on) == 0 or len(oy) == 0:
        return None
    return (float(x1 + on[0] - pad), float(y1 + oy[0] - pad),
            float(x1 + on[-1] + 1 + pad), float(y1 + oy[-1] + 1 + pad))


def tight_ink_ex(bgr, box, gray_max=INK_GRAY, sat_max=INK_SAT, pad=2.0,
                 drop_lines=True):
    """同 tight_ink，但先剔除「线状」连通块（IC 体描边/长线会毁掉紧致框）。"""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    mx, mn, g = sub.max(2), sub.min(2), sub.mean(2)
    m = (((mx - mn) < sat_max) & (g < gray_max)).astype(np.uint8)
    if int(m.sum()) < 6:
        return None
    if drop_lines:
        n, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        H, W = m.shape
        for i in range(1, n):
            bw_, bh_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            ar = st[i, cv2.CC_STAT_AREA]
            if ar < 4:
                continue
            if (bw_ >= 0.75 * W and bh_ <= 4) or (bh_ >= 0.75 * H and bw_ <= 4):
                continue          # 横线 / 竖线（体描边）
            if ar >= 0.55 * H * W:
                continue          # 整块填充
            keep[lab == i] = 1
        m = keep
    if int(m.sum()) < 6:
        return None
    on = np.nonzero(m.sum(0) > 0)[0]
    oy = np.nonzero(m[:, int(on[0]):int(on[-1]) + 1].sum(1) > 0)[0]
    if len(on) == 0 or len(oy) == 0:
        return None
    return (float(x1 + on[0] - pad), float(y1 + oy[0] - pad),
            float(x1 + on[-1] + 1 + pad), float(y1 + oy[-1] + 1 + pad))


def tight_ink_grp(bgr, box, gray_max=INK_GRAY, sat_max=INK_SAT, pad=2.0,
                  near_y=None):
    """剔线后按「同一行字」分组，只取离 near_y 最近的哪一组（避免串到邻脚名）。"""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    mx, mn, g = sub.max(2), sub.min(2), sub.mean(2)
    m = (((mx - mn) < sat_max) & (g < gray_max)).astype(np.uint8)
    if int(m.sum()) < 6:
        return None
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    H, W = m.shape
    comps = []
    for i in range(1, n):
        bw_, bh_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        ar = st[i, cv2.CC_STAT_AREA]
        if ar < 4:
            continue
        if (bw_ >= 0.75 * W and bh_ <= 4) or (bh_ >= 0.75 * H and bw_ <= 4):
            continue
        if ar >= 0.55 * H * W:
            continue
        comps.append((st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                      bw_, bh_))
    if not comps:
        return None
    # 按 y 重叠贪心成组
    comps.sort(key=lambda c: (c[1], c[0]))
    groups = []
    for c in comps:
        placed = False
        for gp in groups:
            oy = min(gp[3], c[1] + c[3]) - max(gp[1], c[1])
            if oy > 0.45 * min(gp[3], c[3]):
                gp[0] = min(gp[0], c[0]); gp[1] = min(gp[1], c[1])
                gp[2] = max(gp[2], c[0] + c[2]); gp[3] = max(gp[3], c[1] + c[3])
                placed = True
                break
        if not placed:
            groups.append([c[0], c[1], c[0] + c[2], c[1] + c[3]])
    if near_y is None:
        near_y = (y1 + y2) / 2.0
    best = min(groups, key=lambda gp: abs((gp[1] + gp[3]) / 2.0
                                          + y1 - float(near_y)))
    return (float(x1 + best[0] - pad), float(y1 + best[1] - pad),
            float(x1 + best[2] + pad), float(y1 + best[3] + pad))


def read_up(rr, bgr, box, scale=2.0, role="pin_name"):
    """把小裁片放大后再读（小框在 CRNN 输入尺寸下细节不足）。"""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = bgr[y1:y2, x1:x2]
    up = cv2.resize(sub, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
    H, W = up.shape[:2]
    t, c = rr.read_many(up, [((0.0, 0.0, float(W), float(H)), role)])[0]
    return ((t or "").strip(), float(c))


def ink_count(bgr, box, gray_max=INK_GRAY, sat_max=INK_SAT):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0
    sub = bgr[y1:y2, x1:x2].astype(np.int16)
    mx, mn, g = sub.max(2), sub.min(2), sub.mean(2)
    return int((((mx - mn) < sat_max) & (g < gray_max)).sum())


def side_of(pt, bb):
    """pt 落在 bbox(图像帧) 的哪条边；返回 (边名, 内向单位向量)。"""
    x1, y1, x2, y2 = bb
    ds = {"L": abs(pt[0] - x1), "R": abs(pt[0] - x2),
          "T": abs(pt[1] - y1), "B": abs(pt[1] - y2)}
    s = min(ds, key=lambda k: ds[k])
    vec = {"L": (1.0, 0.0), "R": (-1.0, 0.0),
           "T": (0.0, 1.0), "B": (0.0, -1.0)}[s]
    return s, vec


def strip_box(pt, bb, half, depth=DEPTH, inset=1.0):
    """贴边、以脚点为中心、往体内 depth 的带（图像帧）。"""
    s, _v = side_of(pt, bb)
    x1, y1, x2, y2 = bb
    if s in ("L", "R"):
        xe = x1 + inset if s == "L" else x2 - inset
        xa, xb = (xe, xe + depth) if s == "L" else (xe - depth, xe)
        return (min(xa, xb), pt[1] - half, max(xa, xb), pt[1] + half)
    ye = y1 + inset if s == "T" else y2 - inset
    ya, yb = (ye, ye + depth) if s == "T" else (ye - depth, ye)
    return (pt[0] - half, min(ya, yb), pt[0] + half, max(ya, yb))


def _libbest(libset, text):
    """把读出串吸附到库内合法引脚名（返回 (名字, 相似度) 或 None）。"""
    nt = libpins.norm(text or "")
    if not nt or not libset:
        return None
    best, bs = None, 0.0
    for ln, orig in libset.items():
        if abs(len(ln) - len(nt)) > 3:
            continue
        r = difflib.SequenceMatcher(None, nt, ln).ratio()
        if r > bs:
            best, bs = orig, r
    return None if best is None else (best, bs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    run = Path(a.run)

    from _textseg_eval import ROLE_OF  # noqa: F401  (保持与检出口径一致)
    eng = TextEngine()
    rr = eng.recognizer()
    ctx = libpins.new_ctx()
    ctx.load()
    cfg = M.EvalConfig()
    tot = collections.Counter()
    arm = collections.Counter()
    shown = 0
    records = []
    print("%-6s %-5s %-6s %-8s %-8s %s" %
          ("case", "comp", "pin", "gt名", "最好候选", "体内墨像素"))
    print("-" * 78)
    for c in [x for x in discover(DEFAULT_DATA_ROOT) if x.split == a.split][:a.n]:
        raw = _find_pred(run, c.case_id)
        if raw is None:
            continue
        gt = M.normalise_doc(load_json(Path(c.json_path)))
        pred = M.normalise_doc(raw)
        _c, keymap = M.assign_components(gt, pred, cfg, {})
        pred2 = M.rewrite_keys(pred, keymap)
        box = {k for k, v in (gt.get("components") or {}).items()
               if M.canon_type((v or {}).get("type")) in M.IC_TYPES}
        asg = _find(run, c.case_id, "assigned")
        s1 = _find(run, c.case_id, "stage1") or {}
        if not asg or not s1:
            continue
        bgr = imread_unicode(s1["image"]["path"])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bw3 = cv2.cvtColor(np.where(gray < 170, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        s1c = {x.get("id"): x for x in (s1.get("components") or [])}
        det = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **H_CFG)]
        det_read = []
        if det:
            det_read = [((t or "").strip(), float(cf)) for t, cf in
                        rr.read_many(gray, [(b, "pin_name") for b in det])]
        # 槽位：key -> {号: (cid, k, rec)}
        slots = collections.defaultdict(dict)
        for cid, cc in (asg.get("comps") or {}).items():
            key = ((cc.get("designator") or {}).get("text") or "").strip()
            for k, rec in (cc.get("pins") or {}).items():
                num = (((rec or {}).get("number") or {}).get("text") or "").strip()
                if num:
                    slots[key].setdefault(num, (cid, int(k), rec))
        # 每颗 IC 的号间距（决定带宽度）
        for comp in sorted(box):
            g = {M.pin_key_to_number(pk): v for pk, v in
                 ((gt.get("pins") or {}).get(comp) or {}).items()}
            p = {M.pin_key_to_number(pk): v for pk, v in
                 ((pred2.get("pins") or {}).get(comp) or {}).items()}
            miss = sorted(set(g) - set(p), key=lambda z: (len(str(z)), str(z)))
            if not miss:
                continue
            # 这颗 IC 在官方库里的合法引脚名集合（型号 -> 名字）
            libset = {}
            libkey = ""
            part = ""
            _cid0 = next(iter((slots.get(comp) or {}).values()), None)
            if _cid0:
                _cc = asg["comps"].get(_cid0[0]) or {}
                pn = ((_cc.get("partname") or {}) or {}).get("text") or comp
                part = str(pn)
                _k = ctx.key_for(str(pn))
                if _k:
                    libkey = str(_k)
                    libset = {libpins.norm(n): n for n in (ctx.names_of(_k) or {})}
            pts_all = []
            for _n, (_cid, _k, _r) in (slots.get(comp) or {}).items():
                pl = (s1.get("pins") or {}).get(_cid) or []
                if _k < len(pl):
                    pts_all.append(pl[_k])
            pitch = 20.0
            if len(pts_all) > 2:
                yy = sorted(v[1] for v in pts_all)
                dd = [yy[i + 1] - yy[i] for i in range(len(yy) - 1)]
                dd = [d for d in dd if d > 3]
                if dd:
                    pitch = float(np.median(dd))
            half = max(7.0, min(26.0, pitch * 0.62))
            for num in miss:
                sl = (slots.get(comp) or {}).get(str(num))
                if sl is None:
                    tot["no_slot"] += 1
                    continue
                cid, k, rec = sl
                pl = (s1.get("pins") or {}).get(cid) or []
                if k >= len(pl):
                    tot["no_point"] += 1
                    continue
                pt = (float(pl[k][0]), float(pl[k][1]))
                cinfo = s1c.get(cid) or {}
                bb = cinfo.get("bbox_img") or (asg["comps"][cid].get("bbox_img"))
                if not bb:
                    tot["no_bbox"] += 1
                    continue
                bb = [float(v) for v in bb]
                gtname = str((g[num] or {}).get("pinname") or "")
                nb = ((rec.get("number") or {}).get("box"))
                tot["pin"] += 1
                cands = {}
                # A) 图里已有的文字检出框
                for j, b in enumerate(det):
                    cb = _ctr(b)
                    if not cb:
                        continue
                    if nb and _ctr(nb) and math.hypot(cb[0] - _ctr(nb)[0],
                                                      cb[1] - _ctr(nb)[1]) < 3.0:
                        continue
                    if math.hypot(cb[0] - pt[0], cb[1] - pt[1]) <= R_NEAR:
                        cands.setdefault("det", (det_read[j][0], det_read[j][1]))
                # B) 体内带 -> 暗墨紧致框
                sb = strip_box(pt, bb, half)
                n_ink = ink_count(bgr, sb)
                tb = tight_ink(bgr, sb)
                if tb is not None:
                    t, cf = rr.read_many(gray, [(tb, "pin_name")])[0]
                    cands["strip"] = ((t or "").strip(), float(cf))
                    # B2) 同一紧致框在二值图上重读
                    t2, cf2 = rr.read_many(bw3, [(tb, "pin_name")])[0]
                    cands["bw"] = ((t2 or "").strip(), float(cf2))
                # B3) 更宽的带 + 更松的墨阈值（吃掉抗锯齿/浅墨）
                sb2 = strip_box(pt, bb, half * 1.25, depth=DEPTH + 10.0)
                tb2 = tight_ink(bgr, sb2, gray_max=205, sat_max=95)
                if tb2 is not None:
                    t3, cf3 = rr.read_many(gray, [(tb2, "pin_name")])[0]
                    cands["strip2"] = ((t3 or "").strip(), float(cf3))
                # B4) 内缩（跳过体描边）+ 剔线状连通块
                for _nm, _ins, _gm, _sm in (("s_nl2", 2.0, INK_GRAY, INK_SAT),
                                            ("s_nl4", 4.0, INK_GRAY, INK_SAT),
                                            ("s_nl4w", 4.0, 205, 95),
                                            ("i6", 6.0, INK_GRAY, INK_SAT),
                                            ("i8", 8.0, INK_GRAY, INK_SAT)):
                    _sb = strip_box(pt, bb, half * 1.1, depth=DEPTH + 6.0,
                                    inset=_ins)
                    _tb = tight_ink_ex(bgr, _sb, gray_max=_gm, sat_max=_sm)
                    if _tb is not None:
                        _t, _c = rr.read_many(gray, [(_tb, "pin_name")])[0]
                        cands[_nm] = ((_t or "").strip(), float(_c))
                # B5) 行分组：只取离脚最近的那一组字
                for _nm, _ins in (("grp4", 4.0), ("grp6", 6.0)):
                    _sb = strip_box(pt, bb, half * 1.35, depth=DEPTH + 12.0,
                                    inset=_ins)
                    _tb = tight_ink_grp(bgr, _sb, near_y=pt[1], pad=1.0)
                    if _tb is not None:
                        _t, _c = rr.read_many(gray, [(_tb, "pin_name")])[0]
                        cands[_nm] = ((_t or "").strip(), float(_c))
                        _u = read_up(rr, gray, _tb, 2.0)
                        if _u:
                            cands[_nm + "up"] = _u
                # C) 号框沿最近的边镜像进体内
                s, vec = side_of(pt, bb)
                if nb:
                    nb = [float(v) for v in nb]
                    nw = nb[2] - nb[0]
                    nh = nb[3] - nb[1]
                    ctr = _ctr(nb)
                    if s in ("L", "R"):
                        edge = bb[0] if s == "L" else bb[2]
                        d = (2.0 if s == "L" else -2.0) * 1.0
                        mx1 = edge + (2.0 if s == "L" else -2.0 - nw)
                        if s == "L":
                            mb = (edge + 2.0, ctr[1] - nh / 2.0,
                                  edge + 2.0 + nw, ctr[1] + nh / 2.0)
                        else:
                            mb = (edge - 2.0 - nw, ctr[1] - nh / 2.0,
                                  edge - 2.0, ctr[1] + nh / 2.0)
                    else:
                        edge = bb[1] if s == "T" else bb[3]
                        if s == "T":
                            mb = (ctr[0] - nw / 2.0, edge + 2.0,
                                  ctr[0] + nw / 2.0, edge + 2.0 + nh)
                        else:
                            mb = (ctr[0] - nw / 2.0, edge - 2.0 - nh,
                                  ctr[0] + nw / 2.0, edge - 2.0)
                    t, cf = rr.read_many(gray, [(mb, "pin_name")])[0]
                    cands["mir"] = ((t or "").strip(), float(cf))
                hit = [n for n, (t, _c) in cands.items() if _norm(t) == _norm(gtname)]
                # D) 官方库模糊解码：把读歪的串吸附到该型号的合法引脚名
                if libset and len(libset) <= 400:
                    best, bs = None, 0.0
                    for _n, (t, _c) in sorted(cands.items()):
                        nt = libpins.norm(t)
                        if not nt:
                            continue
                        for ln, orig in libset.items():
                            r = difflib.SequenceMatcher(None, nt, ln).ratio()
                            if r > bs:
                                best, bs = orig, r
                    if best and bs >= 0.6:
                        cands["lib"] = (best, bs)
                        tot["lib_emit"] += 1
                        if _norm(best) == _norm(gtname):
                            tot["lib_right"] += 1
                        else:
                            tot["lib_wrong"] += 1
                        if _norm(best) == _norm(gtname):
                            hit.append("lib")
                for _n in ("strip", "strip2", "bw", "mir", "det",
                           "s_nl2", "s_nl4", "s_nl4w", "i6", "i8",
                           "grp4", "grp4up", "grp6", "grp6up"):
                    if _n in cands:
                        tot["emit_" + _n] += 1
                        if _norm(cands[_n][0]) == _norm(gtname):
                            tot["right_" + _n] += 1
                # 投票：≥2 个臂给同一串才算「有共识」
                _vs = collections.defaultdict(int)
                for _n, (t, _c) in cands.items():
                    if _norm(t):
                        _vs[_norm(t)] += 1
                if _vs:
                    _win, _cnt = max(_vs.items(), key=lambda z: z[1])
                    if _cnt >= 2:
                        tot["vote_emit"] += 1
                        if _win == _norm(gtname):
                            tot["vote_right"] += 1
                        else:
                            tot["vote_wrong"] += 1
                if not n_ink:
                    tot["no_ink"] += 1
                else:
                    tot["has_ink"] += 1
                if hit:
                    tot["rescued"] += 1
                    for n in hit:
                        arm["hit_" + n] += 1
                else:
                    arm["miss"] += 1
                if a.show and shown < a.show:
                    shown += 1
                    best = "|".join("%s:%s" % (n, t) for n, (t, _c) in cands.items())
                    print("%-6s %-5s %-6s %-8s %-8s %d" %
                          (c.case_id, comp, num, gtname, best[:60], n_ink))
                _lb = {}
                if libset:
                    for _k2, _v2 in cands.items():
                        _m = _libbest(libset, _v2[0])
                        if _m:
                            _lb[_k2] = [_m[0], round(_m[1], 3)]
                records.append({
                    "case": c.case_id, "comp": comp, "pin": str(num),
                    "gt": gtname, "ink": int(n_ink),
                    "part": part, "libkey": libkey, "lib_n": len(libset),
                    "gt_in_lib": bool(_norm(gtname) and
                                      _norm(gtname) in libset),
                    "lib_best": _lb,
                    "cands": {k: [v[0], round(float(v[1]), 4)]
                              for k, v in cands.items()},
                })
    print("-" * 78)
    print("号已读对但被丢的脚：%d   其中救回 %d（%.1f%%）"
          % (tot["pin"], tot["rescued"], 100.0 * tot["rescued"] / max(1, tot["pin"])))
    print("  体内这个位置有墨 %d / 无墨 %d（无墨=图上就没印名字）"
          % (tot["has_ink"], tot["no_ink"]))
    print("  各臂命中：%s   全miss %d" % (dict(arm), arm["miss"]))
    for _n in ("det", "strip", "strip2", "bw", "mir", "s_nl2", "s_nl4",
               "s_nl4w", "i6", "i8", "grp4", "grp4up", "grp6", "grp6up"):
        print("    %-7s 出串 %3d  正确 %3d  准确率 %.3f"
              % (_n, tot["emit_" + _n], tot["right_" + _n],
                 1.0 * tot["right_" + _n] / max(1, tot["emit_" + _n])))
    print("    库模糊   出串 %3d  正确 %3d  错 %3d  准确率 %.3f"
          % (tot["lib_emit"], tot["lib_right"], tot["lib_wrong"],
             1.0 * tot["lib_right"] / max(1, tot["lib_emit"])))
    print("    ≥2臂共识 出串 %3d  正确 %3d  错 %3d  准确率 %.3f"
          % (tot["vote_emit"], tot["vote_right"], tot["vote_wrong"],
             1.0 * tot["vote_right"] / max(1, tot["vote_emit"])))
    print("  其它：no_slot %d  no_point %d  no_bbox %d"
          % (tot["no_slot"], tot["no_point"], tot["no_bbox"]))
    if a.dump:
        Path(a.dump).write_text(json.dumps(records, ensure_ascii=False,
                                           indent=1), encoding="utf-8")
        print("  明细 %d 条 -> %s" % (len(records), a.dump))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
