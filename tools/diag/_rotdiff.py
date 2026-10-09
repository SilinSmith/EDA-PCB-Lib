# -*- coding: utf-8 -*-
"""对比两种旋转裁片口径（定位 `pinrot.fix()` 与 `_c6num.py` 探针读数不一致的原因）。

A) 探针口径：先从原图裁条带 → 把条带整块转 90° → `_tight_ink_grp`
B) 模块口径：整图转 90° → 把条带框映射进旋转帧 → `_tight_ink_grp`

对同一个 (cid, pin, field) 把两者的「条带图」与「紧致框图」并排存成 PNG，便于肉眼看。
用法：python -X utf8 eda-pcb\tools\diag\_rotdiff.py <cid> <pin> [field] [run_dir] [case]
"""
import json
import os
import sys

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "eda-pcb"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "stage2"))
os.environ.setdefault("STAGE2_PINROT", "1")

import cv2                                            # noqa: E402
from text import TextEngine                           # noqa: E402
from assemble import build_doc                        # noqa: E402
import pinname as PN                                  # noqa: E402
import pinrot as PR                                   # noqa: E402

CID = sys.argv[1] if len(sys.argv) > 1 else "c0002"
PIN = int(sys.argv[2]) if len(sys.argv) > 2 else 0
FIELD = sys.argv[3] if len(sys.argv) > 3 else "name"
RUN = Path(sys.argv[4]) if len(sys.argv) > 4 else (ROOT / "work" / "_fast_out_c6")
CASE = sys.argv[5] if len(sys.argv) > 5 else "6"
OUT = ROOT / "work" / "_rotdbg"
OUT.mkdir(parents=True, exist_ok=True)


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


s1 = json.loads((RUN / "runs" / "stage1" / ("%s.stage1.json" % CASE)).read_text(encoding="utf-8"))
a = json.loads((RUN / "runs" / "stage2" / "_assigned" / ("%s.json" % CASE)).read_text(encoding="utf-8"))
comps = {c.get("id"): c for c in s1["components"]}
c = comps[CID]
bb = [float(v) for v in c["bbox_img"]]
pts = s1["pins"][CID]
pt = (float(pts[PIN][0]), float(pts[PIN][1]))
bgr = imread_u(s1["image"]["path"])
eng = TextEngine(device="cpu")
rec = eng.recognizer()

tl = list((a["comps"][CID].get("pins") or {}).items())
ys = sorted(float(pts[int(k)][1]) for k, _p in tl if str(k).isdigit() and int(k) < len(pts))
dd = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
dd = [d for d in dd if d > 3]
pitch = float(sorted(dd)[len(dd) // 2]) if dd else 20.0
half = max(PN.HALF_LO, min(PN.HALF_HI, pitch * PN.HALF_FRAC)) * 1.1

box0 = (PN._strip_box(pt, bb, half, depth=PR.NAME_DEPTH) if FIELD == "name"
        else PR._out_box(pt, bb, half, PR.NUM_DEPTH))
side = PN._side_of(pt, bb)
print("cid=%s bb=%s pt=%s side=%s pitch=%.1f half=%.2f box0=%s"
      % (CID, [round(v) for v in bb], [round(v) for v in pt], side, pitch, half,
         [round(v) for v in box0]))

role = "pin_name" if FIELD == "name" else "pin_number"
res = {}
for which in ("0", "cw", "ccw"):
    code = PR._ROT[which]
    # ---- B) 模块口径：整图旋转 + 映射框 ----
    rimg = bgr if code is None else cv2.rotate(bgr, code)
    rbox = PR._map_box(box0, bgr.shape, which)
    rpt = PR._map_pt(pt, bgr.shape, which)
    x0, y0, x1, y1 = [int(round(v)) for v in rbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rimg.shape[1], x1), min(rimg.shape[0], y1)
    tb_mod = PN._tight_ink_grp(rimg, [float(x0), float(y0), float(x1), float(y1)],
                               near_y=rpt[1])
    # ---- A) 探针口径：先裁条带再旋转 ----
    sx0, sy0, sx1, sy1 = [int(round(v)) for v in box0]
    crop = bgr[max(0, sy0):sy1, max(0, sx0):sx1].copy()
    if code is None:
        rcrop = crop
        cpt = (pt[0] - sx0, pt[1] - sy0)
    else:
        rcrop = cv2.rotate(crop, code)
        cpt = PR._map_pt((pt[0] - sx0, pt[1] - sy0), crop.shape, which)
    tb_prb = PN._tight_ink_grp(rcrop, [0.0, 0.0, float(rcrop.shape[1]), float(rcrop.shape[0])],
                               near_y=None if code is None else cpt[1])
    rg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    rgc = cv2.cvtColor(rcrop, cv2.COLOR_BGR2GRAY)
    # 内容等价性检查：整图旋转后裁的窗口 vs 先裁后转的窗口
    sub_full = rimg[y0:y1, x0:x1]
    same = (sub_full.shape == rcrop.shape and sub_full.size and
            bool(np.array_equal(sub_full, rcrop)))
    if not same:
        d = (sub_full.shape, rcrop.shape)
        diff = None
        if sub_full.shape == rcrop.shape:
            diff = int(np.abs(sub_full.astype(int) - rcrop.astype(int)).max())
        print("     !! 内容不等 shape=%s maxdiff=%s" % (d, diff))
        cv2.imwrite(str(OUT / ("%s_p%d_%s_%s_fullwin.png" % (CID, PIN, FIELD, which))),
                    sub_full)
        cv2.imwrite(str(OUT / ("%s_p%d_%s_%s_croprot.png" % (CID, PIN, FIELD, which))),
                    rcrop)
    got_mod = rec.read_many(rg, [(tb_mod, role)])[0] if tb_mod is not None else ("-", 0.0)
    got_prb = rec.read_many(rgc, [(tb_prb, role)])[0] if tb_prb is not None else ("-", 0.0)
    # 交叉读：模块大图 + 探针框（搬到全图坐标）、探针小图 + 模块框（搬到小图坐标）
    got_c = ("-", 0.0)
    if tb_prb is not None:
        tb_c = [tb_prb[0] + x0 - sx0 * 0 + 0, tb_prb[1], tb_prb[2], tb_prb[3]]
        tb_c = [x0 + tb_prb[0] - (0 if which == "0" else 0), y0 + tb_prb[1],
                x0 + tb_prb[2], y0 + tb_prb[3]]
        got_c = rec.read_many(rg, [(tb_c, role)])[0]
    got_d = ("-", 0.0)
    if tb_mod is not None:
        tb_d = [tb_mod[0] - x0, tb_mod[1] - y0, tb_mod[2] - x0, tb_mod[3] - y0]
        got_d = rec.read_many(rgc, [(tb_d, role)])[0]
    res[which] = (tb_mod, got_mod, tb_prb, got_prb)
    print("%-4s mod box=%s tb=%s -> %r(%.3f) | prb tb=%s -> %r(%.3f) | 交叉 %r / %r"
          % (which, [x0, y0, x1, y1],
             None if tb_mod is None else [round(v) for v in tb_mod], got_mod[0], got_mod[1],
             None if tb_prb is None else [round(v) for v in tb_prb], got_prb[0], got_prb[1],
             got_c[0], got_d[0]))
    # 变体读：紧致框 ± pad、原条带框、2× 放大 —— 看能不能把 'GND' 稳定读出来
    if tb_mod is not None:
        variants = []
        for pad in (0, 1, 2, 3, 4):
            b = [tb_mod[0] - pad, tb_mod[1] - pad, tb_mod[2] + pad, tb_mod[3] + pad]
            variants.append(("pad%d" % pad, b))
        variants.append(("raw", [float(x0), float(y0), float(x1), float(y1)]))
        line = []
        for nm, b in variants:
            try:
                t, c = rec.read_many(rg, [(b, role)])[0]
            except Exception as e:                # noqa: BLE001
                t, c = "ERR:" + type(e).__name__, 0.0
            line.append("%s=%r(%.2f)" % (nm, t, c))
        print("     变体 " + " ".join(line))
        # 放大 2× 再读（把紧致框抠出来放大，模仿探针「先裁后转」的尺度）
        b = [int(tb_mod[0]), int(tb_mod[1]), int(tb_mod[2]), int(tb_mod[3])]
        sub = rimg[max(0, b[1]):b[3], max(0, b[0]):b[2]]
        if sub.size:
            up = cv2.resize(sub, (sub.shape[1] * 2, sub.shape[0] * 2),
                            interpolation=cv2.INTER_CUBIC)
            t, c = rec.read_many(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY),
                                 [([0.0, 0.0, float(up.shape[1]), float(up.shape[0])], role)])[0]
            print("     放大2x -> %r(%.2f)" % (t, c))
    # 存图：上=模块口径条带（含紧致框），下=探针口径
    def _tile(img, tb):
        v = img.copy()
        if tb is not None:
            cv2.rectangle(v, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])),
                          (0, 0, 255), 1)
        return v

    top, bot = _tile(rimg[y0:y1, x0:x1], None if tb_mod is None else
                     [tb_mod[0] - x0, tb_mod[1] - y0, tb_mod[2] - x0, tb_mod[3] - y0]), \
        _tile(rcrop, tb_prb)
    w = max(top.shape[1], bot.shape[1])
    canvas = np.full((top.shape[0] + bot.shape[0] + 6, w + 2, 3), 200, np.uint8)
    canvas[0:top.shape[0], 1:1 + top.shape[1]] = top
    canvas[top.shape[0] + 6:top.shape[0] + 6 + bot.shape[0], 1:1 + bot.shape[1]] = bot
    cv2.imwrite(str(OUT / ("%s_p%d_%s_%s.png" % (CID, PIN, FIELD, which))), canvas)

print("图存到", OUT)
