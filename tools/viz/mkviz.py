# -*- coding: utf-8 -*-
"""逐批次识别可视化：把一个 case 的 GT / 各阶段产物 / 候选池 / 取景框画成可叠加的图层。

产出（默认 `work\\viz\\`）：
    assets\\<case>\\base.png     原图（缩放后）
    assets\\<case>\\gt.png       GT 元件 + GT 引脚 + GT 键/Name
    assets\\<case>\\ours.png     我方元件（按配对结果着色）+ 我方键
    assets\\<case>\\texts.png   stage1 文字框 + 它被读成了什么
    assets\\<case>\\regions.png 融合层给每个槽位建的取景框（focus 亮框）
    assets\\<case>\\pins.png    引脚点 + 读引脚用的裁剪窗
    assets\\<case>\\net.png     网络标签
    cases\\<case>.json          全部结构化数据（槽位候选、阶段操作、失配账本）

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\viz\\mkviz.py 1 0001 8 0048
    python work\\tools\\viz\\mkviz.py --tier quickset
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from PIL import Image, ImageDraw, ImageFont                       # noqa: E402

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, find_case  # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components, evaluate_case  # noqa: E402
from schematic_parser.metrics import IC_TYPES, is_unkeyed, canon_type              # noqa: E402
from cvparts import hypo as HY                                     # noqa: E402
from fusion import claims as CL                                    # noqa: E402
from fusion import priors as PR                                    # noqa: E402
from fusion import solve as SV                                     # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"
VIZ = ROOT / "work" / "viz"

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
_FONTS: Dict[int, Any] = {}

# 图层配色
C_GT_IC = (0, 200, 83, 255)
C_GT_OTHER = (0, 200, 83, 150)
C_GT_MISS = (255, 145, 0, 255)
C_OK = (41, 121, 255, 255)
C_EXTRA = (255, 23, 68, 255)
C_TYPEMIS = (255, 145, 0, 255)
C_TEXT = (0, 188, 212, 255)
C_TEXT_READ = (255, 235, 59, 255)
C_REGION = (170, 0, 255, 180)
C_FOCUS = (255, 0, 255, 255)
C_PIN = (255, 193, 7, 255)
C_PINWIN = (255, 112, 67, 200)
C_NET = (118, 255, 3, 230)


def font(sz: int):
    if sz not in _FONTS:
        try:
            _FONTS[sz] = ImageFont.truetype(FONT_PATH, sz)
        except Exception:                                          # noqa: BLE001
            _FONTS[sz] = ImageFont.load_default()
    return _FONTS[sz]


def label(dr: ImageDraw.ImageDraw, xy, text: str, fill, sz: int = 13,
          pad: int = 2, bg=(0, 0, 0, 170)):
    """带底色的短标签（保证在任意底图上可读）。"""
    if not text:
        return
    f = font(sz)
    try:
        x0, y0, x1, y1 = dr.textbbox((0, 0), text, font=f)
    except Exception:                                              # noqa: BLE001
        x0, y0, x1, y1 = 0, 0, len(text) * sz // 2, sz
    w, h = x1 - x0, y1 - y0
    x, y = xy
    dr.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=bg)
    dr.text((x, y), text, font=f, fill=fill)


def new_layer(w: int, h: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def dash_rect(dr, box, fill, width=2, dash=8, gap=6):
    x0, y0, x1, y1 = box
    def seg(a, b, horiz):
        p = a
        while p < b:
            q = min(p + dash, b)
            if horiz:
                dr.line([p, a_y, q, a_y], fill=fill, width=width)
            else:
                dr.line([a_x, p, a_x, q], fill=fill, width=width)
            p = q + gap
    a_x, a_y = x0, y0
    seg(x0, x1, True)
    a_y = y1
    seg(x0, x1, True)
    a_x = x0
    seg(y0, y1, False)
    a_x = x1
    seg(y0, y1, False)


# ---------------------------------------------------------------- 数据装配

def gt_box_img(b, H):
    """GT 的 bbox 在 JSON 空间（y 已翻转）⇒ 换回图像空间。"""
    x0, y0, x1, y1 = b
    return [x0, H - y1, x1, H - y0]


def load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def cid_gt_key(s1: dict, gt: dict) -> Dict[str, str]:
    """`cid -> GT 键`：用**框重叠**判，两边都在 JSON 空间（stage1 的 `bbox_json` 与 GT 的 `bbox`）。"""
    gts = []
    for k, c in (gt.get("components") or {}).items():
        b = c.get("bbox")
        if b:
            gts.append((k, [float(x) for x in b]))
    out = {}
    for c in (s1.get("components") or []):
        b = c.get("bbox_json") or c.get("bbox_img")
        if not b:
            continue
        b = [float(x) for x in b]
        best, bi = None, 0.0
        for k, gb in gts:
            v = _iou(b, gb)
            if v > bi:
                best, bi = k, v
        if best is not None and bi >= 0.5:
            out[str(c.get("id"))] = best
    return out


def build_case(cid: str, pred_dir: Path, vlm_dir: Path, cfg: EvalConfig,
               want_pool: bool) -> Optional[dict]:
    case = find_case(cid, DEFAULT_DATA_ROOT)
    if case is None:
        print("  %-16s 找不到 GT case，跳过" % cid)
        return None
    gt = load_json(Path(case.json_path)) if case.json_path else None
    pred = load_json(pred_dir / ("%s.json" % cid))
    if gt is None or pred is None:
        print("  %-16s GT 或预测缺，跳过（gt=%s pred=%s）" % (cid, gt is not None, pred is not None))
        return None
    s1 = load_json(RUN / "stage1" / ("%s.stage1.json" % cid))
    asg = load_json(RUN / "stage2" / "_assigned" / ("%s.json" % cid))
    s2asg = RUN / "stage2" / "_assigned" / ("%s.json" % cid)
    pinidx = load_json(RUN / "stage2" / ("%s.pinindex.json" % cid))
    vlm = load_json(vlm_dir / ("%s.json" % cid)) if vlm_dir else None
    if s1 is None:
        print("  %-16s stage1 缺，跳过" % cid)
        return None
    H = float((s1.get("image") or {}).get("h") or 0)
    W = float((s1.get("image") or {}).get("w") or 0)

    res, keymap = assign_components(gt, pred, cfg)
    cs = evaluate_case(gt, pred, cid, cfg)
    # keymap: 预测键 -> GT 键；反过来看「GT 键配上了没有」
    km_inv = {}
    for pk, gk in keymap.items():
        km_inv.setdefault(gk, pk)

    out: Dict[str, Any] = {
        "id": cid,
        "split": case.key,
        "png": case.png_path,
        "json": case.json_path,
        "w": W, "h": H,
        "n_pred_comps": len(pred.get("components") or {}),
        "score": cs.as_dict() if hasattr(cs, "as_dict") else {
            "case_id": cid, "score": cs.score, "diagnostic": dict(cs.diagnostic)},
        "paths": {
            "s1": str(RUN / "stage1" / ("%s.stage1.json" % cid)),
            "assigned": str(s2asg),
            "pred": str(pred_dir / ("%s.json" % cid)),
            "vlm": str(vlm_dir / ("%s.json" % cid)) if vlm_dir else "",
            "pinindex": str(RUN / "stage2" / ("%s.pinindex.json" % cid)),
        },
        "gt": {"comps": [], "n_pins": 0, "nets": []},
        "ours": {"comps": [], "nets": []},
        "texts": [],
        "slots": [],
        "regions": [],
        "pinwins": [],
        "ops": [],
        "labels": [],
    }
    if not hasattr(cs, "as_dict"):
        out["score"] = {
            "case_id": cid, "score": round(cs.score, 4),
            "component": cs.component.as_dict(), "pin": cs.pin.as_dict(),
            "net_hypergraph": cs.net_hypergraph.as_dict(), "net_line": cs.net_line.as_dict(),
            "diagnostic": {k: round(v, 6) for k, v in cs.diagnostic.items()},
        }

    # ---- GT 元件 ----
    for k, c in (gt.get("components") or {}).items():
        b = c.get("bbox")
        if not b:
            continue
        out["gt"]["comps"].append({
            "key": k, "type": c.get("type"), "Name": c.get("Name"), "value": c.get("value"),
            "box": gt_box_img(b, H), "matched": k in km_inv, "pred_key": km_inv.get(k),
            "is_ic": (c.get("type") in IC_TYPES),
        })
    out["gt"]["n_pins"] = len(gt.get("pins") or {})
    for nk, n in (gt.get("nets") or {}).items():
        out["gt"]["nets"].append({"key": nk, "hyperGraph": n.get("hyperGraph"),
                                  "label": n.get("label")})

    # ---- 我方元件（带配对判决）----
    gt_keys = set((gt.get("components") or {}).keys())
    for k, c in (pred.get("components") or {}).items():
        gk = keymap.get(k)
        if gk is None:
            verdict = "coord" if str(k) in {str(v) for v in keymap.values()} else "extra"
            if k in gt_keys:
                verdict = "key"
        else:
            verdict = "key"
        b = c.get("bbox") or [0, 0, 0, 0]
        out["ours"]["comps"].append({
            "key": k, "type": c.get("type"), "Name": c.get("Name"), "value": c.get("value"),
            "box": [b[0], H - b[3], b[2], H - b[1]], "gt_key": gk, "verdict": verdict,
            "n_pins": len(((pred.get("pins") or {}).get(k) or {})),
        })
    for nk, n in (pred.get("nets") or {}).items():
        out["ours"]["nets"].append({"key": nk, "hyperGraph": n.get("hyperGraph"),
                                    "label": n.get("label")})
    return {"case": case, "gt": gt, "pred": pred, "s1": s1, "asg": asg, "vlm": vlm,
            "pinidx": pinidx, "H": H, "W": W, "keymap": keymap, "res": res,
            "out": out, "pred_dir": pred_dir, "vlm_dir": vlm_dir, "cfg": cfg,
            "want_pool": want_pool}


# ---------------------------------------------------------------- 画图层

def draw_layers(B: dict) -> Dict[str, Image.Image]:
    """图层只画**几何**（不画文字）——文字与高亮由网页按「当前选中项」叠上去。

    这样默认视图不会被上百个标签糊成一团，选中某一项时才显示它的文字。
    """
    out = B["out"]
    H, W = B["H"], B["W"]
    s1 = B["s1"]
    gt = B["gt"]

    # ---- GT 元件 / 引脚 ----
    im, dr = new_layer(int(W), int(H))
    for c in out["gt"]["comps"]:
        box = c["box"]
        if c["matched"]:
            dr.rectangle(box, outline=C_GT_IC if c["is_ic"] else C_GT_OTHER, width=2)
        else:
            dash_rect(dr, box, C_GT_MISS, width=3)
    for k, plist in (gt.get("pins") or {}).items():
        for p in (plist or []):
            pt = p.get("point") if isinstance(p, dict) else None
            if pt:
                x, y = float(pt.get("x", 0)), H - float(pt.get("y", 0))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = float(p[0]), H - float(p[1])
            else:
                continue
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=C_GT_IC)
    gtl = im

    # ---- 我方元件 ----
    im, dr = new_layer(int(W), int(H))
    for c in out["ours"]["comps"]:
        v = c["verdict"]
        col = C_OK if v == "key" else (C_TYPEMIS if c["gt_key"] else C_EXTRA)
        dr.rectangle(c["box"], outline=col, width=2)
    ourl = im

    # ---- stage1 文字框 ----
    im, dr = new_layer(int(W), int(H))
    for t in out["texts"]:
        dr.rectangle(t["box"], outline=C_TEXT, width=1)
    txtl = im

    # ---- 引脚点 + 引脚裁剪窗 ----
    im, dr = new_layer(int(W), int(H))
    for k, plist in (s1.get("pins") or {}).items():
        for p in (plist or []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                dr.ellipse([float(p[0]) - 2, float(p[1]) - 2,
                            float(p[0]) + 2, float(p[1]) + 2], fill=C_PIN)
    for w in out.get("pinwins") or []:
        dr.rectangle(w["box"], outline=C_PINWIN, width=1)
    pinl = im

    # ---- 网络标签 ----
    im, dr = new_layer(int(W), int(H))
    for lb in out["labels"]:
        dr.rectangle(lb["box"], outline=C_NET, width=2)
    netl = im

    return {"gt": gtl, "ours": ourl, "texts": txtl, "pins": pinl, "net": netl}


def main():
    ap = argparse.ArgumentParser(description="逐批次识别可视化：渲染一个 case 的全部图层")
    ap.add_argument("cases", nargs="*", help="case 名（默认取 --tier）")
    ap.add_argument("--tier", default="vizdemo", help="tiers.json 里的档名")
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fusion_out220no4"),
                    help="融合产物目录（含 <case>.json 与 _assigned/）")
    ap.add_argument("--vlm", default=str(ROOT / "work" / "_fusion_vlm"),
                    help="VLM 补强目录（画 vlm:enrich 候选的来源）")
    ap.add_argument("--out", default=str(VIZ))
    ap.add_argument("--max-side", type=int, default=1500, help="底图最长边缩到这个尺寸")
    ap.add_argument("--no-pool", action="store_true", help="不重建候选池（快，但看不到槽位候选）")
    a = ap.parse_args()

    tier_p = ROOT / "work" / "tools" / "eval" / "tiers.json"
    cases = list(a.cases)
    if not cases:
        try:
            cases = json.loads(tier_p.read_text(encoding="utf-8")).get(a.tier) or []
        except Exception:                                          # noqa: BLE001
            cases = []
    if not cases:
        print("没有 case 可渲染（--cases 或 --tier）")
        return 1

    out_root = Path(a.out)
    (out_root / "assets").mkdir(parents=True, exist_ok=True)
    (out_root / "cases").mkdir(parents=True, exist_ok=True)
    pred_dir = Path(a.pred)
    vlm_dir = Path(a.vlm) if a.vlm else None
    cfg = EvalConfig()
    ok = 0
    for cid in cases:
        print("== %s" % cid)
        B = build_case(cid, pred_dir, vlm_dir, cfg, not a.no_pool)
        if B is None:
            continue
        out = B["out"]
        H, W = B["H"], B["W"]
        scale = min(1.0, float(a.max_side) / max(W, H))
        ow, oh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))

        # ---- stage1 文字框 + 读数（读数到 _assigned 里找）----
        reads = {}
        asg = B["asg"] or {}
        for c in (asg.get("comps") or {}).values():
            dg = c.get("designator") or {}
            if dg.get("box_i") is not None:
                reads[int(dg["box_i"])] = {"text": dg.get("text"), "canon": dg.get("canon"),
                                           "conf": dg.get("conf"), "kind": "designator"}
        for bl in (asg.get("net_labels") or []):
            if bl.get("box_i") is not None:
                reads[int(bl["box_i"])] = {"text": bl.get("text"), "canon": bl.get("net"),
                                           "conf": bl.get("conf"), "kind": "net"}
        for i, t in enumerate(B["s1"].get("texts") or []):
            r = reads.get(i)
            out["texts"].append({"i": i, "box": t.get("bbox_img"),
                                 "read": (r or {}).get("text"),
                                 "canon": (r or {}).get("canon"),
                                 "conf": (r or {}).get("conf"),
                                 "kind": (r or {}).get("kind")})
        for bl in (asg.get("net_labels") or []):
            if bl.get("box"):
                out["labels"].append({"box": bl["box"], "text": bl.get("text") or bl.get("net")})

        # ---- 引脚读数窗（`_boxspace.py` 已实测：`_assigned` 里的 box 全是**图像空间**）----
        try:
            km = CL.vlm_keymap(B["s1"], asg)            # 装配键 -> cid（必须喂 stage1 文档）
            cid2key = {v: k for k, v in (km or {}).items()}
        except Exception:                                          # noqa: BLE001
            cid2key = {}
        cid2gt = cid_gt_key(B["s1"], B["gt"])            # cid -> GT 键（用框重叠判，两边都是 JSON 空间）
        out["cid2gt"] = cid2gt
        out["cid2key"] = cid2key
        for cidk, c in (asg.get("comps") or {}).items():
            for k, p in (c.get("pins") or {}).items():
                for kind in ("number", "name"):
                    e = (p or {}).get(kind)
                    if not (e and e.get("box")):
                        continue
                    out["pinwins"].append({
                        "cid": cidk, "key": cid2key.get(cidk), "k": k, "kind": kind,
                        "box": e["box"], "read": e.get("text"),
                        "conf": e.get("conf"),
                        "gt": (B["pinidx"] or {}).get(cidk, {}).get(str(k)),
                    })

        # ---- 元件级引脚对照（我方 `pin_<N>` vs GT 引脚列表；点坐标换到图像空间）----
        comp_pins = {}
        for k, pg in (B["pred"].get("pins") or {}).items():
            rows = []
            for pk, pv in (pg or {}).items():
                pt = ((pv or {}).get("point") or {})
                rows.append({"pin": pk, "pinname": (pv or {}).get("pinname"),
                             "x": pt.get("x"), "y": (H - float(pt["y"])) if pt.get("y") is not None else None})
            comp_pins[k] = {"ours": rows}
        for c in out["ours"]["comps"]:
            gk = c.get("gt_key")
            g = comp_pins.setdefault(c["key"], {})
            g["gt_key"] = gk
            g["verdict"] = c["verdict"]
            g["box"] = c["box"]
            g["Name"] = c.get("Name")
            g["value"] = c.get("value")
            if gk:
                gl = []
                gp = (B["gt"].get("pins") or {}).get(gk)
                it = gp.items() if isinstance(gp, dict) else enumerate(gp or [])
                for pk, p in it:
                    if isinstance(p, dict):
                        pt = p.get("point") or {}
                        gl.append({"pin": str(pk), "pinname": p.get("pinname"),
                                   "x": pt.get("x"),
                                   "y": (H - float(pt["y"])) if pt.get("y") is not None else None})
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        gl.append({"pin": str(pk), "pinname": None,
                                   "x": p[0], "y": H - float(p[1])})
                g["gt"] = gl
        out["comp_pins"] = comp_pins

        # ---- 候选池：真实重建（纯 CPU）----
        if B["want_pool"]:
            try:
                bgr = CL.load_image(B["s1"])
                pool = CL.from_stage2(cid, B["s1"], asg, bgr=bgr, vlm=B["vlm"])
                PR.Priors().annotate(pool)
                sol = SV.solve(pool)
                assign = (sol or {}).get("assign") or {}
                for sid in pool.slot_ids():
                    s = pool.hy.slots.get(sid)
                    if s is None:
                        continue
                    reg = pool.region(sid)
                    cands = [{"v": str(x.value), "s": round(float(x.score), 3),
                              "src": str(x.source), "ev": (x.evidence or {})}
                             for x in s.alts]
                    cid_of = sid.split(".")[1] if len(sid.split(".")) >= 3 else None
                    gt_key = cid2gt.get(cid_of) if cid_of else None
                    raw = assign.get(sid)
                    a_val = a_sc = a_src = None
                    if raw is not None:
                        a_val = getattr(raw, "value", raw[0] if isinstance(raw, (tuple, list)) else raw)
                        a_sc = getattr(raw, "score", None)
                        a_src = getattr(raw, "source", None)
                        if a_sc is None and isinstance(raw, (tuple, list)) and len(raw) >= 3:
                            a_val, a_sc, a_src = raw[0], raw[1], raw[2]
                    out["slots"].append({
                        "sid": sid, "cid": cid_of,
                        "key": cid2key.get(cid_of) if cid_of else None,
                        "gt": gt_key,
                        "chosen": str(s.chosen) if s.chosen is not None else None,
                        "by": s.decided_by,
                        "resolved": bool(s.resolved()) if hasattr(s, "resolved") else None,
                        "assign": (str(a_val) if a_val is not None else None),
                        "assign_score": (round(float(a_sc), 3) if isinstance(a_sc, (int, float)) else None),
                        "assign_src": (str(a_src) if a_src else None),
                        "cands": cands,
                    })
                    if reg is not None:
                        bb = getattr(reg, "box", None)
                        fo = getattr(reg, "focus", None)
                        if bb or fo:
                            out["regions"].append({"sid": sid, "box": list(bb) if bb else None,
                                                   "focus": list(fo) if fo else None,
                                                   "note": getattr(reg, "note", "")})
                out["ops"].append({"stage": "fusion", "name": "候选池重建",
                                   "n": len(out["slots"]),
                                   "note": "slots=%d  已解=%d  未解=%d"
                                           % (len(out["slots"]), len(assign),
                                              len(out["slots"]) - len(assign))})
            except Exception as exc:                               # noqa: BLE001
                import traceback
                traceback.print_exc()
                out["ops"].append({"stage": "fusion", "name": "候选池重建失败",
                                   "n": 0, "note": repr(exc)[:200]})

        # ---- 阶段统计 ----
        st = B["s1"].get("stats") or {}
        out["ops"].insert(0, {"stage": "stage1", "name": "检测 + 文字框 + 导线 + 引脚",
                              "n": len(B["s1"].get("components") or []),
                              "note": "元件 %d｜文字框 %d｜导线 %s｜bigic %s｜%.1fs"
                                      % (len(B["s1"].get("components") or []),
                                         len(B["s1"].get("texts") or []),
                                         st.get("n_wires", "?"), st.get("n_bigic", "?"),
                                         float(st.get("total_s") or 0))})
        if B["asg"]:
            nz = sum(1 for v in (B["asg"].get("desig_slots") or {}).values() if v)
            out["ops"].insert(1, {"stage": "stage2", "name": "文字归属 + 识别 + 装配",
                                  "n": nz,
                                  "note": "元件 %d｜有文字候选的元件 %d｜文字 %s(认领 %s/自由 %s)｜网络标签 %d"
                                          % (len(B["asg"].get("comps") or {}), nz,
                                             B["asg"].get("n_texts"), B["asg"].get("n_owned"),
                                             B["asg"].get("n_free"),
                                             len(B["asg"].get("net_labels") or []))})
        nd = 0
        for c in (B["asg"].get("comps") or {}).values():
            for p in (c.get("pins") or {}).values():
                if (p or {}).get("number") or (p or {}).get("name"):
                    nd += 1
        out["ops"].insert(2, {"stage": "stage2", "name": "引脚号 / 引脚名读出",
                              "n": nd, "note": "pinindex 元件 %d｜引脚 %d"
                              % (len(B["pinidx"] or {}),
                                 sum(len(v) for v in (B["pinidx"] or {}).values()))})

        # ---- 渲染 ----
        src_png = Path(B["case"].png_path)
        im0 = Image.open(src_png).convert("RGB")
        if scale < 1.0:
            im0 = im0.resize((ow, oh), Image.LANCZOS)
        d = out_root / "assets" / cid
        d.mkdir(parents=True, exist_ok=True)
        im0.save(d / "base.png")
        layers = draw_layers(B)
        if scale < 1.0:
            layers = {k: v.resize((ow, oh), Image.LANCZOS) for k, v in layers.items()}
        for k, v in layers.items():
            v.save(d / (k + ".png"))
        out["scale"] = scale
        # ⚠ 页面里的框位置一律按「原图像素」算百分比（图层 PNG 会被拉伸到容器宽度，
        #   盒子按原图像素 → 百分比 = x/W，缩放后依然对齐）。
        #   所以这里给的是**原图** W/H，不是 PNG 的像素尺寸（后者另记 png_w/png_h）。
        out["ow"], out["oh"] = W, H
        out["png_w"], out["png_h"] = ow, oh
        out["assets"] = "assets/%s" % cid
        (out_root / "cases" / ("%s.json" % cid)).write_text(
            json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
        print("   → %s（分 %.2f，槽位 %d，取景框 %d）"
              % (d, float(out["score"].get("score") or 0), len(out["slots"]), len(out["regions"])))
        ok += 1
    print("\n完成 %d 张，产物在 %s" % (ok, out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
