"""引脚窗口 A/B：`pin_pitch_frac=0`（旧，垂直半宽 = cross）vs `0.45`（新，按引脚间距收窄）。

只跑**指定元件的引脚网格**（每元件 1~3 次 VLM 调用），不跑整条流水线。
判据：把模型报回的 (编号, 名称) 与 GT 引脚按最近点配对，统计
「名称完全对」「编号完全对」「都対」以及逐条错例。

用法：
    python work\\tools\\diag\\_pinab.py [case:KEY ...]
默认 `8:U1`（10GT 里引脚 name_mismatch 最重的一张）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D                                       # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components           # noqa: E402
from stage2.text import imread_unicode                                       # noqa: E402
from vlm import enrich as E                                                  # noqa: E402
from vlm import client                                                       # noqa: E402

PRED_DIR = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage2")

jobs = []
for a in sys.argv[1:]:
    if a.startswith("--"):
        continue
    if ":" in a:
        c, k = a.split(":", 1)
        jobs.append((c, k))
    else:
        jobs.append((a, None))
if not jobs:
    jobs = [("8", "U1")]

cfg = EvalConfig()
base = dict(E.DEFAULTS)
base.update(designator=False, desig_all=False, desig_fallback=False,
            value=False, value_batch=1, name=False, preflight=False)

ARMS = [("OLD   frac=0    edge=0", 0.0, False),
        ("EDGE  frac=0    edge=1", 0.0, True),
        ("HALF  frac=0.45 edge=0", 0.45, False),
        ("NEW   frac=0.45 edge=1", 0.45, True)]
grand = {t: [0, 0, 0, 0] for t, _, _ in ARMS}        # 名对 / 号对 / 都对 / 总

client.preflight()

for cid, key in jobs:
    case = D.find_case(cid)
    gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
    doc = json.loads((PRED_DIR / f"{case.case_id}.json").read_text(encoding="utf-8"))
    bgr = imread_unicode(case.png_path)
    _f1, km = assign_components(gt, doc, cfg, {})
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}
    keys = [key] if key else [k for k in pins if k in km and len(pins[k] or {}) >= 3]
    for k in keys:
        gkey = km.get(k)
        if gkey is None:
            print(f"[{cid}] {k}: 没配上 GT 键，跳过"); continue
        H = bgr.shape[0]
        bb = comps[k]["bbox"]                                   # JSON 空间（y 向上）
        bb_img = (bb[0], H - bb[3], bb[2], H - bb[1])           # → 图像空间
        pts = [(v["point"]["x"], H - v["point"]["y"])           # 图像空间
               for v in (pins[k] or {}).values()]
        ctr = ((bb_img[0] + bb_img[2]) / 2.0, (bb_img[1] + bb_img[3]) / 2.0)
        win = E.adaptive_window(bb_img, base["inner"], base["cross"], base["outer"], base["ref"])
        gp = (gt.get("pins") or {}).get(gkey) or {}
        gpts = [(kk, vv.get("pinname", ""), vv["point"]["x"], H - vv["point"]["y"])
                for kk, vv in gp.items()]
        pitch = E._pin_pitch(pts)
        print(f"\n=== case {cid} · {k} → GT {gkey} | n_pred={len(pts)} n_gt={len(gpts)} "
              f"| win(outer,inner,cross)=({win[0]:.1f},{win[1]:.1f},{win[2]:.1f}) "
              f"pitch={pitch:.1f}")
        print("   GT 引脚:", ", ".join(f"{n}={v}" for n, v, *_ in gpts))
        for tag, frac, edge in ARMS:
            o = dict(base); o["pin_pitch_frac"] = frac; o["pin_edge_dir"] = edge
            stat: dict = dict(desig=0, grid=0, value=0, value_set=0)
            ans = E.read_pins(bgr, pts, ctr, win, o, stat, bbox=bb_img)
            okn = oknum = okboth = 0
            detail = []
            for i, q in enumerate(pts):
                got = ans.get(i)
                if got is None:
                    detail.append(f"#{i}=<空>"); continue
                num, name = got
                # GT 里离这个点最近的那条
                j = min(range(len(gpts)), key=lambda t: (gpts[t][2]-q[0])**2 + (gpts[t][3]-q[1])**2)
                gname = gpts[j][1]
                gnum = gpts[j][0].replace("pin_", "")
                n_ok = (str(name).strip().upper() == str(gname).strip().upper())
                q_ok = (str(num).strip() == str(gnum).strip())
                okn += n_ok; oknum += q_ok; okboth += (n_ok and q_ok)
                detail.append(f"#{i} got({num}:{name}) gt({gnum}:{gname}){'' if n_ok else ' ✗名'}")
            n = len(pts)
            g = grand[tag]
            g[0] += okn; g[1] += oknum; g[2] += okboth; g[3] += n
            print(f"  [{tag}] 名对 {okn}/{n}  号对 {oknum}/{n}  都对 {okboth}/{n}"
                  f"  cross_perp={stat.get('cross_perp')}")
            if frac == 0.0 and not edge or True:
                print("      " + " | ".join(detail))

print("\n" + "=" * 70)
for tag, _ in ARMS:
    n, b, o = grand[tag][0], grand[tag][2], grand[tag][3]
    print(f"{tag}: 名称对 {n}/{o} = {n/max(1,o):.3f}   都对 {b}/{o} = {b/max(1,o):.3f}")
