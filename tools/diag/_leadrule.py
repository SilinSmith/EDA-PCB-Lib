# -*- coding: utf-8 -*-
"""**远场引线规则**（`STAGE1_PIN_LEAD` 的设计验证，只读、不跑流水线）。

背景：两脚元件的真引线常常是**导线端点**，而它离框边 9~10px（> `TOUCH_TOL=6`）⇒ 现在
判不了「接触」，候选就落到贴在框边的**文字笔画**（"R1"/"10K"，我们这边的导线提取器把它
当成了 wire）上，轴都是错的（case 7 的 R1 就是标本，见 `_leadmiss.py`）。

候选规则（只对**非 IC** 元件、只沿**主轴**）：
对框的两条主轴边各找一个「引线端点」= 段端点满足
  · 沿轴方向落在框外 `[0, reach]` 内；
  · 侧向（垂直轴）离框中心 ≤ `lat` px；
取**离框边最近**的那个；把候选点放在该端点位置。

本探针在官方 10 张上量：GT 两脚元件引脚里有多少能被这条规则**命中**（≤6px）、
以及规则会不会**凭空造出**候选（离任何 GT 引脚都 >6px = 假候选）。

    python -X utf8 eda-pcb/tools/diag/_leadrule.py --run work/_fast_out_fin --split 10GT
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover                     # noqa: E402

TWO_PIN = frozenset({"r", "c", "l", "d", "led", "fuse", "crystal_2pin", "crystal"})
REF_RE = re.compile(r"([A-Za-z]+\d+)\.(\d+)")


def gtnet_of(gt):
    """GT 网成员索引：(位号, 脚号) -> net_k。"""
    out = {}
    for k, v in (gt.get("nets") or {}).items():
        hg = (v or {}).get("hyperGraph") or ""
        for m in REF_RE.finditer(hg):
            out[(m.group(1), m.group(2))] = k
    return out


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def lead_cands(bb, segs, reach=18.0, lat=4.0):
    """主轴两侧的引线端点候选（图像坐标）。"""
    x0, y0, x1, y1 = [float(v) for v in bb]
    wide = (x1 - x0) >= (y1 - y0)
    out = []
    for si, sg in enumerate(segs):
        for ex, ey in ((sg[0][0], sg[0][1]), (sg[1][0], sg[1][1])):
            if wide:
                d = max(x0 - ex, ex - x1)          # 沿轴：框外为正
                if d <= 0 or d > reach:
                    continue
                if abs(ey - (y0 + y1) / 2.0) > lat:
                    continue
            else:
                d = max(y0 - ey, ey - y1)
                if d <= 0 or d > reach:
                    continue
                if abs(ex - (x0 + x1) / 2.0) > lat:
                    continue
            out.append((float(ex), float(ey), float(d), si))
    # 每侧只留离框边最近的那个
    best = []
    for side in (0, 1):
        if wide:
            pool = [c for c in out if (c[0] > x1) == bool(side)]
        else:
            pool = [c for c in out if (c[1] > y1) == bool(side)]
        if pool:
            pool.sort(key=lambda c: c[2])
            best.append(pool[0])
    return best


def net_of_point(pt, segs, seg2net, join=12.0):
    """候选点挂到最近**线段**（点到线段距离，近似 nets.py 沿段采样的 PIN_JOIN_DIST）所属的 stage1 网。"""
    bd, bsi = 1e9, None
    for si, sg in enumerate(segs):
        x1, y1 = float(sg[0][0]), float(sg[0][1])
        x2, y2 = float(sg[1][0]), float(sg[1][1])
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / L2))
        d = math.hypot(pt[0] - (x1 + t * dx), pt[1] - (y1 + t * dy))
        if d < bd:
            bd, bsi = d, si
    if bd > join or bsi is None:
        return None, bsi, bd
    return seg2net.get(bsi), bsi, bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_fin")
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--reach", type=float, default=18.0)
    ap.add_argument("--lat", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=6.0)
    ap.add_argument("--bynet", action="store_true", help="按「候选所在段 → stage1 网 → 是否含 GT 真伙伴」判定")
    ap.add_argument("--mode", default="", choices=["", "current", "ideal", "lead"],
                    help="current=stage1 现有取点 / ideal=主轴两侧 10px 理想点 / lead=远场引线端点")
    ap.add_argument("--od", type=float, default=10.0, help="ideal 模式的主轴外伸（PIN_OUT）")
    ap.add_argument("--join", type=float, default=12.0, help="候选挂到最近段端点的最大距离（PIN_JOIN_DIST）")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = (ROOT / run).resolve()
    cases = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split]
    tot = collections.Counter()
    for c in cases[:a.n]:
        st_p = run / "runs" / "stage1" / ("%s.stage1.json" % c.case_id)
        if not st_p.exists():
            continue
        st = load(st_p)
        H = float((st.get("image") or {}).get("h") or 0.0)
        gt = load(c.json_path)
        segs = ((st.get("wire_detail") or {}).get("segments") or [])
        gtc = gt.get("components") or {}
        stc = {x["id"]: x for x in (st.get("components") or [])}
        # 按网判定所需的索引
        g2n = gtnet_of(gt)
        p2n = {}
        pname = {}
        seg2net = {}
        netmember = {}
        if a.bynet:
            for si, net in enumerate(st.get("nets") or []):
                netmember[net.get("id")] = [(cid, i) for cid, i in (net.get("pins") or [])]
                for j in (net.get("seg_index") or []):
                    seg2net[j] = net.get("id")
            pp = run / "final_results" / ("%s_Pred.json" % c.case_id)
            if pp.exists():
                pred = load(pp)
                idx = pred.get("__pin_index") or {}
                # cid -> 位号：从 stage2 的 _assigned 里取（final 的 components 是按位号索引的）
                ap = run / "runs" / "stage2" / "_assigned" / ("%s.json" % c.case_id)
                if ap.exists():
                    asc = (load(ap).get("comps") or {})
                    for cid, v in asc.items():
                        ds = (v or {}).get("designator") or {}
                        nm = ds.get("canon") or ds.get("text")
                        if nm:
                            pname[cid] = nm
                for cid, v in (pred.get("components") or {}).items():
                    pname.setdefault(cid, (v or {}).get("Name") or cid)
                for cid, mp in idx.items():
                    for k, pk in (mp or {}).items():
                        if pk:
                            p2n[(cid, int(k))] = (pname.get(cid, cid), str(pk).split("_")[-1])
        stat = collections.Counter()
        for name, g in gtc.items():
            if str(g.get("type") or "").lower() not in TWO_PIN:
                continue
            bb = g.get("bbox") or []
            if len(bb) != 4:
                continue
            gx, gy = (bb[0] + bb[2]) / 2.0, H - (bb[1] + bb[3]) / 2.0
            best = None
            for cid, s in stc.items():
                sb = s.get("bbox_img") or []
                if len(sb) != 4:
                    continue
                d = math.hypot((sb[0] + sb[2]) / 2.0 - gx, (sb[1] + sb[3]) / 2.0 - gy)
                if d < 40.0 and (best is None or d < best[0]):
                    best = (d, cid, s)
            if not best:
                continue
            s = best[2]
            mc = best[1]
            if a.mode in ("ideal", "combo"):
                sb = [float(v) for v in (s.get("bbox_img") or [0, 0, 0, 0])]
                w = (sb[2] - sb[0]) >= (sb[3] - sb[1])
                if w:
                    ide = [(sb[0] - a.od, (sb[1] + sb[3]) / 2.0),
                           (sb[2] + a.od, (sb[1] + sb[3]) / 2.0)]
                else:
                    ide = [((sb[0] + sb[2]) / 2.0, sb[1] - a.od),
                           ((sb[0] + sb[2]) / 2.0, sb[3] + a.od)]
                cands = [(x, y, a.od, None) for x, y in ide]
                if a.mode == "combo":
                    leads = lead_cands(s.get("bbox_img"), segs, a.reach, a.lat)
                    out2 = []
                    for sd, (x, y) in enumerate(ide):
                        if net_of_point((x, y), segs, seg2net, a.join)[0] is not None:
                            out2.append((x, y, a.od, None))
                            continue
                        lp = [cd for cd in leads if (1 if ((cd[0] > sb[2]) if w else (cd[1] > sb[3])) else 0) == sd]
                        out2.append(lp[0] if lp else (x, y, a.od, None))
                    cands = out2
            elif a.mode == "current":
                cands = [(float(p[0]), float(p[1]), 0.0, None)
                         for p in ((st.get("pins") or {}).get(mc) or [])]
            else:
                cands = lead_cands(s.get("bbox_img"), segs, a.reach, a.lat)
            gp = (gt.get("pins") or {}).get(name) or {}
            gpts = []
            for pk, pv in gp.items():
                pt = (pv or {}).get("point") or {}
                if "x" in pt:
                    gpts.append((pt["x"], H - pt["y"]))
            stat["pins"] += len(gpts)
            for gp_ in gpts:
                hit = any(math.hypot(cd[0] - gp_[0], cd[1] - gp_[1]) <= a.tol for cd in cands)
                if hit:
                    stat["hit"] += 1
                else:
                    stat["miss"] += 1
            for cd in cands:
                near = any(math.hypot(cd[0] - gp_[0], cd[1] - gp_[1]) <= a.tol for gp_ in gpts)
                if near:
                    stat["cand_true"] += 1
                else:
                    stat["cand_spurious"] += 1
                    if a.verbose:
                        print("     假候选 %s(%s) (%.1f,%.1f) d=%.1f GT脚 %s"
                              % (name, g.get("type"), cd[0], cd[1], cd[2],
                                 [(round(p[0]), round(p[1])) for p in gpts]))
                if not a.bynet:
                    continue
                # 按网判：候选落在哪个 stage1 网里、该网里有没有这个 GT 脚的真伙伴
                sb2 = s.get("bbox_img") or [0, 0, 0, 0]
                wide = (sb2[2] - sb2[0]) >= (sb2[3] - sb2[1])
                side = 1 if ((cd[0] > sb2[2]) if wide else (cd[1] > sb2[3])) else 0
                if not gpts:
                    stat["bynet_nogt"] += 1
                    continue
                gp_c = max(gpts, key=lambda p: p[0]) if wide else max(gpts, key=lambda p: p[1])
                gp_c = gp_c if side == 1 else (
                    min(gpts, key=lambda p: p[0]) if wide else min(gpts, key=lambda p: p[1]))
                gkey = None
                for pk, pv in gp.items():
                    pt = (pv or {}).get("point") or {}
                    if "x" not in pt:
                        continue
                    if abs(pt["x"] - gp_c[0]) < 1e-6 and abs(H - pt["y"] - gp_c[1]) < 1e-6:
                        gkey = (name, str(pk).split("_")[-1])
                if not gkey or gkey not in g2n:
                    stat["bynet_nogt"] += 1
                    continue
                gnet = g2n[gkey]
                partners = {r for r, n in g2n.items() if n == gnet and r != gkey}
                nid = seg2net.get(cd[3]) if cd[3] is not None else \
                    net_of_point((cd[0], cd[1]), segs, seg2net, a.join)[0]
                refs = set()
                for (cid2, i2) in netmember.get(nid, []):
                    r = p2n.get((cid2, i2))
                    if r:
                        refs.add(r)
                if not nid:
                    stat["bynet_noseg"] += 1
                elif refs & partners:
                    stat["bynet_ok"] += 1
                    if a.verbose:
                        print("     ✓引线 %s.%s seg%d -> %s 伙伴 %s"
                              % (name, gkey[1], cd[3], nid, sorted(refs & partners)[:3]))
                else:
                    stat["bynet_bad"] += 1
                    if a.verbose:
                        print("     ✗引线 %s.%s seg%d -> %s refs %s 期望伙伴 %s"
                              % (name, gkey[1], cd[3], nid, sorted(refs)[:4], sorted(partners)[:4]))
        print("  case %-6s GT两脚引脚 %3d ｜ 规则命中 %3d / 未中 %3d ｜ 候选 %3d（真 %3d / **假 %3d**）"
              % (c.case_id, stat["pins"], stat["hit"], stat["miss"],
                 stat["cand_true"] + stat["cand_spurious"], stat["cand_true"],
                 stat["cand_spurious"]))
        if a.bynet:
            print("         └ 按网判：候选 %d ｜ 落到含真伙伴的网 **OK %d** ｜ 落错网 %d ｜ 无段 %d ｜ 无 GT 脚 %d"
                  % (stat["bynet_ok"] + stat["bynet_bad"] + stat["bynet_noseg"] + stat["bynet_nogt"],
                     stat["bynet_ok"], stat["bynet_bad"], stat["bynet_noseg"], stat["bynet_nogt"]))
        tot.update(stat)
    print("\n== 汇总（reach %.0f / lat %.0f / tol %.0f）==" % (a.reach, a.lat, a.tol))
    print("  GT 两脚引脚 %d ｜ 命中 %d（%.1f%%）｜ 未中 %d ｜ 候选 %d（真 %d / 假 %d）"
          % (tot["pins"], tot["hit"], 100.0 * tot["hit"] / max(1, tot["pins"]),
             tot["miss"], tot["cand_true"] + tot["cand_spurious"],
             tot["cand_true"], tot["cand_spurious"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
