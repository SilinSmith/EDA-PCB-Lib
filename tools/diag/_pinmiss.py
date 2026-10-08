# -*- coding: utf-8 -*-
"""把「GT 有、我方没有」的 IC 引脚按**成因**分桶（取代 2026-10 之前的占位版）。

背景（`_libopp.py` 量出来的）：官方 9 张的 120 个 GT IC 引脚里，我方
`ok 58`、`lib_fix 6`、`no_lib 1`、**`missing 55`（45.8%）**，而引脚**号**通道
`extra 0`（我们交出的号全是 GT 有的号）⇒ 这 55 个是「压根没发出来」的引脚。

本工具把每个 missing 引脚追到 stage1/stage2 的原始读数上，分三桶：
  A `name_read`   —— 这颗 IC 的原始读数里**读到了同一个引脚名**（含编辑距离 1）
                     ⇒ 名读对了，是**绑定/闸门**把它丢了（最可修）
  B `name_other`  —— 引脚点附近 R 像素内**有**名字读数，但内容不是 GT 的名字
                     ⇒ 识别错 or 绑错脚
  C `no_name`     —— 附近没有任何名字读数 ⇒ **检测/召回**没拿到（要动检测）
另附几何旁证：`attach_near`（stage1 在该点附近有附着点）与最近名字读数框中心的距离。

用法（从仓库根跑）：
    python -X utf8 work\\tools\\diag\\_pinmiss.py \\
        --pred work\\_ckptab\\base\\runs\\stage2 \\
        --stage1 work\\_fast_out_reg9\\runs\\stage1 \\
        --cases 1,2,3,4,5,7,8,9,10 --radius 30
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, assign_components, canon_type, rewrite_keys)
from schematic_parser.data import IC_TYPES, pin_key_to_number  # noqa: E402

GT9 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"


def _ctr(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())


def _dist1(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="文档级产物目录（含 <case>.json）")
    ap.add_argument("--stage1", required=True, help="含 <case>.stage1.json 的目录")
    ap.add_argument("--assigned", default=None,
                    help="含 _assigned/<case>.json 的目录（默认 <pred>/_assigned）")
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--gt", default=str(GT9))
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    predroot = Path(a.pred)
    s1root = Path(a.stage1)
    asgroot = Path(a.assigned) if a.assigned else (predroot / "_assigned")
    gtroot = Path(a.gt)
    cfg = EvalConfig()

    bucket = Counter()
    Esub = Counter()
    A1sub = Counter()
    cross = Counter()
    rows = []

    for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
        gtf = gtroot / c / ("%s.json" % c)
        pf = predroot / ("%s.json" % c)
        sf = s1root / ("%s.stage1.json" % c)
        af = asgroot / ("%s.json" % c)
        if not (gtf.exists() and pf.exists() and sf.exists() and af.exists()):
            print("  ! 跳过 case %s（缺文件）" % c)
            continue
        gt = json.loads(gtf.read_text(encoding="utf-8"))
        pred = json.loads(pf.read_text(encoding="utf-8"))
        s1 = json.loads(sf.read_text(encoding="utf-8"))
        asg = json.loads(af.read_text(encoding="utf-8"))
        H = float(s1["image"]["h"])

        _, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)

        s1comps = []
        for comp in s1.get("components") or []:
            b = comp.get("bbox_json") or []
            if len(b) == 4:
                s1comps.append((comp.get("id"), _ctr(b)))
        attach_by_id = {x.get("id"): x for x in (s1.get("attachments") or [])}
        s1_pins = s1.get("pins") or {}

        n_miss = n_a = n_b = n_c = n_att = 0
        n_dn = n_dw = n_e = n_near = 0
        n_a1 = n_a2 = n_a3 = n_a4 = 0
        for gk, grec in (gt.get("components") or {}).items():
            if canon_type(grec.get("type")) not in IC_TYPES:
                continue
            if gk not in (pred2.get("components") or {}):
                continue
            gb = grec.get("bbox") or []
            if len(gb) != 4 or not s1comps:
                continue
            gc = _ctr(gb)
            cid, _pos_cid = min(s1comps, key=lambda t: _dist(t[1], gc))
            d_cid = _dist(_pos_cid, gc)   # ← 以前这里直接解包成位置元组，verbose 会炸
            rec = (asg.get("comps") or {}).get(cid) or {}
            raw_pins = rec.get("pins") or {}
            names = []
            for _idx, pr in raw_pins.items():
                nm = pr.get("name")
                if nm and nm.get("text"):
                    box = nm.get("box") or []
                    names.append((nm["text"], _ctr(box) if len(box) == 4 else None))
            hits = (attach_by_id.get(cid) or {}).get("hits") or []

            for pk, pv in (gt.get("pins", {}).get(gk) or {}).items():
                if gk in (pred2.get("pins") or {}) and pk in pred2["pins"][gk]:
                    continue
                n_miss += 1
                gname = (pv or {}).get("pinname") or ""
                pt = (pv or {}).get("point") or {}
                if "x" not in pt or "y" not in pt:
                    continue
                p_img = (float(pt["x"]), H - float(pt["y"]))
                d_att = min([_dist((float(h["at"][0]), float(h["at"][1])), p_img)
                             for h in hits if h.get("at")], default=None)
                has_att = d_att is not None and d_att <= a.radius
                n_att += 1 if has_att else 0
                same = [t for t, _c in names if _norm(gname) and _norm(t) == _norm(gname)]
                near1 = [t for t, _c in names
                         if _norm(gname) and _dist1(_norm(t), _norm(gname))]
                cand = sorted([(t, _dist(cc, p_img)) for t, cc in names if cc is not None],
                              key=lambda z: z[1])
                nearest = cand[0] if cand else None
                # ★ 我方在这一位置上到底发没发引脚？（发了但键/名不同 ⇒ 是**号**读错，不是没召回）
                mine = []
                for pk2, pv2 in ((pred2.get("pins") or {}).get(gk) or {}).items():
                    pp = (pv2 or {}).get("point") or {}
                    if "x" not in pp or "y" not in pp:
                        continue
                    q = (float(pp["x"]), H - float(pp["y"]))
                    mine.append((pk2, (pv2 or {}).get("pinname") or "", _dist(q, p_img)))
                mine.sort(key=lambda z: z[2])
                m0 = mine[0] if mine else None
                if m0 and m0[2] <= a.radius:
                    if _norm(m0[1]) == _norm(gname) and _norm(gname):
                        tag = "D 发了同位同名(键/号不同)"
                        if same:
                            n_dn += 1
                        else:
                            n_dw += 1
                    else:
                        tag = "E 发了同位但名不同"
                        n_e += 1
                        # 名字是同一颗 IC 的**别的脚**的吗？（= 名/脚绑错，off-by-one）
                        others = []
                        for pk3, pv3 in (gt.get("pins", {}).get(gk) or {}).items():
                            if pk3 == pk:
                                continue
                            if _norm((pv3 or {}).get("pinname")) != _norm(m0[1]):
                                continue
                            q3 = (pv3 or {}).get("point") or {}
                            if "x" not in q3 or "y" not in q3:
                                continue
                            others.append(_dist((float(q3["x"]), H - float(q3["y"])), p_img))
                        if others and min(others) <= a.radius:
                            Esub["swap"] += 1
                        else:
                            Esub["other"] += 1
                        _gn = str(pin_key_to_number(pk) or "").strip()
                        _mn = str(pin_key_to_number(m0[0]) or "").strip()
                        if _gn and _mn == _gn:
                            Esub["键=正解"] += 1
                        else:
                            Esub["键也错"] += 1
                        if a.verbose:
                            print("        · E GT键=%s(号%r) 我方键=%s(号%r) 名=%r"
                                  % (pk, _gn, m0[0], _mn, m0[1]))
                    n_near += 1
                elif same or near1:
                    # A：名读到了却没发 —— 再细分「读数里到底有没有号 / 有没有桩」
                    hit_idx = [(i, pr) for i, pr in raw_pins.items()
                               if _norm((pr.get("name") or {}).get("text"))
                               and _norm((pr.get("name") or {}).get("text")) == _norm(gname)]
                    with_num = [1 for _i, pr in hit_idx
                                if (pr.get("number") or {}).get("text")]
                    numtxt = " / ".join((pr.get("number") or {}).get("text", "")
                                        for _i, pr in hit_idx if (pr.get("number") or {}).get("text"))
                    stubs = s1_pins.get(cid) or []
                    d_stub = min([_dist((float(q[0]), float(q[1])), p_img)
                                  for q in stubs if len(q) >= 2], default=None)
                    has_stub = d_stub is not None and d_stub <= a.radius
                    if with_num and has_stub:
                        tag = "A1 名+号都读到、桩也在(逻辑丢)"
                        n_a1 += 1
                        _gn = str(pin_key_to_number(pk) or "").strip()
                        _toks = re.findall(r"\d+", numtxt)
                        if _gn and _gn in _toks:
                            A1sub["号里含正解"] += 1
                        else:
                            A1sub["号里没有正解"] += 1
                        if a.verbose:
                            print("        · A1 GT号=%r 读数=%r tokens=%s"
                                  % (_gn, numtxt, _toks))
                    elif with_num:
                        tag = "A2 名+号都读到，但该处无桩"
                        n_a2 += 1
                    elif has_stub:
                        tag = "A3 名读到、桩在、号没读到"
                        n_a3 += 1
                    else:
                        tag = "A4 名读到但桩也没有"
                        n_a4 += 1
                    if same:
                        n_a += 1
                    else:
                        n_a += 1
                    if a.verbose:
                        print("      → A 细分 %s idx=%s num=%r 桩距离=%s"
                              % (tag, [i for i, _ in hit_idx], numtxt,
                                 ("%.1f" % d_stub) if d_stub is not None else "-"))
                elif nearest and nearest[1] <= a.radius:
                    tag = "B 附近名读到别的"
                    n_b += 1
                else:
                    tag = "C 附近无名"
                    n_c += 1
                bucket[tag] += 1
                cross["%s|%s" % (tag, "attach" if has_att else "no-attach")] += 1
                if a.verbose:
                    print("    %s %s.%s comp=%s(d=%.1f) %-14r att=%-6s near=%-8s %r | mine=%s %-10r d=%s" % (
                        c, gk, pk, cid, d_cid, gname,
                        ("%.1f" % d_att) if has_att else "-",
                        ("%.1f" % nearest[1]) if nearest else "-",
                        nearest[0] if nearest else "",
                        m0[0] if m0 else "-", m0[1] if m0 else "",
                        ("%.1f" % m0[2]) if m0 else "-"))
        rows.append((c, n_miss, n_a, n_b, n_c, n_att))
        print("  case %-3s 缺 %2d | 名读到却没发 %2d [A1 名+号+桩都有 %2d / A2 名+号有但无桩 %2d / "
              "A3 名+桩但无号 %2d / A4 只有名 %2d] | 同位名不同 %2d | 同位同名号不同 %2d | 无名 %2d"
              % (c, n_miss, n_a, n_a1, n_a2, n_a3, n_a4, n_e, n_dn, n_c))

    tot = sum(bucket.values())
    print("\n===== 汇总（%d 个 missing 引脚）=====" % tot)
    for k, v in bucket.most_common():
        print("  %-20s %4d  %5.1f%%" % (k, v, 100.0 * v / tot if tot else 0.0))
    print("----- 与附着点交叉 -----")
    for k, v in sorted(cross.items()):
        print("  %-34s %4d" % (k, v))
    print("----- E 细分（我方在该位置发的名，是不是同一颗 IC 别的脚的名字）-----")
    print("  E-swap 绑到隔壁脚 %4d | E-other 其他 %4d" % (Esub["swap"], Esub["other"]))
    print("  E 的键：我方号=GT号 %4d | 号也错 %4d" % (Esub["键=正解"], Esub["键也错"]))
    print("  A1 读到的号文本里：含 GT 号 %4d | 不含 %4d"
          % (A1sub["号里含正解"], A1sub["号里没有正解"]))
    print("\n提示：A = 名读到了却没发（闸门/无附着点）；B = 附近读到的是别的名；"
          "C = 附近没有名字读数（检测没拿到）；\n      D = **我们在同一位置已经发了引脚、名字也对，只是键（引脚号）不同** "
          "⇒ 是「号」的问题；E = 同位已发但名字不同。")


if __name__ == "__main__":
    main()
