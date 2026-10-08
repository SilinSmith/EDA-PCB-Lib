# -*- coding: utf-8 -*-
"""**文字识别到底抽不抽风**：把 GT IC 引脚的号/名，与「位置上最近的那个我方槽位」逐条对账。

和 `_pinnamepair.py` 的区别：那份**按号查槽位**（号读错了就查不到，混进"没有槽位"），
这份**按位置查槽位**（先绑定几何，再看读出来的字符串对不对），才能把
「读错了」和「根本没这个槽位」分开。

    python work/tools/diag/_ocrstat.py --split 10GT --run work/_full220
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser import metrics as M                                  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402

NEAR = 45.0


def cls(ours: str, gt: str) -> str:
    if not ours:
        return "空"
    if ours == gt:
        return "完全对"
    if ours in gt:
        return "**截断**（真值含我方）"
    if gt in ours:
        return "多读了字符"
    return "整串读错"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "work" / "_full220"))
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)

    st = collections.Counter()
    numk = collections.Counter()
    namek = collections.Counter()
    widths = collections.defaultdict(list)
    rows = []
    for c in [x for x in discover(DEFAULT_DATA_ROOT) if x.split == a.split][:a.n]:
        f = run / "runs" / "stage2" / "_assigned" / (c.case_id + ".json")
        s1f = run / "runs" / "stage1" / (c.case_id + ".stage1.json")
        if not f.exists():
            print("!! 缺 %s" % f)
            continue
        asg = load_json(f)
        H = float((load_json(s1f).get("image") or {}).get("h") or 0) if s1f.exists() else 0.0
        gt = M.normalise_doc(load_json(Path(c.json_path)))
        box = {k for k, v in (gt.get("components") or {}).items()
               if M.canon_type((v or {}).get("type")) in M.IC_TYPES}
        bydesig = collections.defaultdict(list)
        for cid, cc in (asg.get("comps") or {}).items():
            key = ((cc.get("designator") or {}).get("text") or "").strip()
            for s, rec in (cc.get("pins") or {}).items():
                bydesig[key].append((s, rec))
        for comp in box:
            sl = bydesig.get(comp) or []
            for pk, info in ((gt.get("pins") or {}).get(comp) or {}).items():
                gnum = str(M.pin_key_to_number(pk))
                gname = ((info or {}).get("pinname") or "")
                p = (info or {}).get("point") or {}
                if not p:
                    continue
                gx, gy = float(p["x"]), float(p["y"])
                st["gt"] += 1
                # 最近槽位：号框心 or 名框心，哪个近算哪个
                best = None
                for s, rec in sl:
                    for tag in ("number", "name"):
                        bb = (rec.get(tag) or {}).get("box")
                        if not bb or len(bb) < 4:
                            continue
                        cx = (float(bb[0]) + float(bb[2])) / 2.0
                        cy = float(bb[1]) + float(bb[3]) / 2.0
                        d = math.hypot(gx - cx, gy - (H - (float(bb[1]) + float(bb[3])) / 2.0))
                        if best is None or d < best[0]:
                            best = (d, tag, rec)
                if best is None or best[0] > NEAR:
                    numk["无近邻槽位"] += 1
                    namek["无近邻槽位"] += 1
                    st["no_slot"] += 1
                    continue
                d, tag, rec = best
                st["slot"] += 1
                otx = (((rec.get("number") or {}).get("text")) or "").strip()
                onm = (((rec.get("name") or {}).get("text")) or "").strip()
                numk[cls(otx, gnum)] += 1
                namek[cls(onm, gname)] += 1
                for tag2, o, g in (("number", otx, gnum), ("name", onm, gname)):
                    bb = (rec.get(tag2) or {}).get("box")
                    if bb and len(bb) >= 4 and o and o != g and o in g:
                        widths[len(g)].append((float(bb[2]) - float(bb[0]),
                                               float(bb[3]) - float(bb[1]), o, g))
                if a.show and (otx != gnum or onm != gname) and len(rows) < a.show:
                    rows.append((c.case_id, comp, gnum, gname, otx, onm, round(d, 1)))
    print("run=%s split=%s" % (run, a.split))
    print("GT IC 引脚 %d；**45px 内有我方槽位（几何绑定成立）** %d（%.0f%%）、没有 %d"
          % (st["gt"], st["slot"], 100.0 * st["slot"] / max(st["gt"], 1), st["no_slot"]))
    print()
    print("在这个「几何已经绑对」的子集里：")
    print("  号：", dict(numk.most_common()))
    print("  名：", dict(namek.most_common()))
    if widths:
        print()
        print("「截断」样本的**文字框像素宽度**（框高度 ~17px ⇒ 一个数字宽约 10~11px、步距约 12~13px）：")
        for n in sorted(widths):
            ws = [w[0] for w in widths[n]]
            sample = ", ".join("%s→%s(w=%.0f)" % (w[3], w[2], w[0]) for w in widths[n][:6])
            print("   %d 位真值 %d 条：框宽 %.0f~%.0f（期望 ≥ %d）｜%s"
                  % (n, len(ws), min(ws), max(ws), int(10 * n), sample))
    if rows:
        print()
        print("样例（case, 元件, GT号, GT名, 我号, 我名, 距离）：")
        for r in rows:
            print("   %-5s %-5s %-6s %-10s %-6s %-10s %.1f" % r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
