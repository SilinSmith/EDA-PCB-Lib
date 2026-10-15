# -*- coding: utf-8 -*-
"""**GT 的 `point` 是「框边接点」还是「引脚名/号那行文字的中心」？**

已知（会打架的两条旧结论）：
  * `_pinpos.py` 的注释说「GT 的引脚点贴在**符号框边上**」；
  * `_pinsnap.py`（n=353，T0..T5）实测：吸附到**我方框边**只剩 8/178、吸附到
    **真值框边**只剩 5、全局平移扫描最优就是恒等 ⇒ **真值点根本不在框边上**。
  * `_gtpinplace.py`：GT 引脚 `point` **100% 落在元件 bbox 之外**，中位 9px。
  * `_pinoracle.py --split 10GT`：名字对的有 59 根，其中 22 根距离 8.0/8.1/8.2/29.1/30.1，
    多个样本是**同一个方向的 8px 常量**（gt x=1334.0 vs pr x=1326.0）。

三条合起来指向同一个解释：**GT 的 `point` 是那根引脚的「名字/号」文字的框心**
（标签画在框外 ~9px 处），而我们的点是从框沿追出去的接点。

本脚本不改代码，只用现成产物量：对每根 GT IC 引脚，在 `_assigned` 里按**号文本**找到
对应槽位，比较 GT `point` 到 ① 号文字框心 ② 名文字框心 ③ 两者中点 的距离。

    python work/tools/diag/_pintext.py --split 10GT
    python work/tools/diag/_pintext.py --split 200_train_cases --n 60 --examples 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (IC_TYPES, canon_type, normalise_doc,  # noqa: E402
                                      pin_key_to_number)

TOL = 5.0


def _ctr(box):
    if not box or len(box) < 4:
        return None
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--run", default=str(ROOT / "work" / "_full220"))
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)

    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split][:a.n]
    st = Counter()
    ex = []
    for c in cs:
        q = run / "runs" / "stage2" / "_assigned" / (c.case_id + ".json")
        s1p = run / "runs" / "stage1" / (c.case_id + ".stage1.json")
        if not q.exists() or not s1p.exists():
            print("!! 缺产物 %s" % c.case_id)
            continue
        asg = load_json(q)
        H = float((load_json(s1p).get("image") or {}).get("h") or 0)
        gt = normalise_doc(load_json(Path(c.json_path)))
        box = {k for k, v in (gt.get("components") or {}).items()
               if canon_type((v or {}).get("type")) in IC_TYPES}
        # 我方：designator 文本 -> {号文本: 槽位}
        ours = {}
        for _cid, comp in (asg.get("comps") or {}).items():
            key = ((comp.get("designator") or {}).get("text") or "").strip()
            bucket = ours.setdefault(key, {})
            for _slot, rec in (comp.get("pins") or {}).items():
                num = (((rec or {}).get("number") or {}).get("text") or "").strip()
                if num:
                    bucket.setdefault(num, rec)
        for comp in box:
            gpin = (gt.get("pins") or {}).get(comp) or {}
            bucket = ours.get(comp) or {}
            for pk, info in gpin.items():
                pt = (info or {}).get("point") or {}
                if not pt:
                    continue
                gx, gy = float(pt.get("x", 0)), float(pt.get("y", 0))
                st["gt"] += 1
                rec = bucket.get(pin_key_to_number(pk))
                if rec is None:
                    st["no_slot"] += 1
                    continue
                st["slot"] += 1
                nb = (rec.get("number") or {}).get("box")
                mb = (rec.get("name") or {}).get("box")
                ds = {}
                for tag, b in (("num", nb), ("name", mb)):
                    ct = _ctr(b)
                    if ct is None:
                        continue
                    ds[tag] = math.hypot(gx - ct[0], gy - (H - ct[1]))
                    st["n_" + tag] += 1
                    if ds[tag] <= TOL:
                        st["ok_" + tag] += 1
                if nb and mb:
                    c1, c2 = _ctr(nb), _ctr(mb)
                    ds["mid"] = math.hypot(gx - (c1[0] + c2[0]) / 2.0,
                                           gy - (H - (c1[1] + c2[1]) / 2.0))
                    st["n_mid"] += 1
                    if ds["mid"] <= TOL:
                        st["ok_mid"] += 1
                if ds:
                    st["best_le5"] += int(min(ds.values()) <= TOL)
                    # 「号」和「名」哪个更近
                    if "num" in ds and "name" in ds:
                        st["closer_name" if ds["name"] < ds["num"] else "closer_num"] += 1
                if a.examples and len(ex) < a.examples and ds:
                    ex.append((c.case_id, comp, pk, (info or {}).get("pinname"),
                               (round(gx, 1), round(gy, 1)),
                               {k: round(v, 1) for k, v in ds.items()}))
    print("split=%s  run=%s" % (a.split, run))
    print("GT IC 引脚 **按号能找到槽位**：%d / %d" % (st["slot"], st["gt"]))
    print()
    print("| 候选输出点 | 有该文字框 | 距离 ≤5px | 命中率 |")
    print("|---|---|---|---|")
    for tag, label in (("num", "「号」文字框心"), ("name", "「名」文字框心"),
                       ("mid", "号心与名心的中点")):
        n, ok = st["n_" + tag], st["ok_" + tag]
        print("| %s | %d | %d | %s |" % (label, n, ok,
                                         ("%.1f%%" % (100.0 * ok / n)) if n else "—"))
    n, ok = st["slot"], st["best_le5"]
    print("| **三个候选里最近的那个** | %d | %d | %s |"
          % (n, ok, ("%.1f%%" % (100.0 * ok / n)) if n else "—"))
    if st["closer_name"] + st["closer_num"]:
        tot = st["closer_name"] + st["closer_num"]
        print()
        print("两者都有框时，更近的是：「名」 %d（%.0f%%）／「号」 %d（%.0f%%）"
              % (st["closer_name"], 100.0 * st["closer_name"] / tot,
                 st["closer_num"], 100.0 * st["closer_num"] / tot))
    if ex:
        print()
        print("样例（gt点, 到各候选的距离）：")
        for cid, comp, pk, nm, gp, ds in ex:
            print("  %-6s %-6s %-8s gt名=%-10s gt=%s %s" % (cid, comp, pk, nm, gp, ds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
