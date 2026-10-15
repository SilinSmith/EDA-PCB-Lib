# -*- coding: utf-8 -*-
"""量「CRNN 在官方渲染图上读文字的错读率」——按套（10GT / 10_GTcase / train200）× role 分开。

为什么单独一个探针：`mk_official_textset.py` 只保留「读数与 GT 足够像」的行（那是为了拿它当
标签训练），而改进方向要先知道**分母**有多大。这里把每个「框 ↔ GT 文字」配对都算进来，只要
坐标系与出处两道门过了，就统计 exact / CER / 混淆对。

框来自我们自己的检测，标签来自 GT：
  · 元件：`stage1.components[i].bbox_json`（官方约定，y 从底数）配 GT `components[设计号].bbox`
  · 引脚：`stage1.pins[cid][j]` 端点翻 y 后配 GT `pins[设计号][pin_K].point`
读数取 `_assigned` 里的槽位（`_by` 含 `vlm` 的丢掉，见 `mk_official_textset.ocr_by`）。

用法：
  python -X utf8 work\tools\dev\_textrate.py --cases-file work\_cases220.txt
  python -X utf8 work\tools\dev\_textrate.py --limit 60 --max-print 12
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(WORK / "tools" / "dev"))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import mk_official_textset as M       # noqa: E402
import stage2_eval as SE              # noqa: E402

ROLES = ("designator", "value", "partname", "pin_name", "pin_number")


def suite(n: str) -> str:
    if re.match(r"^\d{1,2}$", n):
        return "10GT"
    if re.match(r"^\d{4}$", n):
        return "train200"
    return "10_GTcase"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage1", default=str(WORK / "_full220" / "runs" / "stage1"))
    ap.add_argument("--assigned", default=str(WORK / "_f220_A" / "_assigned"))
    ap.add_argument("--no-pins", action="store_true")
    ap.add_argument("--max-print", type=int, default=8)
    a = ap.parse_args()

    names = [s.strip() for s in Path(a.cases_file).read_text(
        encoding="utf-8").strip().split(",") if s.strip()]
    if a.limit:
        names = names[:a.limit]

    idx, digits = M.build_index()
    s1dir, asdir = Path(a.stage1), Path(a.assigned)
    # (suite, role) -> [n, exact, cer_num, cer_den]
    agg = defaultdict(lambda: [0, 0, 0, 0])
    conf = defaultdict(Counter)          # (suite, role) -> Counter((gt, ocr))
    skip = Counter()
    from PIL import Image
    for name in names:
        s1p, asp = s1dir / ("%s.stage1.json" % name), asdir / ("%s.json" % name)
        if not s1p.is_file() or not asp.is_file():
            skip["缺产物"] += 1
            continue
        case, _how = SE.resolve(name, idx, digits)
        if case is None:
            skip["缺 GT"] += 1
            continue
        gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
        s1 = json.loads(s1p.read_text(encoding="utf-8"))
        asg = json.loads(asp.read_text(encoding="utf-8"))
        img = str(case.png_path) if getattr(case, "png_path", None) else None
        if not img:
            skip["缺图"] += 1
            continue
        img_h = Image.open(img).size[1]
        gt_comps, gt_pins = gt.get("components") or {}, gt.get("pins") or {}
        s1c = {c["id"]: c for c in (s1.get("components") or []) if c.get("id")}
        cmap = M.match_comps(gt_comps, s1c)
        st = suite(name)

        def add(role, gt_text, ocr_text):
            if not gt_text or not ocr_text:
                return
            g, o = str(gt_text), str(ocr_text)
            k = (st, role)
            agg[k][0] += 1
            d = M.edit_dist(o, g, 99)
            agg[k][2] += d
            agg[k][3] += len(g)
            if o == g:
                agg[k][1] += 1
            else:
                conf[k][(g, o)] += 1

        for cid, gk in cmap.items():
            if M.junk_key(gk):
                continue
            g = gt_comps[gk] or {}
            ac = (asg.get("comps") or {}).get(cid) or {}
            for slot, role, gt_text in (
                    ("designator", "designator", gk if M.DESIG.match(gk) else None),
                    ("value", "value", g.get("value"))):
                e = ac.get(slot) or {}
                if e.get("box") and M.ocr_by(e):
                    add(role, gt_text, e.get("text"))
            gn = g.get("Name")
            e = ac.get("partname") or {}
            if gn and gn != gk and not M.junk_key(gn) and not M.DESIG.match(str(gn)) \
                    and e.get("box") and M.ocr_by(e):
                add("partname", gn, e.get("text"))
            if a.no_pins:
                continue
            ours = (s1.get("pins") or {}).get(cid) or []
            gps = gt_pins.get(gk) or {}
            pmap = M.match_pins(ours, gps, img_h)
            apins = asg.get("comps", {}).get(cid, {}).get("pins") or {}
            for j, pk in pmap.items():
                ae = apins.get(str(j)) or {}
                gp = gps.get(pk) or {}
                en = ae.get("name") or {}
                if en.get("box") and M.ocr_by(en) and not M.junk_key(str(gp.get("pinname"))):
                    add("pin_name", gp.get("pinname"), en.get("text"))
                en = ae.get("number") or {}
                if en.get("box") and M.ocr_by(en):
                    add("pin_number", M.pin_number(pk), en.get("text"))

    print("跳过:", dict(skip))
    suites = ("10GT", "10_GTcase", "train200")
    for role in ROLES:
        line = ["%-12s" % role]
        for st in suites:
            v = agg.get((st, role))
            if not v or not v[0]:
                line.append("%-22s" % "-")
            else:
                line.append("%s n=%-5d exact=%5.1f%% cer=%.4f" % (
                    st, v[0], 100.0 * v[1] / v[0], v[2] / max(1, v[3])))
        print("  ".join(line))
    tot = {}
    for st in suites:
        n = sum(agg[(st, r)][0] for r in ROLES)
        ex = sum(agg[(st, r)][1] for r in ROLES)
        dn = sum(agg[(st, r)][3] for r in ROLES)
        dd = sum(agg[(st, r)][2] for r in ROLES)
        tot[st] = (n, ex, dd, dn)
        if n:
            print("ALL %-10s n=%-5d exact=%5.1f%% cer=%.4f" % (
                st, n, 100.0 * ex / n, dd / max(1, dn)))
    # 所有套合起来
    n = sum(v[0] for v in tot.values())
    ex = sum(v[1] for v in tot.values())
    dd = sum(v[2] for v in tot.values())
    dn = sum(v[3] for v in tot.values())
    print("ALL 合计     n=%-5d exact=%5.1f%% cer=%.4f" % (n, 100.0 * ex / max(1, n), dd / max(1, dn)))
    if a.max_print:
        print("\n错误最多的配对（GT -> 读数）:")
        allc = Counter()
        for k, c in conf.items():
            for pair, cnt in c.items():
                allc[(k[0], k[1]) + pair] += cnt
        for (st, role, g, o), cnt in allc.most_common(a.max_print):
            print("  %-10s %-12s %-14s -> %-14s x%d" % (st, role, g, o, cnt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
