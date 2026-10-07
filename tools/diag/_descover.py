# -*- coding: utf-8 -*-
"""图上到底印了哪些位号——把 stage1 的**全部**文字框过一次 CRNN（designator 角色），
收集所有位号形状的读法，再和 GT 的键集合对比。

回答的问题：那 2191 个「检测对、类型对、位置对、就是没位号」的元件，
**位号是不是压根没印在图上**（或印成了别的字），还是印了但我们没读到。

    python work\\tools\\diag\\_descover.py --cases 0001,0002,0003 --all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import stage2_eval as SE                                  # noqa: E402
from stage2.text import TextEngine, imread_unicode        # noqa: E402

DES = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="逗号分隔的 case id；留空则取 --n 张")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--stage1", default=str(ROOT / "work" / "_crnntrunc" / "runs" / "stage1"))
    ap.add_argument("--all", action="store_true", help="逐条打印 GT 对不上的键")
    a = ap.parse_args()

    ex, di = SE.build_index(SE.DEFAULT_DATA_ROOT)
    s1dir = Path(a.stage1)
    names = [s.strip() for s in a.cases.split(",") if s.strip()]
    if not names:
        names = [p.name.replace(".stage1.json", "") for p in sorted(s1dir.glob("*.stage1.json"))[: a.n]]
    eng = TextEngine()

    tot_gt = tot_hit = tot_got = 0
    for nm in names:
        try:
            c, _ = SE.resolve(nm, ex, di)
        except Exception as e:
            print("skip", nm, e)
            continue
        s1f = s1dir / (nm + ".stage1.json")
        if not s1f.is_file():
            print("no stage1", nm)
            continue
        s1 = json.loads(s1f.read_text(encoding="utf-8"))
        g = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        keys = [k for k in (g.get("components") or {}) if not str(k).startswith("\u2205")]
        bgr = imread_unicode(c.png_path)
        boxes = [tuple(float(v) for v in t["bbox_img"]) for t in (s1.get("texts") or [])]
        out = eng.read(bgr, boxes, ["designator"]) if boxes else []
        got = {}
        for rr in out:
            txt = (rr.get("reads") or {}).get("designator", ("", 0.0))[0]
            if txt and DES.match(txt):
                got.setdefault(txt.upper(), 0)
                got[txt.upper()] += 1
        hit = sum(1 for k in keys if str(k).upper() in got)
        tot_gt += len(keys)
        tot_hit += hit
        tot_got += len(got)
        print("%-10s 文字框 %4d  能读出位号形状的种类 %3d  GT键 %3d  覆盖 %3d (%.2f)"
              % (nm, len(boxes), len(got), len(keys), hit, hit / max(1, len(keys))))
        if a.all:
            miss = [k for k in keys if str(k).upper() not in got]
            print("     GT 读不到：", " ".join(sorted(map(str, miss))[:40]))
            print("     图上读到：", " ".join(sorted(got)[:40]))
    print("合计 GT键 %d  被图上读法覆盖 %d = %.3f" % (tot_gt, tot_hit, tot_hit / max(1, tot_gt)))


if __name__ == "__main__":
    main()
