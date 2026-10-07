# -*- coding: utf-8 -*-
"""逐颗 IC 直接调 `enrich.read_designator`，看位号通道**到底读出了什么**。

`_ickey.py` 实测（快速回归集 19 张）：GT 有真名的 IC 元件 59 个，键配上 24（0.407）；
没配上的 35 个里 **`整颗没输出` 0 个、`类型不是 IC` 0 个、`兜底键 pNNN` 24 个（68.6%）、
`读到别的位号` 11 个**。⇒ 框其实全都找到了、type 也都对，**卡在「位号没读出来」**。
而 `enrich.py` 的 `desig_fallback` 本来就该去读这些 `pNNN` 元件 —— 这个脚本查它读到了什么。

用法：
    python work\\tools\\diag\\_ickeyprobe.py --cases 0024,0019 --out work\\_ickeyprobe.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402

IC = ("box", "ic", "boxic")
FKEY = re.compile(r"^p\d{3,4}$")
DESIG = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="0024,0019,0086,0103")
    ap.add_argument("--out", default=str(ROOT / "work" / "_ickeyprobe.json"))
    ap.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()

    from vlm import enrich                     # noqa: E402
    from vlm.enrich import DEFAULTS            # noqa: E402
    from stage2.text import imread_unicode     # noqa: E402

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    opts = dict(DEFAULTS)
    if a.margin:
        opts["desig_margin"] = a.margin

    rows = []
    for cid in [x.strip() for x in a.cases.split(",") if x.strip()]:
        c, _how = resolve(cid, exact, digits)
        pr = json.loads(pred_file(Path(a.pred), cid).read_text(encoding="utf-8"))
        gt = load_json(c.json_path)
        bgr = imread_unicode(str(c.png_path))
        H = bgr.shape[0]
        gkeys = {k for k, v in (gt.get("components") or {}).items()
                 if str(v.get("type") or "").lower() in IC}
        print("== case %s  图 %dx%d  GT 的 IC 键: %s"
              % (cid, bgr.shape[1], H, ",".join(sorted(gkeys)[:20])))
        for key, cv_ in (pr.get("components") or {}).items():
            if str(cv_.get("type") or "").lower() not in IC:
                continue
            bb = cv_.get("bbox")
            if not bb:
                continue
            ib = (bb[0], H - bb[3], bb[2], H - bb[1])
            stat = dict(desig=0, grid=0, value=0, value_set=0)
            got = enrich.read_designator(bgr, ib, opts, stat, "box")
            hit = "OK" if got and any(got.upper() == g.upper() for g in gkeys) else ""
            print("   %-8s -> %-10s 调用 %d  %s" % (key, got, stat["desig"], hit))
            rows.append(dict(case=cid, key=key, got=got, calls=stat["desig"],
                             gt_ics=sorted(gkeys)))
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n写出", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
