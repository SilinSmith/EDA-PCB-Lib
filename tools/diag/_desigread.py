"""直接审问位号读取链路：对一个 `box` 元件，`DESIG_PROMPT` 到底回了什么？

背景（`_desigsrc.py` 在 220 张上的实测）：594 个没对上的 GT `box` 里
**294 个（49.5%）我们的键是占位符 `pNNN`**，而这 294 个**全部**类型 ∈
`IC = ("box","ic","boxic")` ⇒ **位号读取本来就为它们发起过**，
问题出在「读到的东西」或「读到之后的处置」，不是「没去读」。

本脚本不猜，直接打印模型的**原样回复**，四类结局一眼可分：
  * `NONE`              ⇒ 裁剪/提示词问题（位号不在框外 60px 内？）
  * 邻居位号（`U1`…）    ⇒ 取景太大，看到隔壁
  * 器件型号（`LM385`）  ⇒ **它长得也像位号**（`DESIG_RE` 是 `^[A-Za-z]{1,3}\\d{1,4}$`，
                          `LM385`/`MAX232`/`IR4428` 全部合法）⇒ 提示词没说清区别
  * 乱码/空             ⇒ 模型看不清

用法：
  python work\\tools\\diag\\_desigread.py --case 0002 --key p001
  python work\\tools\\diag\\_desigread.py --case 0002 --all
  python work\\tools\\diag\\_desigread.py --dir work\\_submit_out220r36\\runs\\fusion --case 0030 --all --margin 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

import cv2                                                                # noqa: E402
from vlm import client, enrich as E                                       # noqa: E402


def find_json(pdir: Path, case: str):
    for pat in (f"{case}.json", f"final_results/{case}_Pred.json",
                f"runs/fusion/{case}.json", f"runs/stage1/{case}.stage1.json",
                f"{case}.stage1.json"):
        p = pdir / pat
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out"))
    ap.add_argument("--case", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--margin", type=float, default=60.0)
    ap.add_argument("--model", default="")
    ap.add_argument("--save", default="", help="把每个裁片存成 PNG 到该目录，便于肉眼核对")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = find_json(Path(a.dir), a.case)
    if not p:
        print("找不到 %s 的产物（试过 %s/...）" % (a.case, a.dir))
        return
    doc = json.loads(p.read_text(encoding="utf-8"))
    comps = doc.get("components") or {}
    from schematic_parser import data as DD                               # noqa: E402
    c = DD.find_case(a.case)
    if not c:
        print("找不到 %s 的图片（data root = %s）" % (a.case, DD.DEFAULT_DATA_ROOT))
        return
    img = Path(c.png_path)
    bgr = cv2.imdecode(__import__("numpy").fromfile(str(img), dtype="uint8"), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    print("产物 %s\n图片 %s  %dx%d  元件 %d 个" % (p, img.name, W, H, len(comps)))

    opts = dict(E.DEFAULTS)
    opts["desig_margin"] = a.margin
    if a.model:
        opts["model"] = a.model
    stat = dict(desig=0, grid=0, value=0, value_set=0, name=0, name_set=0)

    rows = []
    for k, v in comps.items():
        ct = str((v or {}).get("type") or "").lower()
        if ct not in E.IC:
            continue
        if a.key and k != a.key:
            continue
        if a.all and not E.FKEY_RE.match(k):
            continue
        bb = (v or {}).get("bbox")
        if not bb:
            continue
        ib = (bb[0], H - bb[3], bb[2], H - bb[1])
        rows.append((k, ct, bb, ib))
    if not rows:
        print("没有匹配的元件（--key %s / --all）" % (a.key or "(未给)"))
        return

    for k, ct, bb, ib in rows:
        h, w = bgr.shape[:2]
        m = opts["desig_margin"]
        x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
        x1 = min(w, int(ib[2] + m)); y1 = min(h, int(ib[3] + m))
        crop = bgr[y0:y1, x0:x1]
        if a.save:
            od = Path(a.save)
            od.mkdir(parents=True, exist_ok=True)
            op = od / ("%s_%s.png" % (a.case, k))
            ok2, buf = cv2.imencode(".png", crop)
            if ok2:
                buf.tofile(str(op))
        raw, dt = client.ask(crop, E.DESIG_PROMPT, model=opts["model"])
        first = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
        clean = __import__("re").sub(r"[^0-9A-Za-z?]", "", first)
        ok = bool(clean and E.DESIG_RE.match(clean))
        got = E.read_designator(bgr, ib, opts, stat, ct)
        print("\n%-10s type=%-8s bbox=%s  crop=%dx%d"
              % (k, ct, [int(x) for x in bb], x1 - x0, y1 - y0))
        print("   raw      = %r  (%.1fs)" % ((raw or "").strip()[:120], dt))
        print("   清洗后   = %r  合法位号=%s" % (clean, ok))
        print("   read_designator() → %r" % (got,))


if __name__ == "__main__":
    main()
