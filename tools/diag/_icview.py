# -*- coding: utf-8 -*-
"""把「GT 是 IC、键没配上」的元件在原图上画出来，人眼核对**图上到底有没有印那个位号**。

`_ickeyprobe.py` 实测：`0024` 的 `pNNN` 元件 VLM 读出了 `K1..K15`（GT 却叫 `U1..U16`）；
`0019` 读出 `U1` 而 GT 是 `u1`（**大小写**）；而 `0086/0075/0042/0092/0048` 全部返回 `None`
（各 2 次调用，margin 60 与 132）。这个脚本把后一类的原图裁出来，好判断是
「图上没有」还是「有但我们裁不到」。

用法：`python work\\tools\\diag\\_icview.py --cases "0042,0086,0075" --pad 40 --scale 1.0 --out work\\preview\\_icnokey.png`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402

IC = ("box", "ic", "boxic")


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="0042,0086,0075")
    ap.add_argument("--pad", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "work" / "preview" / "_icnokey.png"))
    a = ap.parse_args()

    import cv2
    from stage2.text import imread_unicode

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    panels = []
    for cid in [x.strip() for x in a.cases.split(",") if x.strip()]:
        c, _how = resolve(cid, exact, digits)
        pr = json.loads(pred_file(Path(a.pred), cid).read_text(encoding="utf-8"))
        gt = load_json(c.json_path)
        bgr = imread_unicode(str(c.png_path))
        H = bgr.shape[0]
        im = bgr.copy()
        for key, cv_ in (pr.get("components") or {}).items():
            if str(cv_.get("type") or "").lower() not in IC:
                continue
            bb = cv_.get("bbox")
            if not bb:
                continue
            cv2.rectangle(im, (int(bb[0]), int(H - bb[3])), (int(bb[2]), int(H - bb[1])),
                          (0, 0, 255), 2)
            cv2.putText(im, str(key), (int(bb[0]), int(H - bb[3]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for key, gv in (gt.get("components") or {}).items():
            if str(gv.get("type") or "").lower() not in IC:
                continue
            bb = gv["bbox"]
            cv2.rectangle(im, (int(bb[0]), int(H - bb[3])), (int(bb[2]), int(H - bb[1])),
                          (255, 0, 0), 1)
            cv2.putText(im, "GT:" + str(key), (int(bb[0]), int(H - bb[3]) - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if a.scale != 1.0:
            im = cv2.resize(im, None, fx=a.scale, fy=a.scale)
        panels.append((cid, im))

    # 拼图：按列数排，行高取最大
    rows = []
    for i in range(0, len(panels), a.cols):
        rows.append(panels[i:i + a.cols])
    W = max(sum(p[1].shape[1] for p in r) for r in rows)
    Hh = sum(max(p[1].shape[0] for p in r) for r in rows)
    canvas = 255 * __import__("numpy").ones((Hh, W, 3), dtype="uint8")
    y = 0
    for r in rows:
        x = 0
        h = max(p[1].shape[0] for p in r)
        for cid, im in r:
            canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
            x += im.shape[1]
        y += h
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), canvas)
    print("写出", a.out, canvas.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
