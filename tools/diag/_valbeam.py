# -*- coding: utf-8 -*-
"""r/c `value` 到底能不能靠「后处理/词表」救：把 CRNN 每个字的**前 3 名**打出来。

动机（见 `work\\_HANDOFF_20261009.md` 与 b897）：官方 9 张上 component 的 7 对
`text_mismatch` **全部**是 r/c 的 `value`，而且 6/7 在 case 1，形态高度集中：

    C159  GT '1uF'   -> '1uH'      (F/H)
    C160  GT '10nF'  -> '10uF'     (n/u)
    C154  GT '10nF'  -> '10uF'
    C152  GT '10nF'  -> '10uF'
    C153  GT '10nF'  -> '10uF'
    C155  GT '10nF'  -> None       (整条没读出)
    C162  GT '0.1μF' -> None

如果正确的字只是**第二名**（两者对数概率差很小），那么加一层「单位词表 + 重打分」
的 beam 就能白捡；如果正确字的概率远低于错字，那是模型本身没看清，只能回去喂样本
（`work\\_HANDOFF_20261009.md` §2 第 4 条）。

用法：
    python -X utf8 work\\tools\\diag\\_valbeam.py --case 1 --filter "F|H|u|n|Ω"
    python -X utf8 work\\tools\\diag\\_valbeam.py --case 1 --bbox 1000,229,1040,250 --role value
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA\eda-pcb")
WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(EDA))

from textrec.charset import BLANK                       # noqa: E402
from textrec.dataset import crop_gray                   # noqa: E402
from textrec.infer import Recognizer, WIDTH_REDUCTION, _role_id   # noqa: E402

DEFAULT_CKPT = WORK / "ocr_mix" / "runs" / "best.pt"


def find_png(case: str) -> Path | None:
    for root in (WORK / "full220" / "images", WORK / "full220",
                 EDA / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"):
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.png")):
            if p.stem == case:
                return p
    return None


def find_stage1(case: str, s1: Path | None) -> dict | None:
    cands = [s1] if s1 else []
    cands += [WORK / "_fast_out_wl_on" / "runs" / "stage1" / ("%s.stage1.json" % case),
              WORK / "_full220" / "runs" / "stage1" / ("%s.stage1.json" % case)]
    for c in cands:
        if c and Path(c).is_file():
            return json.loads(Path(c).read_text(encoding="utf-8"))
    return None


def topk_block(rec: Recognizer, img: Image.Image, bbox, role: str, k: int = 3):
    """单条裁片前向，返回 [(char, [(ch, logp), ...]), ...]（只列真正吐出的字）。"""
    crop = crop_gray(img, bbox, rec.img_h, rec.max_w)
    a = np.asarray(crop, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    x = torch.from_numpy(a)[None, None, :, :].to(rec.device)
    r = torch.tensor([_role_id(role)], dtype=torch.long, device=rec.device)
    with torch.no_grad():
        lp = rec.model(x, r).log_softmax(2)[:, 0, :]      # T, C
    T = lp.shape[0]
    tmax = max(1, min(T, a.shape[1] // WIDTH_REDUCTION))
    lp = lp[:tmax]
    vals, idx = lp.topk(k, dim=1)
    out, prev = [], None
    for t in range(lp.shape[0]):
        i = int(idx[t, 0])
        if i == prev or i == BLANK:
            prev = i
            continue
        prev = i
        alts = []
        for j in range(k):
            c = rec.charset.itos.get(int(idx[t, j]), "")
            alts.append(("%s" % ("{blank}" if int(idx[t, j]) == BLANK else c),
                         round(float(vals[t, j]), 3)))
        out.append((rec.charset.itos.get(i, "?"), alts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--role", default="value")
    ap.add_argument("--filter", default=None, help="正则；只打印读出的文本命中它的框")
    ap.add_argument("--bbox", default=None, help="x1,y1,x2,y2；只分析这一个框")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--max-print", type=int, default=30)
    a = ap.parse_args()

    import re
    pat = re.compile(a.filter) if a.filter else None

    png = Path(a.png) if a.png else find_png(a.case)
    if png is None or not Path(png).is_file():
        print("[err] 找不到图：%s" % a.case)
        return 2
    s1 = find_stage1(a.case, Path(a.stage1) if a.stage1 else None)
    if s1 is None:
        print("[err] 找不到 stage1 产物（用 --stage1 指定）")
        return 2
    img = Image.open(png).convert("L")
    rec = Recognizer.load(a.ckpt, device="cpu")
    print("图 = %s   文字框 = %d   ckpt = %s" % (png.name, len(s1["texts"]), Path(a.ckpt).name))

    if a.bbox:
        x1, y1, x2, y2 = [float(v) for v in a.bbox.split(",")]
        boxes = [(a.bbox, [x1, y1, x2, y2])]
    else:
        boxes = [(t["id"], t["bbox_img"]) for t in s1["texts"]]

    items = [(bb, a.role) for _id, bb in boxes]
    res = rec.read_many(img, items)
    shown = 0
    for (pid, bb), (txt, conf) in zip(boxes, res):
        if conf < a.min_conf:
            continue
        if pat and not pat.search(txt or ""):
            continue
        if shown >= a.max_print:
            break
        shown += 1
        print("\n%s bbox=[%s] role=%s  ->  %r  conf=%.4f"
              % (pid, ",".join("%.1f" % v for v in bb), a.role, txt, conf))
        for ch, alts in topk_block(rec, img, bb, a.role, a.topk):
            flag = ""
            if len(alts) > 1 and (alts[0][1] - alts[1][1]) < 0.7:
                flag = "   <== 前二接近"
            print("     %-3s %s%s" % (ch, "  ".join("%s %.2f" % x for x in alts), flag))
    if not shown:
        print("[info] 没有框命中 filter=%r" % a.filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
