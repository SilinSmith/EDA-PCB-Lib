# -*- coding: utf-8 -*-
"""role 敏感性体检：模型多依赖喂进去的 role？（纯推理，不训练、不看 GT 坐标）

为什么要这个：§5.163 (C) 断言「把网标/脚号/页面家具误标成 value 会污染 value 的 role 嵌入」。
这个断言**可以直接量**：把 ruler 上每张图的全部框统一换成另一个 role 再读一遍，看 exact 怎么动。
- 若换 role 只掉几个点 ⇒ role 输入是弱条件，误标危害有限；
- 若掉很多 ⇒ 误标确实是重标签噪声。

用法（仓根执行）：
  python -X utf8 tools/diag/_rolesens.py                       # 默认交付权重 + work/fullpop/ruler.jsonl
  python -X utf8 tools/diag/_rolesens.py --roles value,net_label
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
if str(EDA) not in sys.path:
    sys.path.insert(0, str(EDA))

from PIL import Image                                        # noqa: E402
from textrec.infer import Recognizer                         # noqa: E402
from textrec.train import levenshtein                        # noqa: E402

ALL_ROLES = ("designator", "value", "pin_name", "pin_number", "net_label", "note")


def load(p: Path):
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(EDA / "weights" / "textrec_best.pt"))
    ap.add_argument("--manifest", default=r"Y:\workspace\EDA\work\fullpop\ruler.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--roles", default=",".join(ALL_ROLES),
                    help="要统一试的 role 列表（逗号分隔）")
    a = ap.parse_args(argv)

    rows = [r for r in load(Path(a.manifest)) if r.get("split") == "real"]
    if not rows:
        print("!! manifest 里没有 split=real 的行：%s" % a.manifest)
        return 2
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    print("[data] %d case / %d 条" % (len(by_img), len(rows)))

    rec = Recognizer.load(Path(a.ckpt), device=a.device)
    print("[recognizer] " + rec.debug())

    maps = {"correct": None}
    for role in [x.strip() for x in a.roles.split(",") if x.strip()]:
        maps["all=" + role] = role

    acc = {k: defaultdict(lambda: [0, 0, 0, 0]) for k in maps}   # tag -> role -> [ed, len, n, exact]
    for img_path in sorted(by_img):
        items = sorted(by_img[img_path], key=lambda r: tuple(r["bbox"]))
        with Image.open(img_path) as im0:
            im = im0.convert("L")
            for tag, forced in maps.items():
                pairs = [(r["bbox"], forced if forced else r["role"]) for r in items]
                preds = rec.read_many(im, pairs)
                for r, (pred, _c) in zip(items, preds):
                    s = acc[tag][r["role"]]
                    s[0] += levenshtein(pred, r["text"])
                    s[1] += max(1, len(r["text"]))
                    s[2] += 1
                    s[3] += int(pred == r["text"])

    cols = ("value", "pin_name", "pin_number", "net_label", "designator")
    print("\n%-16s %-9s %s" % ("读法", "ALL ex", " ".join("%-9s" % c for c in cols)))
    for tag in maps:
        cells, tot = [], [0, 0]
        for role in cols:
            s = acc[tag].get(role)
            cells.append("%.4f" % (s[3] / s[2]) if s and s[2] else "-")
            if s:
                tot[0] += s[3]
                tot[1] += s[2]
        print("%-16s %-9s %s" % (tag, "%.4f" % (tot[0] / max(1, tot[1])),
                                 " ".join("%-9s" % c for c in cells)))
    print("\n注：用 ruler 的 GT 去**挑**一个 role 映射 = 在尺子上拟合，不可当交付决策；"
          "本表只用来判断「role 输入有多强」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
