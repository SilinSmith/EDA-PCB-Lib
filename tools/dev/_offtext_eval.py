# -*- coding: utf-8 -*-
"""在**官方渲染图的域内语料**上评一个 CRNN 权重：只走生产路径（`textrec.infer.Recognizer`）。

为什么不用 `textrec.eval_real`：那一套是 `Dataset300`（另一套真实渲染），它**不含**官方
10GT / 200_train_cases 的栅格；域内适配要看的正是后者。这里的标签来自 GT（坐标系与出处两道门
见 `mk_official_textset.py`）。

用法：
  python -X utf8 work\tools\dev\_offtext_eval.py --ckpt work\ocr_mix\runs\best.pt
  python -X utf8 work\tools\dev\_offtext_eval.py --ckpt work\ocr_official_v1\best.pt --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from textrec import config as C                     # noqa: E402
from textrec.infer import Recognizer                # noqa: E402


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(WORK / "official_text" / "manifest.jsonl"))
    ap.add_argument("--ckpt", default=str(WORK / "ocr_mix" / "runs" / "best.pt"))
    ap.add_argument("--split", default="val", help="用哪批 case：全部=all；或 val/train（按 case 切）")
    ap.add_argument("--val-cases", default="1,5", help="与 mk_textrec_manifest.py 保持一致")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("--max-print", type=int, default=10)
    a = ap.parse_args()

    import re
    rows = [json.loads(x) for x in Path(a.corpus).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if int(r.get("dist", 0)) <= a.max_dist]
    cases = sorted({r["case"] for r in rows})
    val_cases = {s.strip() for s in a.val_cases.split(",") if s.strip()} & set(cases)
    if a.val_every > 0:
        for i, c in enumerate(cases):
            if re.match(r"^\d{4}$", c) and i % a.val_every == 0:
                val_cases.add(c)
    if a.split == "val":
        rows = [r for r in rows if r["case"] in val_cases]
    elif a.split == "train":
        rows = [r for r in rows if r["case"] not in val_cases]

    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    rec = Recognizer.load(a.ckpt, device=a.device)
    print("[ckpt] %s  rows=%d  cases=%d  device=%s" % (a.ckpt, len(rows), len(by_img), a.device))

    stat = defaultdict(lambda: [0, 0, 0, 0])       # role -> n, exact, ed, len
    conf = Counter()
    from PIL import Image
    for p, rs in by_img.items():
        img = Image.open(p).convert("L")
        items = [(r["bbox"], r["role"]) for r in rs]
        got = rec.read_many(img, items)
        for r, (txt, _conf) in zip(rs, got):
            g = str(r["text"])
            exact = int(txt == g)
            ed = lev(txt, g)
            s = stat[r["role"]]
            s[0] += 1
            s[1] += exact
            s[2] += ed
            s[3] += max(1, len(g))
            if not exact:
                conf[(r["role"], g, txt)] += 1
    n = ex = ed = ln = 0
    for role in sorted(stat):
        v = stat[role]
        n += v[0]
        ex += v[1]
        ed += v[2]
        ln += v[3]
        print("  %-12s n=%-5d exact=%6.2f%%  cer=%.4f" % (
            role, v[0], 100.0 * v[1] / max(1, v[0]), v[2] / max(1, v[3])))
    print("  %-12s n=%-5d exact=%6.2f%%  cer=%.4f" % (
        "ALL", n, 100.0 * ex / max(1, n), ed / max(1, ln)))
    if a.max_print:
        print("  还没读对的（GT -> pred）:")
        for (role, g, t), c in conf.most_common(a.max_print):
            print("    %-11s %-14s -> %-14s x%d" % (role, g, t, c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
