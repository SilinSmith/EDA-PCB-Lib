"""2x2 探针：{交付权重, 域内微调权重} x {紧框裁片, 管线外扩裁片}。

问题：上一次域内微调（紧框语料）在管线里 −4.45，假设是「语料裁片口径 ≠ stage2 运行时
裁片口径」（stage2 先 `pad_box` 再裁）。本探针用**同一批官方裁剪行**、同一条文本，
把两种口径各读一遍，量各角色的 exact / CER。

若假设成立：紧框语料训出来的权重在「外扩口径」上明显掉，而交付权重掉得少 ⇒ 用外扩
口径重建语料（mk_official_textset.py 的 `pad_box`）是正确修法。

用法：
  python -X utf8 work\\tools\\dev\\_padchk.py --ckpt-a work\\ocr_mix\\runs\\best.pt \
      --ckpt-b work\\ocr_official_v1\\best.pt --val-cases "1,2,3,4,5,6,7,8,9,10" --val-every 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]      # ...\EDA\work
EDA = WORK.parent                               # ...\EDA
sys.path.insert(0, str(EDA / "eda-pcb"))

from textrec.config import ROLE_ID  # noqa: E402


def edit_dist(a, b, cap=99):
    a, b = str(a), str(b)
    if abs(len(a) - len(b)) >= cap:
        return cap
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pad_box(box, pad_min=2.0, pad_frac=0.0, pad_min_y=3.0, pad_frac_y=0.20,
            pad_max=200.0):
    """与 stage2/text.py:164 同算法（默认值对齐 stage2/text.py:69-80）。"""
    h = max(1.0, float(box[3]) - float(box[1]))
    px = min(pad_max, max(pad_min, pad_frac * h))
    py = min(pad_max, max(pad_min_y, pad_frac_y * h))
    return [float(box[0]) - max(0.0, px), float(box[1]) - max(0.0, py),
            float(box[2]) + max(0.0, px), float(box[3]) + max(0.0, py)]


def load_rows(a):
    val_fixed = {s.strip() for s in a.val_cases.split(",") if s.strip()}
    rows, cases = [], []
    for ln in Path(a.corpus).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("role") not in ROLE_ID:
            continue
        rows.append(r)
        cases.append(r["case"])
    if a.only_val:
        fixed = set(val_fixed)
        cs = sorted(set(cases))
        for i, c in enumerate(cs):
            if a.val_every > 0 and re.match(r"^\d{4}$", c) and i % a.val_every == 0:
                fixed.add(c)
        rows = [r for r in rows if r["case"] in fixed]
    if a.limit:
        rows = rows[: a.limit]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(WORK / "official_text" / "manifest.jsonl"),
                    help="紧框语料（含 bbox = 紧框、text = GT）")
    ap.add_argument("--ckpt-a", default=str(WORK / "ocr_mix" / "runs" / "best.pt"))
    ap.add_argument("--ckpt-b", default=str(WORK / "ocr_official_v1" / "best.pt"))
    ap.add_argument("--only-val", action="store_true", default=True)
    ap.add_argument("--all", dest="only_val", action="store_false")
    ap.add_argument("--val-cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pinnum-frac-y", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump", default="", help="把逐条判定写成 jsonl")
    a = ap.parse_args()

    from PIL import Image
    from textrec.infer import Recognizer

    rows = load_rows(a)
    print("rows=%d  ckpt_a=%s  ckpt_b=%s" % (len(rows), Path(a.ckpt_a).name, Path(a.ckpt_b).name))
    recs = {}
    for tag, p in (("A", a.ckpt_a), ("B", a.ckpt_b)):
        if Path(p).exists():
            recs[tag] = Recognizer.load(p, a.device)
        else:
            print("[warn] 缺 %s: %s" % (tag, p))

    by_img = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["image"]].append(i)

    verdict = {}
    for p, idxs in sorted(by_img.items()):
        try:
            gray = Image.open(p).convert("L")
        except Exception as e:
            print("[warn] 打不开 %s: %s" % (p, e))
            continue
        tight, padded = [], []
        for i in idxs:
            r = rows[i]
            fy = a.pinnum_frac_y if r["role"] == "pin_number" else 0.20
            tight.append(([float(v) for v in r["bbox"]], r["role"]))
            padded.append((pad_box(r["bbox"], pad_frac_y=fy), r["role"]))
        for tag, rec in recs.items():
            for mode, items in (("tight", tight), ("pad", padded)):
                try:
                    got = rec.read_many(gray, items)
                except Exception as e:
                    print("[warn] 读失败 %s/%s: %s" % (tag, mode, e))
                    continue
                for i, (txt, conf) in zip(idxs, got):
                    verdict.setdefault(i, {})["%s_%s" % (tag, mode)] = (txt, float(conf))

    stats = defaultdict(lambda: [0, 0, 0.0])          # key -> [n, exact, cer_sum]
    cross = defaultdict(int)
    dump = []
    for i, r in enumerate(rows):
        gt = str(r["text"])
        v = verdict.get(i)
        if not v:
            continue
        row = {"case": r["case"], "role": r["role"], "text": gt, "was": r.get("was")}
        for k, (txt, conf) in v.items():
            d = edit_dist(txt, gt, 99)
            row[k] = txt
            row[k + "_d"] = d
            s = stats[(r["role"], k)]
            s[0] += 1
            s[1] += int(d == 0)
            s[2] += d / max(1, len(gt))
        for k in ("A_tight", "A_pad", "B_tight", "B_pad"):
            if k in v:
                cross[k] += int(edit_dist(v[k][0], gt, 99) == 0)
        dump.append(row)

    cells = ["A_tight", "A_pad", "B_tight", "B_pad"]
    roles = sorted({r["role"] for r in rows})
    print("\n%-12s %-8s %6s %8s %8s" % ("role", "cell", "n", "exact", "cer"))
    for role in roles + ["ALL"]:
        for c in cells:
            if role == "ALL":
                n = sum(stats[(r, c)][0] for r in roles)
                ex = sum(stats[(r, c)][1] for r in roles)
                ce = sum(stats[(r, c)][2] for r in roles)
            else:
                n, ex, ce = stats[(role, c)]
            if not n:
                continue
            print("%-12s %-8s %6d %7.2f%% %8.4f" % (role, c, n, 100.0 * ex / n, ce / n))
    print("\n总体 exact:", {c: cross[c] for c in cells})
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for row in dump:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("写好", a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
