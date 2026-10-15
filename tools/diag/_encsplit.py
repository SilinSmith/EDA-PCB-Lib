# -*- coding: utf-8 -*-
"""可表示性切分：按「模型自带字表能否表示」切 real split，把结构性问题（标签空间里没有这个字）
与能力问题分开；并可按「该框是否进过训练/验证清单」再切一刀（训练集重叠会让读数虚高）。
见 docs/RESULTS.md §5.158 / §5.159。

用法：python -X utf8 eda-pcb/tools/diag/_encsplit.py [--ckpt ...] [--charset-mode model|json]
      [--holdout-from work\\ocr_real\\manifest.jsonl] [--limit N]
"""
import argparse
import collections
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from PIL import Image                                     # noqa: E402
from textrec.infer import Recognizer                      # noqa: E402

MANIFEST = r"Y:\workspace\EDA\eda-pcb\textrec\data\manifest.jsonl"
CHARSET = r"Y:\workspace\EDA\eda-pcb\textrec\data\charset.json"
BINS = ("1-2", "3-4", "5-7", "8-11", "12+")


def lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return n + m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def lb(n: int) -> str:
    return "1-2" if n <= 2 else ("3-4" if n <= 4 else ("5-7" if n <= 7 else ("8-11" if n <= 11 else "12+")))


def row_key(r: dict) -> "tuple":
    bb = tuple(round(float(x), 1) for x in (r.get("bbox") or []))
    return (os.path.basename(str(r.get("image"))), bb)


def load_jsonl(path: str):
    out = []
    with io.open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def _st(rows):
    n = len(rows)
    ex = sum(int(p == r["text"]) for r, p in rows)
    ed = sum(lev(p, r["text"]) for r, p in rows)
    ln = sum(max(1, len(r["text"])) for r, p in rows)
    return n, ex / max(1, n), ed / max(1, ln)


def report(rows, chars, title, with_missing=True):
    """rows = [(rec, pred, miss_chars)]"""
    print("\n=== %s ===" % title)
    tot = collections.Counter()
    missing_by_role = collections.defaultdict(collections.Counter)
    for r, pred, miss in rows:
        enc = not miss
        k = "enc" if enc else "un"
        tot[(r["role"], k)] += 1
        tot[(r["role"], k + "_ex")] += int(pred == r["text"])
        tot[(r["role"], k + "_ed")] += lev(pred, r["text"])
        tot[(r["role"], k + "_len")] += max(1, len(r["text"]))
        for c in miss:
            missing_by_role[r["role"]][c] += 1

    print("role         | %-28s | %-28s" % ("表内可表示 enc", "表中缺字 unencodable"))
    print("%-12s | %7s %7s %7s | %7s %7s %7s" % ("", "n", "exact", "CER", "n", "exact", "CER"))
    roles = sorted({r["role"] for r, _, _ in rows})
    for role in roles:
        ne, nu = tot[(role, "enc")], tot[(role, "un")]
        print("%-12s | %7d %7.4f %7.4f | %7d %7.4f %7.4f" % (
            role, ne, tot[(role, "enc_ex")] / max(1, ne), tot[(role, "enc_ed")] / max(1, tot[(role, "enc_len")]),
            nu, tot[(role, "un_ex")] / max(1, nu), tot[(role, "un_ed")] / max(1, tot[(role, "un_len")])))
    enc_rows = [(r, p) for r, p, m in rows if not m]
    un_rows = [(r, p) for r, p, m in rows if m]
    ne, ee, ce = _st(enc_rows)
    nu, eu, cu = _st(un_rows)
    print("%-12s | %7d %7.4f %7.4f | %7d %7.4f %7.4f" % ("ALL", ne, ee, ce, nu, eu, cu))

    print("\n表内可表示子集（enc）按 role × 长度：n / exact")
    print("%-12s %s" % ("role", " ".join("%14s" % b for b in BINS)))
    for role in roles:
        cells = []
        for b in BINS:
            sub = [(r, p) for r, p in enc_rows if r["role"] == role and lb(len(r["text"])) == b]
            cells.append("%5d/%.4f" % (len(sub), sum(int(p == r["text"]) for r, p in sub) / max(1, len(sub))))
        print("%-12s %s" % (role, " ".join("%14s" % c for c in cells)))
    for tag, rs in (("ALL-enc", enc_rows), ("ALL(全部)", [(r, p) for r, p, _ in rows])):
        cells = []
        for b in BINS:
            sub = [(r, p) for r, p in rs if lb(len(r["text"])) == b]
            cells.append("%5d/%.4f" % (len(sub), sum(int(p == r["text"]) for r, p in sub) / max(1, len(sub))))
        print("%-12s %s" % (tag, " ".join("%14s" % c for c in cells)))

    if with_missing:
        print("\n各 role 缺字 top（表里没有、因此永远不可能输出）")
        for role in roles:
            c = missing_by_role[role]
            if c:
                tot_miss = sum(1 for r, _, m in rows if r["role"] == role and m)
                print("%-12s 缺字样本 %d/%d  %s" % (
                    role, tot_miss, tot[(role, "enc")] + tot[(role, "un")],
                    " ".join("%s×%d" % (k, v) for k, v in c.most_common(10))))
        allmiss = collections.Counter()
        for role in roles:
            allmiss.update(missing_by_role[role])
        print("\n全量缺字：", " ".join("%s×%d" % (k, v) for k, v in allmiss.most_common(24)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"Y:\workspace\EDA\eda-pcb\weights\textrec_best.pt")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--charset", default=CHARSET)
    ap.add_argument("--charset-mode", choices=("model", "json"), default="model",
                    help="用部署模型自带的字表（默认，权威）还是 charset.json")
    ap.add_argument("--holdout-from", default="",
                    help="把该清单里出现过的框标成 in-train，分「held-out / in-train」两块报（训练集重叠会让读数虚高）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    rec = Recognizer.load(Path(a.ckpt), device=a.device)
    print("[recognizer] " + rec.debug())
    chars_model = list(getattr(rec.charset, "chars", []))
    chars_json = list(json.load(io.open(a.charset, encoding="utf-8"))["chars"])
    print("[charset] 模型自带 %d 字 / charset.json %d 字 | 模型独有 %s | json 独有 %s" % (
        len(chars_model), len(chars_json),
        "".join(sorted(set(chars_model) - set(chars_json))),
        "".join(sorted(set(chars_json) - set(chars_model)))))
    chars = set(chars_model if a.charset_mode == "model" else chars_json)
    print("[charset] 本次判据用 %s（%d 字）" % (a.charset_mode, len(chars)))

    recs = [r for r in load_jsonl(a.manifest) if r.get("split") == "real"]
    by_img = collections.OrderedDict()
    for r in recs:
        by_img.setdefault(r["image"], []).append(r)
    keys = sorted(by_img)
    if a.limit:
        keys = keys[: a.limit]

    rows = []
    for k in keys:
        rs = by_img[k]
        with Image.open(k) as im:
            g = im.convert("L")
            preds = rec.read_many(g, [(r["bbox"], r["role"]) for r in rs])
            for r, (pred, _conf) in zip(rs, preds):
                miss = sorted({c for c in r["text"] if c not in chars})
                rows.append((r, pred, miss))

    if a.holdout_from:
        trained = {row_key(r) for r in load_jsonl(a.holdout_from)}
        ho = [x for x in rows if row_key(x[0]) not in trained]
        it = [x for x in rows if row_key(x[0]) in trained]
        if not ho:
            print("[warn] --holdout-from 覆盖了全部 real 框，检查清单是否搞错")
        print("\n=== 训练集重叠：eval real %d 条，其中 in-train %d (%.1f%%)、held-out %d (%.1f%%) ===" % (
            len(rows), len(it), 100.0 * len(it) / max(1, len(rows)), len(ho), 100.0 * len(ho) / max(1, len(rows))))
        report(ho, chars, "held-out（未出现在训练/验证清单里）—— 干净的尺子")
        report(it, chars, "in-train（训练/验证用过，读数必然虚高）", with_missing=False)
    else:
        report(rows, chars, "全量（未按训练集重叠切分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
