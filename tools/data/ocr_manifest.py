"""把 Dataset300 的 CVJsonStd 变成 textrec 能吃的真实语料清单。

为什么它能用：实测 `texts[].location` 的框和图纸**像素级对齐**
（见 `Y:\workspace\EDA\work\preview\cvjson_align_0059.png`，22 个框全部严丝合缝），
而且 `content` 就是画在纸上的字。所以这是**精确框 + 正确内容**的真实语料，
比从官方 GT 最近邻猜框干净得多。

跑法（CWD = Y:\\workspace\\EDA\\eda-pcb）：
    $env:PYTHONIOENCODING='utf-8'
    python Y:\\workspace\\EDA\\work\\ocr_manifest.py
产物：Y:\\workspace\\EDA\\work\\ocr_real\\manifest.jsonl
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from textrec import config as cfg

sys.stdout.reconfigure(encoding="utf-8")


def _args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--val-every", type=int, default=5)
    return ap.parse_args()


ARGS = _args()

ROOT = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset300我不知道干嘛用的先放置"
OUT = r"Y:\workspace\EDA\work\ocr_real"

ROLE_MAP = {
    "designator": "designator",
    "value": "value",
    "pinName": "pin_name",
    "pinname": "pin_name",
    "pinNumber": "pin_number",
    "pinnumber": "pin_number",
    "netlabel": "net_label",
    "netLabel": "net_label",
}

VAL_SHEETS = set()   # 运行时按 case 列表填充


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*_CVJsonStd.json")))
    cases = []
    for f in files:
        d = os.path.dirname(f)
        cid = os.path.basename(d)
        pngs = (glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
        if not pngs:
            print(f"  !! {cid} 没有图，跳过")
            continue
        cases.append((cid, f, pngs[0]))
    cases.sort()
    print(f"可用 case {len(cases)}")

    # 按 case 抽 val（每 5 个抽 1 个），避免同图文字泄漏
    for k, (cid, _, _) in enumerate(cases):
        if k % 5 == 0:
            VAL_SHEETS.add(cid)
    print(f"val case {len(VAL_SHEETS)}: {sorted(VAL_SHEETS)}")

    rows = []
    for cid, jf, img in cases:
        j = json.load(open(jf, encoding="utf-8"))
        split = "val" if cid in VAL_SHEETS else "train"
        for tid, t in (j.get("texts") or {}).items():
            text = (t.get("content") or "").strip()
            if not text or len(text) > cfg.MAX_TEXT_LEN:
                continue
            loc = t.get("location") or {}
            try:
                bbox = [float(loc["x1"]), float(loc["y1"]), float(loc["x2"]), float(loc["y2"])]
            except Exception:
                continue
            if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 3:
                continue
            role = ROLE_MAP.get((t.get("details") or "").strip(), "note")
            if role == "note":
                continue      # 说明性长文本既不是位号也不是参数，且把字符表撑到 425
            rows.append({
                "image": img, "case": cid, "src": "cvjson", "split": split,
                "role": role, "role_id": cfg.ROLE_ID[role],
                "text": text, "bbox": [round(v, 2) for v in bbox],
            })

    p = os.path.join(OUT, "manifest.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from textrec.charset import Charset

    counter = Counter(ch for r in rows for ch in r["text"])
    # 语料只有 3800 条，按频次砍会把 J / Z / 小写字母这些**合法但少见**的字符一起砍掉，
    # 所以强制保留一份基础 ASCII（位号、引脚名、参数值都在这套字符里），
    # 再靠 min_freq 把汉字和装饰符号挡在外面。
    BASE = ("0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "_-.+/%∅")
    cs = Charset.from_corpus(counter, min_freq=ARGS.min_freq, extra=BASE)
    cs.save(os.path.join(OUT, "charset.json"))
    n_bad = sum(1 for r in rows if cs.try_encode(r["text"]) is None)
    print(f"\n字符表 {cs.num_classes} 类（min_freq={ARGS.min_freq}）"
          f"  被丢弃字符 {len(cs.meta['dropped'])}  不可编码样本 {n_bad}")

    print(f"\n写出 {p}: {len(rows)} 条")
    print("  split:", dict(Counter(r["split"] for r in rows)))
    print("  role :", dict(Counter(r["role"] for r in rows)))
    by = defaultdict(Counter)
    for r in rows:
        by[r["split"]][r["role"]] += 1
    for s in sorted(by):
        print(f"  {s:<6}", dict(by[s]))
    chars = Counter(ch for r in rows for ch in r["text"])
    print(f"  字符集 {len(chars)}: {''.join(sorted(chars))}")
    print(f"  频次<20 的字符: {sorted(c for c, n in chars.items() if n < 20)}")
    cs = Counter()
    for r in rows:
        cs[(r["case"], r["split"])] += 1
    print("  每 case 条数 前 8:", sorted(cs.items())[:8])


if __name__ == "__main__":
    main()
