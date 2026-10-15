# -*- coding: utf-8 -*-
"""把「真实语料」和「合成语料里挑出来的那几类」混成一份 CRNN 训练清单。

为什么需要它
------------
出厂 `eda-pcb/weights/textrec_best.pt`（= `work/ocr_real/runs/best.pt`）是**只在
真实语料上**训出来的：`work/ocr_real/train.log` 头部写得很清楚——

    [charset] 71 chars + blank = 72 classes
    [data] train 2977 (丢弃未知字符 56)   val 785 (丢弃 11)
    [model] CRNN 参数 7.76M  类别 72      13 s/epoch   best CER 0.1871

也就是说它见过 2977 张真实裁片 111 遍，但**没见过合成语料**。而合成语料
（`textrec/data/manifest.jsonl`，366k 条）里有 87k 条 `pin_number`——线上最痛的
那一类（3 位号被 CTC 吞掉一位：`113`→`13`）。反过来，纯合成训出来的模型在真实
图上泛化不了（域差），所以两边都不能扔。

这份脚本做的是：**train = 真实 2977 条全要 + 从合成里按角色/长度挑一批**，
**val 固定沿用真实的 785 条**（同一把尺子，才能和出厂模型比）。

挑法（都是确定性采样，同 seed 同结果）
--------------------------------------
* `pin_number`：按文本长度分层（1/2/3 位，>3 并入 3），每层默认 4000 条 ⇒ 直接
  把「2 位占 79%」的先验扳平（合成里 1/2/3 位是 8909/70694/7660）。
  这是 `textrec/train.py --pin-num-balance equal` 的「数据侧」等价物，但只挑不复制，
  总步数可控。
* 其它角色按 `--designator / --value / --pin-name / --net-label` 各取上限。
* 合成记录一律带 `split="train"`；`src` 字段保留原值（`synth`/`cvjson`）便于查来源。

用法
----
    cd eda-pcb
    python -X utf8 tools\\dev\\mix_crnn_manifest.py --out Y:\\workspace\\EDA\\work\\crnn_mix
    # 顺带把 resume 用的字符表也导出（出厂模型是 71 字，与现在 data/charset.json 的 67 字不同）
    python -X utf8 tools\\dev\\mix_crnn_manifest.py --out ... --charset-from Y:\\workspace\\EDA\\work\\ocr_real\\runs\\last.pt

产物：`<out>/manifest.jsonl`（+ `charset.json`，若给了 `--charset-from`）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve()
EDA = HERE.parents[2]                      # eda-pcb/
DEFAULT_SYNTH = EDA / "textrec" / "data" / "manifest.jsonl"
DEFAULT_REAL = Path(r"Y:\workspace\EDA\work\ocr_real\manifest.jsonl")


def load_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def take_stratified(recs: List[dict], per_bucket: int, seed: int,
                    max_bucket: int = 3) -> List[dict]:
    """按 `len(text)` 分层，每层最多取 per_bucket 条（确定性）。"""
    buckets: Dict[int, List[dict]] = {}
    for r in recs:
        k = min(max(1, len(str(r.get("text") or ""))), max_bucket)
        buckets.setdefault(k, []).append(r)
    rng = random.Random(seed)
    out: List[dict] = []
    for k in sorted(buckets):
        pool = sorted(buckets[k], key=lambda r: (r["image"], tuple(r["bbox"]), r["text"]))
        if per_bucket and len(pool) > per_bucket:
            pool = rng.sample(pool, per_bucket)
        out.extend(pool)
    return out


def take_plain(recs: List[dict], n: int, seed: int) -> List[dict]:
    pool = sorted(recs, key=lambda r: (r["image"], tuple(r["bbox"]), r["text"]))
    if n and len(pool) > n:
        pool = random.Random(seed).sample(pool, n)
    return pool


def dump_charset(ckpt: Path, out: Path) -> None:
    """把 checkpoint 自带的字符表导成 `Charset.load` 能读的 json。

    必须这么做：出厂模型是 **71 字**（含 `Ω ∅ / %`），而现在的
    `textrec/data/charset.json` 是 **67 字**（加了 `丝 螺`、去掉了 `Ω / %`）。
    拿 67 字的表去 `--resume` 出厂模型，输出头对不上。
    """
    import torch
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    chars = list(ck["charset"])
    (out / "charset.json").write_text(
        json.dumps({"chars": chars,
                    "meta": {"from": str(ckpt), "n": len(chars),
                             "note": "导出自 checkpoint 自带字符表"}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("[charset] 从 %s 导出 %d 字 -> %s" % (ckpt.name, len(chars), out / "charset.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="混一份 CRNN 训练清单（真实 + 合成挑样）")
    ap.add_argument("--synth", default=str(DEFAULT_SYNTH))
    ap.add_argument("--real", default=str(DEFAULT_REAL))
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--pin-per-len", type=int, default=4000,
                    help="pin_number 每个长度桶取多少条（默认 4000）")
    ap.add_argument("--designator", type=int, default=4000)
    ap.add_argument("--value", type=int, default=3000)
    ap.add_argument("--pin-name", type=int, default=2000, dest="pin_name")
    ap.add_argument("--net-label", type=int, default=1000, dest="net_label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--charset-from", default="", help="从该 ckpt 导出 charset.json")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    synth = load_jsonl(Path(args.synth))
    real = load_jsonl(Path(args.real))

    # 合成清单里混着 src=="real" 的真实记录（那是给 eval_real 用的），这里要排掉，
    # 否则真实样本会被算两遍。
    syn_train = [r for r in synth if r.get("split") == "train" and r.get("src") != "real"]
    real_train = [r for r in real if r.get("split") == "train"]
    real_val = [r for r in real if r.get("split") == "val"]
    print("[in] synth train %d（已排掉 src=real 的 %d 条）" %
          (len(syn_train), len([r for r in synth if r.get("src") == "real"])))
    print("[in] real  train %d  val %d" % (len(real_train), len(real_val)))

    by_role: Dict[str, List[dict]] = {}
    for r in syn_train:
        by_role.setdefault(r["role"], []).append(r)

    picked: List[dict] = []
    picked += take_stratified(by_role.get("pin_number", []), args.pin_per_len, args.seed)
    for role, n in (("designator", args.designator), ("value", args.value),
                    ("pin_name", args.pin_name), ("net_label", args.net_label)):
        picked += take_plain(by_role.get(role, []), n, args.seed)

    for r in picked:
        r["split"] = "train"
    train = real_train + picked
    train.sort(key=lambda r: (r["image"], tuple(r["bbox"])))
    val = sorted(real_val, key=lambda r: (r["image"], tuple(r["bbox"])))

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in train + val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def cnt(rs: List[dict]) -> str:
        c = Counter(r["role"] for r in rs)
        return " ".join("%s %d" % (k, c[k]) for k in sorted(c))

    print("[out] train %d （真实 %d + 合成 %d）" % (len(train), len(real_train), len(picked)))
    print("        真实部分: " + cnt(real_train))
    print("        合成部分: " + cnt(picked))
    pn = [r for r in picked if r["role"] == "pin_number"]
    print("        pin_number 长度: " + str(dict(sorted(Counter(
        min(len(str(r["text"])), 3) for r in pn).items()))))
    print("[out] val %d（沿用真实 785，同一把尺子）" % len(val))
    print("[out] " + str(out / "manifest.jsonl"))

    if args.charset_from:
        dump_charset(Path(args.charset_from), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
