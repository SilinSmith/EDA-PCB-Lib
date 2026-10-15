# -*- coding: utf-8 -*-
"""按 sheet 留出，生成「全人口 vs 旧规则」两臂的真实训练清单 + 干净 ruler 清单（§5.161）。

为什么需要它：评测用的 `textrec/data/manifest.jsonl`（`split=="real"`）里，`value` 人口
3,160 条中 **2,734 条（86.5%）** 来自 `CVJsonStd` 的 `texts[].details` 为空/`searchName`
的框，而训练清单 `work/ocr_real/manifest.jsonl` 由 `tools/data/ocr_manifest.py:90` 把这类
框一律当 `note` 丢弃 ⇒ 训练侧只有 426 条同种群 `value`。32 张 sheet 全都进过旧训练，所以
现交付模型没有干净的 sheet 基线；干净比较只能来自「同一 recipe 的两臂」，ruler 必须
**按 sheet 整体留出**。

产出（默认 `Y:\\workspace\\EDA\\work\\fullpop\\`）：
  ruler.jsonl            ruler sheet 上全部行（`split="real"`）→ 直接喂
                         `python -X utf8 -m textrec.eval_real --ckpt <候选> --manifest ruler.jsonl`
  control/real.jsonl     训练 sheet 上「旧规则」行（也出现在 `work/ocr_real/manifest.jsonl` 里）+ 公共 val
  treatment/real.jsonl   训练 sheet 上「全部」行 + 同一份公共 val
两臂的 val 完全相同（取自训练 sheet，绝不取 ruler），保证早停口径一致。

接下来（每臂）：
  python -X utf8 -m tools.dev.mix_crnn_manifest --real work/fullpop/<臂>/real.jsonl \
      --out work/fullpop/<臂> --charset-from <起点 ckpt>
  python -X utf8 -m textrec.train --resume <起点 ckpt> --out work/fullpop/<臂>/run \
      --manifest work/fullpop/<臂>/manifest.jsonl      # --charset 留空（§5.160 护栏）
判据（先写死）：ruler 上 `value` 的 exact 提升，且 5+ 字 exact 不降、其它 role 不降超过 1pp。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
WORK = EDA.parent / "work"
DEFAULT_MAN = EDA / "textrec" / "data" / "manifest.jsonl"
DEFAULT_REAL = WORK / "ocr_real" / "manifest.jsonl"
DEFAULT_OUT = WORK / "fullpop"


def key(image: str, bbox) -> tuple:
    return (os.path.basename(image).lower(),
            tuple(round(float(v), 1) for v in bbox))


def comp(name: str, rows: list) -> None:
    c = collections.Counter(r["role"] for r in rows)
    buck = collections.Counter()
    for r in rows:
        n = len(r["text"])
        buck["5+" if n >= 5 else ("3-4" if n >= 3 else "1-2")] += 1
    print("  %-10s n=%-6d %s | 长度 %s" % (name, len(rows), dict(c), dict(buck)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MAN), help="评测清单（含 split=='real' 行）")
    ap.add_argument("--real", default=str(DEFAULT_REAL), help="旧训练清单（定义「旧规则」人口）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--ruler-every", type=int, default=6, help="每 N 张 sheet 抽 1 张当 ruler")
    ap.add_argument("--val", type=int, default=800, help="公共 val 行数（按 role 分层）")
    ap.add_argument("--long-min", type=int, default=5, dest="long_min",
                    help="第三臂 treatment_long 只补长度 >= 此值的增量行")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    real = [json.loads(ln) for ln in Path(a.manifest).open(encoding="utf-8")
            if json.loads(ln).get("split") == "real"]
    old_keys = {key(d["image"], d["bbox"])
                for d in (json.loads(ln) for ln in Path(a.real).open(encoding="utf-8"))}

    sheets = sorted({r["case"] for r in real})
    ruler = {s for i, s in enumerate(sheets) if i % max(1, a.ruler_every) == 0}
    print("sheet 共 %d：ruler %d 张 = %s" % (len(sheets), len(ruler), sorted(ruler)))
    print("训练 sheet %d 张" % (len(sheets) - len(ruler)))

    is_old = lambda r: key(r["image"], r["bbox"]) in old_keys
    ruler_rows = [r for r in real if r["case"] in ruler]
    tr_rows = [r for r in real if r["case"] not in ruler]
    ctrl_rows = [r for r in tr_rows if is_old(r)]

    # 公共 val：从训练 sheet 全人口按 role 分层抽，两臂用同一份
    rnd = random.Random(a.seed)
    by_role = collections.defaultdict(list)
    for r in sorted(tr_rows, key=lambda r: (r["image"], tuple(r["bbox"]), r["text"])):
        by_role[r["role"]].append(r)
    per = max(1, a.val // max(1, len(by_role)))
    val_rows, rest = [], []
    for role in sorted(by_role):
        rs = by_role[role]
        val_rows += rs[:per]
        rest += rs[per:]
    if len(val_rows) < a.val:
        need = a.val - len(val_rows)
        val_rows += rest[:need]
        rest = rest[need:]
    rnd.shuffle(val_rows)
    val_ids = {id(r) for r in val_rows}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list, is_val) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                d = dict(r)
                d["split"] = "val" if is_val(r) else ("real" if path.name == "ruler.jsonl" else "train")
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    dump(out / "ruler.jsonl", ruler_rows, lambda r: False)
    ctrl_tr = [r for r in ctrl_rows if id(r) not in val_ids]
    treat_tr = [r for r in tr_rows if id(r) not in val_ids]
    # 第三臂：只补「长度 >= long_min」的增量行。动机见 RESULTS §5.163：模型是 role 条件模型
    # （role_emb 32 维拼进 LSTM 每一步），而全人口的增量行里有大量被「空 details → value」规则
    # 误标的网标/脚号/页面家具（GND、裸数字、嘉立创EDA…），会污染 value 的 role 嵌入。
    long_extra = [r for r in tr_rows if id(r) not in val_ids and not is_old(r)
                  and len(str(r["text"])) >= a.long_min]
    long_tr = ctrl_tr + long_extra
    dump(out / "control" / "real.jsonl", ctrl_tr + val_rows, lambda r: id(r) in val_ids)
    dump(out / "treatment" / "real.jsonl", treat_tr + val_rows, lambda r: id(r) in val_ids)
    dump(out / "treatment_long" / "real.jsonl", long_tr + val_rows, lambda r: id(r) in val_ids)

    print("\nruler.jsonl（干净尺子，split=real；喂 textrec.eval_real --manifest）")
    comp("ruler", ruler_rows)
    print("control/real.jsonl（旧规则人口）")
    comp("train", ctrl_tr)
    comp("val", val_rows)
    print("treatment/real.jsonl（训练 sheet 上全部框）")
    comp("train", treat_tr)
    comp("val", val_rows)
    print("treatment_long/real.jsonl（control + 只补长度 >= %d 的增量行）" % a.long_min)
    comp("train", long_tr)
    comp("val", val_rows)
    print("\nvalue 人口：control train %d  vs  treatment train %d  vs  treatment_long train %d"
          % (sum(1 for r in ctrl_tr if r["role"] == "value"),
             sum(1 for r in treat_tr if r["role"] == "value"),
             sum(1 for r in long_tr if r["role"] == "value")))
    print("增量行：treatment %d 行，其中 treatment_long 只取 %d 行（长度 >= %d）"
          % (len(treat_tr) - len(ctrl_tr), len(long_extra), a.long_min))
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
