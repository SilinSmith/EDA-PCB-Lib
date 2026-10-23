# -*- coding: utf-8 -*-
"""自训练收字：用**交付模型自己的读数**给 200 张官方页面的检测框打标。

动机（§5.356 / §5.361）：页面级 GT 只够做弱配对（框级正确率 56%），而
GT-exact 收字已经在 armB/armC/armD 里吃掉；剩下的域对齐空间是「模型在这批
图上看到的、置信度够高的读法」。这不是新知识，只是把模型自己的判断在
本域上复习一遍（self-training / distillation），必须混入合成锚点语料以免遗忘。

纯度闸（与 `harvest_labels.py` 同源）：
* `marker_frac` ≤ 0.25 —— 红/绿强饱和像素（引脚红点、导线）占比过高就丢；
* `conf` ≥ `--conf`（默认 0.6）；
* 串长 1~13、全部字符在字符表内；
* 同一张图内同一 (role, text) 只保留最靠前的一批框没必要 —— 框级去重就够。

用法：
  python -X utf8 eda-pcb\\tools\\dev\\self_labels.py --limit 3 --dbg
  python -X utf8 eda-pcb\\tools\\dev\\self_labels.py            # 200 张全量
产物：`work/_man_self.jsonl`、`work/_self_stats.json`。
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvest_labels import ROLES_ALL, ROLE_ID, marker_frac, ok_text   # noqa: E402

EDA = Path(__file__).resolve().parents[2]
REPO = EDA.parent
POOL = EDA / "yolo_part" / "Dataset(waitforpreprocess)" / "200_train_cases"

# 逐角色的形状闸：自训练的标签必须「像一个位号/值/脚号/脚名」，否则模型会把
# 图形碎片（`+ ◁ ▷`、红点、箭头）当成高置信度文字喂回去。
import re as _re                                                   # noqa: E402
_ROLE_RE = {
    "designator": _re.compile(r"^[A-Z]{1,3}\d{1,4}[A-Z]?$"),
    "pin_number": _re.compile(r"^\d{1,3}$"),
    "value": _re.compile(r"^\d+(?:\.\d+)?[A-Za-z%]{0,3}$"),
    "pin_name": _re.compile(r"^[A-Z][A-Z0-9_/+.-]{1,11}$"),
}


def role_ok(role: str, text: str) -> bool:
    r = _ROLE_RE.get(role)
    return bool(r and r.match(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(POOL))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="work/_man_self.jsonl")
    ap.add_argument("--stats", default="work/_self_stats.json")
    ap.add_argument("--conf", type=float, default=0.90)
    ap.add_argument("--strict", type=int, default=1, help="1 = 只收符合角色形状的串")
    ap.add_argument("--marker", type=float, default=0.25)
    ap.add_argument("--dbg", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(EDA))
    from stage2.text import TextEngine, detect_texts, imread_unicode  # noqa: E402

    engine = TextEngine()
    root = Path(a.root)
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    if a.limit:
        cases = cases[:a.limit]
    man, stats = [], {}
    for k, cdir in enumerate(cases, 1):
        png = next(iter(sorted(cdir.glob("*.png"))), None)
        if png is None:
            continue
        bgr = imread_unicode(png)
        boxes = detect_texts(bgr)
        reads = engine.read(bgr, boxes, ROLES_ALL)
        for bi, rec in enumerate(reads):
            bb = rec.get("bbox")
            if bb is None:
                continue
            frac = marker_frac(bgr, bb)
            if frac > a.marker:
                continue
            best = None
            for role in ROLES_ALL:
                got = rec["reads"].get(role)
                if not got:
                    continue
                txt, conf = got[0], got[1]
                if not ok_text(txt) or conf < a.conf:
                    continue
                if a.strict and not role_ok(role, txt):
                    continue
                if best is None or conf > best[1]:
                    best = (txt, conf, role)
            if best is None:
                continue
            txt, conf, role = best
            split = "val" if (zlib.crc32(cdir.name.encode("utf-8")) % 10) == 0 else "train"
            man.append(dict(image=str(png), case=cdir.name, src="self", role=role,
                            role_id=ROLE_ID[role], text=txt,
                            bbox=[round(float(v), 1) for v in bb],
                            tier="self", conf=round(float(conf), 3), split=split))
            s = stats.setdefault(role, {"n": 0, "conf": []})
            s["n"] += 1
            s["conf"].append(round(float(conf), 3))
        if a.dbg:
            print("  %-10s boxes %4d 收 %3d" % (cdir.name, len(boxes),
                                               sum(1 for m in man if m["case"] == cdir.name)))
        if k % 10 == 0:
            print("  ... %d/%d 组，累计 %d 条" % (k, len(cases), len(man)))

    (REPO / a.out).write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man), encoding="utf-8")
    tot = {r: {"n": v["n"], "conf_p50": (sorted(v["conf"])[len(v["conf"]) // 2]
                                        if v["conf"] else 0.0)} for r, v in stats.items()}
    (REPO / a.stats).write_text(json.dumps(tot, ensure_ascii=False, indent=2), encoding="utf-8")
    tr = sum(1 for m in man if m["split"] == "train")
    print("\n记录 %d（train %d / val %d）" % (len(man), tr, len(man) - tr))
    print("按角色：", json.dumps(tot, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
