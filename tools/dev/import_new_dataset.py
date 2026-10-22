# -*- coding: utf-8 -*-
"""把 `新补充训练集2` 的手标 GT 接进 CRNN 训练清单。

为什么需要它
-----------
交付格式是**页面级 GT**（`components{位号:{bbox,value}}` + `pins{位号:{pin_N:{pinname,point}}}`），
坐标系是「像素单位 + 原点在左下（Y 向上）」。训练清单要的是**框级**、且框必须是图像像素
坐标（原点左上）。所以这里做两件事：

1. 只认 `case*` 前缀的组（同事口径：48 组重做，非 case 前缀不用），按 json 的 sha256 去重
   （`dataset/` 里 34 组与 `新补充训练集(1)/` 逐字节相同）；
2. 把 GT 的坐标先搬进「图像像素 / 原点左上」这套框架，再**原样复用**已有的
   `build_manifest_from_gt.py --gt`（弱配对：`stage2.text.detect_texts` 出候选文字框，
   位号/值按「文字框中心到元件框矩形」距离、引脚号按「到引脚点」距离配对）。

坐标变换（实测，见 `docs/DATASET_NEW.md`）
-----------------------------------------
* 像素口径（48 组 case）：`x' = x`，`y' = H - y`（48/48 页导线命中率 0.66~0.81，不翻只有 0.03）；
* 归一化口径（10 组 SCH_，本次不用）：`x' = x*W`，`y' = y*H`（命中 0.70/0.75，翻转反而塌）。

产物
----
    work\\_newdata_stage\\case_XXXX\\...   GT 坐标已改正的临时页目录（--gt 的输入）
    work\\_boxes_new.jsonl               框级中间文件 {case,image,bbox,text,role}
    work\\_man_new.jsonl                 训练清单（字段与 manifest.build_real 一致）

用法
----
    cd Y:\\workspace\\EDA
    python -X utf8 eda-pcb\\tools\\dev\\import_new_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(__file__).resolve().parents[2]                       # eda-pcb/
REPO = EDA.parent                                              # Y:\workspace\EDA
if str(EDA) not in sys.path:
    sys.path.insert(0, str(EDA))
if str(EDA / "tools" / "dev") not in sys.path:
    sys.path.insert(0, str(EDA / "tools" / "dev"))

import build_manifest_from_gt as B                             # noqa: E402  已有转换器
from textrec.manifest import write                             # noqa: E402

DEFAULT_SRC = REPO / "新补充训练集2"
DEFAULT_STAGE = EDA / "work" / "_newdata_stage"
DEFAULT_BOXES = REPO / "work" / "_boxes_new.jsonl"
DEFAULT_OUT = REPO / "work" / "_man_new.jsonl"
CASE_PREFIX = "case"


# --------------------------------------------------------------------------
# 发现 case 组
# --------------------------------------------------------------------------
def discover(root: Path) -> Tuple[List[dict], Counter]:
    """-> ([{case,json,png,sub,sha}], 统计)。json 与同名 png 成对，扁平目录与
    `<NN>/` 这类一层子目录一视同仁（两种都在交付里出现）。"""
    stat: Counter = Counter()
    groups: List[dict] = []
    for jp in sorted(root.rglob("*.json")):
        stat["json"] += 1
        ip = jp.with_suffix(".png")
        if not ip.is_file():
            stat["drop_no_image"] += 1
            continue
        if not jp.stem.startswith(CASE_PREFIX):
            stat["drop_not_case"] += 1
            continue
        rel = jp.parent.relative_to(root)
        sub = rel.parts[0] if rel.parts else "."
        groups.append({"case": jp.stem, "json": jp, "png": ip, "sub": sub,
                       "sha": hashlib.sha256(jp.read_bytes()).hexdigest()})
    return groups, stat


def dedupe(groups: List[dict]) -> Tuple[List[dict], Counter]:
    """同一份 json（sha256 相同）只留一份；优先留非 `dataset` 的原始组。"""
    stat: Counter = Counter()
    keep: Dict[str, dict] = {}
    for g in sorted(groups, key=lambda g: (g["sub"] == "dataset", g["sub"], g["case"])):
        if g["sha"] in keep:
            stat["drop_dup"] += 1
            stat["drop_dup:" + g["sub"]] += 1
            continue
        keep[g["sha"]] = g
    out = sorted(keep.values(), key=lambda g: g["case"])
    names = Counter(g["case"] for g in out)
    clash = [c for c, n in names.items() if n > 1]
    if clash:
        print("  ! case 名重复（内容不同）：%s" % clash)
    return out, stat


# --------------------------------------------------------------------------
# 坐标改正
# --------------------------------------------------------------------------
def to_pixel(doc: dict, W: int, H: int) -> Tuple[dict, str, Counter]:
    """把 doc 里所有 bbox / point 搬到图像像素坐标（原点左上）。返回 (doc', 口径, 统计)。"""
    st: Counter = Counter()
    xs: List[float] = []
    for c in (doc.get("components") or {}).values():
        b = c.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            xs += [float(b[0]), float(b[2])]
    for pl in (doc.get("pins") or {}).values():
        for pv in pl.values():
            pt = pv.get("point") or {}
            if "x" in pt:
                xs.append(float(pt["x"]))
    if not xs:
        return doc, "empty", st
    norm = max(abs(v) for v in xs) <= 1.5
    st["norm" if norm else "pixel"] += 1

    def m(x: float, y: float) -> Tuple[float, float]:
        if norm:
            return x * W, y * H
        return x, H - y

    for c in (doc.get("components") or {}).values():
        b = c.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            x0, y0 = m(float(b[0]), float(b[1]))
            x1, y1 = m(float(b[2]), float(b[3]))
            c["bbox"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    for pl in (doc.get("pins") or {}).values():
        for pv in pl.values():
            pt = pv.get("point")
            if isinstance(pt, dict) and "x" in pt:
                pt["x"], pt["y"] = m(float(pt["x"]), float(pt["y"]))
    return doc, ("norm" if norm else "pixel-yflip"), st


def stage(groups: List[dict], stage_root: Path) -> Tuple[Path, Counter]:
    """写出 --gt 能吃的临时页目录：<stage>/<case>/{<case>.png,<case>.json}。"""
    from PIL import Image
    st: Counter = Counter()
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    for g in groups:
        W, H = Image.open(g["png"]).size
        doc = json.loads(g["json"].read_text(encoding="utf-8-sig"))
        doc, mode, s = to_pixel(doc, W, H)
        st.update(s)
        d = stage_root / g["case"]
        d.mkdir(parents=True, exist_ok=True)
        (d / (g["case"] + ".json")).write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        dst = d / (g["case"] + ".png")
        try:
            dst.hardlink_to(g["png"])
        except OSError:
            shutil.copy2(g["png"], dst)
        g["mode"] = mode
    return stage_root, st


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="新补充训练集2 -> 框级 JSONL -> 训练清单")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--stage", default=str(DEFAULT_STAGE))
    ap.add_argument("--boxes", default=str(DEFAULT_BOXES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--radius", type=float, default=B.__dict__.get("RADIUS", 40.0))
    ap.add_argument("--val-every", type=int, default=B.VAL_EVERY_DEFAULT)
    args = ap.parse_args(argv)

    root = Path(args.src)
    if not root.is_dir():
        print("!! 数据根不存在：%s" % root)
        return 2

    groups, st1 = discover(root)
    print("[发现] %d 组 case 前缀；非 case 跳过 %d；缺图 %d"
          % (len(groups), st1["drop_not_case"], st1["drop_no_image"]))
    for k in sorted(k for k in st1 if k.startswith("subdir:")):
        pass
    per_sub = Counter(g["sub"] for g in groups)
    print("[发现] 分文件夹：%s" % dict(per_sub))

    groups, st2 = dedupe(groups)
    print("[去重] 逐字节重复的副本丢 %d 组（%s）；唯一组 %d"
          % (st2["drop_dup"], dict((k.split(":")[1], v) for k, v in st2.items()
                                  if k.startswith("drop_dup:")), len(groups)))

    stage_root, st3 = stage(groups, Path(args.stage))
    modes = Counter(g["mode"] for g in groups)
    print("[坐标] 口径：%s  ->  stage=%s" % (dict(modes), stage_root))

    recs, info = B.build_from_gt([stage_root], args.radius, args.val_every, "hash",
                                 quiet=True)

    boxes = Path(args.boxes)
    boxes.parent.mkdir(parents=True, exist_ok=True)
    with boxes.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"case": r["case"], "image": r["image"],
                                "bbox": r["bbox"], "text": r["text"],
                                "role": r["role"]}, ensure_ascii=False) + "\n")
    print("[框级] 写出 %d 行 -> %s" % (len(recs), boxes))

    man, n_box = B.build_from_boxes([boxes], [stage_root])
    for r in man:                                  # --boxes 模式恒 train，这里按 case 重切
        r["split"] = B.case_split(r["case"], args.val_every)
    man.sort(key=lambda r: (r["case"], r["bbox"][1], r["bbox"][0]))
    info2 = dict(info, boxes_mode=True, n_box=n_box,
                 n_case=len({r["case"] for r in man}))
    B.summarise(man, info2)

    per_role = Counter(r["role"] for r in man)
    hs: Dict[str, List[float]] = {}
    for r in man:
        hs.setdefault(r["role"], []).append(float(r["bbox"][3]) - float(r["bbox"][1]))
    print("\n[清单] case %d  记录 %d  split=%s"
          % (len({r["case"] for r in man}), len(man), dict(Counter(r["split"] for r in man))))
    print("  %-12s %7s %10s %10s" % ("role", "records", "中位高 px", "p90 高 px"))
    for role in sorted(per_role, key=lambda k: -per_role[k]):
        v = sorted(hs[role])
        print("  %-12s %7d %10.1f %10.1f"
              % (role, per_role[role], v[len(v) // 2], v[int(0.9 * (len(v) - 1))]))
    write(man, Path(args.out))
    print("[提示] 训练：$env:PYTHONPATH='%s'; python -X utf8 -m textrec.train --manifest %s "
          "--smoke --device cpu --limit 500 --out %s"
          % (EDA, args.out, EDA / "work" / "_train_new_smoke"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
