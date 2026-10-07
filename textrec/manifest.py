# -*- coding: utf-8 -*-
"""把合成语料与真实语料的文字标注汇总成一份「裁剪清单」。

输入
----
合成侧 `合成原理图数据集/texts.jsonl`，每行一个 case::

    {"case": "synth_0001",
     "texts": [{"role": "pin_number", "content": "18", "comp": "U1",
                "bbox": [937.13, 592.67, 944.8, 597.0]}, ...]}

`bbox` 是**左上角原点**的图像像素（`synth/emit.py` 已经翻好轴）。

真实侧 Dataset300 的 `<id>-<source>_CVJsonStd.json`，文字在顶层 `texts` 字典里
（键是 `T_1`/`T_2`…，必须遍历 `.values()`）::

    {"content": "R1", "details": "designator", "category": "params",
     "location": {"x1": 12.0, "y1": 30.0, "x2": 22.0, "y2": 36.0}}

产物 `textrec/data/manifest.jsonl`，每行::

    {"image": <绝对路径>, "case": "synth_0001", "src": "synth",
     "split": "train"|"val"|"real", "role": "pin_number", "role_id": 3,
     "text": "18", "bbox": [x1, y1, x2, y2]}

按 case 排序，这样训练时同一个 case 的裁剪会连续命中同一张图的缓存。

划分
----
* 合成：**按 case 抽 val**（`case 序号 % VAL_EVERY == 0`），绝不按裁剪抽——
  同一张图里的文字高度相关，按裁剪抽会泄漏。
* 真实：全部进 `split="real"`，只用于 `eval_real`，**绝不参与训练**。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from . import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------
# 真实语料的 role 映射
# --------------------------------------------------------------------------
# CVJsonStd 的 `details` 字段是自由文本，实测见过 designator / pinNumber / pinName 等。
_REAL_ROLE_MAP = {
    "designator": "designator",
    "designatorname": "designator",
    "pinnumber": "pin_number",
    "pinnum": "pin_number",
    "pinname": "pin_name",
    "netlabel": "net_label",
    "netname": "net_label",
    "params": "value",
    "param": "value",
    "value": "value",
}


def _real_role(details: str, category: str) -> str:
    for key in (details, category):
        if not key:
            continue
        r = _REAL_ROLE_MAP.get(str(key).strip().lower().replace("_", ""))
        if r:
            return r
    return "note"


def _loc_to_bbox(loc) -> "list | None":
    """`location` 可能是 dict（x1/y1/x2/y2 或 left/top/right/bottom）或 4 元素列表。"""
    if loc is None:
        return None
    if isinstance(loc, (list, tuple)) and len(loc) >= 4:
        try:
            return [float(loc[0]), float(loc[1]), float(loc[2]), float(loc[3])]
        except (TypeError, ValueError):
            return None
    if isinstance(loc, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(k in loc for k in keys):
            try:
                return [float(loc[k]) for k in keys]
            except (TypeError, ValueError):
                return None
        alt = ("left", "top", "right", "bottom")
        if all(k in loc for k in alt):
            try:
                l, t, r, b = (float(loc[k]) for k in alt)
                return [l, t, r, b]
            except (TypeError, ValueError):
                return None
    return None


def _ok(bbox) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1 and (x2 - x1) >= 2 and (y2 - y1) >= 2


_DROP_LONG = [0]


def _ok_text(text) -> bool:
    if text is None:
        return False
    n = len(str(text))
    if n == 0:
        return False
    if n > C.MAX_TEXT_LEN:
        _DROP_LONG[0] += 1
        return False
    return True


# --------------------------------------------------------------------------
# 合成侧
# --------------------------------------------------------------------------
def _synth_split(case: str) -> str:
    try:
        idx = int(case.rsplit("_", 1)[-1])
    except ValueError:
        return "train"
    return "val" if idx % C.VAL_EVERY == 0 else "train"


def build_synth(texts_path: Path = None, images_dir: Path = None,
                limit_cases: int = 0) -> List[dict]:
    texts_path = Path(texts_path or C.SYNTH_TEXTS)
    images_dir = Path(images_dir or C.SYNTH_IMAGES)
    if not texts_path.is_file():
        raise FileNotFoundError("找不到合成文字标注: %s" % texts_path)

    out: List[dict] = []
    n_case = 0
    with texts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            case = rec.get("case")
            if not case:
                continue
            img = images_dir / ("%s.png" % case)
            if not img.is_file():
                continue
            n_case += 1
            if limit_cases and n_case > limit_cases:
                break
            split = _synth_split(case)
            for t in rec.get("texts", []):
                bbox = t.get("bbox")
                text = t.get("content")
                role = t.get("role") or "note"
                if not _ok_text(text) or not _ok(bbox):
                    continue
                out.append({
                    "image": str(img),
                    "case": case,
                    "src": "synth",
                    "split": split,
                    "role": role,
                    "role_id": C.ROLE_ID.get(role, C.N_ROLES - 1),
                    "text": text,
                    "bbox": [round(float(v), 2) for v in bbox],
                })
    print("[manifest] 合成 %d 个 case -> %d 条裁剪" % (n_case, len(out)))
    return out


# --------------------------------------------------------------------------
# 真实侧
# --------------------------------------------------------------------------
def build_real(root: Path = None) -> List[dict]:
    root = Path(root) if root else C.DATASET300
    if root is None or not root.is_dir():
        print("[manifest] 跳过真实侧：找不到 Dataset300 目录")
        return []

    out: List[dict] = []
    for jp in sorted(root.glob("*/*_CVJsonStd.json")):
        img = None
        stem = jp.name[: -len("_CVJsonStd.json")]
        for ext in (".png", ".jpg", ".jpeg", ".PNG"):
            cand = jp.parent / (stem + ext)
            if cand.is_file():
                img = cand
                break
        if img is None:
            continue
        try:
            doc = json.loads(jp.read_text(encoding="utf-8-sig"))
        except Exception as exc:                       # noqa: BLE001
            print("  ! 读不动 %s: %s" % (jp.name, exc))
            continue
        texts = doc.get("texts") or {}
        if isinstance(texts, dict):
            texts = list(texts.values())
        case = stem
        for t in texts:
            if not isinstance(t, dict):
                continue
            content = t.get("content")
            bbox = _loc_to_bbox(t.get("location"))
            if not _ok_text(content) or not _ok(bbox):
                continue
            role = _real_role(t.get("details"), t.get("category"))
            out.append({
                "image": str(img),
                "case": case,
                "src": "real",
                "split": "real",
                "role": role,
                "role_id": C.ROLE_ID.get(role, C.N_ROLES - 1),
                "text": str(content),
                "bbox": [round(float(v), 2) for v in bbox],
            })
    n_cases = len({r["case"] for r in out})
    print("[manifest] 真实 %d 个 case -> %d 条裁剪" % (n_cases, len(out)))
    return out


# --------------------------------------------------------------------------
# 落盘 / 读取
# --------------------------------------------------------------------------
def write(records: Iterable[dict], path: Path = None) -> Path:
    path = Path(path or C.MANIFEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print("[manifest] 写出 %d 行 -> %s" % (n, path))
    return path


def load(path: Path = None, splits: "tuple | None" = None) -> List[dict]:
    path = Path(path or C.MANIFEST)
    out: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if splits and r.get("split") not in splits:
                continue
            out.append(r)
    return out


def charset_counter(records: Iterable[dict], charset=None) -> Counter:
    """统计字符频次。给了 charset 就只统计表内字符（用于 val 口径一致）。"""
    c: Counter = Counter()
    for r in records:
        c.update(r["text"])
    return c


def summarise(records: List[dict]) -> None:
    by_split: Dict[str, int] = Counter(r["split"] for r in records)
    by_role: Dict[str, Counter] = {}
    lens: Dict[str, List[int]] = {}
    for r in records:
        by_role.setdefault(r["split"], Counter())[r["role"]] += 1
        lens.setdefault(r["role"], []).append(len(r["text"]))
    print("\n[manifest] 汇总")
    print("  split:", dict(by_split))
    for split in sorted(by_role):
        print("  %-6s roles: %s" % (split, dict(by_role[split])))
    print("  %-12s %8s %8s %8s %8s" % ("role", "n", "len_p50", "len_p90", "len_max"))
    for role in sorted(lens):
        v = sorted(lens[role])
        p = lambda q: v[min(len(v) - 1, int(len(v) * q))]      # noqa: E731
        print("  %-12s %8d %8d %8d %8d" % (role, len(v), p(.5), p(.9), max(v)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="建 textrec 裁剪清单与字符表")
    ap.add_argument("--out", default=str(C.MANIFEST))
    ap.add_argument("--charset", default=str(C.CHARSET_PATH))
    ap.add_argument("--min-freq", type=int, default=C.DEFAULTS["min_freq"])
    ap.add_argument("--limit-cases", type=int, default=0,
                    help="只用前 N 个合成 case（冒烟用）")
    ap.add_argument("--no-real", action="store_true", help="不收集真实侧")
    ap.add_argument("--stats-only", action="store_true", help="只读现有清单做统计")
    args = ap.parse_args(argv)

    if args.stats_only:
        recs = load(Path(args.out))
        summarise(recs)
        return 0

    recs = build_synth(limit_cases=args.limit_cases)
    if not args.no_real:
        recs += build_real()

    # 按 case 排序，保证同一张图的裁剪在训练时连续命中缓存。
    recs.sort(key=lambda r: (r["case"], r["bbox"][1], r["bbox"][0]))

    write(recs, Path(args.out))
    summarise(recs)
    if _DROP_LONG[0]:
        print("  因超过 %d 字符丢弃: %d 条" % (C.MAX_TEXT_LEN, _DROP_LONG[0]))

    train = [r for r in recs if r["split"] == "train"]
    counter = charset_counter(train)
    cs = Charset_from(counter, args.min_freq)
    cs.save(Path(args.charset))
    print("\n[charset] " + cs.describe())
    if cs.meta.get("dropped"):
        print("  丢弃的低频字符: %s" % cs.meta["dropped"])

    # 抽查一下有多少训练样本会因为字符表而作废。
    bad = sum(1 for r in train if cs.try_encode(r["text"]) is None)
    print("  因字符表丢弃的训练样本: %d / %d (%.3f%%)"
          % (bad, len(train), 100.0 * bad / max(1, len(train))))
    return 0


def Charset_from(counter, min_freq):
    """小包装，避免在文件头就把 charset 模块拖进来。"""
    from .charset import Charset
    return Charset.from_corpus(counter, min_freq=min_freq)


if __name__ == "__main__":
    raise SystemExit(main())
