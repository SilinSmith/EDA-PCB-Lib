# -*- coding: utf-8 -*-
"""位号到底读不读得出来？—— 量这条链的**天花板**，把「认不出」和「归错元件」分开。

对每个官方 case：GT 里 key 不以 `∅` 开头的元件就是有位号文字的元件。
拿 stage1 产物里已存的文字框，用 `designator` 角色读一遍，问三件事：
  1. 整张图上**有没有**任何一个框读出的串正好等于 GT 位号？（识别天花板）
  2. 那个框离 GT 元件的距离是多少？（几何可及性）
  3. 每个框按「离元件最近」归属之后，有多少元件能拿到正确位号？（当前做法）

    cd Y:\workspace\EDA\eda-pcb
    python Y:\workspace\EDA\work\_des_ceiling.py [--limit 20] [--split 10GT,10_GTcase,200_train_cases]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from stage2.text import TextEngine, read_stage1_texts  # noqa: E402
from stage2.assign import TYPE_PREFIX  # noqa: E402
from schematic_parser.data import discover, load_json  # noqa: E402

S1 = ROOT / "work" / "stage1_all14"
DATA_ROOT = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)"
SPLITS = ("10GT", "10_GTcase", "200_train_cases")


def snap(text: str, prefix: str) -> str:
    """把读出的位号前缀换成期望前缀，其余必须是纯数字：`Rl4`/`Bl4`/`R14` -> `R14`。"""
    import re
    u = re.sub(r"[^0-9A-Za-z]", "", (text or "")).upper()
    m = re.match(r"^([A-Z]{1,4})(\d{1,4})$", u)
    return (prefix + m.group(2)) if m else ""


def gt_index():
    out = {}
    try:
        for c in discover(DATA_ROOT, SPLITS):
            out[str(c.case_id)] = c
    except Exception as exc:
        print("[warn] discover 失败: %r" % (exc,))
    return out


def rank(path: Path) -> str:
    return path.name[: -len(".stage1.json")]


def dist_to_rect(p, r):
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return float(np.hypot(dx, dy))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--split", default="10GT,10_GTcase")
    ap.add_argument("--cases", default="")
    args = ap.parse_args(argv)

    want = {s.strip() for s in args.split.split(",") if s.strip()}
    files = sorted(S1.glob("*.stage1.json"))
    # stage1 产物里的 `case` 是**字符串 id**（不是 dict）⇒ 用源图路径判定 split。
    GT10 = {"0025", "0030", "0036", "0037", "0042", "0059", "0065", "0070", "0099", "0100"}
    picked = []
    for f in files:
        cid = rank(f)
        d = json.loads(f.read_text(encoding="utf-8"))
        p = ((d.get("image") or {}).get("path") or "").replace("/", "\\")
        if "10_GTcase" in p:
            sp = "10_GTcase"
        elif "200_train_cases" in p:
            sp = "200_train_cases"
        elif "\\10GT\\" in p or p.endswith("10GT"):
            sp = "10GT"
        else:
            sp = "10GT" if (cid.isdigit() and int(cid) <= 10 and not cid.startswith("0")) else "?"
        if args.cases:
            if cid in {c.strip() for c in args.cases.split(",") if c.strip()}:
                picked.append((cid, d))
        elif sp in want:
            picked.append((cid, d))
    picked = picked[: args.limit]

    eng = TextEngine()
    print(eng.describe())
    gti = gt_index()

    n_gt = 0          # 有位号文字的 GT 元件
    n_readable = 0    # 全图存在一个框正好读出该位号
    n_nearest_ok = 0  # 「离元件最近的框」就是它
    n_snap = 0        # 前缀矫正后读得出（`Rl4`/`Bl4` -> `R14`）
    n_snap_nearest = 0
    dists = []

    for cid, doc in picked:
        gt = gti.get(cid)
        if gt is None:
            continue
        gdoc = load_json(gt.json_path)
        H = float(((doc.get("image") or {}).get("h")) or 0) or None
        if H is None:
            from stage2.text import imread_unicode
            H = float(imread_unicode((doc.get("image") or {}).get("path")).shape[0])
        gcomps = gdoc.get("components") or {}
        boxes = read_stage1_texts(doc)
        if not boxes:
            continue
        reads = eng.read(_img(doc), boxes, roles=("designator",))
        texts = [r["reads"].get("designator", ("", 0.0))[0].strip() for r in reads]

        got_ok, got_all = 0, 0
        for key, c in gcomps.items():
            key = (key or "").strip()
            if not key or key.startswith("∅"):
                continue
            got_all += 1
            bb = c.get("bbox")
            if not bb:
                continue
            b = (float(bb[0]), H - float(bb[3]), float(bb[2]), H - float(bb[1]))
            hits = [i for i, t in enumerate(texts) if t == key]
            if hits:
                got_ok += 1
                d = min(dist_to_rect(((boxes[i][0] + boxes[i][2]) / 2,
                                      (boxes[i][1] + boxes[i][3]) / 2), b) for i in hits)
                dists.append(d)
            # 前缀矫正臂：把读出的字母前缀换成该类型应有的前缀（其余必须是纯数字）
            pref = TYPE_PREFIX.get(str(c.get("type") or ""), "")
            cand = [pref[0]] if pref else []
            hh = [i for i, t in enumerate(texts)
                  if any(snap(t, p) == key for p in cand)]
            if hh:
                n_snap += 1
            ds = [(dist_to_rect(((x[0] + x[2]) / 2, (x[1] + x[3]) / 2), b), i)
                  for i, x in enumerate(boxes)]
            ds.sort()
            if hits and ds and ds[0][1] in hits:
                n_nearest_ok += 1
            if hh and ds and ds[0][1] in hh:
                n_snap_nearest += 1
        n_gt += got_all
        n_readable += got_ok
        print("  %-26s 有位号元件 %3d  读得出 %3d (%.2f)" % (cid, got_all, got_ok,
                                                            got_ok / max(1, got_all)))

    print("\n合计：有位号文字的 GT 元件 %d" % n_gt)
    print("  ① 识别天花板（全图任一框读出正确位号）  %d  = %.3f" % (n_readable, n_readable / max(1, n_gt)))
    print("  ② 且「离元件最近的框」正好是它          %d  = %.3f" % (n_nearest_ok, n_nearest_ok / max(1, n_readable)))
    print("  ③ 两者同时满足（=当前「最近框」做法的天花板）  %d  = %.3f" % (n_nearest_ok, n_nearest_ok / max(1, n_gt)))
    print("  ④ 前缀矫正后读得出                          %d  = %.3f" % (n_snap, n_snap / max(1, n_gt)))
    print("  ⑤ 前缀矫正 + 最近框是它                      %d  = %.3f" % (n_snap_nearest, n_snap_nearest / max(1, n_gt)))
    if dists:
        d = np.array(dists)
        print("  命中框到元件框距离 p10 %.1f p50 %.1f p90 %.1f" % tuple(np.percentile(d, [10, 50, 90])))
    return 0


def _img(doc):
    from stage2.text import imread_unicode
    return imread_unicode((doc.get("image") or {}).get("path"))


if __name__ == "__main__":
    raise SystemExit(main())
