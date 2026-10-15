# -*- coding: utf-8 -*-
"""**贴章元件引脚槽位审计**：`stage2/assemble.py:stamp_pins` 把「stage1 的 4 个
候选取点」映射成 `pin_1`/`pin_2` 两个槽位时，会不会把**两个真脚判成同一个号**、
而把**旁侧幻影脚**判成另一个号。

机制（case 1 的六颗去耦电容，2026-10 实测）：
* `c0008`(C151) 的 stage1 候选取点（图像帧）= `[顶, 右, 底, 左]`；
* `stamp_pins` 里 `side = [F, F, T, F]`（竖长框、`s1=-1`），贪心 `seen` 循环按顺序
  发号 ⇒ `[2, 1, 2, 1]` —— **顶(idx0) 与底(idx2) 都拿到 `2``，右/左幻影才拿到 `1`；
* 于是 `num_of[('c0008',0)]=num_of[('c0008',2)]='2'` ⇒ 两条 stage1 网
  （`net_20` 含 U1 的那条、`net_31` 那条**电源梳**）在 `merge_shared_pins` 里被合并；
  而含幻影脚的 `net_4` 单独留下 ⇒ 交付文档里出现 `net_10(C151.2,...)` +
  六条单成员网 `net_4(C151.1)` …，GT 的 `net_3`（六颗电容的 pin_1 + 电源符号）全军覆没。

本探针只读 `runs/stage1/*.stage1.json`，**不改任何东西**：

    python -X utf8 eda-pcb/tools/diag/_stampchk.py --run work/_fast_out_fin --split 10GT
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from stage2 import assemble as A                                              # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json       # noqa: E402


def one(run: Path, cid: str, verbose=False):
    p = run / "runs" / "stage1" / ("%s.stage1.json" % cid)
    if not p.exists():
        print("  !! 缺 %s" % p)
        return None
    st = load_json(p)
    H = float((st.get("image") or {}).get("h") or 0.0)
    st_pins = st.get("pins") or {}
    stat = collections.Counter()
    rows = []
    for c in st.get("components") or []:
        ctype = str(c.get("type") or "")
        if ctype not in A.PIN_STAMP_TYPES:
            continue
        pts = st_pins.get(c["id"]) or []
        if not len(pts):
            continue
        st_out = A.stamp_pins(c, ctype, pts, H)
        stat["n_comp"] += 1
        stat["n_cand"] += len(pts)
        nums = [n for n, _xy in st_out]
        if len(set(nums)) != len(nums):
            stat["n_collide"] += 1
            rows.append((c["id"], ctype, len(pts), nums))
        # 幻影脚（沿轴投影近 0）抢到号：用 stamp_pins 内部同一套几何复算
        bbox = c.get("bbox_json") or []
        raw = c.get("bbox_raw_img") or []
        if len(bbox) == 4:
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            if len(raw) == 4 and (raw[2] - raw[0]) > 0 and (raw[3] - raw[1]) > 0:
                tall = (raw[3] - raw[1]) >= (raw[2] - raw[0])
            else:
                tall = (bbox[3] - bbox[1]) >= (bbox[2] - bbox[0])
            axis_y = (not tall) if ctype in A.PIN_STAMP_PERP else tall
            for (n, _xy), pt in zip(st_out, pts):
                pr = ((H - float(pt[1])) - cy) if axis_y else (float(pt[0]) - cx)
                if abs(pr) < 0.30 * A.PIN_STAMP_DEFAULT_SPAN:
                    stat["n_phantom_slot"] += 1
                    if verbose:
                        print("     幻影抢号 %s(%s) num=%s |proj|=%.1f"
                              % (c["id"], ctype, n, abs(pr)))
        if verbose:
            print("   %s %-3s %d 取点 -> %s" % (c["id"], ctype, len(pts), nums))
    print("  case %-6s 贴章元件 %d ｜ 取点 %d ｜ **同号冲突 %d** ｜ 幻影抢号 %d"
          % (cid, stat["n_comp"], stat["n_cand"], stat["n_collide"],
             stat["n_phantom_slot"]))
    for r in rows[:8]:
        print("     冲突 %s(%s) 取点%d -> %s" % r)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_fin")
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--case", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_absolute():
        run = (ROOT / run).resolve()
    cases = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split]
    if a.case:
        cases = [c for c in cases if c.case_id == a.case]
    tot = collections.Counter()
    for c in cases[:a.n]:
        s = one(run, c.case_id, a.verbose)
        if s:
            tot.update(s)
    print("\n== 汇总 ==")
    print("  贴章元件 %d ｜ 取点 %d ｜ 同号冲突 %d（%.1f%% 的元件）｜ 幻影抢号 %d"
          % (tot["n_comp"], tot["n_cand"], tot["n_collide"],
             100.0 * tot["n_collide"] / max(1, tot["n_comp"]), tot["n_phantom_slot"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
