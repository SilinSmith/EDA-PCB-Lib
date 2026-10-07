"""GT 的 `box`（大 IC）为什么键（位号）对不上：**没检出** 还是 **检出但位号错**？

判据链（`pin_f1` 的口径，`schematic_parser/metrics.py:426-464`）：

* `pin_f1` 只算 GT 里 `canon_type(type) in IC_TYPES = {box, ic, boxic}` 的元件；
* 配对键 = **元件键**，先按身份配、不按几何配 —— 键不对上的元件，它**全部**引脚下游
  只能记成 fn，我们挂在别的键下的引脚连 fp 都不记（`flat()` 过滤掉了）。
* 19 张定点档实测：GT box 引脚 1211 里 **628（51.9%）** 死在键没对上。

所以必须分清两种情形，修法完全不同：

  * **A 类（检出但键错）**：GT 的框位置上有我们的预测元件，只是键（位号）不一样
    ⇒ 修 **stage2 的位号绑定 / LLM 读位号**。
  * **B 类（根本没检出）**：GT 的框位置上一片空白
    ⇒ 修 **stage1 检测**（`bigic.py` 的 `MIN_SIDE=60` 等闸门）。

用法：
  python work\\tools\\diag\\_boxkey.py --dir work\\_fast_out
  python work\\tools\\diag\\_boxkey.py --dir work\\_fast_out --show 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import data as D                                    # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, assign_components, canon_type, normalise_doc, rewrite_keys,
)


def _bbox(v):
    b = (v or {}).get("bbox") or (v or {}).get("box")
    if not b or len(b) < 4:
        return None
    x1, y1, x2, y2 = [float(t) for t in b[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def find_pred(cid, pdir):
    for cand in (pdir / f"{cid}.json",
                 pdir / "final_results" / f"{cid}_Pred.json",
                 pdir / "runs" / "fusion" / f"{cid}.json"):
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    want = {x.strip() for x in a.cases.split(",") if x.strip()} or None
    cases = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)
             if c.json_path and (not want or c.case_id in want)}
    pdir = Path(a.dir)
    cfg = EvalConfig()

    tot = {"gt": 0, "matched": 0, "A": 0, "B": 0, "A_bert": 0}
    rows = []
    for cid, case in sorted(cases.items()):
        f = find_pred(cid, pdir)
        if not f:
            continue
        gt_n = normalise_doc(D.load_json(case.json_path))
        pred_n = normalise_doc(json.loads(f.read_text(encoding="utf-8")))
        _c, keymap = assign_components(gt_n, pred_n, cfg, {})
        pred2 = rewrite_keys(pred_n, keymap)

        boxk = [k for k, v in gt_n["components"].items()
                if canon_type((v or {}).get("type")) in D.IC_TYPES]
        mapped = set(keymap.values())
        pboxes = [(k, _bbox(v)) for k, v in pred2["components"].items()]
        pboxes = [(k, b) for k, b in pboxes if b]
        allp = [(k, _bbox(v)) for k, v in pred_n["components"].items()]
        allp = [(k, b) for k, b in allp if b]

        for k in boxk:
            tot["gt"] += 1
            if k in mapped:
                tot["matched"] += 1
                continue
            gb = _bbox(gt_n["components"].get(k))
            if not gb:
                continue
            hit = sorted(((round(_iou(gb, b), 3), pk) for pk, b in pboxes
                          if _iou(gb, b) > 0.05), reverse=True)
            if len(hit) == 0:                      # 连重叠都没有 ⇒ 没检出
                # 放宽：中心落在 GT 框里的原始预测（未改写键）
                cx, cy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                inside = [k2 for k2, b2 in allp
                          if b2[0] <= cx <= b2[2] and b2[1] <= cy <= b2[3]]
                tot["B"] += 1
                rows.append((cid, k, "B 未检出", gb, inside[:4], []))
            else:
                tot["A"] += 1
                rows.append((cid, k, "A 检出但键错", gb, [], hit[:4]))

    print("=== GT box 元件键对账（%s） ===" % pdir)
    print("GT box 元件 %d · 键对上 %d (%.1f%%) · 键没对上 %d"
          % (tot["gt"], tot["matched"],
             100.0 * tot["matched"] / max(1, tot["gt"]), tot["gt"] - tot["matched"]))
    print("  A 类（框位置上有我们的预测，只是键不同）: %d  (%.1f%% of 没对上)"
          % (tot["A"], 100.0 * tot["A"] / max(1, tot["A"] + tot["B"])))
    print("  B 类（框位置上一片空白 ⇒ 根本没检出）   : %d  (%.1f%%)"
          % (tot["B"], 100.0 * tot["B"] / max(1, tot["A"] + tot["B"])))
    print()
    for r in rows[:max(0, a.show)]:
        cid, k, tag, gb, inside, hit = r
        w, h = gb[2] - gb[0], gb[3] - gb[1]
        print("  %-6s %-10s %s  GT框 %.0fx%.0f" % (cid, k, tag, w, h))
        if inside:
            print("        中心落在它里面的预测元件: %s" % (inside,))
        if hit:
            print("        与它重叠的（键改写后）: %s" % (hit,))


if __name__ == "__main__":
    main()
