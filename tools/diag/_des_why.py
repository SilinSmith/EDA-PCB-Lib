# -*- coding: utf-8 -*-
"""位号挂载归因：把「GT 名字可读但我们没挂上位号」的元件拆成三类。

    python work\tools\diag\_des_why.py [产物目录] [--limit N]

产物目录默认 `work\stage2_best`，需要同目录下有 `_assigned\<case>.json`。

对每一个「我们的 key 是兜底 `pNNN`、而 GT 的 Name 可读」的配对元件，问三个问题：

* `no_slot`   —— `desig_slots` 里根本没这颗元件：**连候选都没产生**（框没检出 / 没被归属 / 读法没通过 strict）
* `slot_miss` —— 有候选，但候选中**没有** GT 那个位号：**读错**（`Rl4`/`Cl5`/`O1` 这种）
* `slot_hit`  —— 候选中**正好有** GT 那个位号却没被选上：**选择策略问题**（一对一匹配 / 打分）

这三类的修法完全不同，所以必须先分开数。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                      # work\tools\diag -> 仓库根
sys.path.insert(0, str(ROOT / "work" / "tools" / "viz"))

import cv2                        # noqa: E402
import numpy as np                # noqa: E402
import comp_cases as CC           # noqa: E402

DIAM = re.compile(r"^\u2205\d+$")
FALLBACK = re.compile(r"^p\d{3,4}$")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def _gt_box(v, H):
    b = v.get("bbox")
    if not b or len(b) != 4:
        return None
    return [float(b[0]), H - float(b[3]), float(b[2]), H - float(b[1])]


def norm(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", s or "").upper()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pred_dir = Path(args[0]) if args else (ROOT / "work" / "stage2_best")
    limit = 10000
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1]) if "=" in a else 10000

    idx = CC.gt_index(CC.DEFAULT_DATA_ROOT)
    stat: Dict[str, int] = {}
    detail: Dict[str, List[str]] = {"no_slot": [], "slot_miss": [], "slot_hit": []}
    by_type: Dict[Tuple[str, str], int] = {}

    cases = [x for x in sorted(pred_dir.glob("*.json")) if not x.name.startswith("_")]
    for f in cases[:limit]:
        case = f.name[:-5]
        rec = idx.get(case) or idx.get(re.sub(r"\D", "", case) or case)
        if rec is None or not rec.json_path:
            continue
        asg_path = pred_dir / "_assigned" / (case + ".json")
        if not asg_path.exists():
            continue
        pdoc = json.loads(f.read_text(encoding="utf-8"))
        asg = json.loads(asg_path.read_text(encoding="utf-8"))
        gdoc = CC.load_json(rec.json_path)
        img = cv2.imdecode(np.fromfile(str(rec.png_path), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        H = img.shape[0]

        pcs = list((pdoc.get("components") or {}).items())
        cids = list((asg.get("comps") or {}).keys())
        slots = asg.get("desig_slots") or {}

        gts = []
        for k, v in (gdoc.get("components") or {}).items():
            b = _gt_box(v, H)
            if b is None:
                continue
            gts.append((k, v, b))

        # 贪心按 IoU 配对
        cand = []
        for gi, (_gk, gv, gb) in enumerate(gts):
            for pi, (_pk, pv) in enumerate(pcs):
                pb = pv.get("bbox")
                if not pb or len(pb) != 4:
                    continue
                v = _iou(gb, [float(x) for x in pb])
                if v >= 0.1:
                    cand.append((v, gi, pi))
        cand.sort(reverse=True)
        used_g, used_p = set(), set()
        for v, gi, pi in cand:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            gk, gv, _gb = gts[gi]
            pk, pv = pcs[pi]
            if not FALLBACK.match(pk or ""):
                continue
            gname = str(gv.get("Name") or "")
            if not gname or DIAM.match(gname):
                continue
            gtype = str(gv.get("type") or "")
            cid = cids[pi] if pi < len(cids) else None
            sl = slots.get(cid) or []
            canons = [norm(s.get("canon") or s.get("text") or "") for s in sl]
            texts = [s.get("text") for s in sl]
            want = norm(gname)
            if not sl:
                cls = "no_slot"
            elif want in canons:
                cls = "slot_hit"
            else:
                cls = "slot_miss"
            stat[cls] = stat.get(cls, 0) + 1
            by_type[(cls, gtype)] = by_type.get((cls, gtype), 0) + 1
            if len(detail[cls]) < 12:
                detail[cls].append("%s gt=%s(%s) slots=%s"
                                   % (case, gname, gtype, texts[:4]))

    tot = sum(stat.values())
    print("目标集合（我们 key 是 pNNN、GT 名可读）: %d" % tot)
    for k in ("no_slot", "slot_miss", "slot_hit"):
        v = stat.get(k, 0)
        print("  %-10s %6d  %5.1f%%" % (k, v, 100.0 * v / tot if tot else 0))
    print()
    for k in ("no_slot", "slot_miss", "slot_hit"):
        print("[%s] 按 GT 类型：" % k)
        rows = sorted(((t, n) for (c, t), n in by_type.items() if c == k),
                      key=lambda x: -x[1])[:12]
        print("   " + "  ".join("%s %d" % (t, n) for t, n in rows))
        for s in detail[k][:6]:
            print("   · " + s)
        print()


if __name__ == "__main__":
    main()
