# -*- coding: utf-8 -*-
"""兜底键（pNNN）的 2191 个元件到底缺什么——逐条查 `_assigned`。

`pNNN` 的含义是「这颗元件最后没拿到位号」，键由 `assemble.py` 的
`key = "p%03d" % i` 生成（i 是 stage1 `components` 里的下标）。

本脚本对每个这样的元件打印它在 `_assigned` 里的 `designator` 条目：
* `text` 为空          -> 压根没有文字框被判给它（归属/检索问题）
* `text` 非空但读废    -> CRNN 读出来的东西不成位号形状（识别问题）
* `text` 是位号但被丢  -> `canon_designator` 因前缀不匹配把它掐掉了（规则问题）

    python work\\tools\\diag\\_pwhy.py --pred work\\_crnntrunc\\final_results --limit 40
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import schematic_parser.metrics as M                    # noqa: E402
from schematic_parser.data import discover               # noqa: E402
from stage2_eval import DEFAULT_DATA_ROOT, build_index, resolve  # noqa: E402


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="final_results 目录")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=25)
    a = ap.parse_args()

    pred = Path(a.pred)
    s2 = pred.parent / "runs" / "stage2"
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    buckets = Counter()
    per_type = {}
    samples = []
    n_comp = 0
    files = sorted(pred.glob("*_Pred.json"))
    if a.limit:
        files = files[: a.limit]
    for f in files:
        stem = f.stem.replace("_Pred", "")
        try:
            c, _how = resolve(stem, exact, digits)
        except Exception:
            continue
        # 产物文件名（= run.py 用的 case id）才是 stage1/stage2 的文件名前缀。
        # GT 的 json 名带 `-jlc_target_named` / `-KiCad_target_named` 后缀，不能用。
        case_id = f.stem.replace("_Pred", "")
        pf = f
        gt = load_json(c.json_path)
        pr = load_json(pf)
        if not gt.get("components") or not pr.get("components"):
            continue
        _f1, km = M.assign_components(gt, pr, M.EvalConfig(), {})
        # keymap: 预测键 -> GT 键；兜底键 pNNN 根本不在 keymap 里（它没配上）
        used_p = set(km)
        gc = gt["components"]
        pc = pr["components"]
        unmatched_gt = [k for k in gc if k not in set(km.values())]
        asf = s2 / "_assigned" / ("%s.json" % case_id)
        s1f = s2.parent / "stage1" / ("%s.stage1.json" % case_id)
        if not asf.is_file() or not s1f.is_file():
            continue
        assigned = load_json(asf).get("comps", {})
        comps = load_json(s1f).get("components", [])
        for pk, pv in pc.items():
            if pk in used_p or not pv.get("bbox"):
                continue
            if not (isinstance(pk, str) and pk.startswith("p") and pk[1:].isdigit()):
                continue
            pty = M.canon_type(pv.get("type"))
            pb = pv["bbox"]
            best = None
            for gk in unmatched_gt:
                gv = gc[gk]
                gb = gv.get("bbox")
                if not gb or M.canon_type(gv.get("type")) != pty:
                    continue
                dd = math.hypot((gb[0] + gb[2]) / 2 - (pb[0] + pb[2]) / 2,
                                (gb[1] + gb[3]) / 2 - (pb[1] + pb[3]) / 2)
                if dd <= 20.0 and (best is None or dd < best[1]):
                    best = (gk, dd, gv)
            if best is None:
                continue
            gk, _dd, gv = best
            n_comp += 1
            idx = int(pk[1:])
            cid = comps[idx]["id"] if idx < len(comps) else None
            info = assigned.get(cid, {}) if cid else {}
            d = info.get("designator") or {}
            txt = d.get("text") or ""
            canon = d.get("canon") or ""
            gtype = M.canon_type(gv.get("type"))
            if not txt:
                b = "A_没有任何文字框判给它"
            elif not canon:
                b = "B_读出来了但被 canon 掐掉"
            else:
                b = "C_有 canon 却不是最终键"
            buckets[b] += 1
            pt = per_type.setdefault(gtype, Counter())
            pt[b] += 1
            if len(samples) < a.samples:
                samples.append((case_id, gk, gtype, txt, canon, b, pk))

    print("扫了 %d 张，兜底键元件 %d 个" % (len(files), n_comp))
    for k, v in buckets.most_common():
        print("  %-28s %6d  %5.1f%%" % (k, v, 100.0 * v / max(1, n_comp)))
    print()
    print("按 GT 类型（前 16）：")
    for t, cnt in sorted(per_type.items(), key=lambda kv: -sum(kv[1].values()))[:16]:
        print("  %-12s n=%-5d %s" % (t, sum(cnt.values()),
                                     "  ".join("%s=%d" % (k.split("_")[0], v) for k, v in cnt.most_common())))
    print()
    print("样例：")
    for case, gk, t, txt, canon, b, pk in samples:
        print("  %-18s GT %-8s(%-8s) <- %-6s assigned.text=%-14r canon=%-12r %s"
              % (case, gk, t, pk, txt, canon, b))


if __name__ == "__main__":
    main()
