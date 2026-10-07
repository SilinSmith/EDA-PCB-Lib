"""**请求合并（batch 节点）**的隔离 A/B：值通道「一格一次」vs「N 格拼一张图」。

只测值通道，不跑整条管线，所以便宜（几百次调用）。

样本口径与 `_vlmval3.py` 一致：**官方 `assign_components` 键匹配成功**、GT 类型是
`r`/`c`（`VAL_TYPES`）、GT `value` 非空、预测有 bbox。这样每个样本都是**会真正参与打分**的元件。

判对口径：`norm(我们写的值) == norm(GT 值)`（只去空白）。
采纳规则与 `eda-pcb/vlm/enrich.py` 完全一致：先过「位号一致」门控，再要求 `VALUE_LIKE`。

用法：
  python work\\tools\\diag\\_valbatch.py --limit 40 --batch 1,6 --cols 2 --side 520
  python work\\tools\\diag\\_valbatch.py --limit 40 --batch 1,6 --side 720 --scale 4
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                       # EDA
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json, discover, SPLITS   # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components, canon_type     # noqa: E402
from vlm import enrich                                                             # noqa: E402

VAL = set(enrich.VAL_TYPES)
norm = lambda s: re.sub(r"\s+", "", s or "")


def pairs_of(case, cfg):
    gt = load_json(Path(case.json_path))
    pf = ROOT / "work" / "_submit_out220z" / "final_results" / ("%s_Pred.json" % case.case_id)
    if not pf.exists():
        return None
    pr = load_json(pf)
    _f1, km = assign_components(gt, pr, cfg, {})
    if not km:
        return None
    gc, pc = gt.get("components") or {}, pr.get("components") or {}
    out = []
    for pk, gk in km.items():
        g, p = gc.get(gk), pc.get(pk)
        if not isinstance(g, dict) or not isinstance(p, dict):
            continue
        ct = canon_type(g.get("type"))
        if ct not in ("r", "c") or ct not in VAL:
            continue
        gv = (g.get("value") or "").strip()
        if not gv or not p.get("bbox"):
            continue
        out.append((pk, gv, p["bbox"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="抽查几张图")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", default="1,6", help="逗号分隔的臂，例如 1,4,6")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--side", type=int, default=520)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--margin", type=float, default=80.0)
    ap.add_argument("--pred", default=None)
    args = ap.parse_args()

    cfg = EvalConfig()
    cases = [c for c in discover(DEFAULT_DATA_ROOT, SPLITS) if c.json_path]
    random.Random(args.seed).shuffle(cases)
    samples, used = [], 0
    for c in cases:
        if used >= args.limit:
            break
        pr = pairs_of(c, cfg)
        if not pr:
            continue
        used += 1
        samples.extend((c, pk, gv, bb) for pk, gv, bb in pr)
    print("图 %d 张｜样本（键匹配上的 r/c，GT 值非空）%d 个" % (used, len(samples)))

    arms = [int(x) for x in str(args.batch).split(",") if x.strip()]
    base = dict(enrich.DEFAULTS)
    base.update(value=True, value_margin=args.margin, value_scale=args.scale,
                value_key_gate=True, value_use_key=False, designator=False,
                name=False, desig_fallback=False)
    res = {}
    for b in arms:
        opts = dict(base, value_batch=b, value_cols=args.cols, value_side=args.side)
        ok = wrong = 0
        t0 = time.time()
        stat = dict(desig=0, grid=0, value=0, value_set=0, name=0, name_set=0)
        by_case = {}
        for c, pk, gv, bb in samples:
            by_case.setdefault(c.case_id, (c, []))[1].append((pk, gv, bb))
        for cid, (c, items) in by_case.items():
            bgr = cv2.imdecode(__import__("numpy").fromfile(str(c.png_path), dtype="uint8"),
                               cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            H = bgr.shape[0]
            iitems = [(pk, (bb[0], H - bb[3], bb[2], H - bb[1])) for pk, _gv, bb in items]
            for (pk, gv, _bb), (kk, val) in zip(items, enrich.read_values(bgr, iitems, opts, stat)):
                if opts["value_key_gate"] and kk and kk.upper() != str(pk).upper():
                    continue
                if not val or not enrich.VALUE_LIKE.search(val):
                    continue
                if norm(val) == norm(gv):
                    ok += 1
                else:
                    wrong += 1
        n = ok + wrong
        res[b] = dict(batch=b, calls=stat["value"], ok=ok, wrong=wrong, total=len(samples),
                      cov=(ok / len(samples) if samples else 0.0),
                      acc=(ok / n if n else 0.0), secs=round(time.time() - t0, 1))
        print("  batch=%-2d  **覆盖 %3d/%d = %.4f**（写回 %4d：对 %3d / 错 %3d）"
              "｜写回精度 %.4f｜调用 %4d 次｜%.1fs"
              % (b, ok, len(samples), ok / max(1, len(samples)), n, ok, wrong,
                 ok / n if n else 0, stat["value"], time.time() - t0))

    if len(arms) > 1:
        a, b = arms[0], arms[-1]
        print("\n结论：batch=%d 覆盖 %.4f（%d 次调用）｜batch=%d 覆盖 %.4f（%d 次调用）"
              "｜Δ覆盖 %+.4f（调用 %+.0f%%）"
              % (a, res[a]["cov"], res[a]["calls"], b, res[b]["cov"], res[b]["calls"],
                 res[b]["cov"] - res[a]["cov"],
                 100.0 * (res[b]["calls"] - res[a]["calls"]) / max(1, res[a]["calls"])))
    (ROOT / "work" / "_valbatch.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
