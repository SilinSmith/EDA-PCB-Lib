"""python -m stage1 —— 把 PNG 跑成第一阶段中间结果（可选接 LLM 裁决）。

例子
----
    cd Y:\\workspace\\EDA\\eda-pcb
    # 四个有 testpoint 的 case，先看单张耗时与产物
    python -m stage1 --cases 0050,0011,0102,0131 --out out\\stage1

    # 官方全部 220 张，8 进程，不跑骨架（只有元件几何）
    python -m stage1 --all --workers 8 --no-wires --out out\\stage1_yolo

    # 带 LLM 裁决（需要 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）
    python -m stage1 --cases 0050 --llm --llm-workers 8 --out out\\stage1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import schematic_parser as S                        # noqa: E402
from stage1.pipeline import run_case, run_cases     # noqa: E402
from stage1 import llm as llm_mod                   # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="逗号分隔的 case id；空则配合 --all")
    ap.add_argument("--all", action="store_true", help="官方全部 220 个 case")
    ap.add_argument("--split", default="", help="只跑某个 split：10GT / 10_GTcase / 200_train_cases")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=r"out\stage1")
    ap.add_argument("--weights", default=None)
    # ★ 2026-07 审计项③：这里以前是 `default=1024`，而真正提交用的
    #   `eda-pcb\submission\run.py:75` 是 `default=1600` ⇒ 用
    #   `python -m stage1 --all` 复现 220 张会**静默跑 1024**，结果对不上账。
    #   现在跟随提交口径。（`stage1\detect.py:104` 的 `Detector.__init__` 与
    #   `stage1\pipeline.py:101` 的 `run_case(imgsz=...)` 签名默认仍是 1024，
    #   那两处只是「直接调库时的兜底」，命令行一律显式传值。）
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=1, help="进程池大小（CPU 密集，不是线程）")
    ap.add_argument("--tile", type=int, default=0, help=">0 时切块推理")
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=2_500_000,
                    help="超过就先降采样再跑骨架（骨架是大图的时间瓶颈）")
    ap.add_argument("--no-wires", action="store_true", help="只出元件几何，最省时间")
    ap.add_argument("--no-cv", action="store_true", help="关掉 CV 专类部件")
    ap.add_argument("--llm", action="store_true", help="对未决槽位并发调 LLM 裁决")
    ap.add_argument("--llm-workers", type=int, default=8)
    a = ap.parse_args()

    from stage1.detect import DEFAULT_WEIGHTS
    weights = a.weights or DEFAULT_WEIGHTS

    if a.all or a.split:
        cases = list(S.discover(S.DEFAULT_DATA_ROOT,
                                splits=(a.split,) if a.split else S.SPLITS))
    else:
        ids = [t.strip() for t in a.cases.split(",") if t.strip()]
        if not ids:
            ap.error("要给 --cases 或 --all")
        cases = []
        for i in ids:
            c = S.find_case(i, S.DEFAULT_DATA_ROOT)
            if c is None:
                print(f"跳过 {i}：找不到")
            else:
                cases.append(c)
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        print("没有可跑的 case")
        return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(weights=weights, imgsz=a.imgsz, conf=a.conf, device=a.device,
              max_pixels=a.max_pixels,
              tiler=(a.tile, a.overlap) if a.tile else None,
              use_cv=not a.no_cv, do_wires=not a.no_wires)
    print(f"case {len(cases)} 个   输出 {out}   workers {a.workers}   "
          f"骨架 {'关' if a.no_wires else '开'}   CV {'关' if a.no_cv else '开'}   "
          f"tile {a.tile or '关'}")
    t0 = time.time()
    res = run_cases(cases, out, workers=a.workers, **kw)
    wall = time.time() - t0

    ok = [r for r in res if r["ok"]]
    bad = [r for r in res if not r["ok"]]
    print(f"\n完成 {len(ok)}/{len(res)}   墙钟 {wall:.1f} s   "
          f"平均 {wall/max(1,len(res)):.1f} s/case   "
          f"串行合计 {sum(r['timing']['total_s'] for r in ok):.1f} s")
    if ok:
        slow = sorted(ok, key=lambda r: -r["timing"]["total_s"])[:5]
        print("最慢 5 个：")
        for r in slow:
            print(f"  {r['case']:<20}{r['timing']['total_s']:>8.1f} s  "
                  f"(yolo {r['timing']['yolo_s']}, wires {r['timing']['wires_s']})")
        print("元件总数", sum(r["stats"]["n_components"] for r in ok),
              " 待裁决", sum(r["stats"]["n_pending"] for r in ok),
              " 已接线", sum(r["stats"]["n_attached"] for r in ok))
    for r in bad:
        print(f"  FAIL {r['case']}: {r['error']}")

    if a.llm:
        from cvparts.hypo import CaseHypotheses
        agg = {"asked": 0, "resolved": 0, "failed": 0, "skipped": 0, "answered": 0}
        for r in ok:
            doc = json.loads(Path(r["out"]).read_text(encoding="utf-8"))
            h = CaseHypotheses.load(r["out"])
            st = llm_mod.arbitrate(h, workers=a.llm_workers)
            for k in agg:
                agg[k] += st.get(k, 0)
            if st.get("resolved"):
                doc["hypotheses"] = h.to_dict()
                doc["stats"]["n_pending"] = sum(
                    1 for c in doc["components"]
                    if c["pending"] and not h.slots.get(c["id"] + ".type", None) is None
                    and not h.slots[c["id"] + ".type"].resolved())
                for c in doc["components"]:
                    sl = h.slots.get(c["id"] + ".type")
                    if sl is not None and sl.resolved():
                        c["type"] = h.folds.get(sl.chosen, sl.chosen)
                        c["pending"] = False
                Path(r["out"]).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"\n[llm] 合计 {agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
