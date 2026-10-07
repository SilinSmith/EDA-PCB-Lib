# -*- coding: utf-8 -*-
"""第三阶段 CLI：读 stage2 的产物与 `_assigned/`，仲裁位号，重新装配并写官方格式 JSON。

    cd Y:\\workspace\\EDA\\eda-pcb
    python -m stage3.run --stage1 Y:\\workspace\\EDA\\work\\stage1_all14 ^
                         --stage2 Y:\\workspace\\EDA\\work\\stage2_out ^
                         --out    Y:\\workspace\\EDA\\work\\stage3_out ^
                         [--limit 20] [--cases 0001,0059] [--mode auto|stub|llm|off]

**为什么必须重新装配而不是改字符串**：官方按 key（位号）匹配元件，改一个键就牵动
`pins` 的键和 `nets[*].hyperGraph` 里的 `(位号.引脚号)` —— 事后改字符串一定会漏。
所以这里把仲裁结果写回 `assigned`，再调用同一个 `assemble.build_doc`。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2.assemble import build_doc                      # noqa: E402
from stage3.arbitrate import apply as apply_choice         # noqa: E402
from stage3.arbitrate import arbitrate as arbitrate_hy     # noqa: E402
from stage3.arbitrate import build as build_hy             # noqa: E402
from stage3.arbitrate import dedup as dedup_desig          # noqa: E402


def process(case: str, stage1_dir: Path, stage2_dir: Path, out_dir: Path,
            mode: str = "auto", log=print, batch: int = 0,
            max_slots: int = 0, workers: int = 0) -> dict:
    t0 = time.time()
    doc = json.loads((stage1_dir / ("%s.stage1.json" % case)).read_text(encoding="utf-8"))
    apath = stage2_dir / "_assigned" / ("%s.json" % case)
    if not apath.exists():
        raise FileNotFoundError("缺少 stage2 的分配结果：%s（重跑 stage2 会生成 _assigned/）" % apath)
    assigned = json.loads(apath.read_text(encoding="utf-8"))

    hy = build_hy(case, assigned, doc.get("components"))
    kw = {}
    if batch:
        kw["batch"] = batch
    if max_slots:
        kw["max_slots"] = max_slots
    if workers:
        kw["workers"] = workers
    rep = arbitrate_hy(hy, mode=mode, log=log, **kw)
    apply_choice(assigned, hy)
    n_named = dedup_desig(assigned)

    pred = build_doc(doc, assigned, case)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("%s.json" % case)).write_text(
        json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    return {"case": case, "n_components": len(assigned.get("comps") or {}),
            "n_named": n_named, "n_slots_total": len(hy.slots),
            "arb": rep, "t": round(time.time() - t0, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="第三阶段：位号仲裁 + 重新装配")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="auto", choices=("auto", "stub", "llm", "off"))
    ap.add_argument("--llm-batch", type=int, default=0,
                    help="一次 LLM 调用裁决几个槽位（0 = 用 stage1.llm 的默认，实测批量是唯一可行用法）")
    ap.add_argument("--llm-max-slots", type=int, default=0,
                    help="每张图最多裁决几个槽位（按候选分排序取前 N；0 = 不限。比赛限时 2 小时）")
    ap.add_argument("--llm-workers", type=int, default=0, help="LLM 并发数（0 = 默认 4）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="")
    a = ap.parse_args(argv)

    s1, s2, out = Path(a.stage1), Path(a.stage2), Path(a.out)
    names = sorted(p.name[:-len(".stage1.json")] for p in s1.glob("*.stage1.json"))
    if a.cases:
        want = {c.strip() for c in a.cases.split(",") if c.strip()}
        names = [n for n in names if n in want]
    if a.limit:
        names = names[:a.limit]
    if not names:
        print("没有找到任何 case")
        return 1

    t0 = time.time()
    rows, n_slots, n_decided, modes = [], 0, 0, {}
    for i, n in enumerate(names, 1):
        try:
            r = process(n, s1, s2, out, mode=a.mode, batch=a.llm_batch,
                        max_slots=a.llm_max_slots, workers=a.llm_workers)
        except Exception as e:
            print("  [%d/%d] %-6s 失败 %s: %s" % (i, len(names), n, type(e).__name__, e))
            continue
        rows.append(r)
        n_slots += r["n_slots_total"]
        n_decided += r["arb"].get("n_decided", 0)
        modes[r["arb"]["mode"]] = modes.get(r["arb"]["mode"], 0) + 1
        print("  [%d/%d] %-6s 元件 %-4d 有位号 %-4d 槽位 %-4d 仲裁 %-3d %s  %.2fs"
              % (i, len(names), n, r["n_components"], r["n_named"],
                 r["n_slots_total"], r["arb"].get("n_decided", 0),
                 r["arb"]["mode"], r["t"]))

    wall = round(time.time() - t0, 1)
    print("\n合计 %d 张，墙钟 %.1fs；位号槽位 %d，仲裁改动 %d；模式 %s"
          % (len(rows), wall, n_slots, n_decided, modes))
    (out / "_stage3_report.json").write_text(json.dumps(
        {"case_mode": a.mode, "modes": modes, "n_slots": n_slots,
         "n_decided": n_decided, "wall_s": wall, "cases": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("产物 -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
