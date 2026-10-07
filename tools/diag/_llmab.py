# -*- coding: utf-8 -*-
"""阶段三 A/B：`mode=off` vs `mode=llm`，同一份 stage2 产物、同一批 case。

问的是**唯一重要的问题**：LLM 仲裁到底有没有换来分数？如果像 3 张冒烟那样
是 0.0000，那它就不该拖慢主流程（2 小时预算是硬约束）。

    python work\tools\diag\_llmab.py --cases 8 --max-slots 0
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
REPO = ROOT / "eda-pcb"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

S1 = ROOT / "work" / "stage1_v6"
S2 = ROOT / "work" / "stage2_v14"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--max-slots", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()

    from stage3.run import process as s3

    avail = sorted(p.stem for p in S2.glob("*.json") if not p.stem.startswith("_"))
    cases = avail[a.skip:a.skip + a.cases]
    print("case 数 %d：%s" % (len(cases), ", ".join(cases)))

    outs, times = {}, {}
    for mode in ("off", "llm"):
        out = ROOT / "work" / ("_llmab_%s" % mode)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        t0 = time.time()
        for c in cases:
            r = s3(c, S1, S2, out, mode=mode, batch=a.batch,
                   max_slots=a.max_slots, log=lambda *_: None)
            print("  %-4s %-28s 槽位 %-4d 仲裁 %-4d %.0fs"
                  % (mode, c, r["n_slots_total"],
                     r.get("arb", {}).get("n_decided", 0), r["t"]))
        times[mode] = time.time() - t0
        outs[mode] = out
        print("== mode=%s 用时 %.1fs（%.1f s/张）" % (mode, times[mode], times[mode] / len(cases)))

    for mode in ("off", "llm"):
        cmd = [sys.executable, str(ROOT / "work" / "tools" / "eval" / "stage2_eval.py"),
               str(outs[mode]), "--quiet", "--json",
               str(ROOT / "work" / ("_llmab_%s.json" % mode))]
        subprocess.run(cmd, cwd=str(REPO))
        d = json.loads((ROOT / "work" / ("_llmab_%s.json" % mode)).read_text(encoding="utf-8"))
        print("== mode=%s 总分 %.4f  用时 %.0fs" % (mode, d.get("total", 0), times[mode]))
        for k in ("component", "pin", "net_h", "net_line"):
            if k in d:
                print("     %-9s %s" % (k, json.dumps(d[k], ensure_ascii=False)))
    print("\n用时：off %.0fs  llm %.0fs  比值 %.1fx"
          % (times["off"], times["llm"], times["llm"] / max(1, times["off"])))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
