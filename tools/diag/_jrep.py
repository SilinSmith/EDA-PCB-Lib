# -*- coding: utf-8 -*-
"""把 `_fusion_report.json` 里 LLM 轮（`rounds[].judge`）的**开关与计数**整行打出来。

为什么需要它：`judge` 报告的字段（`mode` / `ctx` / `n_ctx_slot` / `n_asked` /
`n_agree_top` / `n_new` / `sec` …）是「这条分支到底跑没跑」的唯一现场证据；
只比总分是看不出「新加的路有没有被执行」的（见 `docs/DIAGNOSTICS.md`）。

用法：
    python tools\\diag\\_jrep.py --out work\\_fast_out_ctxA            # 所有 case 的计数
    python tools\\diag\\_jrep.py --out work\\_fast_out_ctxA --rows     # 再逐行打 judge_rows
    python tools\\diag\\_jrep.py --out <产物目录> --case 1            # 只打某一张
    python tools\\diag\\_jrep.py --cmp work\\_fast_out_ctxA work\\_fast_out_ctxB
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# `rows` 太长，默认不打；这几个字段是「一次问了多少 / 回没回依据」的核心。
ROW_KEYS = ("slot", "ans", "res", "mode", "sec", "why")


def find_report(out: Path) -> Path:
    """提交路径把报告写在 `final_results/runs/fusion/` 下，本地跑在 `runs/fusion/` 下。"""
    for rel in ("runs/fusion/_fusion_report.json",
                "final_results/runs/fusion/_fusion_report.json"):
        p = out / rel
        if p.is_file():
            return p
    raise SystemExit("找不到 _fusion_report.json：%s" % out)


def cases_of(d):
    c = d.get("cases") if isinstance(d, dict) else d
    if isinstance(c, dict):
        c = [dict(v, case=k) for k, v in c.items()]
    return c or []


def show(path: Path, only=None, rows=False):
    d = json.loads(path.read_text(encoding="utf-8"))
    print("== %s" % path)
    for c in cases_of(d):
        if only and str(c.get("case")) not in only:
            continue
        print(" case %-4s n_slots %s" % (c.get("case"), c.get("n_slots")))
        for r in c.get("rounds") or []:
            j = r.get("judge") or {}
            if not j:
                print("   round %s  judge: （没跑）" % r.get("round"))
                continue
            keys = sorted(k for k in j if k != "rows")
            print("   round %s  judge:" % r.get("round"))
            for k in keys:
                print("      %-14s %s" % (k, j[k]))
            if rows:
                for x in r.get("judge_rows") or []:
                    print("      · " + "  ".join(
                        "%s=%r" % (k, x.get(k)) for k in ROW_KEYS if k in x))
        for r in c.get("rounds") or []:
            if "n_q_next" in r:
                print("   round %s  n_q_next %s n_task %s"
                      % (r.get("round"), r.get("n_q_next"), r.get("n_task")))
            if r.get("judge", {}).get("err"):
                print("   round %s  judge 抛异常：%s" % (r.get("round"),
                                                        r["judge"]["err"]))
            if r.get("rescan", {}).get("err"):
                print("   round %s  rescan 抛异常：%s" % (r.get("round"),
                                                         r["rescan"]["err"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--cmp", nargs="+", default=None)
    ap.add_argument("--case", default="")
    ap.add_argument("--rows", action="store_true")
    a = ap.parse_args(argv)
    only = set(x for x in a.case.split(",") if x) or None
    if a.cmp:
        for o in a.cmp:
            show(find_report(Path(o)), only, a.rows)
        return 0
    if not a.out:
        raise SystemExit("给 --out <产物目录> 或 --cmp A B")
    show(find_report(Path(a.out)), only, a.rows)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
