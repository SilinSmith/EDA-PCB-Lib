# -*- coding: utf-8 -*-
"""220 张融合层 A/B 的**单进程**驱动器。

为什么要它：`python -X utf8 -m fusion.run <220 个 case> --workers N` 在这台机器上
**会卡死**（多进程池；被 kill 的父进程会留下孤儿 worker，孤儿与后来者抢同一个
输出目录 ⇒ 两个臂都在第 6~7 张处 CPU 归零、停止写盘）。本脚本在**同一个进程**里
按顺序调 `fusion.run.run_case(case, do_llm=False)`，每张 ~10 s，稳且好观察。

用法（环境变量由调用方设好：`FUSION_RUN_DIR`/`FUSION_VLM_DIR`/`FUSION_LLM=0`）：

    python -X utf8 work\\tools\\diag\\_ro220arm.py --out work\\_ro220b
    python -X utf8 work\\tools\\diag\\_ro220arm.py --out work\\_ro220o --cases 0001,0002
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(__file__).resolve().parents[3] / "eda-pcb"
sys.path.insert(0, str(EDA))
os.chdir(str(EDA))

from fusion import run as R                                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录（相对仓库根或绝对）")
    ap.add_argument("--cases", default="", help="逗号分隔；留空用 --cases-file")
    ap.add_argument("--cases-file", default=r"Y:\workspace\EDA\work\_cases220.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=10, help="每多少张打一行进度")
    a = ap.parse_args()

    outp = Path(a.out)
    if not outp.is_absolute():
        outp = EDA.parent / a.out
    outp = outp.resolve()
    outp.mkdir(parents=True, exist_ok=True)
    os.environ["FUSION_OUT_DIR"] = str(outp)
    R.OUT_DIR = outp                       # `run_case` 写的是模块级 OUT_DIR

    if a.cases:
        names = [s.strip() for s in a.cases.split(",") if s.strip()]
    else:
        cf = Path(a.cases_file)
        if not cf.is_absolute():          # 本脚本第 26 行已经 chdir 到 eda-pcb，
            cf = EDA.parent / cf          # 相对名单必须按仓库根解析
        names = [s.strip() for s in
                 cf.read_text(encoding="utf-8").strip().split(",") if s.strip()]
    if a.limit:
        names = names[:a.limit]

    print("[arm] out=%s cases=%d run_dir=%s vlm_dir=%s alt=%s ord=%s"
          % (outp, len(names), R.RUN_DIR, R.VLM_DIR,
             os.environ.get("FUSION_RUN_ALTWINDOW", "(默认)"),
             os.environ.get("FUSION_RUN_ORDER", "(默认)")), flush=True)
    t0 = time.time()
    slow = []
    for i, n in enumerate(names, 1):
        t = time.time()
        try:
            R.run_case(n, do_llm=False)
        except Exception as exc:                                     # noqa: BLE001
            print("[%d/%d] %s FAILED %s: %s" % (i, len(names), n, type(exc).__name__, exc),
                  flush=True)
            continue
        dt = time.time() - t
        if dt >= 60:
            slow.append((n, round(dt)))
        if i % a.every == 0 or i == len(names):
            print("[%d/%d] %s 累计 %.1f min 本张 %.1fs 慢张=%s"
                  % (i, len(names), n, (time.time() - t0) / 60, dt, slow[-3:]), flush=True)
    n_ok = len([f for f in outp.glob("*.json") if not f.name.startswith("_")])
    print("[arm] done in %.1f min, %d/%d 张落盘在 %s"
          % ((time.time() - t0) / 60, n_ok, len(names), outp), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
