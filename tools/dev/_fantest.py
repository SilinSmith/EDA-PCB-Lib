# -*- coding: utf-8 -*-
"""case 级扇出（`-case-workers`）的回归测试：坏图 / 看门狗 / 关掉 / 自动 / 护栏。

用法（仓库根 = `eda-pcb/` 的上一级）：
    python -X utf8 eda-pcb\\tools\\dev\\_fantest.py

五个现场（用户 m42543「判题机同时给全部样例且允许并发」）：
  ① 扇出 3 路 + 一张坏图 ⇒ 三张都有产物（坏图是空壳）、`decision_log.summary.n_lanes=3`、
     `n_shell=1`、每 lane 摘要都在；
  ② `EDA_WALL_BUDGET_S=1` ⇒ 三个 lane 全被强杀（`rc=-9`），父进程仍写出 **3 份空壳** +
     `decision_log.json`、退出码 0（「超时也要留下东西」）；
  ③ `-case-workers 0` ⇒ 走老路（串行），`summary` 里**没有** `n_lanes` 键；
  ④ 默认（`-case-workers -1`）在本机（1 张卡）⇒ 也走老路（自动值 = 1）；
  ⑤ 交付护栏（赛题「2 小时」硬约束）：未设时补 `VLM_TIMEOUT=180` / `VLM_BUDGET_S=1800` /
     `EDA_WALL_BUDGET_S=6900`，用户设过则不覆盖，且不影响产物数。

不碰交付产物：全部在 `work\\_fantest\\` 下自建输入/输出。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[2]                     # `eda-pcb/`
ROOT = PKG.parent                         # 仓库工作区
WORK = ROOT / "work"
RUNPY = PKG / "submission" / "run.py"
SRCIN = WORK / "_fast_in"


def _mk_input(dst: Path) -> Path:
    """两张真图（从 `work/_fast_in` 抄）+ 一张坏图。"""
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    cand = sorted(p for p in SRCIN.glob("*.png"))[:2] if SRCIN.is_dir() else []
    if len(cand) < 2:
        raise SystemExit("跳过：%s 里没有两张以上的 png（先跑一次 fast.py）" % SRCIN)
    names = []
    for i, p in enumerate(cand):
        n = "fa%d.png" % i
        shutil.copyfile(str(p), str(dst / n))
        names.append(n[:-4])
    (dst / "broken.png").write_text("this is not a png at all", encoding="utf-8")
    names.append("broken")
    return dst


def _run(in_dir: Path, out_dir: Path, extra, budget=None, env_extra=None):
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    env = dict(os.environ)
    env.pop("EDA_WALL_BUDGET_S", None)
    if budget is not None:
        env["EDA_WALL_BUDGET_S"] = str(budget)
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    argv = [sys.executable, "-X", "utf8", str(RUNPY),
            "-input", str(in_dir), "-output", str(out_dir),
            "-vlm", "off", "-fusion", "off", "-no-wires"] + list(extra)
    p = subprocess.run(argv, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, cwd=str(PKG))
    log = p.stdout.decode("utf-8", "replace")
    summ = {}
    dl = out_dir / "decision_log.json"
    if dl.exists():
        try:
            summ = (json.loads(dl.read_text(encoding="utf-8")) or {}).get("summary") or {}
        except Exception:                                           # noqa: BLE001
            summ = {}
    preds = sorted(p.name for p in (out_dir / "final_results").glob("*_Pred.json"))
    return {"rc": p.returncode, "log": log, "summary": summ, "preds": preds}


def main() -> int:
    tmp = WORK / "_fantest"
    tin = _mk_input(tmp / "in")
    n_cases = len(list(tin.glob("*.png")))
    ok, bad = [], []

    def chk(name, cond, detail=""):
        (ok if cond else bad).append(name)
        print("  %s %s%s" % ("✓" if cond else "✗", name, ("  " + detail) if detail else ""))

    print("=== ① 扇出 3 路 + 一张坏图 ===")
    r = _run(tin, tmp / "out_fan", ["-case-workers", "3"])
    s = r["summary"]
    chk("退出码 0", r["rc"] == 0, "rc=%s" % r["rc"])
    chk("每张都有产物", len(r["preds"]) == n_cases, "%d/%d" % (len(r["preds"]), n_cases))
    chk("n_lanes == 3", s.get("n_lanes") == 3, "n_lanes=%s" % s.get("n_lanes"))
    chk("n_shell == 1（坏图）", s.get("n_shell") == 1, "n_shell=%s" % s.get("n_shell"))
    chk("每 lane 摘要齐", len(s.get("lanes") or []) >= 2,
        "lanes=%d" % len(s.get("lanes") or []))
    broken = tmp / "out_fan" / "final_results" / "broken_Pred.json"
    body = json.loads(broken.read_text(encoding="utf-8")) if broken.exists() else {}
    chk("坏图是空壳三键", all(k in body for k in ("components", "pins", "nets")),
        "keys=%s" % sorted(body))

    print("=== ② 看门狗 EDA_WALL_BUDGET_S=1 ===")
    r = _run(tin, tmp / "out_wd", ["-case-workers", "3"], budget=1)
    s = r["summary"]
    chk("退出码 0", r["rc"] == 0, "rc=%s" % r["rc"])
    chk("超时也留下 %d 份" % n_cases, len(r["preds"]) == n_cases,
        "%d/%d" % (len(r["preds"]), n_cases))
    chk("n_shell == %d" % n_cases, s.get("n_shell") == n_cases, "n_shell=%s" % s.get("n_shell"))
    chk("lane rc = -9", all(l.get("rc") == -9 for l in (s.get("lanes") or [])),
        "rcs=%s" % [l.get("rc") for l in (s.get("lanes") or [])])
    chk("run.log 提到强杀", "强杀" in r["log"])

    print("=== ③ -case-workers 0（强制串行）===")
    r = _run(tin, tmp / "out_seq", ["-case-workers", "0"])
    s = r["summary"]
    chk("退出码 0", r["rc"] == 0, "rc=%s" % r["rc"])
    chk("走老路（无 n_lanes）", "n_lanes" not in s, "keys=%s" % sorted(s))
    chk("每张都有产物", len(r["preds"]) == n_cases,
        "%d/%d" % (len(r["preds"]), n_cases))

    print("=== ④ 默认（自动）===")
    r = _run(tin, tmp / "out_auto", [])
    s = r["summary"]
    ndev = 0
    try:
        import torch
        ndev = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:                                               # noqa: BLE001
        ndev = 0
    if ndev >= 2:
        chk("多卡自动扇出", (s.get("n_lanes") or 1) >= 2, "n_lanes=%s" % s.get("n_lanes"))
    else:
        chk("单卡自动 = 串行（无 n_lanes）", "n_lanes" not in s, "keys=%s" % sorted(s))
    chk("退出码 0", r["rc"] == 0, "rc=%s" % r["rc"])

    print("=== ⑤ 交付护栏默认值（外部推理预算）===")
    # 赛题硬约束「全部案例 < 2 小时」+ 未完成案例双罚 ⇒ `_platform_budgets()` 只在
    # **用户没设**时补 `VLM_TIMEOUT=180 / VLM_BUDGET_S=1800 / EDA_WALL_BUDGET_S=6900`。
    r = _run(tin, tmp / "out_guard", ["-case-workers", "0"])
    _gl = " ".join(l for l in r["log"].splitlines() if "护栏" in l)[:200]
    chk("未设时补默认", ("VLM_TIMEOUT=180" in r["log"] and "VLM_BUDGET_S=1800" in r["log"]
                        and "EDA_WALL_BUDGET_S=6900" in r["log"]), _gl)
    chk("护栏不改产物数", len(r["preds"]) == n_cases, "%d/%d" % (len(r["preds"]), n_cases))
    r = _run(tin, tmp / "out_guard2", ["-case-workers", "0"],
             env_extra={"VLM_TIMEOUT": "7", "VLM_BUDGET_S": "0"})
    _gl = " ".join(l for l in r["log"].splitlines() if "护栏" in l)[:200]
    chk("已设时不覆盖", ("VLM_TIMEOUT=7" in r["log"] and "VLM_BUDGET_S=0" in r["log"]), _gl)
    chk("预算=0 = 不限（不拒绝）", "超预算被拒 0 次" in r["log"] or "外部推理预算" not in r["log"])

    print("\n%d 项通过，%d 项失败%s"
          % (len(ok), len(bad), ("：" + ", ".join(bad)) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
