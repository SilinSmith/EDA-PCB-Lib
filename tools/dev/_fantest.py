# -*- coding: utf-8 -*-
"""case 级扇出（`-case-workers`）的回归测试：坏图 / 看门狗 / 关掉 / 自动 / 护栏 / 离线 / 增量发布。

用法（仓库根 = `eda-pcb/` 的上一级）：
    python -X utf8 eda-pcb\\tools\\dev\\_fantest.py

八个现场（用户 m42543「判题机同时给全部样例且允许并发」）：
  ① 扇出 3 路 + 一张坏图 ⇒ 三张都有产物（坏图是空壳）、`decision_log.summary.n_lanes=3`、
     `n_shell=1`、每 lane 摘要都在；
  ② `EDA_WALL_BUDGET_S=1` ⇒ 三个 lane 全被强杀（`rc=-9`），父进程仍写出 **3 份空壳** +
     `decision_log.json`、退出码 0（「超时也要留下东西」）；
  ③ `-case-workers 0` ⇒ 走老路（串行），`summary` 里**没有** `n_lanes` 键；
  ④ 默认（`-case-workers -1`）在本机（1 张卡）⇒ 也走老路（自动值 = 1）；
  ⑤ 交付护栏（赛题「2 小时」硬约束）：未设时补 `VLM_TIMEOUT=180` / `VLM_BUDGET_S=1800` /
     `EDA_WALL_BUDGET_S=6900`，用户设过则不覆盖，且不影响产物数。
  ⑥ `-offline` ⇒ 真·零外部调用（`decision_log.config.vlm_client is None` = 父进程从未
     import `vlm.client`）、打印离线档一行、产物件数不变（§5.41）。
  ⑦ 增量发布（§5.47）：起扇出 → 等到第一个产物 → `taskkill /F /T`（= 平台硬杀容器）
     ⇒ 开 `EDA_PUBLISH_S=1` 时输出目录非空、关（`=0`）时输出目录为空。
  ⑧ 发布**升级**（§5.54）：3 路扇出 + `EDA_PUBLISH_S=1` 的产物必须与串行
     **逐字节相同**（从前「只补缺不覆盖」会让阶段二的渐进导出永久胜出）。

不碰交付产物：全部在 `work\\_fantest\\` 下自建输入/输出。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
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


def _kill_after_first(tin: Path, out_dir: Path, watch: str, env_extra=None,
                      budget_s=240.0):
    """起扇出跑，等 `watch` 出现（或进程自己结束）后**杀整棵树**，返回 (命中, 输出目录产物)。

    这是复刻平台行为：平台超时是硬杀容器（§5.46 `Timeout: 1200 seconds`），
    父进程没有收尾的机会（§5.47 增量发布就是为这一刻设计的）。
    """
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    env = dict(os.environ)
    env.pop("EDA_WALL_BUDGET_S", None)
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    argv = [sys.executable, "-X", "utf8", str(RUNPY), "-input", str(tin),
            "-output", str(out_dir), "-vlm", "off", "-fusion", "off", "-no-wires",
            "-case-workers", "3"]
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, cwd=str(PKG))
    t0 = time.time()
    hit = False
    while time.time() - t0 < budget_s:
        if p.poll() is not None:
            break
        if next(iter(out_dir.glob(watch)), None) is not None:
            hit = True
            break
        time.sleep(0.5)
    time.sleep(1.5)                      # 让发布循环再转一圈（EDA_PUBLISH_S=1）
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        p.wait(timeout=60)
    except Exception:                                               # noqa: BLE001
        p.kill()
    preds = sorted(x.name for x in (out_dir / "final_results").glob("*_Pred.json"))
    return hit, preds


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

    print("=== ⑥ 离线档 -offline（零外部调用，赛题 token 分）===")
    # `-vlm off` **不等于**离线：色相提示与 ⓪.9 定向重读不归 `-vlm` 管（§5.41 实测
    # 仍有 19 次调用）。`-offline` 把这两条旁路一起关掉，且**保留**融合层的确定性节点
    # （先验 + 解，不发请求，值 +0.24）。
    r = _run(tin, tmp / "out_offline", ["-case-workers", "0", "-offline"])
    chk("退出码 0", r["rc"] == 0, "rc=%s" % r["rc"])
    chk("每张都有产物", len(r["preds"]) == n_cases,
        "%d/%d" % (len(r["preds"]), n_cases))
    _ol = " ".join(l for l in r["log"].splitlines() if "离线档" in l)[:200]
    chk("打印离线档", "离线档" in r["log"], _ol)
    chk("FUSION_LLM=0", "FUSION_LLM=0" in r["log"], _ol)
    _dl = tmp / "out_offline" / "decision_log.json"
    _vlm = None
    try:
        _vlm = ((json.loads(_dl.read_text(encoding="utf-8")) or {}).get("config") or {}).get("vlm_client")
    except Exception:                                               # noqa: BLE001
        _vlm = "?"
    chk("从未 import vlm.client（真零调用）", _vlm is None, "vlm_client=%s" % _vlm)

    r = _run(tin, tmp / "out_lean", ["-case-workers", "0", "-lean"])
    chk("精简档退出码 0", r["rc"] == 0, "rc=%s" % r["rc"])
    chk("精简档产物数不变", len(r["preds"]) == n_cases,
        "%d/%d" % (len(r["preds"]), n_cases))
    _ll = " ".join(l for l in r["log"].splitlines() if "精简档" in l)[:200]
    chk("打印精简档且 value-batch 4", "精简档" in r["log"] and "value-batch 4" in r["log"], _ll)

    print("=== ⑦ 增量发布（平台硬杀容器：已完成的图必须留在输出目录）===")
    # §5.47：扇出模式原本要等**全部** lane 结束，父进程才把产物收进输出目录；平台超时是
    # 硬杀容器 ⇒ 收尾根本没机会跑，已跑完的图也拿 0 分。`EDA_PUBLISH_S`（默认 15 s）
    # 让父进程边等边发布。这里开/关各杀一次，看输出目录里有没有东西。
    _hit, _preds = _kill_after_first(tin, tmp / "out_pub", "final_results/*_Pred.json",
                                     env_extra={"EDA_PUBLISH_S": "1"})
    chk("硬杀后输出目录非空（开增量发布）", _hit and len(_preds) >= 1,
        "%d 份%s" % (len(_preds), "" if _hit else "（超时未等到）"))
    _hit2, _preds2 = _kill_after_first(tin, tmp / "out_nopub",
                                       "_lanes/lane*/final_results/*_Pred.json",
                                       env_extra={"EDA_PUBLISH_S": "0"})
    chk("硬杀后输出目录为空（关增量发布）", _hit2 and not _preds2,
        "%d 份%s" % (len(_preds2), "" if _hit2 else "（超时未等到）"))

    print("=== ⑧ 发布链要按新旧升级（§5.54：扇出产物必须与串行逐字节相同）===")
    # §5.54：`_run()` 的阶段二**渐进导出**会先把**还没融合**的文档顶进 lane 的
    # `final_results/`，随后各层再覆盖它。父进程从前「只补缺不覆盖」⇒ 抢到的第一版
    # 永久胜出（8 路 10 张实测 9/10 份与串行不同）。现在 `_publish_one` 按源文件
    # mtime 升级，扇出产物必须与串行**逐字节相同**。
    import hashlib

    def _hashes(od: Path, names):
        out = {}
        for n in names:
            f = od / "final_results" / ("%s_Pred.json" % n)
            if f.exists():
                out[n] = hashlib.sha256(f.read_bytes()).hexdigest()
        return out

    _seq = _run(tin, tmp / "out_up_seq", ["-case-workers", "0"])
    _fan = _run(tin, tmp / "out_up_fan", ["-case-workers", "3"],
                env_extra={"EDA_PUBLISH_S": "1"})
    _real = [n[:-4] for n in sorted(p.name for p in tin.glob("*.png")) if n != "broken.png"]
    _hs, _hf = _hashes(tmp / "out_up_seq", _real), _hashes(tmp / "out_up_fan", _real)
    _same = [n for n in _real if _hs.get(n) and _hs.get(n) == _hf.get(n)]
    chk("扇出与串行逐字节相同（%d 张真图）" % len(_real), len(_same) == len(_real),
        "%d/%d 相同%s" % (len(_same), len(_real),
                          "" if len(_same) == len(_real) else "；不同：%s"
                          % [n for n in _real if n not in _same]))
    chk("扇出退出码 0", _fan["rc"] == 0, "rc=%s" % _fan["rc"])
    _pub = [l for l in _fan["log"].splitlines() if "增量发布" in l]
    chk("有增量发布行", bool(_pub), (_pub[-1][:80] if _pub else "无"))
    _n_stage2 = len(list((tmp / "out_up_fan" / "_lanes").glob(
        "lane*/runs/stage2/*.json"))) if (tmp / "out_up_fan" / "_lanes").is_dir() else 0
    print("  （参考：lane 私有目录里 stage2 档 %d 份；主目录 mtime 均已升级）" % _n_stage2)

    print("\n%d 项通过，%d 项失败%s"
          % (len(ok), len(bad), ("：" + ", ".join(bad)) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
