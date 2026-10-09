# -*- coding: utf-8 -*-
"""灾难现场加固的**故障注入**测试（用户 m42543）。

五个现场，每个都是「本机跑得出、判题机上会整批 0 分」的那种：

  ① 非 ASCII / 带空格路径读图 —— `cv2.imread` 在 Windows 上对非 ASCII 路径返回 None，
     旧代码 `raise RuntimeError` ⇒ 那张图零产物。现在走 `_imread_bgr`（`np.fromfile`
     + `cv2.imdecode`）。
  ② 坏图（截断/不是图片）—— 阶段一 `_worker` 吞成 `ok:False`，导出层**必须**写空壳，
     否则 `final_results/<case>_Pred.json` 一份都没有。
  ③ 同名输入（`a.png` + `a.jpg` 同一 stem）—— 阶段一输出同名覆盖，旧代码只出一份
     产物却不报告；现在保序去重 + 显式告警。
  ④ 无 CUDA 但 `-device 0` —— ultralytics 对显式 device 直接抛 ValueError，
     220 张全失败；现在 `_fix_device` 回退 CPU。
  ⑤ 阶段一整段抛异常（模拟 `f.result()` 炸 / import 失败）—— `main` 外层必须兜住，
     补空壳 + 写 `decision_log.json`，返回 3 而不是把异常抛给入口。

用法：python -X utf8 work\\tools\\dev\\_hardtest.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EPCB = ROOT / "eda-pcb"
RUN_PY = EPCB / "submission" / "run.py"
WORK = ROOT / "work" / "_hardtest"
SRC_PNG = ROOT / "work" / "_fast_in" / "1.png"

OK, BAD = "PASS", "FAIL"
results: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((OK if cond else BAD, name, detail))
    print("  [%s] %s%s" % (OK if cond else BAD, name, ("  —— " + detail) if detail else ""))


def run_sub(in_dir: Path, out_dir: Path, extra=()) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-X", "utf8", str(RUN_PY),
           "-input", str(in_dir), "-output", str(out_dir), "-mode", "off", "-vlm", "off"]
    cmd += list(extra)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=900)


def load_run_module():
    """把 `submission/run.py` 按文件路径加载成模块（避开 `stage2.run` 的重名）。"""
    sys.path.insert(0, str(EPCB))
    sys.path.insert(0, str(EPCB / "submission"))
    spec = importlib.util.spec_from_file_location("_sub_run_under_test", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    if not SRC_PNG.exists():
        print("缺少测试图 %s（先跑一次 fast.py --cases 1 生成）" % SRC_PNG)
        return 2
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    # ── ① 非 ASCII / 带空格路径 ────────────────────────────────────────
    print("\n① 非 ASCII + 空格路径读图")
    d1 = WORK / "中文 目录 带空格"
    d1.mkdir(parents=True, exist_ok=True)
    p1 = d1 / "中文 图.png"
    shutil.copyfile(SRC_PNG, p1)
    sys.path.insert(0, str(EPCB))
    from stage1 import pipeline as sp                                    # noqa: E402
    import cv2                                                           # noqa: E402
    got = sp._imread_bgr(p1)
    old = cv2.imread(str(p1), cv2.IMREAD_COLOR)
    ref = cv2.imread(str(SRC_PNG), cv2.IMREAD_COLOR)
    check("旧路径 cv2.imread 对非 ASCII 返回 None（这就是要救的现场）", old is None,
          "cv2.imread -> %r" % (None if old is None else getattr(old, "shape", "?"),))
    check("_imread_bgr 读出真图", got is not None,
          "shape=%s" % (getattr(got, "shape", None),))
    check("与非 ASCII 无关，逐像素与 ASCII 路径读出的相同",
          got is not None and ref is not None and got.shape == ref.shape
          and bool((got == ref).all()))

    # ── ② 坏图 → 空壳 + decision_log ──────────────────────────────────
    print("\n② 坏图（不是图片的 .png）")
    d2 = WORK / "badonly"
    d2.mkdir(parents=True, exist_ok=True)
    (d2 / "broken.png").write_bytes(b"this is not a png at all\n")
    o2 = WORK / "out_bad"
    r2 = run_sub(d2, o2)
    shell = o2 / "final_results" / "broken_Pred.json"
    check("退出码 0（不把异常抛给入口）", r2.returncode == 0, "rc=%d" % r2.returncode)
    check("空壳 <case>_Pred.json 写出来了", shell.exists())
    if shell.exists():
        doc = json.loads(shell.read_text(encoding="utf-8"))
        check("空壳三个键齐全（metrics.py 直接下标也解析得动）",
              all(k in doc for k in ("components", "pins", "nets")),
              "keys=%s" % sorted(doc))
    check("decision_log.json 存在且可解析", (o2 / "decision_log.json").exists())
    log2 = (o2 / "final_results" / "run.log")
    txt2 = log2.read_text(encoding="utf-8", errors="replace") if log2.exists() else ""
    check("run.log 里明确写了「空壳兜底」", "空壳" in txt2)

    # ── ③ 同名输入去重 ────────────────────────────────────────────────
    print("\n③ 同名输入（same.png + same.jpg）")
    d3 = WORK / "dup"
    d3.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC_PNG, d3 / "same.png")
    shutil.copyfile(SRC_PNG, d3 / "same.jpg")
    o3 = WORK / "out_dup"
    r3 = run_sub(d3, o3)
    log3 = (o3 / "final_results" / "run.log")
    txt3 = log3.read_text(encoding="utf-8", errors="replace") if log3.exists() else ""
    n_pred = len(list((o3 / "final_results").glob("*_Pred.json"))) \
        if (o3 / "final_results").is_dir() else 0
    check("退出码 0", r3.returncode == 0, "rc=%d" % r3.returncode)
    check("只导出 1 份 <case>_Pred.json", n_pred == 1, "n=%d" % n_pred)
    check("同名被丢弃这件事写进了 run.log", "同名" in txt3)
    check("渐进导出生效（阶段二完成即落盘）", "渐进导出" in txt3)

    # ── ④ 无 CUDA 时的 CPU 回退 ───────────────────────────────────────
    print("\n④ 无 CUDA 但 -device 0")
    mod = load_run_module()
    import torch                                                         # noqa: E402
    orig = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False                          # type: ignore
        a = types.SimpleNamespace(device="0")
        mod._fix_device(a)
        check("device '0' → 'cpu'", a.device == "cpu", "-> %r" % a.device)
        b = types.SimpleNamespace(device="cuda:1")
        mod._fix_device(b)
        check("device 'cuda:1' → 'cpu'", b.device == "cpu", "-> %r" % b.device)
        c = types.SimpleNamespace(device="cpu")
        mod._fix_device(c)
        check("显式 'cpu' 不动", c.device == "cpu", "-> %r" % c.device)
        torch.cuda.is_available = lambda: True                       # type: ignore
        e = types.SimpleNamespace(device="0")
        mod._fix_device(e)
        check("有卡时不动（本机 CUDA 可用）", e.device == "0", "-> %r" % e.device)
    finally:
        torch.cuda.is_available = orig                               # type: ignore

    # ── ⑤ 阶段一整段抛异常 → _safety_net ───────────────────────────────
    print("\n⑤ 阶段一整段抛异常（模拟进程池 / import 级崩溃）")
    d5 = WORK / "crash"
    d5.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC_PNG, d5 / "c1.png")
    shutil.copyfile(SRC_PNG, d5 / "c2.png")
    o5 = WORK / "out_crash"
    from stage1 import pipeline as sp2                                   # noqa: E402
    orig_rc = sp2.run_cases

    def _boom(*_a, **_k):
        raise RuntimeError("injected: stage1 pool exploded")

    sp2.run_cases = _boom                                               # type: ignore
    try:
        rc = mod.main(["-input", str(d5), "-output", str(o5),
                       "-mode", "off", "-vlm", "off"])
    finally:
        sp2.run_cases = orig_rc                                         # type: ignore
    check("main 返回 3（而不是抛异常）", rc == 3, "rc=%r" % rc)
    fr = o5 / "final_results"
    n5 = len(list(fr.glob("*_Pred.json"))) if fr.is_dir() else 0
    check("两张图都补了空壳", n5 == 2, "n=%d" % n5)
    dl = o5 / "decision_log.json"
    check("decision_log.json 写出来了", dl.exists())
    if dl.exists():
        doc5 = json.loads(dl.read_text(encoding="utf-8"))
        summ = doc5.get("summary") or {}
        check("decision_log 里记了 fatal", bool(summ.get("fatal")),
              "fatal=%r" % (str(summ.get("fatal"))[:60],))
    log5 = fr / "run.log"
    txt5 = log5.read_text(encoding="utf-8", errors="replace") if log5.exists() else ""
    check("run.log 有 traceback 摘要", "injected: stage1 pool exploded" in txt5)

    # ── 汇总 ─────────────────────────────────────────────────────────
    bad = [r for r in results if r[0] == BAD]
    print("\n================ 汇总：%d 项，失败 %d ================"
          % (len(results), len(bad)))
    for st, nm, dt in bad:
        print("  [FAIL] %s  %s" % (nm, dt))
    print("全部通过 ✓" if not bad else "有失败项 ✗")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
