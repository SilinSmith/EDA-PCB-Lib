# -*- coding: utf-8 -*-
"""新 GT 落地后的「重开清单」：一条命令跑完所有**会因 GT 口径变化而重新开闸**的检查。

背景（为什么只有这几步）：
  本仓库的结论全部标定在**当前 GT** 上。经 §5.127–§5.149 的收口，各通道与 GT 口径的关系是：
    * **与口径无关**（新 GT 不必重跑）：检出率、元件键、脚号读取、网络挂接的候选生成；
    * **会重新开闸**（新 GT 必须重跑）：① 引脚锚点约定（当前 = bbox 内 d_bbox 中位 10.0px；
      若改成「红绿导线交点」，§5.129 的「锚点占错项 82%」与 §5.131 的「每(元件,边)恒定偏移」都要重算）；
      ② 库补号闸门（§5.133 要求「库里的号必须与图上读到的号一致」，闸门未实装）。
  因此本脚本只跑：GT 指纹、两把尺子的新基线、锚点诊断、池侧对照——**不做判断，只记录基线**。
  新 GT 上的通过/否决一律用**相对 A/B**（净增益 > 0 且对照不降），不用旧 GT 的绝对阈值。

用法：
  python -X utf8 tools/eval/newgt_reopen.py                # 全量（含交付尺子，慢）
  python -X utf8 tools/eval/newgt_reopen.py --skip-slow    # 跳过交付尺子
  python -X utf8 tools/eval/newgt_reopen.py --dry          # 只打印将要执行的命令
"""
from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = [sys.executable, "-X", "utf8"]


def run(cmd: list[str], logdir: pathlib.Path, name: str, dry: bool) -> tuple[int, str]:
    line = " ".join(str(c) for c in cmd)
    print("\n$ %s" % line, flush=True)
    if dry:
        return 0, ""
    logdir.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    (logdir / ("%s.log" % name)).write_text(out, encoding="utf-8")
    tail = "\n".join(out.strip().splitlines()[-12:])
    print(tail, flush=True)
    print("[exit %d] -> %s" % (p.returncode, logdir / ("%s.log" % name)), flush=True)
    return p.returncode, out


def score_of(text: str) -> str:
    """从 gt_retest 输出里抓 `default` 那一行的总分与四通道（宽松匹配，抓不到就返回空）。"""
    for ln in text.splitlines():
        if "`default`" in ln and "**" in ln:
            cells = [c.strip() for c in ln.split("|")]
            keep = [c for c in cells if c and c != "---"]
            return " | ".join(keep[:8])
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="",
                    help="逗号分隔的**案例号**（如 1,8）；默认空 = 用两个工具的官方默认（10 张）"
                         "。注意别传 `10GT`——那是 split 名，会被当成一个案例号而只跑 1 张。")
    ap.add_argument("--out", default=str(ROOT / "work/_newgt_reopen"))
    ap.add_argument("--skip-slow", action="store_true", help="跳过交付尺子（很慢）")
    ap.add_argument("--with-pool", action="store_true", help="追加 200 案池（很慢，默认不跑）")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = pathlib.Path(a.out) / stamp
    notes: list[str] = []

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                           capture_output=True, text=True).stdout.strip()
    print("HEAD=%s  工作树脏行=%d  输出目录=%s" % (head, len(dirty.splitlines()), logdir))

    # 1) GT 指纹（变了才说明这是新 GT）
    rc, out = run(PY + ["tools/eval/gt_retest.py", "--fingerprint-only"], logdir, "1_fingerprint", a.dry)
    fp = [ln for ln in out.splitlines() if "指纹" in ln or "fingerprint" in ln.lower()]
    notes.append("1) GT 指纹: %s" % (fp[-1].strip() if fp else "(见 1_fingerprint.log)"))

    # 2) 确定性尺子基线（新 GT 上的起点）
    rc, out = run(PY + ["tools/eval/gt_retest.py", "--arms", "default", "--rulers", "deterministic"]
                  + (["--cases", a.cases] if a.cases else []) + ["--tag", "newgt_det_base"],
                  logdir, "2_det_base", a.dry)
    det = score_of(out)
    notes.append("2) 确定性 default: %s" % (det or "(见 2_det_base.log)"))

    # 3) 交付尺子基线（慢；可与 --skip-slow 跳过）
    if a.skip_slow:
        notes.append("3) 交付尺子: 已跳过（--skip-slow）")
    else:
        rc, out = run(PY + ["tools/eval/gt_retest.py", "--arms", "default", "--rulers", "delivery"]
                      + (["--cases", a.cases] if a.cases else []) + ["--tag", "newgt_del_base"],
                      logdir, "3_del_base", a.dry)
        notes.append("3) 交付 default: %s" % (score_of(out) or "(见 3_del_base.log)"))

    # 4) 引脚锚点诊断（新口径下必须重算：恒定边占比 + 神谕上界）
    #    必须读**刚跑出来的**确定性产物，否则会在空目录上给出全 0 的假读数。
    run_dir = ROOT.parent / "work/_fast_out_deterministic_default"
    if not run_dir.exists():
        notes.append("4) 锚点诊断: 跳过（找不到 %s；先跑确定性尺子）" % run_dir)
    else:
        rc, out = run(PY + ["tools/diag/_pinanchor.py", "--run-dir", str(run_dir)]
                      + (["--cases", a.cases] if a.cases else []) + ["--show"],
                      logdir, "4_anchor", a.dry)
        const = [ln for ln in out.splitlines() if "恒定" in ln]
        notes.append("4) 锚点诊断: %s" % (const[-1].strip() if const else "(见 4_anchor.log)"))

    # 5) 池侧对照（同一把尺子上的 200 案泛化面，供新 GT 前后对照；很慢，默认不跑）
    if a.with_pool:
        rc, out = run(PY + ["tools/eval/gt_retest.py", "--pool", "--arms", "default",
                            "--rulers", "deterministic", "--tag", "newgt_pool"], logdir, "5_pool", a.dry)
        notes.append("5) 池侧: %s" % (score_of(out) or "(见 5_pool.log)"))
    else:
        notes.append("5) 池侧: 已跳过（--with-pool 才跑，200 案很慢；旧值 27.6754 见 §5.148）")

    print("\n===== 新 GT 重开清单（只记录基线，不做判断）=====")
    for n in notes:
        print("  " + n)
    print("""
接下来**必须做的两件事**（都不是跑分，而是 A/B）：
  (A) 锚点约定：若新 GT 把引脚锚点改成「红绿导线交点」，用 `tools/diag/_pinanchor.py` 重算
      「每 (元件,边) 恒定偏移」占比；只有当 GT-blind 的偏移估计能在**对照不降**的前提下
      把 pin_f1 抬起来，才谈实装（旧 GT 上：恒定边占 68.3%、真值修正 +1.74 分，
      但色转折探针施行 −9.32 分 ⇒ 旧 GT 上不可 GT-blind 估计）。
  (B) 库补号闸门：`stage2/libpins.py` 目前用库给号（§5.133 记 `TPS2553DBVR` 库 `/FAULT`=4
      而 case6 GT 写 3 ⇒ 一处 −0.16）。必须加「库里的号要出现在图上读到的号里」这道闸，
      再用 `--arms` A/B；判据 = 净增益 > 0 且两口径对照不降。
  * 其余通道（检出/元件键/脚号读取/网络挂接）已证与 GT 口径无关（§5.127/§5.137/§5.149），新 GT 不必重跑。""")
    if not a.dry:
        (logdir / "report.md").write_text(
            "# 新 GT 重开清单\n\nHEAD=%s\n\n" % head + "\n".join("- " + n for n in notes) + "\n",
            encoding="utf-8")
        print("\n报告：%s" % (logdir / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
