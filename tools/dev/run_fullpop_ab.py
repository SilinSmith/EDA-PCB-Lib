# -*- coding: utf-8 -*-
"""按 §5.161 预注册跑「全人口 vs 旧规则」两臂 A/B（真实语料文字线，需要 GPU）。

清单由 `tools/dev/mk_fullpop_manifest.py` 生成（默认 `Y:\\workspace\\EDA\\work\\fullpop\\`）：
  control    = 训练 sheet 上「旧规则」行（`details ∈ ROLE_MAP`，即今天 `ocr_real` 的人口）
  treatment  = 训练 sheet 上**全部**行（`details` 为空 → role `value`，补上被丢掉的 2,734 条种群）
两臂共用同一份 800 行 val、同一合成池、同一超参、同一起点 ckpt；ruler（6 张 sheet，989 框）
**永不进训练**，只用来判分。

用法：
  python -X utf8 tools/dev/run_fullpop_ab.py --dry                 # 只打印将执行的命令
  python -X utf8 tools/dev/run_fullpop_ab.py --arm both            # 全流程（mix → train → eval → 对比）
  python -X utf8 tools/dev/run_fullpop_ab.py --arm treatment --stage eval   # 只补跑评测
判据（§5.162 预注册，先写死）：ruler 上 `value` 的 ≥5 字 exact（n=94）treatment ≥ control + 0.05，
护栏 = ALL ≥5 不降、其它 role 各自不降超过 1pp。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
WORK = EDA.parent / "work"
DEFAULT_OUT = WORK / "fullpop"
DEFAULT_START = WORK / "ocr_mix" / "runs" / "last.pt"
ARMS = ("control", "treatment")


def log_tail(path: Path, n: int = 30) -> None:
    if not path.exists():
        print("  !! 日志不存在：%s" % path)
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("  ---- %s（末 %d 行）----" % (path.name, min(n, len(lines))))
    for ln in lines[-n:]:
        print("  " + ln)


def run(cmd: list, log: Path | None, dry: bool) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    if log is None:
        return subprocess.run([str(c) for c in cmd], cwd=str(EDA)).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(EDA), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    log_tail(log)
    return rc


def _read_text(path: Path) -> str:
    """容错读日志：pwsh 的 `>` 重定向默认写 UTF-16LE，Python 自己写的是 UTF-8。"""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def parse_arm(log: Path) -> dict:
    """从评测日志里抓 by-len 表的关键格（value / ALL 的 >=5 列与 n）与主表 ALL 行。"""
    out: dict = {}
    if not log.exists():
        return out
    txt = _read_text(log)
    for name in ("value", "ALL"):
        m = re.search(r"^%s\s+.*\|\s+([0-9.]+|-)\s+(\d+)\s*$" % name, txt, re.M)
        if m:
            out["%s_ge5" % name] = None if m.group(1) == "-" else float(m.group(1))
            out["%s_ge5_n" % name] = int(m.group(2))
    m = re.search(r"^%s\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s*$" % "ALL", txt, re.M)
    if m:
        out["ALL_n"] = int(m.group(1))
        out["ALL_cer"] = float(m.group(2))
        out["ALL_exact"] = float(m.group(3))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="清单/产物根目录")
    ap.add_argument("--arm", default="both", choices=("control", "treatment", "both"))
    ap.add_argument("--stage", default="all", choices=("mix", "train", "eval", "compare", "all"))
    ap.add_argument("--start-ckpt", default=str(DEFAULT_START))
    ap.add_argument("--epochs", type=int, default=0, help="0 = 用 train.py 默认（12）")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="只打印命令，不执行")
    a = ap.parse_args(argv)

    out = Path(a.out)
    ruler = out / "ruler.jsonl"
    if not ruler.exists():
        print("!! 缺少 %s —— 先跑 tools/dev/mk_fullpop_manifest.py" % ruler)
        return 2
    arms = ARMS if a.arm == "both" else (a.arm,)

    for arm in arms:
        d = out / arm
        print("\n=== %s ===" % arm)
        if a.stage in ("mix", "all"):
            run([sys.executable, "-X", "utf8", "-m", "tools.dev.mix_crnn_manifest",
                 "--real", d / "real.jsonl", "--out", d,
                 "--charset-from", a.start_ckpt],
                d / "mix.log", a.dry)
        if a.stage in ("train", "all"):
            cmd = [sys.executable, "-X", "utf8", "-m", "textrec.train",
                   "--resume", a.start_ckpt, "--out", d / "run",
                   "--manifest", d / "manifest.jsonl",
                   "--batch", a.batch, "--workers", a.workers]
            if a.epochs:
                cmd += ["--epochs", a.epochs]
            run(cmd, d / "train.log", a.dry)
        if a.stage in ("eval", "all"):
            run([sys.executable, "-X", "utf8", "-m", "textrec.eval_real",
                 "--ckpt", d / "run" / "best.pt", "--manifest", ruler,
                 "--no-ov", "--by-len"],
                d / "ruler_eval.log", a.dry)

    if a.stage in ("compare", "all") and (a.arm == "both" or a.stage == "compare"):
        print("\n=== 对比（ruler，判据见 §5.162）===")
        res = {arm: parse_arm(out / arm / "ruler_eval.log") for arm in ARMS}
        for arm in ARMS:
            r = res[arm]
            if not r:
                print("  %-10s （无读数）" % arm)
                continue
            print("  %-10s ALL n=%s exact=%s CER=%s | value>=5 exact=%s (n=%s) | ALL>=5 exact=%s (n=%s)"
                  % (arm, r.get("ALL_n"), r.get("ALL_exact"), r.get("ALL_cer"),
                     r.get("value_ge5"), r.get("value_ge5_n"),
                     r.get("ALL_ge5"), r.get("ALL_ge5_n")))
        c, t = res["control"], res["treatment"]
        if c.get("value_ge5") is not None and t.get("value_ge5") is not None:
            dv = t["value_ge5"] - c["value_ge5"]
            da = (t.get("ALL_ge5") or 0) - (c.get("ALL_ge5") or 0)
            print("  Δ(value>=5) = %+.4f   Δ(ALL>=5) = %+.4f   ⇒ 主判据 %s"
                  % (dv, da, "通过" if dv >= 0.05 and da >= -0.01 else "未通过"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
