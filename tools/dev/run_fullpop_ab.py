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
import json
import math
import re
import subprocess
import sys
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
WORK = EDA.parent / "work"
DEFAULT_OUT = WORK / "fullpop_r8"     # §5.166：人类已裁定换成「目标框最多的前 8 张」ruler
                                      # （旧的 work/fullpop，每 6 张取 1 / 989 框，已作废但留在盘上）
DEFAULT_START = EDA / "weights" / "textrec_best.pt"
# ★ §5.181 更正（2026-10-16）：这里原本写死 `WORK/ocr_mix/runs/last.pt`，而 `last.pt` 是那次训练
# **末段发散**的一轮（存储 epoch=61/完成 e60，val CER 0.21529），交付权重 `best.pt` 是更早的
# 最优轮（存储 epoch=57/完成 e56，val CER 0.16978）。§5.180 的三臂就是从这个坏起点跑的，
# 于是「起点自带紧框偏好」与「微调效应」混在一起。默认改成交付权重（`--start-ckpt` 可覆盖），
# 这样任何训练侧 A/B 的参照物都是「要替换的那个模型」，而不是「同一次训练里更差的轮次」。
ARMS = ("control", "treatment", "treatment_long")


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


def start_epoch_of(ckpt: Path) -> int:
    """读起点 checkpoint 的 `epoch`（train.py 的 `--epochs` 是**绝对**上限，必须 > 它）。"""
    try:
        import torch
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        return int(ck.get("epoch", 0))
    except Exception as e:                                   # pragma: no cover
        print("  !! 读不出起点轮数（%s），按 0 算：%s" % (ckpt, e))
        return 0


def _read_text(path: Path) -> str:
    """容错读日志：pwsh 的 `>` 重定向默认写 UTF-16LE，Python 自己写的是 UTF-8。"""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def parse_arm(log: Path) -> dict:
    """从评测日志里抓 by-len 表（role / >=5 列 / n）、主表 ALL 行。"""
    out: dict = {"roles": {}}
    if not log.exists():
        return out
    txt = _read_text(log)
    row = re.compile(r"^(\S+)\s+(?:[0-9.-]+\s+){5}\|\s+([0-9.]+|-)\s+(\d+)\s*$", re.M)
    for m in row.finditer(txt):
        role = m.group(1)
        out["roles"][role] = {
            "ge5": None if m.group(2) == "-" else float(m.group(2)),
            "n": int(m.group(3)),
        }
    for name in ("value", "ALL"):
        if name in out["roles"]:
            out["%s_ge5" % name] = out["roles"][name]["ge5"]
            out["%s_ge5_n" % name] = out["roles"][name]["n"]
    m = re.search(r"^%s\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s*$" % "ALL", txt, re.M)
    if m:
        out["ALL_n"] = int(m.group(1))
        out["ALL_cer"] = float(m.group(2))
        out["ALL_exact"] = float(m.group(3))
    return out


def _row_key(r: dict) -> tuple:
    """评测行的唯一键：图名 + bbox（0.1px 取整）+ role + GT 文本。"""
    return (Path(r["image"]).name,
            tuple(round(float(x), 1) for x in r["bbox"]),
            r["role"], r.get("text") or r.get("gt") or "")


def load_wrong(path: Path):
    """读 eval_real.py --dump 写的错样本集（键集合）；文件不存在返回 None。"""
    if not path.exists():
        return None
    keys = set()
    for ln in _read_text(path).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        keys.add((Path(d["image"]).name,
                  tuple(round(float(x), 1) for x in d["bbox"]),
                  d["role"], d.get("gt") or ""))
    return keys


def mcnemar_exact(b: int, c: int) -> float:
    """精确双侧 McNemar p（对数空间，避免 `2.0 ** disc` 溢出）。

    p = 2 * P(X <= min(b,c))，X ~ Bin(b+c, 0.5)。★ 坑（2026-10-16）：原实现写
    `sum(comb(disc, i) ...) / 2.0 ** disc`，在 `ruler_big`（n=20,343）上不协调数 ~1,000 ⇒
    `OverflowError: (34, 'Result too large')`（float 上限 1.8e308 ≈ 2**1024）。改成
    `lgamma` 逐项相加（log-sum-exp）后，disc 上千也只是几百项浮点加法。
    """
    disc = b + c
    if disc == 0:
        return 1.0
    if disc > 10000:                      # 大样本走正态近似（带连续性校正），避免长求和
        z = (abs(b - c) - 1.0) / math.sqrt(disc)
        return min(1.0, math.erfc(max(0.0, z) / math.sqrt(2.0)))
    k = min(b, c)
    lg, l2 = math.lgamma, math.log(2.0)
    terms = [lg(disc + 1) - lg(i + 1) - lg(disc - i + 1) - disc * l2 for i in range(k + 1)]
    m = max(terms)
    s = m + math.log(sum(math.exp(t - m) for t in terms))
    return min(1.0, 2.0 * math.exp(s))


def paired_test(ruler: Path, wrong_c, wrong_t, role: str = "value", min_len: int = 5) -> dict:
    """目标格的**配对**比较（McNemar）：b=control 错/treat 对，c=control 对/treat 错。

    为什么需要它：主终点在现 ruler 上只有 94 条（`work/_power.py`：独立样本 MDE=0.18、
    配对 MDE(20% 不协调)=0.15），只看两个点估计的差就是在噪声里读结论。这里给出
    b/c、Δ 的配对 95% CI 与精确 McNemar p。
    """
    rows = [json.loads(ln) for ln in ruler.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [r for r in rows if r.get("role") == role and len(r.get("text", "")) >= min_len]
    n = b = c = 0
    for r in rows:
        k = _row_key(r)
        wc, wt = (k in wrong_c), (k in wrong_t)
        n += 1
        if wc and not wt:
            b += 1
        elif wt and not wc:
            c += 1
    if n == 0:
        return {}
    delta = (b - c) / n
    disc = b + c
    se = math.sqrt(max(0.0, disc - (b - c) ** 2 / n)) / n if disc else 0.0
    p = mcnemar_exact(b, c)
    return {"n": n, "b": b, "c": c, "disc": disc, "delta": delta,
            "ci": (delta - 1.96 * se, delta + 1.96 * se), "p": p}


def verdict(res: dict, treat: str = "treatment", paired: dict = None) -> list:
    """按 §5.162 判据给出结论行（主判据 + 护栏，逐条）。"""
    lines = []
    c, t = res.get("control", {}), res.get(treat, {})
    lines.append("[%s vs control]" % treat)
    cv, tv = c.get("value_ge5"), t.get("value_ge5")
    if cv is None or tv is None:
        return ["读数不足，无法判定（缺 value 的 >=5 格）"]
    dv = tv - cv
    lines.append("主判据 value>=5：control %.4f → treatment %.4f，Δ%+.4f（需要 ≥ +0.0500）⇒ %s"
                 % (cv, tv, dv, "通过" if dv >= 0.05 else "未通过"))
    if paired:
        lo, hi = paired["ci"]
        lines.append("   └ 配对核查（n=%d，不协调 %d：control 错/treat 对 %d，反向 %d，Δ%+.4f）"
                     " 95%%CI [%+.4f, %+.4f]，精确 McNemar p=%.4f"
                     % (paired["n"], paired["disc"], paired["b"], paired["c"],
                        paired["delta"], lo, hi, paired["p"]))
        # §5.165 人类裁定：判据读「Δ + 配对 CI」（不再只看点估计过/不过）
        if dv >= 0.05 and lo > 0:
            lines.append("   └ 读法（Δ+CI）：点估计达线且区间不跨 0 ⇒ **通过**")
        elif dv >= 0.05:
            lines.append("   └ 读法（Δ+CI）：点估计达线但区间跨 0 ⇒ **证据不足**（不可当通过）")
        elif hi > 0:
            lines.append("   └ 读法（Δ+CI）：点估计未达线但区间跨 0 ⇒ **未能证伪**（也不能当通过）")
        else:
            lines.append("   └ 读法（Δ+CI）：点估计未达线且区间全负 ⇒ **未通过**")
        if paired["disc"] and paired["p"] > 0.05:
            lines.append("   └ 注意：不协调对数太少/太偏，本判据在现 ruler 上**不足以**分辨 Δ（见 §5.165 功效分析）")
    ca, ta = c.get("ALL_ge5"), t.get("ALL_ge5")
    if ca is not None and ta is not None:
        da = ta - ca
        lines.append("护栏 ALL>=5：Δ%+.4f（不得 < -0.0100）⇒ %s"
                     % (da, "通过" if da >= -0.01 else "未通过"))
    for role in sorted(set(c.get("roles", {})) | set(t.get("roles", {}))):
        if role in ("value", "ALL"):
            continue
        crc, trc = c.get("roles", {}).get(role), t.get("roles", {}).get(role)
        if not crc or not trc or crc["ge5"] is None or trc["ge5"] is None:
            continue
        d = trc["ge5"] - crc["ge5"]
        flag = "OK" if d >= -0.01 else "**降超 1pp**"
        lines.append("护栏 role %-11s：Δ%+.4f（n %d→%d）⇒ %s"
                     % (role, d, crc["n"], trc["n"], flag))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="清单/产物根目录")
    ap.add_argument("--ruler", default="", help=(
        "评测/配对用的 ruler.jsonl（默认 <out>/ruler.jsonl）。重跑到新 --out 目录时用它"
        "复用冻结尺子，省掉重建清单"))
    ap.add_argument("--arm", default="both",
                    choices=("control", "treatment", "treatment_long", "both"))
    ap.add_argument("--stage", default="all", help=(
        "要跑的阶段，**可用逗号分隔**：mix/train/eval/compare/all（默认 all）。"
        "例：`--stage train,eval,compare` 复用已生成的 manifest，只跑训练/评测/对比"))
    ap.add_argument("--save-every", type=int, default=0, help=(
        "透传给 textrec.train：每 N 轮另存 ckpt_e%%03d.pt（默认 0 = 不存）。"
        "§5.181 需要它把「口径偏好 vs 轮次」画成曲线"))
    ap.add_argument("--start-ckpt", default=str(DEFAULT_START))
    ap.add_argument("--epochs", type=int, default=0,
                    help="绝对轮数上限（>0 时原样传给 train.py；一般别用，用 --add-epochs）")
    ap.add_argument("--add-epochs", type=int, default=12,
                    help="在起点 checkpoint 的轮数上**再加**多少轮（默认 12）。"
                         "train.py 的 --epochs 是绝对上限 `range(start, epochs)`，"
                         "而 C.DEFAULTS['epochs']=40 < 起点第 61 轮 ⇒ 不显式给就会空跑 0 轮"
                         "（§5.179 踩到：best CER 1e9、不产出 best.pt、eval 报 FileNotFoundError）")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="只打印命令，不执行")
    a = ap.parse_args(argv)

    out = Path(a.out)
    ruler = Path(a.ruler) if a.ruler else (out / "ruler.jsonl")
    if not ruler.exists():
        print("!! 缺少 %s —— 先跑 tools/dev/mk_fullpop_manifest.py，"
              "或用 `--ruler <已有的 ruler.jsonl>` 复用冻结尺子" % ruler)
        return 2
    arms = ARMS if a.arm == "both" else (a.arm,)
    bad_arms: list = []
    stages = ({"mix", "train", "eval", "compare"} if a.stage == "all"
              else {s.strip() for s in a.stage.split(",") if s.strip()})
    unknown = stages - {"mix", "train", "eval", "compare"}
    if unknown:
        print("!! 未知阶段 %s（可选 mix/train/eval/compare/all，可逗号分隔）" % sorted(unknown))
        return 2

    for arm in arms:
        d = out / arm
        print("\n=== %s ===" % arm)
        if "mix" in stages:
            run([sys.executable, "-X", "utf8", "-m", "tools.dev.mix_crnn_manifest",
                 "--real", d / "real.jsonl", "--out", d,
                 "--charset-from", a.start_ckpt],
                d / "mix.log", a.dry)
        if "train" in stages:
            st = 0 if a.epochs else start_epoch_of(a.start_ckpt)
            ep = a.epochs or (st + a.add_epochs)
            cmd = [sys.executable, "-X", "utf8", "-m", "textrec.train",
                   "--resume", a.start_ckpt, "--out", d / "run",
                   "--manifest", d / "manifest.jsonl",
                   "--batch", a.batch, "--workers", a.workers,
                   "--epochs", ep]
            if a.save_every:
                cmd += ["--save-every", a.save_every]
            print("  [epochs] 绝对上限 %d（起点第 %d 轮 + %d 轮）"
                  % (ep, st, ep - st))
            run(cmd, d / "train.log", a.dry)
            if not a.dry and not (d / "run" / "best.pt").exists():
                print("  !! 没产出 best.pt —— 该臂无效，跳过 eval（看 %s）" % (d / "train.log"))
                bad_arms.append(arm)
                continue
        if "eval" in stages:
            run([sys.executable, "-X", "utf8", "-m", "textrec.eval_real",
                 "--ckpt", d / "run" / "best.pt", "--manifest", ruler,
                 "--no-ov", "--by-len", "--dump", d / "ruler_errors.jsonl"],
                d / "ruler_eval.log", a.dry)

    if "compare" in stages and (a.arm == "both" or stages == {"compare"}):
        print("\n=== 对比（ruler，判据见 §5.162/§5.163）===")
        if bad_arms:
            print("  !! 本轮有臂无效（没产出 best.pt）：%s —— 下面的读数不可作为判据"
                  % ", ".join(bad_arms))
        res = {arm: parse_arm(out / arm / "ruler_eval.log") for arm in ARMS}
        roles = set()
        for arm in ARMS:
            roles |= set(res[arm].get("roles", {}))
        hdr = "  %-12s" % "role" + "".join("%-11s" % arm for arm in ARMS)
        print(hdr + "  Δ(vs control)")
        for role in sorted(roles):
            cells, base = "", res["control"].get("roles", {}).get(role, {}).get("ge5")
            for arm in ARMS:
                v = res[arm].get("roles", {}).get(role, {}).get("ge5")
                cells += "%-11s" % ("%.4f" % v if v is not None else "-")
            last = res[ARMS[-1]].get("roles", {}).get(role, {}).get("ge5")
            d = "-" if (base is None or last is None) else "%+.4f" % (last - base)
            print("  %-12s%s  %s" % (role, cells, d))
        for arm in ARMS:
            r = res[arm]
            if r.get("ALL_n"):
                print("  [%s] ALL n=%s exact=%s CER=%s"
                      % (arm, r.get("ALL_n"), r.get("ALL_exact"), r.get("ALL_cer")))
        for treat in ("treatment_long", "treatment"):
            if any(res[treat].get("roles")):
                print("\n  ---- 判定（%s，§5.162 预注册判据）----" % treat)
                wc = load_wrong(out / "control" / "ruler_errors.jsonl")
                wt = load_wrong(out / treat / "ruler_errors.jsonl")
                paired = {}
                if wc is None or wt is None:
                    print("  （缺少 %s —— 评测时加 --dump 才能做配对核查）"
                          % "ruler_errors.jsonl")
                else:
                    paired = paired_test(ruler, wc, wt)
                for ln in verdict(res, treat, paired or None):
                    print("  " + ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
