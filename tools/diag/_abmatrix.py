# -*- coding: utf-8 -*-
"""多 checkpoint × 多尺子的配对矩阵（只读 `textrec.eval_real --dump` 写出的错样本）。

为什么需要它：§5.162/§5.166 的判据是「配对 Δ + 95% CI + 精确 McNemar」，而一次 A/B 往往
要在**好几把尺子**（冻结主尺子 / 交付案池外扩 / 紧框 / 大尺子）上同时看，并且既跟 control
比、也跟**交付权重**比。手工拼表出过错（同一批行换口径后符号翻转，见 §5.180），所以这里
把「同一把尺子 × 任意两个 checkpoint 的配对比较」固定成一条命令。

用法：
    python -X utf8 tools/diag/_abmatrix.py --spec <spec.json> [--set NAME] [--endpoint NAME]
    python -X utf8 tools/diag/_abmatrix.py --example        # 打印 spec 模板

spec（JSON 数组）：
    [{"name": "clean",                      # 尺子标签
      "ruler": "Y:/.../ruler_clean.jsonl",  # 尺子清单（键 = image+bbox+role+text）
      "dumps": {"baseline": "Y:/.../baseline/pool_errors.jsonl",
                "control":  "Y:/.../control/pool_errors.jsonl"}}]

readout 纪律（踩过的坑）：
* 键必须用 `_row_key`（图名 + 0.1px bbox + role + GT），它就是判据脚本的口径；
* `n` 是**尺子上的格数**，不是 dump 行数 —— dump 里只有错的那部分；
* 点估计差（两个 exact 相减）**不等于**配对 Δ（b/c 只数翻转的格），两者都要打；
* 缺 dump 的 checkpoint 直接标「缺」，绝不当 0 读（§5.179 的 `best.pt` 缺席教训）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDA / "tools" / "dev"))
from run_fullpop_ab import _row_key, mcnemar_exact  # noqa: E402

EXAMPLE = """[
  {"name": "clean",
   "ruler": "Y:/workspace/EDA/work/pool_r1/ruler_clean.jsonl",
   "dumps": {"baseline": "Y:/workspace/EDA/work/fullpop_r8/baseline/pool_errors.jsonl",
             "control": "Y:/workspace/EDA/work/fullpop_r8/control/pool_errors.jsonl"}}
]"""

ENDPOINTS = [
    ("value>=5", "value", 5),
    ("ALL>=5", None, 5),
    ("designator(全部)", "designator", 0),
    ("value(全部)", "value", 0),
    ("ALL", None, 0),
]


def load_wrong(path: Path):
    """读错样本键集合；文件不存在返回 None（= 该 checkpoint 没有读数）。"""
    if not path.exists():
        return None
    keys = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        keys.add((Path(d["image"]).name,
                  tuple(round(float(x), 1) for x in d["bbox"]),
                  d["role"], d.get("gt") or ""))
    return keys


def paired(rows, wrong_c, wrong_t) -> dict:
    """b = 前一个错/后一个对；c = 前一个对/后一个错；Δ = (b-c)/n，CI 为配对 Wald。"""
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
    return {"n": n, "b": b, "c": c, "disc": disc, "delta": delta,
            "ci": (delta - 1.96 * se, delta + 1.96 * se), "p": mcnemar_exact(b, c)}


def report(name: str, ruler_path: Path, dumps: dict, only=None, refs=None) -> None:
    raw = [json.loads(ln) for ln in ruler_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    wrong = {k: load_wrong(Path(v)) for k, v in dumps.items()}
    print("\n########## %s（%s，%d 行）##########" % (name, ruler_path.name, len(raw)))
    print("[dump] " + "  ".join("%s=%s" % (k, "缺" if v is None else len(v))
                                for k, v in wrong.items()))
    for ename, role, min_len in ENDPOINTS:
        if only and ename not in only:
            continue
        rows = [r for r in raw
                if (role is None or r.get("role") == role) and len(r.get("text", "")) >= min_len]
        if not rows:
            continue
        print("\n=== %s（n=%d）===" % (ename, len(rows)))
        for ck, w in wrong.items():
            if w is None:
                continue
            e = sum(1 for r in rows if _row_key(r) not in w)
            print("  exact  %-16s %.4f" % (ck, e / len(rows)))
        for ref in (refs or wrong.keys()):
            if wrong.get(ref) is None:
                continue
            for arm in wrong:
                if arm == ref or wrong.get(arm) is None:
                    continue
                r = paired(rows, wrong[ref], wrong[arm])
                if not r:
                    continue
                lo, hi = r["ci"]
                print("  vs %-9s %-16s Δ%+.4f  b/c=%d/%d  95%%CI[%+.4f,%+.4f]  p=%.4f"
                      % (ref, arm, r["delta"], r["b"], r["c"], lo, hi, r["p"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多 checkpoint × 多尺子的配对矩阵")
    ap.add_argument("--spec", help="spec JSON 路径")
    ap.add_argument("--set", default=None, help="只跑 spec 里的某个标签")
    ap.add_argument("--endpoint", default=None, help="只打印这些终点（逗号分隔）")
    ap.add_argument("--ref", default=None, help="只跟这些 checkpoint 比（逗号分隔）")
    ap.add_argument("--example", action="store_true", help="打印 spec 模板")
    a = ap.parse_args(argv)
    if a.example or not a.spec:
        print(EXAMPLE)
        return 0
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    only = a.endpoint.split(",") if a.endpoint else None
    refs = a.ref.split(",") if a.ref else None
    for item in spec:
        if a.set and item["name"] != a.set:
            continue
        report(item["name"], Path(item["ruler"]), item["dumps"], only=only, refs=refs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
