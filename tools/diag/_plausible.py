# -*- coding: utf-8 -*-
"""`plausible` 节点（21 节点目标架构里那一格）的第一版落地：**不需要 LLM 的结构性矛盾体检**。

用户 m12477 的原话是「假如 LLM 发现现有的体系中 VCC 和 GND 短接（only LLM 知道这意味着什么）
那很明显这个电路识别错了，要在里面找错误点」。第一版先做**纯几何/电气规则**能做到的部分 ——
因为规则命中率如果很低，就不值得为它花 LLM 的钱。

规则（每条都产出一段「可疑 + 依据」）：
  `R1 电源与地短接`   同一个网络里既有电源轨引脚、又有地轨引脚。
  `R2 去耦电容同一端` 一颗电容的两个脚落在同一个网络里（电容就没起作用了，必是识别错）。
  `R3 地脚不在带 GND 的网` 引脚名叫 GND，但它所在的网络里没有任何「地」名的引脚/标签。
  `R4 网络只有一个脚`  单元素网络（`_netwhy.py` 的 `C` 类就住在这里）。

用法：`python work\\tools\\diag\\_plausible.py [--pred 目录] [--cases 1,2,3] [--all] [--show 8]`
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402

TEN = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
FAST = TEN + ["0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]

GROUND = {"gnd", "gnda", "gndd", "agnd", "dgnd", "pgnd", "vss", "vssa", "0v"}
POWER = {"vcc", "vdd", "vddd", "vdda", "vddio", "3v3", "5v", "12v", "vbat",
         "vbus", "vin", "vout", "avdd", "dvdd", "vdd_3v3", "vcc_3v3"}


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def refs_of(hg: str):
    out = []
    for tok in str(hg or "").strip().strip("()").split(","):
        tok = tok.strip()
        if "." in tok:
            c, _, p = tok.rpartition(".")
            out.append((c.strip(), p.strip()))
    return out


def nm_of(pins, comp, num):
    return str((((pins or {}).get(comp) or {}).get("pin_%s" % num) or {})
               .get("pinname") or "").strip().lower()


def check(pr: dict) -> dict:
    pins = pr.get("pins") or {}
    nets = pr.get("nets") or {}
    r = collections.Counter()
    hits = collections.defaultdict(list)
    for n, rec in nets.items():
        refs = refs_of(rec.get("hyperGraph", ""))
        lab = str(rec.get("label") or "").strip().lower()
        names = [nm_of(pins, c, p) for c, p in refs]
        gs = [x for x in (names + [lab]) if x in GROUND]
        ps = [x for x in (names + [lab]) if x in POWER]
        r["nets"] += 1
        if len(refs) == 1:
            r["R4_单元素网络"] += 1
        if gs and ps:
            r["R1_电源与地短接"] += 1
            hits["R1"].append((n, len(refs), sorted(set(gs))[:3], sorted(set(ps))[:3]))
        for c, p in refs:
            if c not in pins:
                continue
            # 电容两个脚是否落在同一个网里
            if str(c)[:1].upper() not in ("C",):
                continue
        comps_in = collections.Counter(c for c, _ in refs)
        for c, cnt in comps_in.items():
            if cnt >= 2 and str(c)[:1].upper() == "C":
                r["R2_电容同一端"] += 1
                hits["R2"].append((n, c, cnt))
        for c, p in refs:
            if nm_of(pins, c, p) in GROUND and not gs:
                r["R3_地脚不在带地名的网"] += 1
                hits["R3"].append((n, c, p))
    return {"r": r, "hits": hits}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or (FAST if a.all else TEN)
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    agg = collections.Counter()
    for cid in cases:
        c, _how = resolve(cid, exact, digits)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        out = check(pr)
        agg.update(out["r"])
        flag = " ".join("%s=%d" % (k.split("_")[0], out["r"][k])
                        for k in ("R1_电源与地短接", "R2_电容同一端",
                                  "R3_地脚不在带地名的网", "R4_单元素网络") if out["r"][k])
        print("%-7s 网 %3d ｜ %s" % (cid, out["r"]["nets"], flag or "（干净）"))
        if a.show:
            for tag in ("R1", "R2"):
                for h in out["hits"][tag][:a.show]:
                    print("        %s %s" % (tag, h))
    print("\n== 合计")
    for k in ("nets", "R1_电源与地短接", "R2_电容同一端",
              "R3_地脚不在带地名的网", "R4_单元素网络"):
        print("   %-22s %6d" % (k, agg[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
