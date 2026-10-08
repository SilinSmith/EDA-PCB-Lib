# -*- coding: utf-8 -*-
"""元器件库（官方 cpntLibrary）对 GT 的覆盖率体检。

回答三个问题：
  1) GT 里 box/IC 的 Name 有多少能命中官方库（精确 / 归一化 / 模糊）？
  2) GT 里 r/c 的 value 有多少能命中库键（库键里确实混着 `3.3V`、`25MHz` 这类值）
  3) 命中库的 IC，它的 GT pinname 有多少出现在库记录的 name 集合里（决定「引脚名先验」值不值得做）

用法：
    python -X utf8 work\\tools\\diag\\_libcov.py                 # 9 张官方 GT + 200 训练案例
    python -X utf8 work\\tools\\diag\\_libcov.py --fuzzy 0.85
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # work\tools\diag\_libcov.py -> 仓库根
GT9 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
GT200 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "200_train_cases"
LIB = ROOT / "work" / "data" / "cpntLibrary.json"
LIB_OLD = ROOT / "work" / "data" / "BoxICLibrary_200cases.json"

BOX_TYPES = {"box", "ic", "boxic"}
VALUE_TYPES = {"r", "resistor", "c", "capacitor"}


def norm(s: str) -> str:
    """归一化：大小写无关、去掉所有非字母数字（`1.25-2PWB` -> `1252pwb`）。"""
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Lib:
    def __init__(self, raw: dict):
        self.raw = raw
        self.keys = list(raw)
        self.by_norm: dict[str, list[str]] = {}
        for k in self.keys:
            self.by_norm.setdefault(norm(k), []).append(k)

    def exact(self, text: str) -> list[str]:
        t = (text or "").strip()
        return [t] if t in self.raw else []

    def normalized(self, text: str) -> list[str]:
        return self.by_norm.get(norm(text), [])

    def fuzzy(self, text: str, cutoff: float, topk: int = 3) -> list[tuple[str, float]]:
        t = norm(text)
        if not t:
            return []
        return difflib.get_close_matches(t, list(self.by_norm), n=topk, cutoff=cutoff) and [
            (self.by_norm[c][0], difflib.SequenceMatcher(None, t, c).ratio())
            for c in difflib.get_close_matches(t, list(self.by_norm), n=topk, cutoff=cutoff)
        ]

    def pin_names(self, key: str) -> Counter:
        c: Counter = Counter()
        for p in self.raw.get(key, {}).get("pins", []):
            if p.get("name"):
                c[p["name"]] += 1
        return c

    def pin_numbers(self, key: str) -> Counter:
        c: Counter = Counter()
        for p in self.raw.get(key, {}).get("pins", []):
            if p.get("number"):
                c[str(p["number"])] += 1
        return c


def cases(root: Path) -> list[tuple[str, Path]]:
    out = []
    for d in sorted(root.iterdir() if root.is_dir() else []):
        if not d.is_dir():
            continue
        js = sorted(d.glob("*.json"))
        if js:
            out.append((d.name, js[0]))
    return out


def audit(case_list, lib: Lib, cutoff: float, label: str, verbose: int = 0) -> dict:
    st = dict(case=label, n_case=len(case_list))
    buck = Counter()          # IC 名命中桶
    val_buck = Counter()      # r/c value 命中桶
    pin_hit = pin_tot = 0     # pinname 覆盖率
    num_hit = num_tot = 0     # pinnumber 覆盖率
    miss_names: Counter = Counter()
    miss_vals: Counter = Counter()
    for cname, jf in case_list:
        doc = json.loads(jf.read_text(encoding="utf-8"))
        comps = doc.get("components") or {}
        pins = doc.get("pins") or {}
        for k, v in comps.items():
            t = str(v.get("type", "")).lower()
            name = v.get("Name") or ""
            if t in BOX_TYPES:
                if lib.exact(name):
                    buck["exact"] += 1
                elif lib.normalized(name):
                    buck["norm"] += 1
                elif lib.fuzzy(name, cutoff):
                    buck["fuzzy"] += 1
                else:
                    buck["miss"] += 1
                    miss_names[name] += 1
                    if verbose:
                        print("    [IC-miss] %s %s %r" % (cname, k, name))
                # 引脚名/号覆盖率（只看能命中的）
                cand = (lib.exact(name) or lib.normalized(name) or
                        [x[0] for x in lib.fuzzy(name, cutoff)])
                if cand:
                    key = cand[0]
                    ln, lnum = lib.pin_names(key), lib.pin_numbers(key)
                    for pk, pv in (pins.get(k) or {}).items():
                        pn = pv.get("pinname") or ""
                        num = re.sub(r"^pin_", "", pk)
                        pin_tot += 1
                        if pn and pn in ln:
                            pin_hit += 1
                        num_tot += 1
                        if str(num) in lnum:
                            num_hit += 1
            elif t in VALUE_TYPES:
                val = v.get("value") or ""
                if not val:
                    val_buck["empty"] += 1
                elif lib.exact(val) or lib.normalized(val):
                    val_buck["hit"] += 1
                elif lib.fuzzy(val, cutoff):
                    val_buck["fuzzy"] += 1
                else:
                    val_buck["miss"] += 1
                    miss_vals[val] += 1
    st.update(ic=buck, value=val_buck,
              pin_name_cov=(pin_hit / pin_tot if pin_tot else 0.0), pin_n=pin_tot,
              pin_num_cov=(num_hit / num_tot if num_tot else 0.0))
    st["miss_names"] = miss_names.most_common(12)
    st["miss_vals"] = miss_vals.most_common(12)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzzy", type=float, default=0.85)
    ap.add_argument("--lib", default=str(LIB))
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    lib = Lib(load(Path(a.lib)))
    print("库: %s  键数 %d" % (a.lib, len(lib.keys)))
    if Path(a.lib) == LIB and LIB_OLD.exists():
        old = Lib(load(LIB_OLD))
        print("对照组: BoxICLibrary_200cases.json 键数 %d（两库并集 %d）"
              % (len(old.keys), len(set(lib.keys) | set(old.keys))))

    for root, label in ((GT9, "官方9张"), (GT200, "200训练案例")):
        cl = cases(root)
        if not cl:
            print("跳过 %s（未找到 %s）" % (label, root))
            continue
        st = audit(cl, lib, a.fuzzy, label, a.verbose)
        print("\n===== %s（%d 例）=====" % (label, st["n_case"]))
        ic = st["ic"]
        tot = sum(ic.values())
        print("  IC 名命中: 精确 %d / 归一化 %d / 模糊(%.2f) %d / 未命中 %d   合计 %d"
              % (ic["exact"], ic["norm"], a.fuzzy, ic["fuzzy"], ic["miss"], tot))
        if tot:
            print("  可命中率 %.1f%%" % (100.0 * (tot - ic["miss"]) / tot))
        print("  引脚名覆盖率 %.1f%% (n=%d)   引脚号覆盖率 %.1f%%"
              % (100 * st["pin_name_cov"], st["pin_n"], 100 * st["pin_num_cov"]))
        vb = st["value"]
        vt = sum(vb.values())
        if vt:
            print("  r/c value: 命中 %d / 模糊 %d / 未命中 %d / 空 %d   (命中率 %.1f%%)"
                  % (vb["hit"], vb["fuzzy"], vb["miss"], vb["empty"],
                     100.0 * (vb["hit"] + vb["fuzzy"]) / max(1, vt - vb["empty"])))
        if st["miss_names"]:
            print("  未命中的 IC 名 top:", ", ".join("%s×%d" % (n, c) for n, c in st["miss_names"]))
        if st["miss_vals"]:
            print("  未命中的 value top:", ", ".join("%s×%d" % (n, c) for n, c in st["miss_vals"]))


if __name__ == "__main__":
    main()
