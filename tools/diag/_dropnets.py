# -*- coding: utf-8 -*-
"""离线：丢掉「没有任何引脚引用」的网络（可选同时要求无 edges），再用 fast.py --score-only 打分。

用法：
    python -X utf8 work\\_dropnets.py --pred work\\_fast_out_leadon --out drlead [--keep-edged]
    python -X utf8 eda-pcb\\tools\\eval\\fast.py --score-only --cases 1,...,10 --out drlead
--keep-edged = 只丢「无引用 **且** 无 edges」的网（保守版，等价于历史 work/_dropempty.py）。
默认 = 丢掉所有「无引用」的网（哪怕它有 edges）。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import metrics as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-edged", action="store_true")
    a = ap.parse_args()
    src = Path(a.pred)
    if not src.is_absolute():
        src = ROOT / src
    dst = ROOT / "work" / ("_fast_out_" + a.out)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    tot = 0
    for fp in sorted((dst / "final_results").glob("*.json")):
        doc = json.loads(fp.read_text(encoding="utf-8"))
        nets = doc.get("nets") or {}
        drop = []
        for k, v in nets.items():
            v = v or {}
            if M.parse_hypergraph(v.get("hyperGraph") or ""):
                continue
            if a.keep_edged and (v.get("edges") or {}):
                continue
            drop.append(k)
        for k in drop:
            nets.pop(k)
        if drop:
            tot += len(drop)
            fp.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                          encoding="utf-8")
            print("  %s 删 %d 个：%s" % (fp.name, len(drop), ",".join(drop)))
    print("[dropnets] 合计删 %d 个无引用网（keep_edged=%s）-> %s"
          % (tot, a.keep_edged, dst))


main()
