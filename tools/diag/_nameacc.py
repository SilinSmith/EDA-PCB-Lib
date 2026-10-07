"""IC 框的 `Name`（器件型号）到底读对没有 —— `component_text_ok` 对 IC_TYPES 比的就是它，
读错一处是 `fp+=1; fn+=1; 无 tp`（一笔亏三处），权重 0.30。

用法（从 `Y:\\workspace\\EDA` 根跑）：
    python work\\tools\\diag\\_nameacc.py
    python work\\tools\\diag\\_nameacc.py --pred work\\_submit_out220w\\runs\\stage3_vlm
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover  # noqa: E402
from schematic_parser.metrics import IC_TYPES, canon_type              # noqa: E402

FAST = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
        "0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="work/_submit_out220w/runs/stage2")
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    cases = [c for c in a.cases.split(",") if c] or FAST
    pdir = ROOT / a.pred

    idx = {}
    for sp in SPLITS:
        for c in discover(DEFAULT_DATA_ROOT, splits=(sp,)):
            idx.setdefault(c.case_id, c)

    tot = hit = 0
    rows = []
    bad = []
    for cid in cases:
        c = idx.get(cid)
        p = pdir / ("%s.json" % cid)
        if c is None or not p.exists():
            continue
        gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pr = json.loads(p.read_text(encoding="utf-8-sig"))
        n = h = 0
        for k, v in (gt.get("components") or {}).items():
            if canon_type((v or {}).get("type")) not in IC_TYPES:
                continue
            if k not in (pr.get("components") or {}):
                continue
            gname = str((v or {}).get("Name") or "").strip()
            pname = str((pr["components"][k] or {}).get("Name") or "").strip()
            n += 1
            if gname and gname == pname:
                h += 1
            else:
                bad.append((cid, k, gname, pname))
        tot += n
        hit += h
        if n:
            rows.append((cid, c.split, n, h))
    print("预测目录:", a.pred)
    print("%-8s %-18s %6s %6s" % ("case", "split", "IC框", "Name对"))
    for cid, sp, n, h in rows:
        print("%-8s %-18s %6d %6d" % (cid, sp, n, h))
    print("合计 IC 框 %d，Name 完全一致 %d（%.3f）" % (tot, hit, (hit / tot) if tot else 0.0))
    print("---- 错的前 25 个 ----")
    for cid, k, g, p in bad[:25]:
        print("   %-8s %-6s GT=%-26s PR=%s" % (cid, k, g or "(空)", p or "(空)"))


if __name__ == "__main__":
    main()
