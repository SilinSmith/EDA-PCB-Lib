"""第十批·补：把「闸门 A/B 判负」这条 bullet 插到 _wfbullets.py 的 ev_pin 里。

幂等守卫：文件里一旦出现 `_gateab.py` 就跳过（因为插入的文本里带这个标记，
用「old 是否唯一」判不出来 —— old 是 new 的前缀）。
"""
import argparse
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EDA = ROOT.parent / "eda-pcb"
LIVE = ROOT / "flow" / "_wfbullets.py"
SNAP = EDA / "tools" / "diag" / "_wfbullets.py"

MARK = "_gateab.py"

OLD = '附近 45px 没槽位 6",'
NEW = OLD + "\n" + ' "⛔ **那 23 根不是「关掉闸门」就能拿回来的**（离线单变量 A/B `work\\tools\\diag\\_gateab.py`：直接用已落盘的 `runs/stage1` + `runs/stage2/_assigned` 重跑 `assemble.build_doc`、改模块全局切开关 ⇒ **纯 CPU、不占 GPU、不重跑 OCR**，再 `stage2_eval.py` 打分）：**官方 9 张 52.6510 → 52.4987（−0.15，判负）**、`pin.name_mismatch` 16→41、`pin.extra` 6→11，而 **tp 一根没多**（34→34），换来的只有 `net_hypergraph` +0.030；**全量 220 张反而 +0.10**（26.0962 → 26.1978）⇒ **闸门保护的是精度，且效果分档相关**。要真拿那 23 根，必须**把名字读出来**（`VLM_PIN_CHANNEL=1` + `VLM_PIN_MERGE=1`）",'


def apply(path, write):
    txt = path.read_text(encoding="utf-8")
    if MARK in txt:
        print("%s  已有 %s，跳过" % (path, MARK))
        return True
    n = txt.count(OLD)
    print("%s  命中 %d" % (path, n))
    if n != 1:
        print("  ！不唯一，跳过")
        return False
    txt = txt.replace(OLD, NEW)
    if write:
        shutil.copy2(path, str(path) + ".bak10b")
        path.write_text(txt, encoding="utf-8")
        print("  已写入（备份 .bak10b）")
    else:
        print("  dry-run：未写入")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sync", action="store_true")
    a = ap.parse_args()
    ok = apply(LIVE, a.write)
    if a.sync:
        if not ok:
            print("work 那份没改成功，不 sync")
            return 1
        shutil.copy2(LIVE, SNAP)
        print("已同步 -> %s" % SNAP)
    else:
        apply(SNAP, a.write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
