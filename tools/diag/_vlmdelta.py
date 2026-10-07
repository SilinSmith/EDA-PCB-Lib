# -*- coding: utf-8 -*-
"""量 VLM 节点到底改了什么：逐 case 对比 runs/stage2 与 runs/stage3_vlm 的产物。

只读。回答「ev_vlm 这个节点有没有发挥作用」。
用法: python work/tools/diag/_vlmdelta.py [run_dir]
"""
import json
import sys
from collections import Counter
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\_submit_out220r32\runs")


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__err__": str(e)}


def keys_of(d):
    """把一个 case 的产物压成 {路径: 标量}。"""
    out = {}

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + "/" + str(k))
        elif isinstance(o, list):
            out[path + "/#"] = len(o)
            for i, v in enumerate(o):
                walk(v, path + "/%d" % i)
        else:
            out[path] = o

    walk(d, "")
    return out


def main():
    for sub in ("stage1", "stage2", "stage3", "stage3_vlm"):
        d = RUN / sub
        n = len(list(d.glob("*.json"))) if d.is_dir() else -1
        print("  %-12s %d 个文件" % (sub, n))

    s2 = sorted((RUN / "stage2").glob("*.json"))
    vlm = RUN / "stage3_vlm"
    print("\n逐 case 对比 stage2 vs stage3_vlm：")
    tot = Counter()
    changed_cases = 0
    field_counter = Counter()
    for p in s2:
        q = vlm / p.name
        if not q.exists():
            tot["缺 vlm 产物"] += 1
            continue
        a, b = keys_of(load(p)), keys_of(load(q))
        ka, kb = set(a), set(b)
        dif = [k for k in ka & kb if a[k] != b[k]]
        # 归一化字段名（去掉 case 特有的编号）
        import re
        for k in dif:
            field_counter[re.sub(r"/\d+", "/#", k)] += 1
        tot["case 数"] += 1
        tot["字段改动总数"] += len(dif)
        if dif:
            changed_cases += 1
        tot["只在 stage2"] += len(ka - kb)
        tot["只在 vlm"] += len(kb - ka)
    for k, v in tot.items():
        print("  %-16s %d" % (k, v))
    print("  变了的 case 数     %d / %d" % (changed_cases, tot["case 数"]))
    print("\n改动最多的字段路径（归一化后）：")
    for k, c in field_counter.most_common(25):
        print("  %-60s %d" % (k, c))


if __name__ == "__main__":
    main()
