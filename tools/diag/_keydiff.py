"""对比两份文档的元件键集合，看融合层到底改了什么。

用法（workdir=Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_keydiff.py <case> [<base_dir> <new_dir>]

默认 base=work\\_fusion_vlm  new=work\\_fusion_out220nl
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def keys_of(doc):
    return set(doc.get("components") or {})


def main(argv):
    case = argv[1]
    bdir = Path(argv[2]) if len(argv) > 2 else ROOT / "work" / "_fusion_vlm"
    ndir = Path(argv[3]) if len(argv) > 3 else ROOT / "work" / "_fusion_out220nl"
    a = load(bdir / ("%s.json" % case))
    b = load(ndir / ("%s.json" % case))
    ka, kb = keys_of(a), keys_of(b)
    print("case %s" % case)
    print("  基线 %d 键：%s" % (len(ka), sorted(ka)))
    print("  新件 %d 键：%s" % (len(kb), sorted(kb)))
    print("  只在基线：%s" % sorted(ka - kb))
    print("  只在新件：%s" % sorted(kb - ka))
    # 引脚：谁带着引脚
    for tag, d in (("基线", a), ("新件", b)):
        pins = d.get("pins") or {}
        cnt = {k: len(v) for k, v in pins.items()}
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:8]
        print("  %s pins 段：%d 个键，最多的 %s" % (tag, len(pins), top))
    # 网络
    for tag, d in (("基线", a), ("新件", b)):
        nets = d.get("nets") or {}
        if isinstance(nets, dict):
            items = list(nets.items())[:6]
        else:
            items = list(enumerate(nets))[:6]
        print("  %s nets=%d 前几条：%s" % (tag, len(nets), items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
