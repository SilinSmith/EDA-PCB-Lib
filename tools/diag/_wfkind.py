# -*- coding: utf-8 -*-
"""改 `workflow.html` 里节点的 `kind`（徽标），用 `json` 模块安全改写内嵌 graphdata。

为什么不用手改：`graphdata` 是 `<script type="application/json">` 里的一行长 JSON，
直接在编辑器里改 `kind` 很容易把 `\\n` 等转义弄坏（`_wfhead.py` 就是为同一个理由写的）。

用法：
    python work/tools/diag/_wfkind.py --list                 # 列出现有 kind 分布 + 每个 new 节点
    python work/tools/diag/_wfkind.py --set exec know pool   # 把 know/pool 改成 exec
    python work/tools/diag/_wfkind.py --set exec --all-new   # 把**全部** new 改成 exec
    python work/tools/diag/_wfkind.py --set new boxscan      # 反向（需要时）
默认 `--dry`，加 `--write` 才落盘（先备份 `.bakkind`）。

⚠ 必须**在 `_wfbuild.py --write` 之前**跑：`_wfbuild.py` 只重写 bullets/params/布局，
   不碰 `kind`，但它是从 `workflow.html` 读进来再整份写回去的。
"""
import argparse
import io
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "work" / "flow" / "workflow.html"
RE = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)',
                re.S)


def load(path: Path):
    txt = io.open(path, encoding="utf-8").read()
    m = RE.search(txt)
    if not m:
        raise SystemExit("没找到 graphdata：%s" % path)
    return txt, m, json.loads(m.group(2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(SRC))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", default="")
    ap.add_argument("--all-new", action="store_true")
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    path = Path(a.file)
    txt, m, d = load(path)
    by = {n["id"]: n for n in d["nodes"]}

    if a.list or not a.set:
        print("kind 分布：", dict(Counter(n.get("kind") for n in d["nodes"])))
        for n in d["nodes"]:
            if n.get("kind") == "new":
                print("  ★ %-14s %s" % (n["id"], (n.get("name") or "")[:46]))
        return 0

    ids = list(a.ids)
    if a.all_new:
        ids += [n["id"] for n in d["nodes"] if n.get("kind") == "new"]
    ids = [i for i in dict.fromkeys(ids) if i]
    bad = [i for i in ids if i not in by]
    if bad:
        raise SystemExit("没有这些节点：%s" % bad)
    for i in ids:
        n = by[i]
        print("  %-14s %-6s → %s" % (i, n.get("kind"), a.set))
        n["kind"] = a.set

    if not ids:
        print("没有要改的节点")
        return 0
    if not a.write:
        print("（dry-run，加 --write 才落盘）")
        return 0

    bak = path.with_suffix(path.suffix + ".bakkind")
    if not bak.exists():
        shutil.copy2(path, bak)
    new_json = json.dumps(d, ensure_ascii=False, indent=1)
    out = txt[:m.start(2)] + new_json + txt[m.end(2):]
    io.open(path, "w", encoding="utf-8", newline="").write(out)
    print("-> %s（备份 %s）" % (path, bak.name))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
