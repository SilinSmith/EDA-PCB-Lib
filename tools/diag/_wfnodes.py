# -*- coding: utf-8 -*-
"""看 workflow.html 里指定节点的 file / note / params（用法：python _wfnodes.py [id ...]）。"""
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "work" / "flow" / "workflow.html"
RE = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)


def load():
    s = P.read_text(encoding="utf-8")
    m = RE.search(s)
    return s, m, json.loads(m.group(2))


def main():
    want = sys.argv[1:] or ["cropfix", "audit", "boxscan"]
    _, _, d = load()
    for n in d["nodes"]:
        if n["id"] in want:
            print("==== %s  %s" % (n["id"], n.get("title")))
            print("  file  : %s" % n.get("file"))
            print("  sub   : %s" % n.get("sub"))
            print("  note  : %s" % (n.get("note") or "").replace("\n", " | "))
            ps = n.get("params") or {}
            if isinstance(ps, dict):
                for k, v in ps.items():
                    print("  param : %s = %s" % (k, v))
            elif ps:
                print("  params: %s" % ps)
            print()


if __name__ == "__main__":
    main()
