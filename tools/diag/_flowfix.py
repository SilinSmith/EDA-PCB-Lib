"""第三批小修：把外扩参数从 ev_ocr 收敛到 ③ 节点（避免两处并存讲同一件事），
并把 ③ 的 ins/outs 与编号打印出来核对。幂等。"""
import json
import re
from pathlib import Path

HTML = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)


def main():
    s = HTML.read_text(encoding="utf-8")
    m = PAT.search(s)
    g = json.loads(m.group(2))
    for n in g["nodes"]:
        if n["id"] == "ev_ocr":
            n["params"] = [
                ["解码", "按 w//8 截断"],
                ["CHUNK", "256"],
                ["外扩 / 裁剪", "见 ③ 裁剪尺度归一（同一个 stage2/text.py）"],
            ]
    blob = json.dumps(g, ensure_ascii=False, indent=1)
    HTML.write_text(s[: m.start(2)] + blob + s[m.end(2):], encoding="utf-8")

    print("nodes=%d links=%d" % (len(g["nodes"]), len(g["links"])))
    for n in g["nodes"]:
        if n["id"] == "ev_crop":
            print("\n[ev_crop] title=%s  x=%s y=%s w=%s" % (n["title"], n["x"], n["y"], n["w"]))
            print("  file=%s" % n["file"])
            print("  ins =%s" % [i["id"] for i in n["ins"]])
            print("  outs=%s" % [o["id"] for o in n["outs"]])
            for p in n["params"]:
                print("  param %-12s %s" % (p[0], p[1]))
    print("\n证据层编号：")
    for n in g["nodes"]:
        if n["group"] == "ev":
            print("  y=%-5s %s" % (n["y"], n["title"]))
    print("\n进度标记：")
    for n in g["nodes"]:
        nt = n.get("note") or ""
        if "**进度：" in nt:
            print("  %-11s %s" % (n["id"], nt.split("\n")[0][:70]))


if __name__ == "__main__":
    main()
