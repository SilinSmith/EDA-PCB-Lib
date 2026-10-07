"""按 id 比对 workflow.html 与备份（而不是按下标），并扫描残留的旧数字。"""
import json
import re
import sys
from pathlib import Path

HTML = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)

STALE = ["27.33", "49.29", "29.83", "26.34", "0.30h", "0.30×h",
         "PAD_FRAC=0.30", "PAD_FRAC_Y=0.30", "PAD_MIN_Y=2", "pad_min_y=2"]


def load(p):
    s = Path(p).read_text(encoding="utf-8")
    return json.loads(PAT.search(s).group(2))


def main():
    old = load(sys.argv[1]) if len(sys.argv) > 1 else None
    new = load(HTML)
    if old:
        on = {n["id"]: n for n in old["nodes"]}
        nn = {n["id"]: n for n in new["nodes"]}
        print("按 id 比对的真实变化：")
        for k in sorted(set(on) | set(nn)):
            if k not in on:
                print("  + 新增 %s" % k)
            elif k not in nn:
                print("  - 删除 %s" % k)
            elif json.dumps(on[k], ensure_ascii=False, sort_keys=True) != \
                 json.dumps(nn[k], ensure_ascii=False, sort_keys=True):
                dif = []
                for f in sorted(set(on[k]) | set(nn[k])):
                    a = json.dumps(on[k].get(f), ensure_ascii=False, sort_keys=True)
                    b = json.dumps(nn[k].get(f), ensure_ascii=False, sort_keys=True)
                    if a != b:
                        dif.append(f)
                print("  ~ %-11s 字段: %s" % (k, ",".join(dif)))
        print("  links: %d -> %d" % (len(old["links"]), len(new["links"])))

    print("\n残留旧数字扫描：")
    blob = json.dumps(new, ensure_ascii=False)
    for t in STALE:
        c = blob.count(t)
        print("  %-16s %s" % (t, ("⚠ 出现 %d 次" % c) if c else "ok"))

    print("\n各节点 note 长度与首行：")
    for n in new["nodes"]:
        nt = (n.get("note") or "")
        print("  %-11s %5d  %s" % (n["id"], len(nt), nt.split("\n")[0][:60]))


if __name__ == "__main__":
    main()
