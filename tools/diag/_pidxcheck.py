import json, os, sys
sys.path.insert(0, ".")
from stage2.assemble import build_doc
RUN = r"..\work\_submit_out220r36\runs"
ok = bad = 0
for case in ["0001","0002","0003","0004","0005"]:
    s1 = json.load(open(os.path.join(RUN,"stage1",case+".stage1.json"), encoding="utf-8"))
    a  = json.load(open(os.path.join(RUN,"stage2","_assigned",case+".json"), encoding="utf-8"))
    ref= json.load(open(os.path.join(RUN,"stage2",case+".json"), encoding="utf-8"))
    d  = build_doc(s1, a, case)
    pidx = d.pop("__pin_index", None)
    same = (json.dumps(d, ensure_ascii=False, sort_keys=True)
            == json.dumps(ref, ensure_ascii=False, sort_keys=True))
    n = sum(len(v) for v in (pidx or {}).values())
    print("%s  交付文档逐字节等价=%s   pin_index 元件=%d 引脚=%d" % (
        case, same, len(pidx or {}), n))
    ok += same; bad += (not same)
print("---- 等价 %d / 不等价 %d" % (ok, bad))
d = build_doc(json.load(open(os.path.join(RUN,"stage1","0001.stage1.json"), encoding="utf-8")),
              json.load(open(os.path.join(RUN,"stage2","_assigned","0001.json"), encoding="utf-8")), "0001")
pi = d["__pin_index"]
k0 = sorted(pi)[0]
print("样例 0001 %s 的引脚索引->键:" % k0, json.dumps(pi[k0], ensure_ascii=False))
