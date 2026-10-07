# -*- coding: utf-8 -*-
"""把 `case_*.html` 里内嵌的 `<script>` 抽出来，配一套最小 DOM 桩，用 node 真跑一遍。

目的是在把页面交给用户之前，先抓出 JS 的语法错与运行期异常
（静态看代码抓不到 `D.xxx` 拼错这类问题）。
用法：python work/tools/viz/_jstest.py [case_0001.html ...]
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]

STUB = r"""
// ---- 最小 DOM 桩 ----
const _store = {};
function _el(id){
  if(!_store[id]) _store[id] = {
    id, innerHTML:'', textContent:'', title:'', className:'', style:{}, dataset:{},
    classList:{toggle(){},add(){},remove(){}},
    appendChild(){}, addEventListener(){}, querySelectorAll(){return [];},
    querySelector(){return null;},
  };
  return _store[id];
}
global.document = {
  getElementById:_el,
  createElement(){ return {style:{}, dataset:{}, classList:{toggle(){}}, appendChild(){}, title:'', textContent:'', className:''}; },
  querySelectorAll(){ return []; },
  querySelector(){ return null; },
  addEventListener(){},
};
global.location = {href:''};
global.window = global;
global.Event = function(n){ this.type=n; };
"""


def extract(p: pathlib.Path) -> str:
    t = p.read_text(encoding="utf-8")
    i = t.rfind("<script>")
    j = t.rfind("</script>")
    if i < 0 or j < 0:
        raise SystemExit("找不到 <script>：%s" % p)
    return t[i + len("<script>"):j]


def main() -> int:
    args = sys.argv[1:]
    files = ([ROOT / "work" / "viz" / a for a in args] if args
             else sorted((ROOT / "work" / "viz").glob("case_*.html")))
    bad = 0
    for p in files:
        js = STUB + extract(p)
        tmp = ROOT / "work" / "viz" / "_jstest_run.js"
        tmp.write_text(js, encoding="utf-8")
        r = subprocess.run(["node", str(tmp)], capture_output=True, text=True, encoding="utf-8")
        ok = r.returncode == 0
        if not ok:
            bad += 1
        print("%-22s %s" % (p.name, "OK" if ok else "FAIL"))
        if not ok:
            tail = (r.stderr or "").strip().splitlines()
            for line in tail[:14]:
                print("     ", line)
    print("失败 %d / %d" % (bad, len(files)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
