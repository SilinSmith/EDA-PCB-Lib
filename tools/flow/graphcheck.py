# -*- coding: utf-8 -*-
"""`graphcheck.py` —— 工作流图的**接口注册表 + 统一 + 校验器**（用户 m40034 第二项）。

用户原话：「其中的很多模块直接使用的接口你也要统一起来，不然没办法重组」。

病灶（本工具治的四件事）：
1. **端口只是手写的标签**：每项只有 `{id,label}`，没有数据类型，也没有到代码的绑定 ⇒
   把两个节点接起来时没人能判断「这个输出能不能喂给那个输入」。
2. **页面比 `graph.json` 新**：`work/flow/workflow.html`（31 节点 / 65 连线）是历次
   `tools/diag/_flowpatch*.py` 直接补出来的，源头 `graph.json` 落后（26/52）⇒ 没有单一真源。
3. **悬空连线**：`ev_pin.det`、`metrics.file` 被当输出用，但节点没声明这个输出端口
   ⇒ `build.py` 的 `check()` 报错，渲染器 `portPos()` 取不到元素，那三条线**画不出来**。
4. **同名不同义**：`pool.fix` / `fixstr.fixed` / `audit.fix` 是同一个东西的三个名字；
   `rescan.q` / `llm_judge.plan` / `audit.q` 同理 ⇒ 无法重组。

四件事对应四步（`--all` 按序全做）：
* `--sync`  页面内嵌 `graphdata` → `graph.json`（先备份）⇒ 真源归位；
* `--fix`   统一的**接口改写表**：端口改名（同名同义归一）、悬空线改接到真正产出它的端口、
  把「配置/控制」类连线标成 `kind="param"`（不做类型比对）、节点声明实现状态；
* `--fill`  给端口补 `kind`（数据类型）+ `role`（produce/sink），给节点补 `code`（`模块:符号`，只填空）；
* `--check` 校验 + 写机器可读的 `tools/flow/interfaces.json`（供重组用）。

用法（在 `eda-pcb` 下）：
    python tools\\flow\\graphcheck.py --all            # 同步 + 统一 + 补全 + 校验
    python tools\\flow\\graphcheck.py --check          # 只校验
    python tools\\flow\\graphcheck.py --all --strict   # 有错返回码 1
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDA = HERE.parent.parent                      # `eda-pcb`
ROOT = EDA.parent                             # `Y:\workspace\EDA`
GRAPH = HERE / "graph.json"
PAGE = ROOT / "work" / "flow" / "workflow.html"
IFACES = HERE / "interfaces.json"

# ---------------------------------------------------------------- ① 类型表
# **端口名就是类型名**（`ev_ocr.reads -> pool.reads`）—— 这套图一直在用这个约定，
# 本表把它显式化，才谈得上「重组时判断能不能接」。
KINDS = {
    "img": "image", "det": "det", "boxes": "boxes", "reads": "reads",
    "segs": "segments", "pins": "pins", "nets": "nets", "priors": "priors",
    "claims": "claims", "sol": "sol", "bad": "bad", "susp": "susp",
    "q": "question", "answer": "answer", "plan": "plan", "doc": "doc",
    "file": "file", "back": "back", "region": "region", "ledger": "ledger",
    "faith": "faith", "hint": "hint", "engine": "engine",
    "fixread": "reads",          # cropfix 修过之后的读数（语义上是 reads）
}
# 连线类别：data = 载体，要比对类型；param/ctrl = 配置/控制，只要求两端端口存在。
LINK_KINDS = ("data", "param", "ctrl")

# ---------------------------------------------------------------- ② 统一改写表
# 端口改名：同一个东西在不同节点上的三个名字归一（旧名 -> 新名）。
PORT_RENAME = {
    ("pool", "fix"): "fixed",        # ← fixstr.fixed / audit.fix
    ("audit", "fix"): "fixed",
    ("audit", "judge"): "answer",    # ← llm_judge.answer（审计读的是裁决记录）
    ("audit", "q"): "plan",          # ← 要重裁的槽位，本质是任务清单
    ("conflict", "q"): "plan",
    ("llm_judge", "q"): "plan",
    ("rescan", "q"): "plan",
    ("cropfix", "boxes"): "fixread",  # 出去的不是普通文字框，是「修正后的读数」
}
# 悬空线改接到**真正产出**它的端口（页面里那三条线指向了不存在的输出端口）。
LINK_REPOINT = {
    ("ev_pin.det", "pincol.det"): ("own.det", "pincol.det"),
    ("ev_pin.det", "boxpinprior.det"): ("own.det", "boxpinprior.det"),
    ("metrics.file", "gtcheck.file"): ("export.file", "gtcheck.file"),
}
# 配置/控制类连线：不做类型比对（`batch.plan` 是「请求方式」，不是一个数据产物）。
PARAM_LINKS = {
    ("batch.plan", "ev_vlm.engine"),
    ("batch.plan", "llm_judge.engine"),
    ("batch.plan", "rescan.engine"),
    ("boxscan.region", "ev_vlm.engine"),
}
# 合法的「末端产物」：没有下游是设计如此，不算孤儿。
SINK_PORTS = {"gtcheck.faith", "own.ledger"}

# ---------------------------------------------------------------- ③ 代码绑定
# `模块路径[:符号]`，相对 `eda-pcb/`。**符号都核对过**（`--check` 会再核一遍）。
NODE_CODE = {
    "input": "submission/run.py:main",
    "ev_sym": "stage1/detect.py:Detector",
    "ev_text": "cvparts/textbox.py:detect_boxes",
    "ev_crop": "stage2/text.py:pad_box",
    "ev_ocr": "textrec/infer.py:Recognizer",
    "ev_vlm": "vlm/enrich.py:enrich_dir",
    "ev_wire": "stage1/wires.py:extract_wires",
    "ev_pin": "stage1/pintrace.py:trace_from_boxes",
    "ev_net": "stage1/nets.py:build_nets",
    "know": "fusion/priors.py:Priors",
    "pool": "fusion/claims.py:from_stage2",
    "solve": "fusion/solve.py:solve",
    "plausible": "fusion/plausible.py:check",
    "conflict": "fusion/solve.py:open_questions",
    "llm_judge": "fusion/llm_judge.py:judge",
    "rescan": "fusion/rescan.py:run",
    "iterate": "fusion/run.py:run_case",
    "batch": "vlm/enrich.py:enrich_dir",
    "assemble": "stage2/assemble.py:build_doc",
    "export": "submission/run.py:main",
    "metrics": "schematic_parser/metrics.py:evaluate_case",
    "cropfix": "fusion/cropfix.py:run",
    "audit": "fusion/audit.py:run",
    "own": "cvparts/own.py:thin",
    "ev_bridge": "stage1/labels.py:wire_groups",
    "pincol": "stage1/pincol.py:enumerate_pins",
    "ownnum": "stage2/structmatch.py:rebind",
    "boxpinprior": "fusion/priors.py:slot_boxes",
    "gtcheck": "tools/diag/_desigsrc.py",
    # ⑦c..⑦g：队友分支 `codex/improve-pin-network-parsing` 的引脚证据/读数链（已并入，
    # 识别侧开关默认关，等 A/B 后再翻；模块与符号名逐个核对过）。
    "ev_pincolor": "stage1/pincolor.py:extract_color_pins",
    "ev_pinevid": "stage1/pin_evidence.py:build_pin_evidence",
    "ev_pinlane": "stage2/pin_lane_crops.py:pin_lane_boxes",
    "ev_pinread": "stage2/pin_lane_reading.py:read_pin_lanes",
    "ev_pinppocr": "stage2/pin_name_ocr.py:get_pin_name_ocr",
    # fixstr / boxscan 页面自己写着「已证否 / 要补」⇒ 由 NODE_STUB 声明，不编造绑定。
}
NODE_STUB = {"fixstr": "已证否", "boxscan": "要补"}


def kind_of(port_id: str) -> str:
    return KINDS.get(port_id, "object")


# ---------------------------------------------------------------- 读写
def load_page_graph(path: Path) -> dict:
    t = path.read_text(encoding="utf-8")
    m = re.search(r'id=["\']graphdata["\'][^>]*>(.*?)</script>', t, re.S)
    if not m:
        raise SystemExit("[FATAL] %s 里没有 graphdata" % path)
    return json.loads(m.group(1))


def save(g: dict, args) -> None:
    if not args.dry:
        GRAPH.write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")


def do_sync(args) -> dict:
    g = load_page_graph(PAGE)
    if not args.dry:
        shutil.copy2(GRAPH, GRAPH.with_suffix(".json.bak"))
    print("[sync] graph.json <- %s（备份 graph.json.bak）" % PAGE.name)
    print("       节点 %d  连线 %d  分组 %d"
          % (len(g["nodes"]), len(g["links"]), len(g.get("groups", []))))
    save(g, args)
    return g


def do_fix(g: dict, args) -> dict:
    """统一接口：端口改名 + 悬空线改接 + 连线类别 + 实现状态。幂等。"""
    n_ren = n_rep = n_par = n_stub = 0
    for n in g["nodes"]:
        for group in ("ins", "outs"):
            for p in (n.get(group) or []):
                new = PORT_RENAME.get((n["id"], p["id"]))
                if new and new != p["id"]:
                    p["id"] = new
                    n_ren += 1
        if n["id"] in NODE_STUB:
            n["impl"] = False
            n["stub"] = NODE_STUB[n["id"]]
            n_stub += 1
    seen = set()
    out = []
    for L in g.get("links", []):
        a, b = L.get("from", ""), L.get("to", "")
        for side in ("a", "b"):
            key = (a.split(".")[0], a.split(".", 1)[1]) if side == "a" else (b.split(".")[0], b.split(".", 1)[1])
            new = PORT_RENAME.get(key)
            if new:
                a = "%s.%s" % (key[0], new) if side == "a" else a
                b = "%s.%s" % (key[0], new) if side == "b" else b
        rp = LINK_REPOINT.get((a, b))
        if rp:
            a, b = rp
            n_rep += 1
        if (a, b) in PARAM_LINKS:
            L["kind"] = "param"
            n_par += 1
        elif L.get("kind") not in LINK_KINDS:
            L["kind"] = "data"
        k = (a, b)
        if k in seen:
            continue
        seen.add(k)
        L["from"], L["to"] = a, b
        out.append(L)
    g["links"] = out
    print("[fix]  端口改名 %d | 悬空线改接 %d | 标为 param %d | 标注未实现节点 %d"
          % (n_ren, n_rep, n_par, n_stub))
    save(g, args)
    return g


def do_fill(g: dict, args) -> dict:
    """补 `kind`/`role`/`code`（**只填空**，已有值不动）。"""
    n_kind = n_role = n_code = 0
    for n in g["nodes"]:
        if not n.get("code") and n["id"] not in NODE_STUB:
            code = NODE_CODE.get(n["id"], "")
            if code:
                n["code"] = code
                n_code += 1
        for group in ("ins", "outs"):
            for p in (n.get(group) or []):
                if not p.get("kind"):
                    p["kind"] = kind_of(p["id"])
                    n_kind += 1
        for p in (n.get("outs") or []):
            if not p.get("role"):
                p["role"] = "sink" if ("%s.%s" % (n["id"], p["id"])) in SINK_PORTS else "produce"
                n_role += 1
    print("[fill] 补 kind %d 个 / role %d 个 / code %d 个" % (n_kind, n_role, n_code))
    save(g, args)
    return g


def do_check(g: dict, args) -> int:
    bad, warn = [], []
    ins, outs = {}, {}
    for n in g["nodes"]:
        for p in (n.get("ins") or []):
            ins["%s.%s" % (n["id"], p["id"])] = p
        for p in (n.get("outs") or []):
            outs["%s.%s" % (n["id"], p["id"])] = p
    used_out, used_in = set(), set()
    n_data = 0
    for L in g.get("links", []):
        f, t, k = L.get("from", ""), L.get("to", ""), L.get("kind", "data")
        used_out.add(f)
        used_in.add(t)
        if f not in outs:
            bad.append("DANGLING  from %s（没有这个输出端口）" % f)
        if t not in ins:
            bad.append("DANGLING  to   %s（没有这个输入端口）" % t)
        if f in outs and t in ins:
            if k == "data":
                n_data += 1
                if outs[f].get("kind") != ins[t].get("kind"):
                    bad.append("KIND      %s(%s) -> %s(%s)"
                               % (f, outs[f].get("kind"), t, ins[t].get("kind")))
            elif k not in LINK_KINDS:
                bad.append("LINKKIND  %s -> %s 的 kind=%r 不认识" % (f, t, k))
    # relay/直通：同名输入 + 同名输出必须真的有入边（本图当前没有 relay，规则留着）
    for key, p in outs.items():
        if p.get("role") == "relay" and key not in used_out:
            warn.append("RELAY     声明了直通却没人接：%s" % key)
    # 输入端口没人喂
    for key in sorted(ins):
        if key not in used_in:
            warn.append("UNFED     输入端口没人喂：%s" % key)
    # 代码绑定核对
    n_nocode = n_missing_mod = n_missing_sym = n_stub = 0
    for n in g["nodes"]:
        if n.get("impl") is False:
            n_stub += 1
            warn.append("STUB      %s 未实现（%s）" % (n["id"], n.get("stub", "")))
            continue
        code = n.get("code") or ""
        if not code:
            n_nocode += 1
            bad.append("NOCODE    %s 没有代码绑定" % n["id"])
            continue
        mod, _, sym = code.partition(":")
        f = EDA / mod
        if not f.exists():
            n_missing_mod += 1
            bad.append("NOMODULE  %s -> %s（文件不存在）" % (n["id"], mod))
            continue
        if sym:
            src = f.read_text(encoding="utf-8", errors="ignore")
            if not re.search(r"(def|class)\s+%s\b" % re.escape(sym), src):
                n_missing_sym += 1
                warn.append("NOSYMBOL  %s -> %s:%s（文件里没有这个符号）" % (n["id"], mod, sym))
        else:
            warn.append("NOSYM     %s -> %s（只绑了模块，没绑符号）" % (n["id"], mod))
    orphans = sorted(k for k in outs
                     if k not in used_out and outs[k].get("role") != "sink")
    ifaces = {
        "source": str(GRAPH.relative_to(EDA)).replace("\\", "/"),
        "kinds": KINDS, "link_kinds": list(LINK_KINDS),
        "nodes": [{"id": n["id"], "title": n.get("title", ""), "group": n.get("group"),
                   "file": n.get("file", ""), "code": n.get("code", ""),
                   "impl": n.get("impl", True),
                   "ins": [{"id": p["id"], "kind": p.get("kind"), "role": p.get("role", "consume")}
                           for p in (n.get("ins") or [])],
                   "outs": [{"id": p["id"], "kind": p.get("kind"), "role": p.get("role")}
                            for p in (n.get("outs") or [])]}
                  for n in g["nodes"]],
        "links": g.get("links", []),
    }
    if not args.dry:
        IFACES.write_text(json.dumps(ifaces, ensure_ascii=False, indent=1), encoding="utf-8")
    n_dang = len([b for b in bad if b.startswith("DANGLING")])
    n_kind_err = len([b for b in bad if b.startswith("KIND")])
    print("[check] 节点 %d（in 端口 %d / out 端口 %d）连线 %d（其中 data %d）"
          % (len(g["nodes"]), len(ins), len(outs), len(g.get("links", [])), n_data))
    print("        悬空 %d | 类型不符 %d | 无 code %d | 模块缺失 %d | 符号缺失 %d | 未实现节点 %d"
          % (n_dang, n_kind_err, n_nocode, n_missing_mod, n_missing_sym, n_stub))
    print("        孤儿输出端口 %d：%s" % (len(orphans), ", ".join(orphans) or "无"))
    for s in bad:
        print("   ✗ " + s)
    for s in warn:
        print("   ! " + s)
    if not bad:
        print("        ✓ 全部连线可解析、data 线类型一致、代码绑定在位")
    print("[check] 接口清单 -> %s" % IFACES.name)
    return 1 if (args.strict and bad) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="页面 graphdata -> graph.json")
    ap.add_argument("--fix", action="store_true", help="统一接口（改名/改接/param/实现状态）")
    ap.add_argument("--fill", action="store_true", help="补 kind/role/code（只填空）")
    ap.add_argument("--check", action="store_true", help="校验 + 写 interfaces.json")
    ap.add_argument("--all", action="store_true", help="sync + fix + fill + check")
    ap.add_argument("--strict", action="store_true", help="有错时返回码 1")
    ap.add_argument("--dry", action="store_true", help="只打印，不写文件")
    a = ap.parse_args()
    if a.all:
        a.sync = a.fix = a.fill = a.check = True
    if not (a.sync or a.fix or a.fill or a.check):
        ap.print_help()
        return 0
    g = do_sync(a) if a.sync else json.loads(GRAPH.read_text(encoding="utf-8"))
    if a.fix:
        g = do_fix(g, a)
    if a.fill:
        g = do_fill(g, a)
    rc = do_check(g, a) if a.check else 0
    return rc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
