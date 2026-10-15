# -*- coding: utf-8 -*-
"""量「围绕元件 box 的重扫」到底缺什么：按 cid 统计位号/引脚槽位与取景框。

    python work\\tools\\diag\\_boxgap.py [case ...]

对每张 case 报：
  * 元件总数（stage1 文档）
  * 有 `comp.<cid>.key` 槽位的元件数、其中有取景框的
  * 完全**没有位号候选**的元件数（真正的「零候选」），其中有多少连槽位都没有
  * 引脚槽位总数 / 有取景框的引脚槽位数
"""
import json, os, sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PCB = r"Y:\workspace\EDA\eda-pcb"
sys.path.insert(0, PCB)

RUN = Path(os.environ.get("FUSION_RUN_DIR",
                          r"Y:\workspace\EDA\work\_submit_out220r36\runs"))
VLM_DIR = Path(os.environ.get("FUSION_VLM_DIR", r"Y:\workspace\EDA\work\_fusion_vlm"))

from fusion import claims as CL  # noqa: E402


def one(case):
    s1 = RUN / "stage1" / ("%s.stage1.json" % case)
    asg = RUN / "stage2" / "_assigned" / ("%s.json" % case)
    if not s1.exists() or not asg.exists():
        print("  ! 缺文件 %s" % case)
        return
    doc = json.loads(s1.read_text(encoding="utf-8"))
    assigned = json.loads(asg.read_text(encoding="utf-8"))
    vp = VLM_DIR / ("%s.json" % case)
    vlm = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else None

    print("=" * 76)
    print("case %s   vlm_doc=%s" % (case, bool(vlm)))
    print("  stage1 文档顶层键：%s" % sorted(doc.keys()))
    pins = doc.get("pins") or {}
    comps = doc.get("components") or []
    cids = [c.get("id") for c in comps]
    n_inter = len(set(pins) & set(cids))
    print("  stage1 pins：%s %d 个键；与 components.id 交集 %d / %d 个元件"
          % (type(pins).__name__, len(pins), n_inter, len(comps)))
    if pins:
        k0 = list(pins)[0]
        print("    样例 pins[%r] = %s" % (k0, json.dumps(pins[k0], ensure_ascii=False)[:160]))

    pool = CL.from_stage2(case, doc, assigned, bgr=None, vlm=vlm)
    slots = pool.slot_ids()
    print("  池子槽位 %d 个" % len(slots))
    kind = Counter(s.split(".")[1] if s.startswith("comp.") else s.split(".")[0]
                   for s in slots)
    print("    槽位类型：%s" % dict(kind))

    comp_ids = set()
    for c in comps:
        comp_ids.add(c.get("id"))
    asg_comps = set((assigned.get("comps") or {}))
    print("  元件：stage1 %d 个 / _assigned %d 个" % (len(comp_ids), len(asg_comps)))

    key_slots = [s for s in slots if s.startswith("comp.") and s.endswith(".key")]
    key_cids = {s.split(".")[1] for s in key_slots}
    key_reg = [s for s in key_slots if pool.region(s)]
    print("  位号槽位 %d 个（覆盖 %d 个元件，占 %.1f%%）；其中**有取景框**的 %d 个"
          % (len(key_slots), len(key_cids), 100.0 * len(key_cids) / max(1, len(comp_ids)),
             len(key_reg)))

    ds = assigned.get("desig_slots") or {}
    ds_cids = set(ds)
    zero = sorted(asg_comps - ds_cids)
    print("  desig_slots 覆盖 %d / %d 个元件（%.1f%%）"
          % (len(ds_cids), len(asg_comps), 100.0 * len(ds_cids) / max(1, len(asg_comps))))
    print("  ★ 零位号候选的元件 %d 个（%.1f%%）；其中有槽位的 %d 个"
          % (len(zero), 100.0 * len(zero) / max(1, len(asg_comps)),
             len(set(zero) & key_cids)))

    # ---- 取景框是按 sid 存的，**槽位可以不存在但有框**（claims 里引脚就是）
    reg = pool.regions
    reg_kind = Counter()
    for s in reg:
        if s.startswith("comp."):
            reg_kind["comp." + s.rsplit(".", 1)[-1]] += 1
        elif s.startswith("pin."):
            reg_kind["pin." + s.rsplit(".", 1)[-1]] += 1
        else:
            reg_kind[s.split(".")[0]] += 1
    print("  取景框共 %d 个：%s" % (len(reg), dict(reg_kind)))

    pin_slots = [s for s in slots if s.startswith("pin.")]
    pin_reg = [s for s in reg if s.startswith("pin.")]
    print("  引脚槽位 %d 个；引脚取景框 %d 个" % (len(pin_slots), len(pin_reg)))
    nb = [s for s in pin_reg if not any(True for _ in [])]
    # 取景框里 focus/box 是不是 None
    none_pt = [s for s in pin_reg
               if pool.region(s) is None
               or getattr(pool.region(s), "focus", None) is None]
    print("    其中取景框为 None 的 %d 个" % len(none_pt))

    # ---- 真正的问题：哪些槽位**一条候选都没有**（问不出来/问不到）
    empty = [s for s in slots if not pool.alts(s)]
    print("  ★ 一条候选都没有的槽位 %d 个：%s" % (len(empty), empty[:10]))
    print("  ★ 有取景框但**没有槽位**的 %d 个：%s"
          % (len([s for s in reg if s not in set(slots)]),
             [s for s in reg if s not in set(slots)][:8]))

    # 每颗元件的引脚数来源
    n_doc = sum(len((pins.get(c) or {})) for c in comp_ids)
    n_asg = sum(len(((assigned.get("comps") or {}).get(c) or {}).get("pins") or {})
                for c in asg_comps)
    print("  引脚点数：stage1 doc['pins'] 合计 %d；_assigned[*]['pins'] 合计 %d"
          % (n_doc, n_asg))

    noslot = [c for c in sorted(asg_comps) if ("comp.%s.key" % c) not in set(key_slots)]
    print("  ★ 连位号槽位都没有的元件 %d 个，前 12：%s" % (len(noslot), noslot[:12]))


for c in (sys.argv[1:] or ["0001", "1", "0002"]):
    one(c)
