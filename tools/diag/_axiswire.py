# -*- coding: utf-8 -*-
"""查「轴层到底有没有出声」：照 `fusion/run.py:run_case` 的方式建池，再直调 `AX.inject`。

用法：python work\tools\diag\_axiswire.py [case]
"""
import json
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
os.environ.setdefault("FUSION_STAGE1_DIR", r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage1")
os.environ.setdefault("FUSION_ASSIGNED_DIR", r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage2\_assigned")

from fusion import axis as AX          # noqa: E402
from fusion import claims as CL        # noqa: E402

VLM_DIR = r"Y:\workspace\EDA\work\_fusion_vlm"


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    case = (sys.argv[1] if len(sys.argv) > 1 else "1")
    s1d = os.environ["FUSION_STAGE1_DIR"]
    asd = os.environ["FUSION_ASSIGNED_DIR"]
    doc = load(os.path.join(s1d, "%s.stage1.json" % case))
    asg = load(os.path.join(asd, "%s.json" % case))
    vp = os.path.join(VLM_DIR, "%s.json" % case)
    vlm = load(vp) if os.path.exists(vp) else None
    pool = CL.from_stage2(case, doc, asg, bgr=CL.load_image(doc), vlm=vlm)

    print("AX.ON =", AX.ON)
    items = AX.items_from_pool(pool)
    print("items_from_pool =", len(items), "（AXIS_MIN = %d）" % AX.AXIS_MIN)
    for it in items:
        print("   %-22s text=%-8s w/h=%.2f/%.2f  n_reads=%d"
              % (it.key, it.text, it.w, it.h, len(it.reads)))
    axes = AX.build_axes(items)
    print("build_axes =", [(a.prefix, len(a.items), round(a.items[0].cy)) for a in axes])
    st = AX.inject(pool)
    print("inject =", {k: v for k, v in st.items() if k != "plan"})
    if st.get("plan"):
        for a in st["plan"]["axes"]:
            print("   PLAN", a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
