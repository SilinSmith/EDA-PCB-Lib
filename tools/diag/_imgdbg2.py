"""临时探针 2：看 from_file 之后 pool.bgr 到底有没有。"""
import sys
from pathlib import Path

PCB = Path(r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, str(PCB))
sys.path.insert(0, str(PCB / "fusion"))

from fusion import claims as CL

RUN = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs")
VLM = Path(r"Y:\workspace\EDA\work\_fusion_vlm")
case = sys.argv[1] if len(sys.argv) > 1 else "0001"
vp = VLM / ("%s.json" % case)
import json
vd = json.loads(vp.read_text("utf-8")) if vp.exists() else None

p = CL.from_file(case, RUN / "stage1" / ("%s.stage1.json" % case),
                 RUN / "stage2" / "_assigned" / ("%s.json" % case), vlm=vd)
print("bgr =", None if p.bgr is None else p.bgr.shape)
print("slot_ids =", len(p.slot_ids()))
print("n_alts comp.c0000.key =", len(p.alts("comp.c0000.key")))
print("region c0000 =", p.region("comp.c0000.key"))
