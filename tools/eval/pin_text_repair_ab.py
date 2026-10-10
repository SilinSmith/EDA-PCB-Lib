"""Replay observed pin geometry while isolating pin text/role repairs.

Neither GT labels nor GT coordinates participate in production recognition.
This diagnostic deliberately keeps upstream detection and wire geometry fixed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
def clean(doc):
    doc=copy.deepcopy(doc);doc.pop("__pin_index",None);return doc
def counts(gt,pred):
    gc={k for k,v in gt["components"].items()if v.get("type")=="box"}
    pc={k for k,v in pred["components"].items()if v.get("type")=="box"}
    g={(c,k):v for c,ps in gt["pins"].items()if c in gc for k,v in ps.items()}
    p={(c,k):v for c,ps in pred["pins"].items()if c in pc for k,v in ps.items()}
    common=set(g)&set(p)
    correct=sum(g[k]["pinname"]==p[k]["pinname"]for k in common)
    return {"gt":len(g),"predicted":len(p),"correct":correct,"matching_number":len(common),
            "name_mismatch":len(common)-correct,"missing_number":len(set(g)-set(p)),"extra_number":len(set(p)-set(g))}
def aggregate(rows):
    out={k:sum(r[k]for r in rows)for k in rows[0]}
    out.update(precision=out["correct"]/out["predicted"]if out["predicted"]else 0,
               recall=out["correct"]/out["gt"]if out["gt"]else 0,
               f1=2*out["correct"]/(out["predicted"]+out["gt"])if out["predicted"]+out["gt"]else 0)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures",type=Path,required=True)
    ap.add_argument("--previous",type=Path,required=True)
    ap.add_argument("--baseline-source",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cases",default="1,2,3,4,5,6,7,8,9,10")
    args=ap.parse_args()
    capture=read(args.fixtures/"capture-verification.json")
    source=read(args.fixtures/"source-verification.json")
    os.environ.update(capture["configuration"])
    import torch,cv2,numpy as np
    torch.set_num_threads(4)
    from stage2 import assemble,structmatch
    from stage2.text import TextEngine
    from stage2.color_crops import read_color_numbers
    from stage2.pin_text_crops import read_pin_texts
    from tools.eval.color_pin_rule_ab import physical_side
    from schematic_parser import metrics
    active=importlib.import_module("stage2.assign")
    old_sm=load("stage2._text_baseline_structmatch",args.baseline_source/"stage2"/"structmatch.py")
    old_assign=load("stage2._text_baseline_assign",args.baseline_source/"stage2"/"assign.py");old_assign.structmatch=old_sm
    old_assemble=load("stage2._text_baseline_assemble",args.baseline_source/"stage2"/"assemble.py")
    assert sha(capture["ocr_weight"])==capture["ocr_weight_sha256"]
    assert sha(source["type_spec_path"])==source["type_spec_sha256"]
    engine=TextEngine(capture["ocr_weight"],device="cpu",allow_missing=False)
    files=("stage2/assign.py","stage2/structmatch.py","stage2/assemble.py","stage2/color_crops.py","stage2/pin_text_crops.py","stage2/run.py")
    source_hashes={f:sha(REPO/f)for f in files}
    arms=("baseline","role_only","digit_only","vertical_only","full","full_edge_midpoint","full_horizontal",
          "full_vertical_names","full_horizontal_names")
    result={"baseline_commit":"5b90d2ec1052307659ed8ee7239ed73b2bb2b49f",
        "scope":"specified 10GT, box chip pins; color diagnostics exclude black case1",
        "evidence_boundary":"Fixed observed color pin geometry, detected components, wire segments and initial OCR; same checkpoint freshly re-read local pin crops; not a fresh upstream end-to-end run or official online score.",
        "coordinate_free_definition":"Exact component key, pin_n and pinname; every coordinate, side and row ignored. Micro counts.",
        "source_hashes":source_hashes,"ocr_checkpoint_sha256":capture["ocr_weight_sha256"],"cases":[],"physical_rows":[],"changes":[]}
    scores={a:[]for a in arms};nr={a:[]for a in arms};cr={a:[]for a in arms}
    trace_path=args.previous/"root-causes"/"all_physical_pin_traces.json"
    traces=read(trace_path)["rows"] if trace_path.exists() else []
    trace_map={(r["case"],r["component"],r["pin"]):r for r in traces}
    if traces:
        result["diagnostic_physical_trace_sha256"]=sha(trace_path)
    cases={c.strip()for c in args.cases.split(",")if c.strip()}
    for src in source["cases"]:
        case=src["case_id"]
        if case not in cases:continue
        assert sha(src["gt_path"])==src["gt_sha256"]and sha(src["image_path"])==src["image_sha256"]
        stage=read(args.previous/"stage1"/f"{case}.stage1.json")
        initial=read(args.fixtures/"reads"/f"{case}.reads.json")
        previous_digits=read(args.previous/"color_number_ocr"/"_extra_reads"/f"{case}.json")
        bgr=cv2.imdecode(np.fromfile(src["image_path"],dtype=np.uint8),cv2.IMREAD_COLOR)
        new_digits=read_color_numbers(stage,initial,engine,bgr)
        vertical=read_pin_texts(stage,initial,engine,bgr)
        all_text=read_pin_texts(stage,initial,engine,bgr,include_horizontal=True)
        save(args.out/"_new_digits"/f"{case}.json",new_digits)
        save(args.out/"_pin_text_reads"/f"{case}.json",vertical)
        save(args.out/"_all_pin_text_reads"/f"{case}.json",all_text)
        gt=read(src["gt_path"]);docs={};assignments={}
        for arm in arms:
            if arm=="baseline":
                binder,builder,reads=old_assign,old_assemble,list(initial)+previous_digits
            else:
                binder,builder=active,assemble
                active.PIN_TEXT_ROLE_GUARD=arm=="role_only" or arm.startswith("full")
                active.PIN_DIGIT_ENTITY=arm=="digit_only" or arm.startswith("full")
                os.environ["STAGE2_PIN_NAME_EDGE_MIDPOINT"]="1" if arm=="full_edge_midpoint" else "0"
                reads=list(initial)+(new_digits if active.PIN_DIGIT_ENTITY else previous_digits)
                if arm in ("vertical_only","full","full_edge_midpoint"):reads+=vertical
                if arm=="full_horizontal":reads+=all_text
                if arm in ("full_vertical_names","full_horizontal_names"):
                    reads += [r for r in (all_text if arm=="full_horizontal_names" else vertical)
                              if r.get("name_only")]
            assigned=binder.assign(stage["components"],stage["pins"],copy.deepcopy(reads),
                pincol=stage.get("pincol"),pin_evidence=stage.get("pin_evidence"))
            doc=clean(builder.build_doc(stage,assigned,case));docs[arm]=doc;assignments[arm]=assigned
            if arm=="baseline":assert doc==read(args.previous/"color_number_ocr"/f"{case}.json"),f"baseline drift {case}"
            score=metrics.evaluate_case(gt,doc,case)
            scores[arm].append(score);cnt=counts(gt,doc);nr[arm].append(cnt)
            if case!="1":cr[arm].append(cnt)
            save(args.out/arm/f"{case}.json",doc);save(args.out/arm/"_assigned"/f"{case}.json",assigned)
        if case=="1":assert docs["baseline"]==docs["full"],"monochrome drift"
        H=stage["image"]["h"]
        for key,gc in gt["components"].items():
            if gc.get("type")!="box":continue
            b=gc["bbox"];box=[b[0],H-b[3],b[2],H-b[1]]
            for pk,gp in gt["pins"].get(key,{}).items():
                point=[gp["point"]["x"],H-gp["point"]["y"]];edge=physical_side(box,point)
                row={"case":case,"component":key,"pin":pk,"gt":gp,"arms":{}}
                trace=trace_map.get((case,key,pk))
                if trace:
                    cid,index=trace["cid"],trace["index"]
                    assert stage["pins"][cid][index]==trace["point_img"],"physical trace drift"
                    row.update(observed_component_id=cid,observed_pin_index=index)
                for arm in arms:
                    pp=docs[arm]["pins"].get(key,{}).get(pk);check=False;distance=None
                    if pp:
                        xy=[pp["point"]["x"],H-pp["point"]["y"]]
                        tangent=abs(xy[1]-point[1])if edge in ("left","right")else abs(xy[0]-point[0])
                        distance=math.dist(xy,point)
                        check=pp["pinname"]==gp["pinname"]and physical_side(box,xy)==edge and tangent<=5
                    info={"prediction":pp,"physical_name_number_row_correct":check,"coordinate_distance":distance}
                    if trace:
                        bound=assignments[arm]["comps"][cid]["pins"][index]
                        name=(bound.get("name")or{}).get("text")
                        number=(bound.get("number")or{}).get("text")
                        info.update(bound_name=name,bound_number=number,
                                    bound_name_correct=name==gp["pinname"],
                                    bound_number_correct=number==pk.removeprefix("pin_"))
                    row["arms"][arm]=info
                result["physical_rows"].append(row)
                if row["arms"]["baseline"]["prediction"]!=row["arms"]["full"]["prediction"]:result["changes"].append(row)
        result["cases"].append({"case":case,"new_digit_reads":len(new_digits),"pin_text_reads":len(vertical),
            "scores":{a:scores[a][-1].as_dict()for a in arms},"coordinate_free":{a:nr[a][-1]for a in arms}})
        print(f"case{case}: coordinate-free {nr['baseline'][-1]['correct']}/{nr['baseline'][-1]['predicted']} -> {nr['full'][-1]['correct']}/{nr['full'][-1]['predicted']}; local score {scores['baseline'][-1].score:.4f}->{scores['full'][-1].score:.4f}",flush=True)
    result["coordinate_free"]={a:{"all10":aggregate(nr[a]),"color2to10":aggregate(cr[a])}for a in arms}
    result["scores"]={a:metrics.aggregate(scores[a])for a in arms}
    color=[r for r in result["physical_rows"]if r["case"]!="1"]
    result["color_physical_correct"]={a:sum(r["arms"][a]["physical_name_number_row_correct"]for r in color)for a in arms}
    if traces:
        result["color_binding_diagnostics"]={a:{
            "name_correct":sum(r["arms"][a].get("bound_name_correct",False)for r in color),
            "number_correct":sum(r["arms"][a].get("bound_number_correct",False)for r in color),
            "both_correct":sum(r["arms"][a].get("bound_name_correct",False)and r["arms"][a].get("bound_number_correct",False)for r in color)
            }for a in arms}
    result["regressions"]=[r for r in color if r["arms"]["baseline"]["physical_name_number_row_correct"]and not r["arms"]["full"]["physical_name_number_row_correct"]]
    result["coordinate_conflicts"]=[r for r in color if r["arms"]["full"]["physical_name_number_row_correct"]and r["arms"]["full"]["coordinate_distance"]>5]
    assert source_hashes=={f:sha(REPO/f)for f in files},"sources changed during evaluation"
    os.environ["STAGE2_PIN_NAME_EDGE_MIDPOINT"]="0"
    save(args.out/"pin-text-repair-ab.json",result)
    print(json.dumps({k:result[k]for k in ("coordinate_free","color_physical_correct","scores")},ensure_ascii=False,indent=2))
    print("physical regressions",len(result["regressions"]))

if __name__=="__main__":main()
