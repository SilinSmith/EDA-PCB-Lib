# -*- coding: utf-8 -*-
"""把 `mkviz.py` 产出的 `work/viz/cases/*.json` 变成可点的网页。

产物：
  work/viz/index.html          批次清单（按 split / 定点档分组，带四分之项分数）
  work/viz/case_<id>.html      单张自包含页（内嵌 JSON + 相对路径图片）

页面要点（对应 m23737「逐批次展示识别结果和各阶段操作」）：
  · 图层开关：原图 / 我方元件框 / GT 元件框 / stage1 文字框 / 引脚读数窗 / 网络标签
  · 点任何框 ⇒ 右侧出详情：我方键、GT 键、判决、候选表（值·分·来源·证据）
  · 选中元件 ⇒ 叠加我方 `pin_<N>` 点与 GT 引脚点 + 引脚对照表
  · 分数卡按 component 0.30 / pin 0.25 / net_hypergraph 0.35 / net_line 0.10 拆开
  · 问题清单：漏检 / 多余 / 读错 / 引脚数差

用法：
  python work/tools/viz/mkview.py [--viz work/viz] [--only 0001,0048]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
VIZ = ROOT / "work" / "viz"
WEIGHTS = {"component": 0.30, "pin": 0.25, "net_hypergraph": 0.35, "net_line": 0.10}
TIER_NAMES = {"low": "低分", "mid": "中位", "high": "高分", "regress": "上轮回归",
              "gt9": "GT9", "quickset": "定点24"}


def slug(cid: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", str(cid))


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 每张的问题清单（纯计算，不改判分）
# ---------------------------------------------------------------------------

def issues_of(D: dict) -> Dict[str, list]:
    out = {"missing": [], "extra": [], "wrong_key": [], "pin_gap": []}
    for c in (D.get("gt") or {}).get("comps") or []:
        if c.get("pred_key") is None:
            out["missing"].append({"key": c.get("key"), "type": c.get("type"),
                                   "Name": c.get("Name"), "box": c.get("box")})
    for c in (D.get("ours") or {}).get("comps") or []:
        if c.get("verdict") == "extra":
            out["extra"].append({"key": c.get("key"), "type": c.get("type"),
                                 "Name": c.get("Name"), "box": c.get("box")})
    for s in D.get("slots") or []:
        if not s.get("sid", "").endswith(".key") or not s.get("gt"):
            continue
        a = s.get("assign")
        if a is None:
            out["wrong_key"].append({"cid": s.get("cid"), "gt": s.get("gt"),
                                     "ours": None, "src": None, "box": None})
        elif str(a) != str(s["gt"]):
            out["wrong_key"].append({"cid": s.get("cid"), "gt": s.get("gt"),
                                     "ours": str(a), "src": s.get("assign_src"),
                                     "box": None})
    sc = (D.get("score") or {}).get("counts") or {}
    out["pin_gap"] = {"ours": sum((1 for k, v in (D.get("comp_pins") or {}).items()
                                   for _ in (v.get("ours") or []))),
                      "gt": sum((1 for k, v in (D.get("comp_pins") or {}).items()
                                 for _ in (v.get("gt") or []))),
                      "pin_f1": ((D.get("score") or {}).get("pin_f1")),
                      "pin_counts": sc.get("pin")}
    cid_box = {}
    for pw in D.get("pinwins") or []:
        cid_box.setdefault(pw.get("cid"), pw.get("box"))
    for w in out["wrong_key"]:
        w["box"] = cid_box.get(w.get("cid"))
    return out


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#101216;--panel:#181b21;--line:#2a2f38;--fg:#e6e9ef;--dim:#98a1b0;
--blue:#2979ff;--red:#ff1744;--orange:#ff9100;--green:#00e676;--cyan:#00e5ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:13px/1.5 "Microsoft YaHei",system-ui,sans-serif}
a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
#top{display:flex;align-items:center;gap:14px;padding:8px 14px;background:var(--panel);
 border-bottom:1px solid var(--line);position:sticky;top:0;z-index:50;flex-wrap:wrap}
#top .chip{padding:2px 9px;border-radius:10px;background:#232833;border:1px solid var(--line)}
#wrap{display:flex;align-items:flex-start;gap:10px;padding:10px}
#left{flex:1 1 auto;min-width:420px;position:sticky;top:52px}
#right{flex:0 0 470px;max-height:calc(100vh - 66px);overflow:auto;padding-right:4px}
#tools{display:flex;gap:12px;flex-wrap:wrap;padding:6px 2px 8px;color:var(--dim)}
#tools label{cursor:pointer;user-select:none}
#stage{position:relative;width:100%;line-height:0;background:#000;
 border:1px solid var(--line);border-radius:4px;overflow:hidden}
#stage img{position:absolute;left:0;top:0;width:100%;height:auto}
#stage img.l0{position:relative}
#ov{position:absolute;left:0;top:0;right:0;bottom:0}
.b{position:absolute;border:1.5px solid;border-radius:2px;cursor:pointer}
.b:hover{outline:2px solid #fff}
.b.sel{outline:2px solid #fff;box-shadow:0 0 0 2px #000 inset}
.b.gt{border-color:var(--green)}
.b.gtdash{border-color:var(--orange);border-style:dashed}
.b.blue{border-color:var(--blue)}
.b.red{border-color:var(--red)}
.b.orange{border-color:var(--orange)}
.b.cyan{border-color:var(--cyan);border-width:1px}
.b.foc{border-color:#fff;border-style:dotted;pointer-events:none}
.b.reg{border-color:#ffffff33;border-style:dotted;pointer-events:none}
.dot{position:absolute;width:5px;height:5px;margin:-2.5px 0 0 -2.5px;border-radius:50%;
 pointer-events:none}
.lab{position:absolute;font-size:11px;line-height:1;padding:1px 3px;border-radius:3px;
 background:#000000cc;white-space:nowrap;pointer-events:none;transform:translateY(-100%)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;
 padding:9px 11px;margin-bottom:9px}
.card h3{margin:0 0 7px;font-size:13px;color:var(--dim);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:3px 5px;border-bottom:1px solid #232833;text-align:left;vertical-align:top}
th{color:var(--dim);font-weight:600}
tr.it{cursor:pointer}tr.it:hover{background:#20242c}
tr.win{background:#12321f}tr.hitgt{background:#0d2b19}
.mono{font-family:Consolas,monospace}
.dim{color:var(--dim)}.sm{font-size:11.5px}
.k{display:inline-block;min-width:52px;color:var(--dim)}
.bar{height:6px;background:#232833;border-radius:3px;overflow:hidden;margin-top:2px}
.bar>i{display:block;height:100%;background:var(--blue)}
.bad{color:var(--red)}.good{color:var(--green)}.warn{color:var(--orange)}
.tag{display:inline-block;padding:0 6px;border-radius:8px;font-size:11px;
 background:#232833;border:1px solid var(--line);margin-left:4px}
#list{max-height:320px;overflow:auto}
#list tr.on{background:#1d2431}
.f.row{display:flex;gap:8px;align-items:baseline;margin:2px 0}
"""

JS = r"""
const W = {component:0.30, pin:0.25, net_hypergraph:0.35, net_line:0.10};
const WN = {component:'元件', pin:'引脚', net_hypergraph:'网络(超图)', net_line:'网络(线)'};
const OW = D.ow, OH = D.oh;
let show = {base:1, ours:1, gt:1, texts:0, pins:1, net:1, lab:0};
let sel = null, selComp = null;

function pctX(v){ return (v/OW*100)+'%'; }
function pctY(v){ return (v/OH*100)+'%'; }
function box(el,b,cls,title,data){
  el.style.left=pctX(b[0]); el.style.top=pctY(b[1]);
  el.style.width=pctX(b[2]-b[0]); el.style.height=pctY(b[3]-b[1]);
  el.className='b '+cls; if(title) el.title=title; if(data) Object.assign(el.dataset,data);
  return el;
}
function mk(cls,b,title,data){
  const d=document.createElement('div');
  return box(d,b,cls,title,data);
}
function esc(s){ return (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function render(){
  const ov=document.getElementById('ov'); ov.innerHTML='';
  // 取景框（最底）
  (D.regions||[]).forEach(r=>{
    if(r.box&&show.pins) ov.appendChild(mk('reg',r.box,'取景框 '+r.sid));
    if(r.focus&&show.pins) ov.appendChild(mk('foc',r.focus,'focus '+r.sid));
  });
  // GT
  if(show.gt) (D.gt&&D.gt.comps||[]).forEach((c,i)=>{
    const e=mk(c.matched?'b gt':'b gtdash', c.box,
      'GT '+c.key+'  ['+c.type+'] '+(c.Name||'')+(c.matched?('  → 我方 '+c.pred_key):'  ✗ 未配上'),
      {t:'gt',i:i});
    if(show.lab&&c.key){const l=document.createElement('div');l.className='lab';
      l.textContent=c.key+(c.matched?'':' ✗');l.style.left=pctX(c.box[0]);l.style.top=pctY(c.box[1]);
      e.appendChild(l);}
    ov.appendChild(e);
  });
  // 我方
  if(show.ours) (D.ours&&D.ours.comps||[]).forEach((c,i)=>{
    const cl=c.verdict==='extra'?'b red':(c.verdict==='type'?'b orange':'b blue');
    const e=mk(cl,c.box,'我方 '+c.key+' ['+c.type+'] '+(c.Name||'')+
      '  '+(c.gt_key?('GT='+c.gt_key):'✗ GT 里没有'),
      {t:'ours',i:i});
    if(show.lab){const l=document.createElement('div');l.className='lab';
      l.textContent=c.key;l.style.left=pctX(c.box[0]);l.style.top=pctY(c.box[1]);e.appendChild(l);}
    ov.appendChild(e);
  });
  // stage1 文字框
  if(show.texts) (D.texts||[]).forEach((t,i)=>{
    ov.appendChild(mk('b cyan',t.box,
      'stage1 文字框 #'+t.i+(t.read?('  读成 '+t.read):'  （未被任何元件认领）'),
      {t:'text',i:i}));
  });
  // 引脚读数窗
  if(show.pins) (D.pinwins||[]).forEach((p,i)=>{
    ov.appendChild(mk('b orange',p.box,
      '引脚'+p.kind+' t'+p.k+'  读出='+(p.read||'∅')+'  GT='+(p.gt||'∅'),
      {t:'pw',i:i}));
  });
  // 网络标签
  if(show.net) (D.labels||[]).forEach((l,i)=>{
    ov.appendChild(mk('b gt',l.box,'网络标签 '+(l.text||''),{t:'net',i:i}));
  });
  // 选中元件的引脚点
  if(selComp!=null){
    const g=D.comp_pins[selComp]; if(g){
      (g.ours||[]).forEach(p=>{ if(p.x==null)return;
        const d=document.createElement('div');d.className='dot';
        d.style.background='#2979ff';d.style.left=pctX(p.x);d.style.top=pctY(p.y);
        d.title='我方 '+p.pin+' = '+(p.pinname||'');ov.appendChild(d);});
      (g.gt||[]).forEach(p=>{ if(p.x==null)return;
        const d=document.createElement('div');d.className='dot';
        d.style.background='#00e676';d.style.left=pctX(p.x);d.style.top=pctY(p.y);
        d.title='GT '+p.pin+' = '+(p.pinname||'');ov.appendChild(d);});
    }
  }
  // 选中的格子
  if(sel){
    const arr={gt:D.gt&&D.gt.comps,ours:D.ours&&D.ours.comps,text:D.texts,pw:D.pinwins,net:D.labels}[sel.t];
    const it=arr&&arr[sel.i];
    if(it){const e=mk('b foc',it.box,'选中');ov.appendChild(e);}
  }
}

function scorecard(){
  const s=D.score||{}, c=s.counts||{};
  let h='<div class="card"><h3>分数分解 &nbsp;<span class="tag">总分 '+esc(s.score)+'</span></h3>';
  ['component','pin','net_hypergraph','net_line'].forEach(k=>{
    const f=s[k+'_f1']||0, cc=c[k]||{};
    h+='<div class="f row"><span class="k">'+WN[k]+' ×'+W[k]+'</span>'+
       '<b style="min-width:56px">'+f.toFixed(4)+'</b>'+
       '<span class="dim sm">tp '+cc.tp+' · fp '+cc.fp+' · fn '+cc.fn+
       ' · P '+(cc.precision||0).toFixed(3)+' · R '+(cc.recall||0).toFixed(3)+'</span></div>'+
       '<div class="bar"><i style="width:'+(f*100)+'%"></i></div>';
  });
  if(s.diagnostic){h+='<div class="dim sm" style="margin-top:6px">'+
    Object.entries(s.diagnostic).map(([k,v])=>k+'='+v).join(' · ')+'</div>';}
  h+='</div>'; document.getElementById('scorecard').innerHTML=h;
}

function opstimeline(){
  let h='<div class="card"><h3>阶段时间线</h3><table>';
  (D.ops||[]).forEach((o,i)=>{
    h+='<tr><td class="dim" style="width:22px">'+(i+1)+'</td><td>'+
       '<b>'+esc(o.name)+'</b><div class="dim sm">'+esc(o.note||'')+'</div></td>'+
       '<td class="mono" style="text-align:right">'+esc(o.n)+'</td></tr>';
  });
  h+='</table><div class="dim sm" style="margin-top:6px">'+esc((D.ops||[]).note||'')+'</div></div>';
  document.getElementById('ops').innerHTML=h;
}

function issuesCard(){
  const it=D.issues||{};
  const g=it.pin_gap||{};
  let h='<div class="card"><h3>问题清单</h3>';
  h+='<div class="f row"><span class="k">漏检</span><b class="bad">'+it.missing.length+
     '</b><span class="dim sm">GT 有、我方没有（配不上的 GT 元件）</span></div>';
  h+='<div class="f row"><span class="k">多余</span><b class="warn">'+it.extra.length+
     '</b><span class="dim sm">我方有、GT 里找不到</span></div>';
  h+='<div class="f row"><span class="k">位号读错</span><b class="bad">'+it.wrong_key.length+
     '</b><span class="dim sm">有 GT 键但选出来的不是它</span></div>';
  h+='<div class="f row"><span class="k">引脚</span><b>'+g.ours+' / '+g.gt+
     '</b><span class="dim sm">我方 / GT，pin_f1 '+(g.pin_f1||0).toFixed(4)+'</span></div>';
  h+='<div style="margin-top:6px;max-height:150px;overflow:auto"><table>';
  it.missing.slice(0,60).forEach(m=>{h+='<tr class="it" data-jump="'+esc(m.key)+'"><td class="mono">GT '+esc(m.key)+
    '</td><td>'+esc(m.Name||'')+'</td><td class="dim">'+esc(m.type)+'</td></tr>';});
  h+='</table></div></div>';
  document.getElementById('issues').innerHTML=h;
}

function detail(){
  const box=document.getElementById('detail');
  if(!sel){box.innerHTML='<div class="card dim">点图上的框看详情。蓝=我方配上、红=多余、绿=GT、黄虚线=GT 未配上、青=stage1 文字框、橙=引脚读数窗。</div>';return;}
  const arr={gt:D.gt&&D.gt.comps,ours:D.ours&&D.ours.comps,text:D.texts,pw:D.pinwins,net:D.labels}[sel.t];
  const it=arr&&arr[sel.i]; if(!it){box.innerHTML='';return;}
  let h='<div class="card"><h3>选中：'+({gt:'GT 元件',ours:'我方元件',text:'stage1 文字框',pw:'引脚读数窗',net:'网络标签'}[sel.t])+'</h3>';
  if(sel.t==='gt'){
    h+='<div class="f row"><span class="k">GT 键</span><b class="good">'+esc(it.key)+'</b></div>'+
       '<div class="f row"><span class="k">名称</span>'+esc(it.Name||'∅')+'</div>'+
       '<div class="f row"><span class="k">类型</span>'+esc(it.type)+(it.is_ic?' <span class="tag">IC</span>':'')+'</div>'+
       '<div class="f row"><span class="k">配上</span>'+(it.matched?('<b class="good">是</b> → 我方 '+esc(it.pred_key)+'</span>'):'<b class="bad">否</b>（这条算漏检 → fn）')+'</div>';
    selComp=it.pred_key; 
  } else if(sel.t==='ours'){
    h+='<div class="f row"><span class="k">我方键</span><b class="mono">'+esc(it.key)+'</b></div>'+
       '<div class="f row"><span class="k">GT 键</span>'+(it.gt_key?('<b class="good">'+esc(it.gt_key)+'</b>'):'<b class="bad">无</b>（多余 → fp）')+'</div>'+
       '<div class="f row"><span class="k">判决</span>'+esc(it.verdict)+'</div>'+
       '<div class="f row"><span class="k">Name</span>'+esc(it.Name||'∅')+'</div>'+
       '<div class="f row"><span class="k">value</span>'+esc(it.value||'∅')+'</div>'+
       '<div class="f row"><span class="k">类型</span>'+esc(it.type)+' · 引脚 '+it.n_pins+'</div>';
    selComp=it.key;
  } else if(sel.t==='text'){
    h+='<div class="f row"><span class="k">#</span>'+it.i+'</div>'+
       '<div class="f row"><span class="k">读出</span>'+esc(it.read||'∅')+' <span class="dim">canon '+esc(it.canon||'∅')+'</span></div>'+
       '<div class="f row"><span class="k">用途</span>'+esc(it.kind||'（未被认领）')+'</div>'+
       (it.conf!=null?('<div class="f row"><span class="k">置信</span>'+it.conf+'</div>'):'');
  } else if(sel.t==='pw'){
    h+='<div class="f row"><span class="k">元件</span>'+esc(it.cid)+' → 键 '+esc(it.key||'∅')+'</div>'+
       '<div class="f row"><span class="k">槽位</span>第 '+it.k+' 脚 · '+esc(it.kind)+'</div>'+
       '<div class="f row"><span class="k">读出</span><b>'+esc(it.read||'∅')+'</b></div>'+
       '<div class="f row"><span class="k">GT</span>'+esc(it.gt||'∅')+'</div>'+
       '<div class="dim sm">这是 stage2 的『引脚号/名』读数窗口。它写进 `_assigned`，但融合层的引脚裁决**进不了交付文档**（doc.apply 只认 comp.*）。</div>';
    selComp=it.key;
  } else {
    h+='<div class="f row"><span class="k">标签</span>'+esc(it.text||'')+'</div>';
  }
  h+='</div>';

  // 该元件相关的槽位 + 候选表
  const cid=it.cid||((D.ours&&D.ours.comps||[])[sel.i]||{}).cid||null;
  let mine=(D.slots||[]).filter(s=>sel.t==='ours'&&s.cid===cidOfKey(it.key));
  if(sel.t==='gt') mine=(D.slots||[]).filter(s=>sel.t==='gt'&&s.gt===it.key&&s.sid.endsWith('.key'));
  if(sel.t==='ours'||sel.t==='gt'){
    mine=(D.slots||[]).filter(s=>s.sid.endsWith('.key')&&((sel.t==='ours'&&s.key===it.key)||(sel.t==='gt'&&s.gt===it.key)));
    mine.forEach(s=>{
      h+='<div class="card"><h3>候选表 &nbsp;<span class="mono dim">'+esc(s.sid)+'</span></h3>';
      const gtk=s.gt;
      h+='<div class="f row"><span class="k">GT</span>'+(gtk?('<b class="good">'+esc(gtk)+'</b>'):'<span class="dim">（这条槽位没有对应 GT）</span>')+
         ' &nbsp;<span class="k">解出</span><b>'+esc(s.assign||'∅')+'</b>'+
         (s.assign_src?('<span class="dim sm"> '+esc(s.assign_src)+'</span>'):'')+'</div>';
      h+='<table><tr><th>值</th><th>分</th><th>来源</th><th>证据</th></tr>';
      (s.cands||[]).forEach((c,ci)=>{
        const isW=s.assign!=null&&String(c.v)===String(s.assign);
        const isG=gtk!=null&&String(c.v)===String(gtk);
        h+='<tr class="'+(isG?'hitgt':(isW?'win':''))+'">'+
           '<td class="mono"><b>'+esc(c.v)+'</b>'+(isG?' <span class="good">← GT</span>':'')+
           (isW?' <span class="dim">← 胜出</span>':'')+'</td>'+
           '<td class="mono">'+c.s+'</td><td class="sm">'+esc(c.src)+'</td>'+
           '<td class="dim sm">'+esc(JSON.stringify(c.ev))+'</td></tr>';
      });
      h+='</table></div>';
    });
    if(!mine.length) h+='<div class="card dim">这个元件没有 `comp.*.key` 槽位（stage2 没给它关联到任何文字）。</div>';
  }

  // 引脚对照
  if(selComp&&D.comp_pins[selComp]){
    const g=D.comp_pins[selComp];
    h+='<div class="card"><h3>引脚对照 · 我方键 <span class="mono">'+esc(selComp)+'</span>'+
       (g.gt_key?(' → GT <b class="good">'+esc(g.gt_key)+'</b>'):' <span class="bad">（GT 里没有）</span>')+'</h3>';
    const n=Math.max((g.ours||[]).length,(g.gt||[]).length);
    h+='<table><tr><th>#</th><th>我方 pin / 名</th><th>GT pin / 名</th></tr>';
    for(let i=0;i<n;i++){
      const o=(g.ours||[])[i]||{}, t=(g.gt||[])[i]||{};
      const same=(String(o.pinname||'')===String(t.pinname||'')&&String(o.pin||'')===String(t.pin||''));
      h+='<tr'+(same?'':' class="bad"')+'><td class="dim">'+(i+1)+'</td>'+
         '<td class="mono">'+esc(o.pin||'')+' / '+esc(o.pinname||'')+'</td>'+
         '<td class="mono">'+esc(t.pin||'')+' / '+esc(t.pinname||'')+'</td></tr>';
    }
    h+='</table><div class="dim sm" style="margin-top:4px">按顺序并列，不是匹配结果；只要同一行两边不同就是错的。</div></div>';
  }
  box.innerHTML=h;
  Array.from(document.querySelectorAll('#list tr')).forEach(tr=>{
    tr.classList.toggle('on', sel.t==='ours'&&tr.dataset.key===it.key);
  });
}

function cidOfKey(k){
  const ks=Object.entries(D.cid2key||{});
  for(const [cid,key] of ks) if(key===k) return cid;
  return null;
}

function listCard(){
  const rows=(D.ours&&D.ours.comps||[]).slice().sort((a,b)=>(a.verdict==='key')-(b.verdict==='key'));
  let h='<div class="card"><h3>我方元件 ('+rows.length+')</h3><div id="list"><table>'+
    '<tr><th>键</th><th>GT</th><th>Name</th><th>类型</th><th>脚</th><th>判决</th></tr>';
  rows.forEach(c=>{
    const cl=c.verdict==='extra'?'bad':(c.verdict==='type'?'warn':'good');
    h+='<tr class="it" data-key="'+esc(c.key)+'"><td class="mono">'+esc(c.key)+'</td>'+
       '<td class="mono">'+esc(c.gt_key||'∅')+'</td><td>'+esc(c.Name||'')+'</td>'+
       '<td class="dim">'+esc(c.type)+'</td><td class="dim">'+c.n_pins+'</td>'+
       '<td class="'+cl+'">'+esc(c.verdict)+'</td></tr>';
  });
  h+='</table></div></div>';
  document.getElementById('listcard').innerHTML=h;
}

function wire(){
  document.getElementById('ov').addEventListener('click',e=>{
    const el=e.target.closest('.b'); if(!el||!el.dataset.t) return;
    sel={t:el.dataset.t,i:+el.dataset.i}; selComp=null; render(); detail();
  });
  document.querySelectorAll('#tools input').forEach(cb=>{
    cb.addEventListener('change',()=>{show[cb.value]=cb.checked?1:0; render();});
  });
  document.getElementById('listcard').addEventListener('click',e=>{
    const tr=e.target.closest('tr.it'); if(!tr) return;
    const i=(D.ours.comps||[]).findIndex(c=>c.key===tr.dataset.key);
    if(i>=0){sel={t:'ours',i:i};selComp=null;render();detail();}
  });
  document.getElementById('issues').addEventListener('click',e=>{
    const tr=e.target.closest('tr.it'); if(!tr) return;
    const i=(D.ours.comps||[]).findIndex(c=>c.gt_key===tr.dataset.jump);
    if(i>=0){sel={t:'ours',i:i};selComp=null;render();detail();}
  });
  document.addEventListener('keydown',e=>{
    if(e.key==='ArrowLeft'&&PREV) location.href=PREV;
    if(e.key==='ArrowRight'&&NEXT) location.href=NEXT;
    if(e.key==='l'||e.key==='L'){const cb=document.querySelector('#tools input[value=lab]');
      cb.checked=!cb.checked;cb.dispatchEvent(new Event('change'));}
  });
}

scorecard(); opstimeline(); issuesCard(); listCard(); render(); detail(); wire();
"""

CASE_TPL = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__</title><style>__CSS__</style></head><body>
<div id="top">
  <a href="index.html">← 批次清单</a>
  <b>__TITLE__</b>
  <span class="chip">总分 <b>__SCORE__</b></span>
  <span class="chip">元件 __CF1__</span><span class="chip">引脚 __PF1__</span>
  <span class="chip">网络 __NF1__</span><span class="chip">线 __LF1__</span>
  <span class="dim sm">__SPLIT__ ｜ __SIZE__ ｜ 预测档 __PRED__</span>
  <span style="margin-left:auto">__NAV__</span>
</div>
<div id="wrap">
  <div id="left">
    <div id="tools">
      <label><input type="checkbox" value="base" checked> 原图</label>
      <label><input type="checkbox" value="ours" checked> 我方元件</label>
      <label><input type="checkbox" value="gt" checked> GT 元件</label>
      <label><input type="checkbox" value="texts"> stage1 文字框</label>
      <label><input type="checkbox" value="pins" checked> 引脚读数窗</label>
      <label><input type="checkbox" value="net" checked> 网络标签</label>
      <label><input type="checkbox" value="lab"> 文字标签</label>
      <span class="dim sm">按 L 切标签 · ←/→ 切图 · 点框看详情</span>
    </div>
    <div id="stage">
      <img class="l0" src="__ASSETS__/base.png">
      <img src="__ASSETS__/gt.png" style="opacity:.85">
      <img src="__ASSETS__/ours.png" style="opacity:.85">
      <img src="__ASSETS__/texts.png" style="opacity:.8">
      <img src="__ASSETS__/pins.png" style="opacity:.8">
      <img src="__ASSETS__/net.png" style="opacity:.8">
      <div id="ov"></div>
    </div>
  </div>
  <div id="right">
    <div id="scorecard"></div>
    <div id="issues"></div>
    <div id="ops"></div>
    <div id="detail"></div>
    <div id="listcard"></div>
  </div>
</div>
<script>
const D = __DATA__;
const PREV = __PREV__, NEXT = __NEXT__;
__JS__
</script></body></html>
"""

INDEX_TPL = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>识别过程可视化 · 批次清单</title><style>__CSS__</style></head><body>
<div id="top"><b>识别过程可视化</b>
 <span class="chip">预测档 __PRED__</span>
 <span class="dim sm">点分数进单张页；单张页里可以开关图层、点框看候选表</span></div>
<div style="padding:12px">
<div class="card"><h3>总览</h3>__SUMMARY__</div>
__BODY__
</div></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default=str(VIZ))
    ap.add_argument("--only", default="")
    ap.add_argument("--pred", default="work\\_fusion_out220no4")
    a = ap.parse_args()
    viz = Path(a.viz)
    cdir = viz / "cases"
    files = sorted(cdir.glob("*.json"))
    if a.only:
        want = set(x.strip() for x in a.only.split(",") if x.strip())
        files = [f for f in files if f.stem in want]
    if not files:
        print("没找到 case JSON：", cdir)
        return 1

    tiers = read_json(ROOT / "work" / "tools" / "eval" / "tiers.json") or {}
    cases = []
    for f in files:
        D = read_json(f)
        if not D:
            continue
        D["issues"] = issues_of(D)
        cases.append(D)
    order = [D["id"] for D in cases]

    for i, D in enumerate(cases):
        cid = D["id"]
        prev = ("case_%s.html" % slug(order[i - 1])) if i > 0 else "null"
        nxt = ("case_%s.html" % slug(order[i + 1])) if i + 1 < len(order) else "null"
        sc = D.get("score") or {}
        nav = []
        if prev != "null":
            nav.append("<a href='%s'>← 上一张</a>" % prev)
        if nxt != "null":
            nav.append("<a href='%s'>下一张 →</a>" % nxt)
        html = (CASE_TPL
                .replace("__CSS__", CSS).replace("__JS__", JS)
                .replace("__NAV__", " &nbsp; ".join(nav) or "<span class='dim'>（只有这一张）</span>")
                .replace("__TITLE__", "%s" % cid)
                .replace("__SCORE__", "%.2f" % float(sc.get("score") or 0))
                .replace("__CF1__", "%.3f" % float(sc.get("component_f1") or 0))
                .replace("__PF1__", "%.3f" % float(sc.get("pin_f1") or 0))
                .replace("__NF1__", "%.3f" % float(sc.get("net_hypergraph_f1") or 0))
                .replace("__LF1__", "%.3f" % float(sc.get("net_line_f1") or 0))
                .replace("__SPLIT__", str(D.get("split") or ""))
                .replace("__SIZE__", "%d×%d" % (int(D.get("ow") or 0), int(D.get("oh") or 0)))
                .replace("__PRED__", a.pred)
                .replace("__ASSETS__", str(D.get("assets") or ""))
                .replace("__PREV__", json.dumps(prev)).replace("__NEXT__", json.dumps(nxt))
                .replace("__DATA__", json.dumps(D, ensure_ascii=False, default=str)))
        (viz / ("case_%s.html" % slug(cid))).write_text(html, encoding="utf-8")
        print("  → case_%s.html（%.2f）" % (slug(cid), float(sc.get("score") or 0)))

    # 清单
    scored = [D for D in cases]
    mean = sum(float((D.get("score") or {}).get("score") or 0) for D in scored) / max(1, len(scored))
    summ = ["<div class='f row'><span class='k'>张数</span><b>%d</b></div>" % len(scored),
            "<div class='f row'><span class='k'>平均</span><b>%.4f</b></div>" % mean]
    body = ""
    groups: List[tuple] = []
    seen = set()
    for name, ids in (tiers.get("tiers") or {}).items():
        got = [D for D in cases if D["id"] in set(ids)]
        if got:
            groups.append(("定点档 · %s（%s）" % (name, TIER_NAMES.get(name, "")), got))
            seen |= set(D["id"] for D in got)
    rest = [D for D in cases if D["id"] not in seen]
    if rest:
        groups.append(("其余", rest))

    def table_for(got):
        s = ("<table><tr><th>case</th><th>总分</th><th>元件</th><th>引脚</th><th>网络</th><th>线</th>"
             "<th>漏检</th><th>多余</th><th>位号错</th><th>split</th></tr>")
        for D in sorted(got, key=lambda x: float((x.get("score") or {}).get("score") or 0)):
            sc = D.get("score") or {}
            it = D.get("issues") or {}
            s += ("<tr class='it'><td><a href='case_%s.html'>%s</a></td>"
                  "<td class='mono'><b>%.2f</b></td><td class='mono'>%.3f</td><td class='mono'>%.3f</td>"
                  "<td class='mono'>%.3f</td><td class='mono'>%.3f</td>"
                  "<td class='bad'>%d</td><td class='warn'>%d</td><td class='bad'>%d</td>"
                  "<td class='dim sm'>%s</td></tr>"
                  % (slug(D["id"]), D["id"], float(sc.get("score") or 0),
                     float(sc.get("component_f1") or 0), float(sc.get("pin_f1") or 0),
                     float(sc.get("net_hypergraph_f1") or 0), float(sc.get("net_line_f1") or 0),
                     len(it.get("missing") or []), len(it.get("extra") or []),
                     len(it.get("wrong_key") or []), str(D.get("split") or "")))
        return s + "</table>"

    by_split: Dict[str, list] = {}
    for D in cases:
        by_split.setdefault(str(D.get("split") or "?").split("/")[0], []).append(D)
    body += "<div class='card'><h3>按批次（数据集 split）</h3></div>"
    for k in sorted(by_split):
        got = by_split[k]
        body += "<div class='card'><h3>%s · %d 张</h3>%s</div>" % (k, len(got), table_for(got))

    body += "<div class='card'><h3>按定点档</h3></div>"
    for title, got in groups:
        body += "<div class='card'><h3>%s · %d 张</h3>%s</div>" % (title, len(got), table_for(got))
    (viz / "index.html").write_text(
        INDEX_TPL.replace("__CSS__", CSS).replace("__PRED__", a.pred)
        .replace("__SUMMARY__", "".join(summ)).replace("__BODY__", body),
        encoding="utf-8")
    print("  → index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
