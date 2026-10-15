# -*- coding: utf-8 -*-
"""把 `ledger.build_ledger` 的结果渲染成**单文件可交互 HTML**。

一个 case 一个文件：`work\\debug\\<case>.html`
- 分数账（100 分怎么被扣掉的，逐层下钻）
- 图纸叠加（GT 蓝虚框 / 预测彩框，可缩放拖动、可开关图层、可筛选判决）
- 逐条对账表（点一行 → 图上高亮并放大过去）
- 阶段步进（阶段一→阶段二→阶段三→VLM→打分，看每一阶段往里加/减了什么）
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

# 判决 -> (颜色, 中文, 出现在哪一侧)
VERDICT_STYLE = {
    "tp":                ("#3fb950", "正确", "both"),
    "missing":           ("#58a6ff", "漏检（GT 有、我们没有）", "gt"),
    "extra":             ("#f85149", "误检（我们有、GT 没有）", "pred"),
    "text_mismatch":     ("#f0883e", "文字不符（一笔亏三处）", "both"),
    "bbox_mismatch":     ("#bc8cff", "外框不符（一笔亏三处）", "both"),
    "type_mismatch":     ("#d29922", "类型不符（一笔亏三处）", "both"),
    "position_mismatch": ("#f0883e", "引脚位置偏（一笔亏三处）", "both"),
    "name_mismatch":     ("#f0883e", "引脚名不符（一笔亏三处）", "both"),
    "no_point":          ("#bc8cff", "缺 point", "both"),
    "malformed":         ("#8b949e", "结构损坏", "both"),
    "partial":           ("#f0883e", "部分对上", "both"),
}

CSS = """
:root{--bg:#0d1117;--pan:#161b22;--bor:#30363d;--fg:#e6edf3;--dim:#8b949e;
--ok:#3fb950;--fn:#58a6ff;--fp:#f85149;--warn:#f0883e;--pur:#bc8cff;--yel:#d29922;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
a{color:#58a6ff}
h1{font-size:20px;margin:0}
h2{font-size:15px;margin:24px 0 8px;padding-bottom:5px;border-bottom:1px solid var(--bor)}
code{background:#21262d;padding:1px 4px;border-radius:3px;font-size:12px}
.wrap{max-width:1720px;margin:0 auto;padding:16px 20px 60px}
.hd{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
padding:14px 18px;background:var(--pan);border:1px solid var(--bor);border-radius:8px}
.big{font-size:34px;font-weight:700;line-height:1}
.mono{font-family:Consolas,"Cascadia Mono",monospace}
.dim{color:var(--dim)}
/* ---- 分数账 ---- */
.led{margin-top:14px;border:1px solid var(--bor);border-radius:8px;overflow:hidden}
.led .row{display:grid;grid-template-columns:130px 1fr 92px 92px 26px;
align-items:center;gap:10px;padding:9px 14px;border-top:1px solid var(--bor);
background:var(--pan);cursor:pointer;user-select:none}
.led .row:first-child{border-top:0}
.led .row:hover{background:#1c2128}
.bar{height:16px;background:#21262d;border-radius:3px;overflow:hidden;display:flex}
.bar i{display:block;height:100%}
.v{text-align:right;font-family:Consolas,monospace}
.caret{color:var(--dim);transition:transform .15s}
.row.open .caret{transform:rotate(90deg)}
.det{display:none;padding:12px 16px 16px;background:#0f141b;border-top:1px solid var(--bor)}
.det.open{display:block}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:4px 8px;border-bottom:1px solid #21262d;text-align:left;vertical-align:top}
th{color:var(--dim);font-weight:600;position:sticky;top:0;background:var(--bg);z-index:2}
tbody tr:hover{background:#1c2128}
tbody tr.sel{background:#1f6feb33;outline:1px solid #1f6feb}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;
font-weight:600;white-space:nowrap}
.gain{color:var(--ok);font-weight:700}
/* ---- 图 ---- */
#stage{position:relative;margin-top:10px;background:#010409;border:1px solid var(--bor);
border-radius:8px;overflow:hidden;height:74vh;min-height:460px;cursor:grab}
#stage.drag{cursor:grabbing}
#world{position:absolute;transform-origin:0 0}
#world img{display:block;image-rendering:pixelated}
#world svg{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}
#world svg .hit{pointer-events:all}
.tools{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0}
.tools button,.tools label{background:#21262d;border:1px solid var(--bor);color:var(--fg);
border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer}
.tools button:hover{background:#30363d}
.tools button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
.tools .sep{width:1px;height:20px;background:var(--bor);margin:0 4px}
#tip{position:fixed;pointer-events:none;background:#161b22ee;border:1px solid var(--bor);
border-radius:6px;padding:7px 10px;font-size:12px;max-width:460px;display:none;z-index:99;
box-shadow:0 6px 20px #0009;white-space:pre-wrap}
/* ---- 阶段 ---- */
.steps{display:flex;gap:0;margin-top:14px;flex-wrap:wrap}
.step{flex:1;min-width:150px;background:var(--pan);border:1px solid var(--bor);
padding:10px 12px;margin-right:-1px;cursor:pointer}
.step:first-child{border-radius:8px 0 0 8px}
.step:last-child{border-radius:0 8px 8px 0;margin-right:0}
.step.on{background:#1f6feb22;border-color:#1f6feb}
.step b{display:block;font-size:12px}
.step .n{font-family:Consolas,monospace;color:var(--dim);font-size:11px;margin-top:3px}
.tabs{display:flex;gap:4px;margin:14px 0 0;flex-wrap:wrap}
.tab{background:var(--pan);border:1px solid var(--bor);border-bottom:0;color:var(--dim);
padding:6px 14px;border-radius:6px 6px 0 0;cursor:pointer;font-size:12px}
.tab.on{background:#1f6feb22;color:var(--fg);border-color:#1f6feb;font-weight:600}
.tabpane{display:none;border:1px solid var(--bor);border-radius:0 8px 8px 8px;padding:12px;
background:#0f141b}
.tabpane.on{display:block}
.warnbox{background:#f8514922;border:1px solid #f85149;border-radius:6px;padding:10px 14px;
margin-top:12px;font-size:12px}
.notebox{background:#1f6feb15;border-left:3px solid #1f6feb;padding:8px 14px;margin:10px 0;
font-size:12px;border-radius:0 6px 6px 0}
.diagbox{background:#3fb95010;border:1px solid #3fb95055;border-radius:6px;
padding:10px 14px;margin:10px 0;font-size:12.5px}
.diagbox ul{margin:6px 0 0 16px;padding:0}
.diagbox li{margin:3px 0}
"""

JS = r"""
const LED = __LED__;
const ST  = __VERDICT_STYLE__;
const IMG = {w: __IMG_W__, h: __IMG_H__};
let view = {x: 0, y: 0, k: 1};
let layers = {cgt:1, cpr:1, pgt:1, ppr:1, ngt:0, npr:0, lgt:0, lpr:0};
let filter = null;          // null = 全部；否则一个判决名
let sel = null;

/* ---------- 坐标：GT/pred 的 bbox 是「左下角 y 向上」 ---------- */
function rect(b){
  if(!b) return null;
  const [x1,y1,x2,y2] = b;
  return {x: x1, y: IMG.h - y2, w: x2-x1, h: y2-y1};
}
function pt(p){ return p ? {x:p[0], y: IMG.h - p[1]} : null; }

/* ---------- 视图 ---------- */
function apply(){
  const w = document.getElementById('world');
  w.style.transform = `translate(${view.x}px,${view.y}px) scale(${view.k})`;
}
function fit(){
  const s = document.getElementById('stage');
  const k = Math.min(s.clientWidth/IMG.w, s.clientHeight/IMG.h)*0.98;
  view.k = k;
  view.x = (s.clientWidth  - IMG.w*k)/2;
  view.y = (s.clientHeight - IMG.h*k)/2;
  apply();
}
function zoomAt(cx, cy, f){
  const k2 = Math.max(0.05, Math.min(40, view.k*f));
  view.x = cx - (cx-view.x)*(k2/view.k);
  view.y = cy - (cy-view.y)*(k2/view.k);
  view.k = k2; apply();
}
function focusOn(r, pad){
  pad = pad || 90;
  const s = document.getElementById('stage');
  const w = r.w + pad*2, h = r.h + pad*2;
  const k = Math.min(s.clientWidth/w, s.clientHeight/h, 12);
  view.k = k;
  view.x = s.clientWidth/2  - (r.x + r.w/2)*k;
  view.y = s.clientHeight/2 - (r.y + r.h/2)*k;
  apply();
}

/* ---------- 画 ---------- */
function esc(s){ return (s===null||s===undefined) ? '' : String(s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function boxsvg(r, verdict, tag, id, dash){
  const c = (ST[verdict]||['#8b949e'])[0];
  const on = judge_visible(verdict);
  if(!on) return '';
  const sw = Math.max(1, 1.6/view.k);
  const hot = (sel === id) ? 3.2/view.k : 0;
  return `<rect class="hit" data-id="${id}" data-tag="${esc(tag)}"
    x="${r.x}" y="${r.y}" width="${Math.max(r.w,1)}" height="${Math.max(r.h,1)}"
    fill="${c}18" stroke="${hot?'#fff':c}" stroke-width="${(sw+hot)}"
    ${dash?'stroke-dasharray="6 4"':''}/>`;
}
function judge_visible(v){
  if(filter && v !== filter) return false;
  return true;
}

function render(){
  const svg = [];
  const c = LED.components, p = LED.pins, n = LED.nets;
  // --- 网络（把同一网络的元件框连起来太乱，这里只画引脚点）
  if(layers.pgt) for(const r of p) if(r.gt_point){
    const q = pt(r.gt_point), col = (ST[r.verdict]||['#8b949e'])[0];
    if(!judge_visible(r.verdict)) continue;
    svg.push(`<circle class="hit" data-id="pin:gt:${esc(r.comp)}:${esc(r.num)}"
      cx="${q.x}" cy="${q.y}" r="${4.5/view.k}" fill="none" stroke="${col}"
      stroke-width="${1.8/view.k}"/>`);
  }
  if(layers.ppr) for(const r of p) if(r.pred_point){
    const q = pt(r.pred_point), col = (ST[r.verdict]||['#8b949e'])[0];
    if(!judge_visible(r.verdict)) continue;
    svg.push(`<circle class="hit" data-id="pin:pr:${esc(r.comp)}:${esc(r.num)}"
      cx="${q.x}" cy="${q.y}" r="${3/view.k}" fill="${col}"/>`);
  }
  // --- 元件
  if(layers.cgt) for(const r of c){
    const q = rect(r.gt_bbox);
    if(q) svg.push(boxsvg(q, r.verdict, 'GT '+r.gt_key, 'c:gt:'+r.gt_key, true));
  }
  if(layers.cpr) for(const r of c){
    const q = rect(r.pred_bbox);
    if(q) svg.push(boxsvg(q, r.verdict, '我们 '+r.pred_key, 'c:pr:'+r.pred_key, false));
  }
  // --- 未配对的元件（只有一侧有框）也要画
  if(layers.cgt) for(const r of c) if(!r.gt_bbox && r.verdict==='extra'){
    // 预测侧已有框，不用补
  }
  document.getElementById('ov').innerHTML = svg.join('');
  document.getElementById('ov').setAttribute('viewBox',`0 0 ${IMG.w} ${IMG.h}`);
  document.getElementById('ov').setAttribute('width', IMG.w);
  document.getElementById('ov').setAttribute('height', IMG.h);
}

/* ---------- 交互 ---------- */
function init(){
  const s = document.getElementById('stage');
  let dragging=false, lx=0, ly=0;
  s.addEventListener('mousedown', e=>{dragging=true;lx=e.clientX;ly=e.clientY;s.classList.add('drag');});
  window.addEventListener('mouseup', ()=>{dragging=false;s.classList.remove('drag');});
  window.addEventListener('mousemove', e=>{
    if(!dragging) return;
    view.x += e.clientX-lx; view.y += e.clientY-ly; lx=e.clientX; ly=e.clientY; apply();
  });
  s.addEventListener('wheel', e=>{
    e.preventDefault();
    const r = s.getBoundingClientRect();
    zoomAt(e.clientX-r.left, e.clientY-r.top, e.deltaY<0?1.15:1/1.15);
  }, {passive:false});
  // 悬停提示
  const tip = document.getElementById('tip');
  s.addEventListener('mousemove', e=>{
    const t = e.target;
    if(t && t.dataset && t.dataset.id){
      const d = describe(t.dataset.id);
      if(d){ tip.style.display='block'; tip.innerHTML=d;
        tip.style.left = (e.clientX+14)+'px'; tip.style.top = (e.clientY+14)+'px'; return; }
    }
    tip.style.display='none';
  });
  s.addEventListener('click', e=>{
    if(e.target && e.target.dataset && e.target.dataset.id){
      const id = e.target.dataset.id; sel = id; render(); markRow(id);
    }
  });
  window.addEventListener('resize', ()=>apply());
  render(); fit();
}

/* 让视口高度贴着图的长宽比，避免上下大片空白 */
function sizeStage(){
  const s = document.getElementById('stage');
  const w = s.clientWidth || 1200;
  const h = w * IMG.h / IMG.w + 2;
  s.style.height = Math.round(Math.max(360, Math.min(window.innerHeight*0.82, h))) + 'px';
}

function describe(id){
  const [kind, side, ...rest] = id.split(':');
  if(kind==='c'){
    const key = rest.join(':');
    const r = LED.components.find(x => side==='gt' ? x.gt_key===key : x.pred_key===key);
    if(!r) return null;
    const st = ST[r.verdict]||['#8b949e','?'];
    return `<b>${side==='gt'?'GT':'我们'} 元件 ${esc(key)}</b>  <span style="color:${st[0]}">${st[1]}</span>
类型: GT <code>${esc(r.gt_type)}</code> / 我们 <code>${esc(r.pred_type)}</code>
值: GT <code>${esc(r.gt_value)}</code> / 我们 <code>${esc(r.pred_value)}</code>
型号: GT <code>${esc(r.gt_name)}</code> / 我们 <code>${esc(r.pred_name)}</code>
${esc(r.reason||'')}`;
  }
  if(kind==='pin'){
    const num = rest.pop(), comp = rest.join(':');
    const r = LED.pins.find(x => x.comp===comp && String(x.num)===String(num));
    if(!r) return null;
    const st = ST[r.verdict]||['#8b949e','?'];
    return `<b>引脚 ${esc(comp)}.${esc(num)}</b>  <span style="color:${st[0]}">${st[1]}</span>
名字: GT <code>${esc(r.gt_name)}</code> / 我们 <code>${esc(r.pred_name)}</code>
${esc(r.reason||'')}`;
  }
  return null;
}

function markRow(id){
  document.querySelectorAll('tbody tr.sel').forEach(tr=>tr.classList.remove('sel'));
  const tr = document.querySelector(`tbody tr[data-id="${CSS.escape(id)}"]`);
  if(tr){ tr.classList.add('sel'); tr.scrollIntoView({block:'center',behavior:'smooth'}); }
}

/* ---------- 表 ---------- */
function pill(v){
  const s = ST[v]||['#8b949e',v];
  return `<span class="pill" style="background:${s[0]}22;color:${s[0]};
    border:1px solid ${s[0]}66">${s[1]}</span>`;
}
function fmt(x){ return (x===null||x===undefined||x==='') ? '<span class=dim>—</span>' : esc(x); }

function buildTables(){
  const c = LED.components, p = LED.pins;
  document.getElementById('tbody-comp').innerHTML = c.map(r=>{
    const id = (r.gt_bbox||!r.gt_key) ? ('c:gt:'+r.gt_key) : ('c:pr:'+r.pred_key);
    return `<tr data-id="${esc(id)}">
      <td class="mono">${fmt(r.gt_key)}</td><td class="mono">${fmt(r.pred_key)}</td>
      <td>${fmt(r.gt_type)}</td><td>${fmt(r.pred_type)}</td>
      <td>${fmt(r.gt_value)}</td><td>${fmt(r.pred_value)}</td>
      <td>${fmt(r.gt_name)}</td><td>${fmt(r.pred_name)}</td>
      <td>${r.centre_dist!==undefined&&r.centre_dist!==null?r.centre_dist:''}</td>
      <td>${pill(r.verdict)}</td><td style="max-width:340px">${esc(r.reason||'')}</td></tr>`;
  }).join('');
  document.getElementById('tbody-pin').innerHTML = p.map(r=>{
    const id = r.gt_point ? `pin:gt:${r.comp}:${r.num}` : `pin:pr:${r.comp}:${r.num}`;
    return `<tr data-id="${esc(id)}">
      <td class="mono">${fmt(r.comp)}</td><td class="mono">${fmt(r.num)}</td>
      <td class="mono">${fmt(r.gt_pin_key)}</td><td class="mono">${fmt(r.pred_pin_key)}</td>
      <td>${fmt(r.gt_name)}</td><td>${fmt(r.pred_name)}</td>
      <td>${r.dist!==undefined?r.dist:''}</td>
      <td>${pill(r.verdict)}</td><td style="max-width:380px">${esc(r.reason||'')}</td></tr>`;
  }).join('');
  document.getElementById('tbody-net').innerHTML = LED.nets.map(r=>`<tr>
      <td class="mono">${fmt(r.gt_net)}</td><td class="mono">${fmt(r.pred_net)}</td>
      <td class="mono">${r.n_gt}</td><td class="mono">${r.n_pred}</td>
      <td class="mono" style="color:var(--ok)">${r.n_inter}</td>
      <td>${pill(r.verdict)}</td>
      <td style="max-width:520px">${esc(r.reason||'')}
        ${(r.missing_pins&&r.missing_pins.length)?'<div class=dim>少: '+esc(r.missing_pins.join(' '))+'</div>':''}
        ${(r.extra_pins&&r.extra_pins.length)?'<div class=dim>多: '+esc(r.extra_pins.join(' '))+'</div>':''}</td></tr>`).join('');
  document.getElementById('tbody-line').innerHTML = LED.lines.map(r=>`<tr>
      <td class="mono">${fmt(r.gt_net)}</td><td class="mono">${fmt(r.pred_net)}</td>
      <td class="mono">${r.n_tp!==undefined?r.n_tp:''}</td>
      <td class="mono" style="color:var(--fp)">${r.n_fp!==undefined?r.n_fp:''}</td>
      <td class="mono" style="color:var(--fn)">${r.n_fn!==undefined?r.n_fn:''}</td>
      <td>${pill(r.verdict)}</td><td style="max-width:520px">${esc(r.reason||'')}</td></tr>`).join('');
  document.querySelectorAll('tbody tr').forEach(tr=>{
    tr.addEventListener('click', ()=>{
      const id = tr.dataset.id; if(!id) return;
      sel = id; render(); markRow(id);
      const [kind,side,...rest] = id.split(':');
      if(kind==='c'){
        const key = rest.join(':');
        const r = LED.components.find(x=> side==='gt' ? x.gt_key===key : x.pred_key===key);
        const q = r && rect(side==='gt' ? r.gt_bbox : r.pred_bbox);
        if(q) focusOn(q);
      } else if(kind==='pin'){
        const r = LED.pins.find(x=> x.comp===rest[0] && String(x.num)===String(rest[1]));
        const q = r && pt(side==='gt' ? r.gt_point : r.pred_point);
        if(q) focusOn({x:q.x-40,y:q.y-40,w:80,h:80});
      }
    });
  });
}

/* ---------- 图层开关 ---------- */
function initTools(){
  document.querySelectorAll('[data-layer]').forEach(b=>{
    b.onclick = ()=>{ const k=b.dataset.layer; layers[k]=layers[k]?0:1;
      b.classList.toggle('on', !!layers[k]); render(); };
  });
  document.querySelectorAll('[data-filter]').forEach(b=>{
    b.onclick = ()=>{ const v=b.dataset.filter;
      filter = (filter===v) ? null : v;
      document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('on', x.dataset.filter===filter));
      render(); };
  });
  document.getElementById('btn-fit').onclick = fit;
  document.querySelectorAll('.led .row').forEach(r=>{
    r.onclick = ()=>{ const d=document.getElementById(r.dataset.det);
      d.classList.toggle('open'); r.classList.toggle('open'); };
  });
  document.querySelectorAll('.tab').forEach(t=>{
    t.onclick = ()=>{ document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
      document.querySelectorAll('.tabpane').forEach(x=>x.classList.remove('on'));
      t.classList.add('on');
      document.getElementById('pane-'+t.dataset.tab).classList.add('on'); };
  });
  document.querySelectorAll('.step').forEach(s=>{
    s.onclick = ()=>{ document.querySelectorAll('.step').forEach(x=>x.classList.remove('on'));
      s.classList.add('on');
      const el = document.getElementById(s.dataset.step); if(el) el.scrollIntoView({behavior:'smooth',block:'center'}); };
  });
}
window.addEventListener('DOMContentLoaded', ()=>{ sizeStage(); buildTables(); initTools(); init(); });
window.addEventListener('resize', ()=>{ sizeStage(); fit(); });
"""


def _bar(frac, color, width=100):
    return ('<div class="bar"><i style="width:%.2f%%;background:%s"></i>'
            '<i style="width:%.2f%%;background:#21262d"></i></div>'
            % (frac * 100, color, (1 - frac) * 100))


def render(led, meta):
    esc = html.escape
    parts = led["score"]["parts"]
    total = led["score"]["total"]
    off = led["score"]["official"]
    bad = [c for c in led["selfcheck"] if not c["ok"]]

    # ---------- 分数账 ----------
    colors = {"component": "#3fb950", "pin": "#58a6ff",
              "net_hypergraph": "#f0883e", "net_line": "#bc8cff"}
    rows = []
    for i, p in enumerate(parts):
        b = p["stats"]["buckets"]
        bl = " ｜ ".join("%s %d" % (VERDICT_STYLE.get(k, ("", k, ""))[1].split("（")[0], v)
                                  for k, v in sorted(b.items(), key=lambda kv: -kv[1]))
        rows.append("""
<div class="row" data-det="det-%d">
  <div><b>%s</b><div class="dim" style="font-size:11px">权重 %.2f</div></div>
  <div>%s<div class="dim" style="font-size:11px;margin-top:2px">%s</div></div>
  <div class="v"><b>%.3f</b><div class="dim" style="font-size:11px">F1</div></div>
  <div class="v"><b style="color:%s">%.3f</b><div class="dim" style="font-size:11px">得分</div></div>
  <div class="caret">&#9654;</div>
</div>
<div class="det" id="det-%d">
  <div class="notebox">%s</div>
  <div class="mono" style="margin-bottom:6px">
    tp <b style="color:var(--ok)">%d</b> ｜
    fp <b style="color:var(--fp)">%d</b> ｜
    fn <b style="color:var(--fn)">%d</b> ｜
    P %.4f ｜ R %.4f ｜ F1 %.4f ｜
    <b>丢 %.3f 分</b>（满分 %.1f）</div>
  <div class="mono dim" style="margin-bottom:8px">%s</div>
  __HEAD__
</div>""" % (i, esc(p["cn"]), p["weight"], _bar(p["f1"], colors.get(p["key"], "#8b949e")),
             bl, p["f1"], colors.get(p["key"], "#8b949e"), p["pts"], i, p["why"],
             p["tp"], p["fp"], p["fn"], p["p"], p["r"], p["f1"], p["lost"], p["max_pts"],
             esc(json.dumps(p["stats"], ensure_ascii=False))))

    # 每个指标的「修好能涨多少」
    for i, h in enumerate(meta.get("headroom", [])):
        items = "".join(
            '<li>%s → <span class="gain">%s</span> <span class="dim">%s</span></li>'
            % (esc(it["what"]),
               ("+%.2f 分" % it["gain"]) if it.get("gain") is not None else "见明细",
               esc(it["note"]))
            for it in h["items"])
        rows[i] = rows[i].replace("__HEAD__",
            ('<div style="margin-top:8px"><b>如果把这一类全修好，这个指标能涨：</b>'
             '<ul style="margin:6px 0 0 18px">%s</ul>'
             '<div class="dim" style="margin-top:4px">注意：这些收益**不能简单相加**'
             '（分子分母同时变），一项全修好之后其余项的边际收益会变。</div></div>'
             % items) if items else
            '<div class="dim">这一项已经没有可归类的错误。</div>')
    if bad:
        rows.insert(0, '<div class="warnbox">⚠ <b>自校验没通过</b>：本页的逐条统计与本地评分器 '
                       '<code>evaluate_case</code> 不一致 —— %s。'
                       '报告数字以本地评分器复算为准，不要用本页的明细。</div>'
                       % esc(json.dumps(bad, ensure_ascii=False)))

    # ---------- 一句话诊断 ----------
    flat = []
    for h in meta.get("headroom", []):
        for it in h["items"]:
            if it.get("gain"):
                flat.append((it["gain"], h["cn"], it["what"], it.get("note", "")))
    flat.sort(key=lambda x: -x[0])
    if flat:
        lis = "".join('<li><span class="gain">+%.2f 分</span> —— %s：%s '
                      '<span class="dim">%s</span></li>' % (g, cn, esc(w), esc(n))
                      for g, cn, w, n in flat[:5])
        diag = ('<div class="diagbox"><b>最大的一处损失有多大（单看这一张图）</b>'
                '<ul>%s</ul>'
                '<div class="dim">这些是<b>每一项单独从当前值修到满分</b>的收益，'
                '彼此不能相加（分子分母会同时变）。真正的意思：'
                '按这个顺序去修，收益最大。</div></div>' % lis)
    else:
        diag = ""
    st = meta.get("stages", {})
    steps = st.get("steps", [])
    stephtml = "".join(
        '<div class="step" data-step="stage-%d"><b>%s</b><div class="n">%s</div></div>'
        % (i, esc(s["name"]), esc(s["n"])) for i, s in enumerate(steps))

    # ---------- 图层工具条 ----------
    tools = """
<div class="tools">
  <button id="btn-fit">适应窗口</button>
  <div class="sep"></div>
  <button data-layer="cgt" class="on">GT 元件框</button>
  <button data-layer="cpr" class="on">我们的元件框</button>
  <button data-layer="pgt" class="on">GT 引脚点</button>
  <button data-layer="ppr" class="on">我们的引脚点</button>
  <div class="sep"></div>
  <span class="dim" style="font-size:12px">只看：</span>
  <button data-filter="missing">漏检</button>
  <button data-filter="extra">误检</button>
  <button data-filter="text_mismatch">文字不符</button>
  <button data-filter="bbox_mismatch">外框不符</button>
  <button data-filter="position_mismatch">引脚位置偏</button>
  <button data-filter="tp">正确</button>
  <div class="sep"></div>
  <span class="dim" style="font-size:12px">滚轮缩放 · 拖动平移 · 点表格行定位</span>
</div>"""

    legend = "".join(
        '<span class="pill" style="background:%s22;color:%s;border:1px solid %s66">%s</span> '
        % (c[0], c[0], c[0], c[1]) for c in VERDICT_STYLE.values())
    legend += ('<br><span class="dim">线型：<b>虚线 = GT 的框</b>，'
               '<b>实线 = 我们输出的框</b>；颜色 = 上面这条判决。'
               '一个位置同时出现「漏检」和「误检」两个框，就是'
               '<b>检测到了但键（位号）对不上</b> —— 那是 FP 和 FN 同时 +1。</span>')

    img_uri = "data:image/png;base64," + meta["img_b64"]
    led_json = json.dumps(led, ensure_ascii=False).replace("</", "<\\/")
    st_json = json.dumps({k: [v[0], v[1]] for k, v in VERDICT_STYLE.items()}, ensure_ascii=False)

    js = (JS.replace("__LED__", led_json)
            .replace("__VERDICT_STYLE__", st_json)
            .replace("__IMG_W__", str(meta["img_w"]))
            .replace("__IMG_H__", str(meta["img_h"])))

    cfg_json = esc(json.dumps(meta.get("config_note") or {}, ensure_ascii=False, indent=1))
    other_html = st.get("detail_html") or ""
    color = "#3fb950" if total >= 60 else ("#f0883e" if total >= 30 else "#f85149")
    head = "".join(rows)
    legend2 = legend
    tools2 = tools
    steps2 = stephtml
    case_esc = esc(led["case"])
    split_esc = esc(led.get("split") or "")
    gt_esc = esc(meta.get("gt_path", ""))
    pred_esc = esc(meta.get("pred_path", ""))

    return (
        '<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">\n'
        '<title>调试 ' + case_esc + ' · %.2f 分</title><style>' % total + CSS + '</style></head><body>\n'
        '<div class="wrap">\n'
        '<div class="hd">\n'
        '  <div><h1 class="mono">' + case_esc + '</h1>\n'
        '    <div class="dim">' + split_esc + ' ｜ GT <code>' + gt_esc + '</code>\n'
        '     ｜ 预测 <code>' + pred_esc + '</code></div></div>\n'
        '  <div style="margin-left:auto;text-align:right">\n'
        '    <div class="big mono" style="color:' + color + '">' + ("%.2f" % total) + '</div>\n'
        '    <div class="dim">/ 100 （本地复算 ' + ("%.2f" % off) + '）</div></div>\n'
        '</div>\n'
        '<h2>一、这 100 分是怎么没的</h2>\n'
        '<div class="notebox">此处为书面规则的本地重建评分，非官方成绩；分数是 <b>加权 F1 之和</b>，不是正确率：'
        '<code>100 × (0.30·元件F1 + 0.25·引脚F1 + 0.35·网络F1 + 0.10·导线F1)</code>。'
        '每一项的 F1 都是<b>整张图</b>算的：图上物品越多，错几个就越致命。'
        '点每一行展开，看这一类错误具体是哪几条、修好能涨多少。</div>\n'
        + diag +
        '<div class="led">' + head + '</div>\n'
        '<h2>二、这张图（把鼠标放上去看每一条的判决）</h2>\n'
        '<div class="notebox">' + legend2 + '</div>\n'
        + tools2 +
        '<div id="stage"><div id="world"><img src="' + img_uri + '" width="' + str(meta["img_w"])
        + '" height="' + str(meta["img_h"]) + '">\n'
        '<svg id="ov" width="' + str(meta["img_w"]) + '" height="' + str(meta["img_h"]) + '"></svg></div></div>\n'
        '<div id="tip"></div>\n'
        '<h2>三、管线步进（IDE 式：每个阶段往里加/减了什么）</h2>\n'
        '<div class="steps">' + steps2 + '</div>\n' + other_html +
        '<h2>四、逐条对账</h2>\n'
        '<div class="tabs">\n'
        '  <div class="tab on" data-tab="comp">元件 ' + str(len(led["components"])) + '</div>\n'
        '  <div class="tab" data-tab="pin">引脚 ' + str(len(led["pins"])) + '</div>\n'
        '  <div class="tab" data-tab="net">网络 ' + str(len(led["nets"])) + '</div>\n'
        '  <div class="tab" data-tab="line">导线 ' + str(len(led["lines"])) + '</div>\n'
        '</div>\n'
        '<div class="tabpane on" id="pane-comp"><table>\n'
        '<thead><tr><th>GT 键</th><th>我们键</th><th>GT 类型</th><th>我们类型</th>'
        '<th>GT 值</th><th>我们的值</th><th>GT 型号</th><th>我们的型号</th><th>中心距</th>'
        '<th>判决</th><th>为什么</th></tr></thead><tbody id="tbody-comp"></tbody></table></div>\n'
        '<div class="tabpane" id="pane-pin"><table>\n'
        '<thead><tr><th>IC</th><th>号</th><th>GT 键</th><th>我们键</th><th>GT 名</th><th>我们名</th>'
        '<th>点距</th><th>判决</th><th>为什么</th></tr></thead><tbody id="tbody-pin"></tbody></table></div>\n'
        '<div class="tabpane" id="pane-net"><table>\n'
        '<thead><tr><th>GT 网络</th><th>我们的网络</th><th>GT 引脚</th><th>我们引脚</th>'
        '<th>命中</th><th>判决</th><th>说明</th></tr></thead><tbody id="tbody-net"></tbody></table></div>\n'
        '<div class="tabpane" id="pane-line"><table>\n'
        '<thead><tr><th>GT 网络</th><th>我们的网络</th><th>对上</th><th>我们多</th><th>少了</th>'
        '<th>判决</th><th>说明</th></tr></thead><tbody id="tbody-line"></tbody></table></div>\n'
        '<h2>五、环境</h2>\n'
        '<div class="mono dim">' + cfg_json + '</div>\n'
        '</div><script>' + js + '</script></body></html>'
    )
