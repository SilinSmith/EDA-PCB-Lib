# -*- coding: utf-8 -*-
"""重建 workflow.html：要点清单 + 按数据流分带布局 + CSS/JS 补丁（阅读面板/分层 tidy）。

用法：
    python work\\tools\\diag\\_wfbuild.py            # 试运行，只打印布局
    python work\\tools\\diag\\_wfbuild.py --write    # 写回两处副本（先备份）

幂等：CSS/JS 补丁以「锚点必须唯一」的方式打，重复跑会明确报错而不是叠两次。
"""
import io, json, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # Y:\workspace\EDA


def _find(*rels):
    for r in rels:
        p = os.path.abspath(os.path.join(ROOT, r))
        if os.path.exists(p):
            return p
    raise SystemExit("[FATAL] 找不到 %s（在 %s 下试过 %s）" % (rels[0], ROOT, rels))


# 本脚本同时存在于 work\tools\diag（工作区）和 eda-pcb\tools\diag（仓库快照）：
# 工作区的 ROOT 是仓库根，仓库快照的 ROOT 是 eda-pcb ⇒ 两处都试。
FLOW = _find("work/flow", "../work/flow")
SRC = _find("work/flow/workflow.html", "../work/flow/workflow.html")
DST = _find("eda-pcb/docs/workflow.html", "docs/workflow.html")

sys.path.insert(0, FLOW)
import _wfbullets as W  # noqa: E402

COLW, X0, Y0, GAP, BANDGAP = 392, 40, 40, 34, 90
VIEW_W, VIEW_H = 1400, 990          # 中央画布的大致可用尺寸（1920 屏 − 左右栏 − 顶栏）


# ---------------------------------------------------------------- 布局
def depths(nodes, links):
    ids = [n["id"] for n in nodes]
    idset = set(ids)
    fwd = []
    for l in links:
        if l.get("kind") == "ctrl":
            continue
        a, b = l["from"].split(".")[0], l["to"].split(".")[0]
        if a != b and a in idset and b in idset:
            fwd.append((a, b))
    adj = {i: [] for i in ids}
    for a, b in fwd:
        adj[a].append(b)
    color, back = {}, set()
    sys.setrecursionlimit(10000)

    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color.get(v) == 1:
                back.add((u, v))
            elif v not in color:
                dfs(v)
        color[u] = 2

    for i in ids:
        if i not in color:
            dfs(i)
    dag = [e for e in fwd if e not in back]
    d = {i: 0 for i in ids}
    for _ in range(len(ids) + 2):
        moved = False
        for a, b in dag:
            if d[b] < d[a] + 1:
                d[b] = d[a] + 1
                moved = True
        if not moved:
            break
    return d, back, dag


def est_height(n):
    h = 30                                              # 表头
    sub = n.get("sub") or ""
    if sub:
        lines = min(3, max(1, (len(sub) + 33) // 34))
        h += 15 * lines + 5
    h += 6
    h += max(len(n.get("ins") or []), len(n.get("outs") or [])) * 16
    par = n.get("params") or []
    if par:
        chars = sum(len(str(p[0])) + len(str(p[1])) + 2 for p in par)
        h += 8 + 20 * max(1, (chars + 42) // 43)
    nb = len(n.get("bullets") or [])
    if nb:
        total = sum(len(b) + 2 for b in n["bullets"])
        h += 12 + 14 * max(nb, (total + 40) // 41)
    return h + 14


def bary_sort(layers, pred, succ, iters=6):
    pos = {}
    for col in layers:
        for i, n in enumerate(col):
            pos[n["id"]] = i
    for it in range(iters):
        rev = it % 2 == 1
        seq = range(len(layers) - 1, -1, -1) if rev else range(len(layers))
        for li in seq:
            col = layers[li]
            key = {}
            for i, n in enumerate(col):
                nb = (succ if rev else pred).get(n["id"]) or []
                vals = [pos[m] for m in nb if m in pos]
                key[n["id"]] = (sum(vals) / len(vals)) if vals else float(i)
            col.sort(key=lambda n: (key[n["id"]], pos[n["id"]]))
            for i, n in enumerate(col):
                pos[n["id"]] = i
    return layers


def col_h(col):
    return sum(est_height(n) for n in col) + GAP * max(0, len(col) - 1)


def bands_height(layers, C):
    tot = 0
    nbands = 0
    for b0 in range(0, len(layers), C):
        band = layers[b0:b0 + C]
        tot += max((col_h(c) for c in band), default=0)
        nbands += 1
    return tot + BANDGAP * max(0, nbands - 1)


def choose_bands(layers):
    """选一个让整图在给定视口里缩得最大的分带宽度。"""
    best = None
    for C in range(1, len(layers) + 1):
        w = C * COLW
        h = bands_height(layers, C)
        k = min((VIEW_W - 60) / max(w, 1), (VIEW_H - 60) / max(h, 1))
        key = (round(k, 4), -w)
        if best is None or key > best[0]:
            best = (key, C, w, h, k)
    return best[1], best[2], best[3], best[4]


def place(layers, C):
    y = Y0
    for b0 in range(0, len(layers), C):
        band = layers[b0:b0 + C]
        H = max((col_h(c) for c in band), default=0)
        for i, col in enumerate(band):
            yy = y + max(0, (H - col_h(col)) / 2)
            for n in col:
                n["x"] = X0 + i * COLW
                n["y"] = int(yy)
                yy += est_height(n) + GAP
        y += H + BANDGAP
    return y


def overlaps(layers):
    bad = []
    for col in layers:
        for i in range(len(col) - 1):
            a, b = col[i], col[i + 1]
            if a["y"] + est_height(a) > b["y"]:
                bad.append((a["id"], b["id"]))
    return bad


def layout(nodes, links):
    d, back, dag = depths(nodes, links)
    maxd = max(d.values()) if d else 0
    layers = [[] for _ in range(maxd + 1)]
    for n in nodes:
        layers[d[n["id"]]].append(n)
    pred, succ = {}, {}
    for a, b in dag:
        pred.setdefault(b, []).append(a)
        succ.setdefault(a, []).append(b)
    layers = bary_sort(layers, pred, succ)
    Cb, _wb, _hb, kb = choose_bands(layers)
    # 单条链：数据流一路向右，最容易读。适应窗口的缩放比只有 ~0.25（本来就不可读），
    # 所以不靠 fit() 保证可读性 —— 靠「默认落在入口节点的 0.75 缩放」+ 右侧阅读面板。
    C = len(layers)
    w = C * COLW
    h = bands_height(layers, C)
    k = min((VIEW_W - 60) / w, (VIEW_H - 60) / h)
    place(layers, C)
    return d, back, layers, C, w, h, k, (Cb, kb)


# ---------------------------------------------------------------- 补丁
CSS_OLD = (
    ".node .sub{color:var(--dim);font-size:10.5px;line-height:1.35;margin-bottom:5px;\n"
    "  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}"
)
CSS_NEW = (
    ".node .sub{color:var(--dim);font-size:10.5px;line-height:1.35;margin-bottom:5px;\n"
    "  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}\n"
    ".node .bul{margin:0 0 5px;padding:3px 4px 4px;background:#0f1620;border:1px solid #212b38;\n"
    "  border-radius:5px;list-style:none;font-size:10px;line-height:1.45;color:#aeb8c4}\n"
    ".node .bul li{position:relative;padding-left:9px}\n"
    ".node .bul li:before{content:'\\25B8';position:absolute;left:1px;color:#4a5b6d}\n"
    ".node .bul b{color:#e6edf3;font-weight:600}\n"
    "#insp .rd{padding-bottom:24px}\n"
    ".rt{font-size:14px;font-weight:700;border-left:3px solid;padding-left:7px;margin:2px 0 6px}\n"
    ".rmeta{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:7px}\n"
    ".rsub{color:#aeb8c4;font-size:11.5px;line-height:1.5;margin-bottom:5px}\n"
    ".rfile{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;color:#7d8590;\n"
    "  margin-bottom:6px;word-break:break-all}\n"
    ".rsec{margin:11px 0 6px;font-size:10px;letter-spacing:.09em;color:#7d8590;\n"
    "  text-transform:uppercase;border-top:1px solid #212b38;padding-top:7px}\n"
    ".rports div{margin-bottom:3px;line-height:1.9}\n"
    ".rports i{display:inline-block;background:#1b2430;border:1px solid #2b3543;border-radius:4px;\n"
    "  padding:0 5px;margin:0 3px 3px 0;font-style:normal;font-size:10.5px;color:#9aa5b1}\n"
    ".rpar .rp{display:grid;grid-template-columns:100px 1fr;gap:6px;margin-bottom:5px;\n"
    "  font-size:11px;align-items:start}\n"
    ".rpar .rp b{color:#e6edf3;font-weight:600;font-family:ui-monospace,Consolas,monospace;\n"
    "  font-size:10.5px;word-break:break-all}\n"
    ".rpar .rp span{color:#aeb8c4}\n"
    ".rbul{padding-left:0;list-style:none;margin:0}\n"
    ".rbul li{position:relative;padding-left:12px;margin-bottom:6px;font-size:11.5px;\n"
    "  line-height:1.55;color:#c9d1d9}\n"
    ".rbul li:before{content:'\\25B8';position:absolute;left:0;color:#58a6ff}\n"
    ".rbul code,.rnote code{background:#1b2430;border-radius:3px;padding:0 3px;\n"
    "  font-family:ui-monospace,Consolas,monospace;font-size:10.5px;color:#79c0ff}\n"
    ".rbul b,.rnote b{color:#e6edf3}\n"
    ".rnote{white-space:pre-wrap;font-family:ui-monospace,Consolas,monospace;font-size:10.5px;\n"
    "  line-height:1.6;color:#8b949e}"
)

TOP_OLD = '    <button onclick="fit()">适应窗口</button>'
TOP_NEW = ('    <button onclick="fit()">适应窗口</button>\n'
           '    <button onclick="toggleMode()" id="mbtn" class="pri">✎ 编辑</button>')

JS = [
    # 1) rich() 助手
    (r"""function renderNodes(){
  const box = document.getElementById('nodes');""",
     r"""function rich(s){ return esc(s).replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>'); }
function md(s){
  return esc(s).replace(/`([^`]+)`/g,'<code>$1</code>')
               .replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');
}
function renderNodes(){
  const box = document.getElementById('nodes');"""),
    # 2) 节点正面渲染要点清单
    (r"""    if(n.sub) h += '<div class="sub">'+esc(n.sub)+'</div>';""",
     r"""    if(n.sub) h += '<div class="sub">'+esc(n.sub)+'</div>';
    if(n.bullets && n.bullets.length)
      h += '<ul class="bul">' + n.bullets.map(b=>'<li>'+rich(b)+'</li>').join('') + '</ul>';"""),
    # 3) fit() 用真实高度
    (r"""  for(const n of ns){ x1=Math.min(x1,n.x); y1=Math.min(y1,n.y);
    x2=Math.max(x2,n.x+(n.w||210)); y2=Math.max(y2,n.y+230); }""",
     r"""  for(const n of ns){ const el=document.getElementById('nd_'+n.id);
    const hh = el ? el.offsetHeight : 260;
    x1=Math.min(x1,n.x); y1=Math.min(y1,n.y);
    x2=Math.max(x2,n.x+(n.w||210)); y2=Math.max(y2,n.y+hh); }"""),
    # 4) renderInsp 拆成「阅读 / 编辑」两态
    (r"""function renderInsp(){
  const box = document.getElementById('insp');
  if(!sel){ box.innerHTML = '<div class="empty">点一个节点或一条连线看详情。<br><br>'
    + '这个页面是<b>可编辑</b>的：拖节点、拖圆点连线、按 Delete 删除、'
    + '改完点「改动说明」把变更贴回给我，我按新流程改代码。</div>'; return; }
  if(sel.type==='link'){""",
     r"""let MODE = 'read';
function toggleMode(){
  MODE = MODE==='read' ? 'edit' : 'read';
  const b = document.getElementById('mbtn');
  if(b){ b.textContent = MODE==='read' ? '✎ 编辑' : '👁 阅读'; b.className = MODE==='read'?'pri':''; }
  renderInsp();
}
const KIND_ZH = {exec:'现在有',param:'标尺',data:'数据',new:'★要新写',cand:'候选',dead:'停用'};
const KIND_C  = {exec:'#3fb950',param:'#d29922',data:'#58a6ff',new:'#f85149',cand:'#f85149',dead:'#6e7681'};
function renderInsp(){ return MODE==='read' ? renderRead() : renderEdit(); }
function renderRead(){
  const box = document.getElementById('insp');
  if(!sel){ box.innerHTML = '<div class="empty">点一个节点看它的<b>完整说明</b>（要点 + 详细记录）。<br><br>'
    + '顶部「✎ 编辑」切到编辑态：拖节点、拖圆点连线、按 Delete 删除；'
    + '改完点「改动说明」把变更贴回给我，我按新流程改代码。</div>'; return; }
  if(sel.type==='link'){
    const L = G.links.find(x=>linkId(x)===sel.id);
    if(!L){ sel=null; return renderRead(); }
    box.innerHTML = '<div class="rd"><h4 style="margin-top:2px">连线</h4>'
      + '<div class="rt" style="border-left-color:#4a5b6d">'+esc(L.from)+' → '+esc(L.to)+'</div>'
      + '<div class="rmeta"><span class="tag">'+esc(L.kind)+'</span></div>'
      + '<div class="rnote">data = 数据流（一条记录往下传）\nctrl = 控制面（预算/开关决定对面跑不跑）</div></div>';
    return;
  }
  const n = nodeById(sel.id); if(!n){ sel=null; return renderRead(); }
  const g = groupById(n.group);
  const kc = KIND_C[n.kind]||'#8b949e', kn = KIND_ZH[n.kind]||n.kind;
  const ins = n.ins||[], outs = n.outs||[], par = n.params||[], bl = n.bullets||[];
  let h = '<div class="rd">';
  h += '<h4 style="margin-top:2px">'+esc(n.id)+'</h4>';
  h += '<div class="rt" style="border-left-color:'+g.color+'">'+esc(n.title)+'</div>';
  h += '<div class="rmeta"><span class="kbadge" style="color:'+kc+';border-color:'+kc+'">'+kn+'</span>'
     + '<span class="tag">'+esc(g.title)+'</span></div>';
  if(n.sub) h += '<div class="rsub">'+esc(n.sub)+'</div>';
  if(n.file) h += '<div class="rfile">'+esc(n.file)+'</div>';
  if(ins.length || outs.length){
    h += '<div class="rsec">端口</div><div class="rports">'
      + (ins.length ? '<div><span class="lab">入 </span>'+ins.map(p=>'<i>'+esc(p.label||p.id)+'</i>').join('')+'</div>' : '')
      + (outs.length? '<div><span class="lab">出 </span>'+outs.map(p=>'<i>'+esc(p.label||p.id)+'</i>').join('')+'</div>' : '')
      + '</div>';
  }
  if(par.length){
    h += '<div class="rsec">参数（'+par.length+'）</div><div class="rpar">'
      + par.map(p=>'<div class="rp"><b>'+esc(p[0])+'</b><span>'+esc(p[1])+'</span></div>').join('') + '</div>';
  }
  if(bl.length){
    h += '<div class="rsec">要点清单（'+bl.length+'）</div><ul class="rbul">'
      + bl.map(b=>'<li>'+md(b)+'</li>').join('') + '</ul>';
  }
  if(n.note) h += '<div class="rsec">详细记录</div><div class="rnote">'+md(n.note)+'</div>';
  h += '</div>';
  box.innerHTML = h;
}
function renderEdit(){
  const box = document.getElementById('insp');
  if(!sel){ box.innerHTML = '<div class="empty">点一个节点或一条连线看详情。</div>'; return; }
  if(sel.type==='link'){"""),
    # 5) renderEdit 内部的自调用改名
    (r"""  const n = nodeById(sel.id); if(!n){ sel=null; return renderInsp(); }""",
     r"""  const n = nodeById(sel.id); if(!n){ sel=null; return renderEdit(); }"""),
    # 6) 检视器里能编辑要点清单
    (r"""    + '<div class="row"><span>说明</span><textarea oninput="setNode(\'note\',this.value)">'+esc(n.note||'')+'</textarea></div>'""",
     r"""    + '<div class="row"><span>说明</span><textarea oninput="setNode(\'note\',this.value)">'+esc(n.note||'')+'</textarea></div>'
    + '<div class="row"><span>要点清单（每行一条，**粗体**，画在节点正面）</span>'
      + '<textarea style="min-height:130px" oninput="setBullets(this.value)">'+esc((n.bullets||[]).join('\n'))+'</textarea></div>'"""),
    # 7) setBullets()
    (r"""function setNode(k,v){ const n=nodeById(sel.id); if(n){ n[k]=v; renderNodes(); renderList(); save(); renderStat(); } }""",
     r"""function setNode(k,v){ const n=nodeById(sel.id); if(n){ n[k]=v; renderNodes(); renderList(); save(); renderStat(); } }
function setBullets(v){ const n=nodeById(sel.id); if(n){ n.bullets=v.split('\n').map(s=>s.trim()).filter(Boolean); renderNodes(); save(); } }"""),
    # 8) 新节点默认带空 bullets
    (r"""params:[], note:''});""",
     r"""params:[], bullets:[], note:''});"""),
    # 9) tidy() 改成按数据流分带
    (r"""function tidy(){
  const cols = {};
  for(const g of G.groups) cols[g.id] = {x:40 + G.groups.indexOf(g)*360, y:90};
  for(const n of G.nodes){
    const c = cols[n.group] || (cols[n.group] = {x:40,y:90});
    n.x = c.x; n.y = c.y; c.y += 236;
  }
  renderAll(); fit();
}""",
     r"""function layerOf(){
  // 最长路径分层；先用 DFS 剔除回边，保证是 DAG。
  const ids = G.nodes.map(n=>n.id), idset = new Set(ids);
  const fwd = [];
  for(const l of G.links){
    if(l.kind==='ctrl') continue;
    const a=l.from.split('.')[0], b=l.to.split('.')[0];
    if(a!==b && idset.has(a) && idset.has(b)) fwd.push([a,b]);
  }
  const adj = {}; ids.forEach(i=>adj[i]=[]);
  fwd.forEach(([a,b])=>adj[a].push(b));
  const color = {}, back = new Set();
  const dfs = u => { color[u]=1;
    for(const v of adj[u]){ if(color[v]===1) back.add(u+'>'+v);
      else if(color[v]===undefined) dfs(v); }
    color[u]=2; };
  ids.forEach(i=>{ if(color[i]===undefined) dfs(i); });
  const dag = fwd.filter(([a,b])=>!back.has(a+'>'+b));
  const d = {}; ids.forEach(i=>d[i]=0);
  for(let k=0;k<ids.length+2;k++){ let moved=false;
    for(const [a,b] of dag) if(d[b]<d[a]+1){ d[b]=d[a]+1; moved=true; }
    if(!moved) break; }
  return d;
}
function estH(n){
  let h = 30;
  if(n.sub) h += 15*Math.min(3,Math.max(1,Math.ceil(n.sub.length/34))) + 5;
  h += 6 + 16*Math.max((n.ins||[]).length,(n.outs||[]).length);
  const par = n.params||[];
  if(par.length){ const c = par.reduce((s,p)=>s+String(p[0]).length+String(p[1]).length+2,0);
    h += 8 + 20*Math.max(1,Math.ceil(c/43)); }
  const nb = (n.bullets||[]).length;
  if(nb){ const t = n.bullets.reduce((s,b)=>s+b.length+2,0);
    h += 12 + 14*Math.max(nb,Math.ceil(t/41)); }
  return h + 14;
}
function tidy(){
  const d = layerOf();
  const layers = {}; let maxd = 0;
  for(const n of G.nodes){ const k=d[n.id]||0; maxd=Math.max(maxd,k);
    (layers[k]=layers[k]||[]).push(n); }
  // 层内按前驱重心排序，减少交叉
  const pos = {};
  for(let k=0;k<=maxd;k++) (layers[k]||[]).forEach((n,i)=>{ pos[n.id]=i; });
  const pred = {};
  for(const l of G.links){ if(l.kind==='ctrl') continue;
    const a=l.from.split('.')[0], b=l.to.split('.')[0];
    if(a!==b) (pred[b]=pred[b]||[]).push(a); }
  for(let it=0; it<6; it++){
    const rev = it%2===1;
    for(let k = rev?maxd:0; rev?k>=0:k<=maxd; k += rev?-1:1){
      const col = layers[k]||[]; if(!col.length) continue;
      const key = {};
      col.forEach((n,i)=>{ const nb = rev ? [] : (pred[n.id]||[]);
        const vals = rev ? [] : nb.map(m=>pos[m]).filter(v=>v!==undefined);
        key[n.id] = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : i; });
      col.sort((a,b)=> key[a.id]-key[b.id] || pos[a.id]-pos[b.id]);
      col.forEach((n,i)=>{ pos[n.id]=i; });
    }
  }
  const COLW=392, X0=40, Y0=40, GAP=34, BANDGAP=90;
  const C = Math.max(1, Math.min((G.layout && G.layout.bands) || 7, maxd+1));
  const colH = col => col.reduce((s,n,i)=>s+estH(n)+(i?GAP:0),0);
  let y = Y0;
  for(let b0=0; b0<=maxd; b0+=C){
    const ks = []; for(let k=b0;k<Math.min(b0+C,maxd+1);k++) ks.push(k);
    let bandH = 0; ks.forEach(k=>{ bandH = Math.max(bandH, colH(layers[k]||[])); });
    ks.forEach((k,i)=>{
      const col = layers[k]||[];
      let yy = y + Math.max(0,(bandH-colH(col))/2);
      col.forEach(n=>{ n.x = X0 + i*COLW; n.y = yy; yy += estH(n)+GAP; });
    });
    y += bandH + BANDGAP;
  }
  renderAll(); fit();
}
function startView(){
  // 单条链宽 5000+px，fit() 只能给 ~0.25 缩放（字太小读不了）。
  // 默认落在入口节点附近、0.75 缩放；要看全貌按「适应窗口」。
  const first = G.nodes.find(n=>n.id==='input') || G.nodes[0]; if(!first) return;
  const r = cv.getBoundingClientRect();
  view.k = 0.75;
  view.x = 40 - first.x*view.k;
  view.y = Math.max(20, r.height*0.34 - (first.y+130)*view.k);
  applyView();
}"""),
    # 10) 首屏用可读缩放（不是整图 fit）
    ("renderAll(); fit();\n</script>",
     "renderAll(); startView();\n</script>"),
]


def apply_patch(text, old, new, tag):
    if new in text:
        return text, False
    c = text.count(old)
    if c != 1:
        raise SystemExit("[FATAL] 锚点 %s 出现 %d 次（应为 1）" % (tag, c))
    return text.replace(old, new, 1), True


def main():
    write = "--write" in sys.argv
    raw = io.open(SRC, encoding="utf-8").read()

    m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)',
                  raw, re.S)
    if not m:
        raise SystemExit("[FATAL] 找不到 graphdata")
    d = json.loads(m.group(2))

    nb = ns = npar = 0
    for n in d["nodes"]:
        i = n["id"]
        if i in W.BULLETS:
            n["bullets"] = W.BULLETS[i]
            nb += 1
        if i in W.SUB:
            n["sub"] = W.SUB[i]
            ns += 1
        if i in W.PARAMS:
            n["params"] = W.PARAMS[i]
            npar += 1
    print("内容：%d 个节点有要点清单 / %d 个改了副标题 / %d 个改了参数" % (nb, ns, npar))
    print("要点总数：%d 条 / %d 字" % (sum(len(v) for v in W.BULLETS.values()),
                                  sum(len(b) for v in W.BULLETS.values() for b in v)))

    dep, back, layers, C, w, h, k, (Cb, kb) = layout(d["nodes"], d["links"])
    d.setdefault("layout", {})["bands"] = C
    print("\n回边 %d 条：%s" % (len(back), sorted("%s>%s" % t for t in back)))
    print("按数据流分层 %d 列（单条链）；整图 %d×%d ⇒ 适应窗口 k≈%.3f"
          % (len(layers), w, h, k))
    print("（作为对比：分带最优是 %d 列/带，k≈%.3f —— 差别不大，取可读性更好的单链）" % (Cb, kb))
    print("\n布局：")
    for bi, b0 in enumerate(range(0, len(layers), C)):
        print("  ── 带 %d（层 %d..%d）" % (bi + 1, b0, min(b0 + C, len(layers)) - 1))
        for kk in range(b0, min(b0 + C, len(layers))):
            col = layers[kk]
            print("     第 %d 层 x=%d h=%d：%s" % (kk, col[0]["x"] if col else 0, col_h(col),
                  ", ".join("%s(y=%d)" % (n["id"], n["y"]) for n in col)))
    bad = overlaps(layers)
    print("\n列内重叠：%s" % ("无" if not bad else bad))

    if not write:
        print("\n（试运行，未写盘。加 --write 落盘）")
        return

    jt = json.dumps(d, ensure_ascii=False, indent=1)
    out = raw[:m.start(2)] + "\n" + jt + "\n" + raw[m.end(2):]

    out, c1 = apply_patch(out, CSS_OLD, CSS_NEW, "CSS")
    out, ct = apply_patch(out, TOP_OLD, TOP_NEW, "TOP")
    done = []
    for i, (o, nw) in enumerate(JS):
        out, ch = apply_patch(out, o, nw, "JS#%d" % (i + 1))
        if ch:
            done.append(i + 1)
    print("补丁：CSS %s｜工具栏 %s｜JS %s"
          % ("已打" if c1 else "已是新版", "已打" if ct else "已是新版",
             ("已打 " + str(done)) if done else "已是新版"))

    m2 = re.search(r'<script type="application/json" id="graphdata">(.*?)</script>', out, re.S)
    d2 = json.loads(m2.group(1))
    assert len(d2["nodes"]) == len(d["nodes"]) and len(d2["links"]) == len(d["links"])
    ids = {n["id"] for n in d2["nodes"]}
    ports = set()
    for n in d2["nodes"]:
        for p in (n.get("ins") or []) + (n.get("outs") or []):
            ports.add(n["id"] + "." + p["id"])
    bad_ep = [l for l in d2["links"] if l["from"] not in ports or l["to"] not in ports]
    print("校验：%d 节点 / %d 连线；悬空端点 %d；缺口 %d；k≈%.3f"
          % (len(d2["nodes"]), len(d2["links"]), len(bad_ep),
             len({n["id"] for n in d2["nodes"]} - ids), k))

    for p in (SRC, DST):
        if os.path.exists(p):
            shutil.copy2(p, p + ".bak8")
            print("  备份 %s" % (p + ".bak8"))
        io.open(p, "w", encoding="utf-8", newline="\n").write(out)
        print("  写出 %s（%d B）" % (p, len(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
