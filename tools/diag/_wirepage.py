# -*- coding: utf-8 -*-
"""_wirepage.py —— 生成独立的「导线 workflow.html」单文件页面（不依赖任何外部资源）。

    python work\\tools\\diag\\_wirepage.py                  # 写 work\\flow\\导线workflow.html
    python work\\tools\\diag\\_wirepage.py --also-docs      # 额外再写一份到 eda-pcb\\docs\\
    python work\\tools\\diag\\_wirepage.py --write-docs     # 只写 eda-pcb\\docs\\（父代理批准后才用）

做法与 `work\\tools\\flow\\build.py` 完全一致：**把 `_wire_stages.json` 内嵌进 HTML 的
`<script type="application/json" id="graphdata">`**，双击即可打开，不需要服务器
（`file://` 下 `fetch('x.json')` 会被 CORS 挡掉，所以必须内嵌）。

数据源：`work\\_wire_stages.json`（本目录另有 README 说明字段）。
页面内容：中部是「导线识别分阶段节点图」（可拖/可滚/点节点看细节），
左下是主图「容差 vs tp」曲线，右下是「空网抢配对」单格。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "work" / "_wire_stages.json"
DOC = ROOT / "work" / "_wire_stages.md"
RECIPE = ROOT / "work" / "_wire_page_recipe.md"
OUT = ROOT / "work" / "flow" / "导线workflow.html"
DOCS_OUT = ROOT / "eda-pcb" / "docs" / "导线workflow.html"

# 主图原始数字（work\tools\diag\_wirefix.py 实测，官方 9 张离线重算）
TOL = [5, 10, 20, 40, 60]
TP = [148, 239, 316, 346, 346]
FP = [857, 766, 689, 659, 659]
FN = [405, 314, 237, 207, 207]
F1 = [0.1900, 0.3068, 0.4056, 0.4442, 0.4442]
SCORE = [58.63, 59.91, 60.78, 61.23, 61.23]
COVER = {"tol5": 14.6, "tol10": 65.3, "tol20": 91.9, "tol40": 97.5}

EXTRA = {
    "curves": {"tol": TOL, "tp": TP, "fp": FP, "fn": FN, "f1": F1, "score": SCORE,
               "cover": COVER,
               "note": "官方 9 张（case 1,2,3,4,5,7,8,9,10）离线重算；"
                       "脚本 work\\tools\\diag\\_wirefix.py（用 work\\_f220_A\\*.json 现成产物，不动管线）。"
                       "「覆盖」= GT 边在 tol 内能找到一条 stage1 线段的比例（_wireskel.py）。"},
    "gate": {"with_net": {"tp": 148, "fp": 857, "fn": 405, "f1": 0.1876},
             "no_net": {"tp": 189, "fp": 816, "fn": 364, "f1": 0.2426},
             "note": "把网级配对（metrics.py:477-494 match_nets）整个拿掉、只做全局边匈牙利"
                     "（line_tol=5）：tp 148→189、F1 0.1876→0.2426。脚本 _wirepairs.py。"},
    "empty": {"case1_net": 12, "case1_pred_net": 17, "gt9_empty_net": 27,
              "tp_before": 148, "tp_after": 154, "fp_before": 857, "fp_after": 851,
              "fn_before": 405, "fn_after": 399, "score_before": 58.63, "score_after": 58.18,
              "note": "删掉 edges={} 的空网后 tp/fp/fn 都变好（148→154、405→399），但 mean score "
                      "58.63→58.18（−0.45）：它同时让 case 4 −2.11 / 9 −3.30 / 5 −0.58 掉分。脚本 _wirefix.py B 臂。"},
    "cands": [
        {"name": "(a) 端点吸附到真交点（支撑直线求交）", "where": "eda-pcb/stage2/assemble.py:259-307 snap_edge_ends（现版把端点投到邻居的**有限线段**垂足上，位移中位 2.0px、吸完更靠 GT 端点 340 / 更远 325，净收益=0）",
         "measure": "离线穷举 radius 20~80 / maxmove 8~40 / 夹角 30~90：最好一档 radius40+minang40 ⇒ tp 148→152、fp 857→853、fn 405→401、F1 0.1900→0.1951（移动 904 个端点，零损失）",
         "risk": "低——只在同一 net 内搬端点、位移上限 40px、不动网结构；最坏情况与现版一样是 0 收益",
         "verdict": "可以做，但只值 +0.0055 F1（×0.10 权重 ≈ +0.055 总分）；不是主项"},
        {"name": "(b) 端点改用拟合直线求交而不是骨架端点", "where": "同上，只是候选邻居从「有限线段」换成「支撑直线 + 夹角门禁」",
         "measure": "就是 (a) 的实测对象；与现管线在同一批产物上跑，基线 148/857/405 一条没变 ⇒ 现管线的 snap **确实在执行**，只是搬错了地方",
         "risk": "低","verdict": "等价于 (a)"},
        {"name": "(c) 同一 junction 簇端点聚类取质心", "where": "eda-pcb/stage1/labels.py:148-166 _rebuild_nodes（已有 tol=1.0 分桶）/ stage2 装配前",
         "measure": "离线实测（同 net、半径 6px、只动 >=2 点的簇，共动 1635 个端点）：tp/fp/fn **148/857/405 一位不变**，F1 0.1900",
         "risk": "低","verdict": "实测无收益——因为簇内端点的均值仍在 5px 噪声带里，官方只看两个端点坐标"},
        {"name": "(d) 把碎边并回一条长边", "where": "eda-pcb/stage2/assemble.py:227 STAGE2_EDGE_MERGE=0 / :310 merge_collinear",
         "measure": "方向性判断（未做全量实测）：我们 1005 条边 vs GT 553 条（1.82×），合并只会减少 fp；但现口径下 fn 405 里没有任何一条能靠并边变成 tp（tp 的 148 条已经在那 5px 内）",
         "risk": "中——EDGE_MERGE 开过一版：19 张快测上「平的且多爆两个零」",
         "verdict": "已知无收益，别做"},
        {"name": "★★ 网级配对口径（最大的那一环）", "where": "eda-pcb/schematic_parser/metrics.py:477-494 match_nets —— 只看 net_pin_set 交叠，交叠 1 个引脚就配上",
         "measure": "把网级门禁整个拿掉、只做全局边匈牙利（tol=5）：tp 148→189、fp 857→816、fn 405→364、F1 0.1876→0.2426；派生 19 张 tp 35→220、F1 0.0160→0.1003",
         "risk": "这是评测口径、不能改，但**我们能改自己的行为**：少输出「单脚空网」、把引脚挂到正确的网上、让我们的网与 GT 网一对一",
         "verdict": "★ 建议先做这个（0.1876→0.2426 的空间，比端点那一档大 10 倍）"},
        {"name": "★★ 上游元件/位号（6 张爆 0 的根因）", "where": "eda-pcb/stage2/assemble.py:164 STAGE2_NET_DROP_UNKEYED=1 + :660-668 位号槽位",
         "measure": "0048：`_assigned/0048.json` 的 desig_slots={} ⇒ 39/39 元件被丢 ⇒ keymap 空 ⇒ 每个 net refs 空 ⇒ :900 if not refs: continue 把 32 个网整网丢掉 ⇒ nets={} ⇒ 导线 tp0/fp0/fn128",
         "risk": "改的是元件键，会连带 component(0.30) 与 pin(0.25)，必须小步验证",
         "verdict": "★ 与上一条同源：元件对不上 ⇒ 网对不上 ⇒ 导线全丢"}],
    "six": {"0048": {"edges": 0, "verdict": "真的一条都没输出（nets={}；stage1 有 32 网 305 段，被 assemble.py:900 if not refs 全丢；39 个预测元件键命中 0 个）"},
            "0092": {"edges": 133, "verdict": "输出了但全判错（7 网 133 边；GT 51 网 156 边；fn 里 94 条来自「网没配上」）"},
            "0019": {"edges": 16, "verdict": "输出了但全判错且远少于 GT（s1 52 网 → s2 仅 2 网 1 边 → 输出 16 边；键命中 6/35）"},
            "0024": {"edges": 144, "verdict": "输出了但全判错（s1 36 网 252 段 → s2 5 网 16 边 → 输出 144 边）"},
            "0086": {"edges": 154, "verdict": "输出了但全判错（s1 105 网 476 段 → s2 2 网 15 边 → 输出 154 边；键命中 6/67）"},
            "0042": {"edges": 25, "verdict": "输出了但全判错（s1 21 网 74 段 → s2 4 网 25 边 → 输出 25 边；键命中 1/16）"}},
    "funnel": {"gt9": {"gt_edges": 553, "pred_edges": 1005, "s1_segIn": 1730, "s2_edges": 979,
                       "fn_net": 26, "fn_noedge": 123, "fn_geom": 256, "keys": "116/219"},
               "d19": {"gt_edges": 2824, "pred_edges": 1563, "s1_segIn": 4777, "s2_edges": 1076,
                       "fn_net": 1403, "fn_noedge": 640, "fn_geom": 746, "keys": "195/829"}},
}

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>导线识别 workflow · 分阶段</title>
<style>
:root{--bg:#12151c;--pn:#1b202b;--bd:#2c3444;--tx:#dfe6f2;--dim:#8b98ad;--ac:#5cc8ff;--ok:#5ad07a;--wn:#f0b429;--er:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.65 "Microsoft YaHei",system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--bd);display:flex;gap:14px;flex-wrap:wrap;align-items:baseline}
h1{font-size:16px;margin:0}h2{font-size:14px;margin:0 0 6px}
.sub{color:var(--dim);font-size:12px}
.wrap{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(320px,.85fr);gap:12px;padding:12px;align-items:start}
.card{background:var(--pn);border:1px solid var(--bd);border-radius:10px;padding:12px}
#cvwrap{overflow:hidden;position:relative;height:78vh;min-height:520px;cursor:grab;background:
 repeating-linear-gradient(0deg,#161a22,#161a22 24px,#181d26 24px,#181d26 25px),
 repeating-linear-gradient(90deg,#161a22,#161a22 24px,#181d26 24px,#181d26 25px)}
#cv{position:absolute;left:0;top:0;transform-origin:0 0}
.node{position:absolute;width:250px;background:#1e2431;border:1px solid var(--bd);border-left:5px solid var(--ac);border-radius:8px;padding:8px 10px;cursor:pointer}
.node.data{border-left-color:#a78bfa}.node.param{border-left-color:var(--wn)}.node.exec{border-left-color:var(--ok)}
.node.sel{outline:2px solid #fff;box-shadow:0 0 0 4px rgba(92,200,255,.25)}
.node .t{font-weight:600;font-size:13px}.node .s{color:var(--dim);font-size:11px;margin-top:3px}
.node .g{color:#5cc8ff;font-size:10px;margin-top:4px;opacity:.85}
svg.links{position:absolute;left:0;top:0;pointer-events:none}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid var(--bd);padding:3px 6px;text-align:left;vertical-align:top}
th{color:var(--dim);font-weight:500}
code,kbd{background:#0e1219;border:1px solid var(--bd);border-radius:4px;padding:0 4px;font-size:11.5px}
ul{margin:6px 0 0 16px;padding:0}li{margin:3px 0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px}
.kv b{color:#9fb3cd;font-weight:500}
.tag{display:inline-block;border:1px solid var(--bd);border-radius:999px;padding:0 8px;font-size:11px;color:var(--dim);margin-right:6px}
b.hl{color:var(--wn)}.mono{font-family:Consolas,monospace;font-size:11.5px}
#detail{max-height:74vh;overflow:auto}
</style></head><body>
<header>
<h1>导线识别 workflow · 从骨架像素到 net_line 实例</h1>
<span class="sub">官方 9 张实测：net_line F1 <b class="hl">0.1876</b>（tp 148 / fp 857 / fn 405），权重 0.10</span>
<span class="sub">主图在这页左下角：容差 vs tp</span>
</header>
<div class="wrap">
  <div class="card" style="padding:0">
    <div id="cvwrap"><svg class="links" id="links"></svg><div id="cv"></div></div>
  </div>
  <div>
    <div class="card" id="detail" style="margin-bottom:12px"><h2>节点详情</h2><div class="sub">点左边任意阶段节点。</div></div>
    <div class="card" style="margin-bottom:12px">
      <h2>★ 主图：容差 vs tp（官方 9 张）</h2>
      <svg id="curve" width="100%" height="230" viewBox="0 0 520 230"></svg>
      <div class="sub" id="curvetip"></div>
    </div>
    <div class="card" style="margin-bottom:12px" id="gatecard"></div>
    <div class="card" id="emptycard"></div>
    <div class="card" id="doc" style="margin-top:12px">
      <h2>分阶段说明（可直接嵌进文档的正文）</h2>
      <div class="sub">每个阶段：做什么 · 代码落点 · 关键参数与默认值 · 中间产物 · 实测吞吐 · 已知失效模式 + 数字</div>
      <div id="docbody"></div>
      <details style="margin-top:12px"><summary style="cursor:pointer;color:#5cc8ff">页面生成 / 新增独立页面的最小做法（第 5 项）</summary>
        <div id="recipebody"></div></details>
    </div>
  </div>
</div>
<script type="application/json" id="graphdata">__GRAPH__</script>
<script type="application/json" id="docmd">__DOC__</script>
<script type="application/json" id="recipemd">__RECIPE__</script>
<script>
const G = JSON.parse(document.getElementById('graphdata').textContent);
const X = __EXTRA__;
const cv = document.getElementById('cv'), svg = document.getElementById('links'), wrap = document.getElementById('cvwrap');
const NW = 250, NH = 96, COL = 330, ROW = 132;
const stageNodes = G.nodes.filter(n => n.group !== '评测标尺（不自算）');
const byId = {}; G.nodes.forEach(n => byId[n.id] = n);
// 简单分层布局：按链接做最长路径分层，同层纵向排列
const ids = stageNodes.map(n => n.id);
const adj = {}; ids.forEach(i => adj[i] = []);
G.links.forEach(l => { const a = l.from.split('.')[0], b = l.to.split('.')[0];
  if (a !== b && adj[a] && adj[b]) adj[a].push(b); });
const dep = {}; ids.forEach(i => dep[i] = 0);
for (let k = 0; k < ids.length + 2; k++)
  G.links.forEach(l => { const a = l.from.split('.')[0], b = l.to.split('.')[0];
    if (dep[a] !== undefined && dep[b] !== undefined && dep[b] < dep[a] + 1) dep[b] = dep[a] + 1; });
const cols = {}; ids.forEach(i => { (cols[dep[i]] = cols[dep[i]] || []).push(i); });
const pos = {};
Object.keys(cols).sort((a,b)=>a-b).forEach(c => { const col = cols[c];
  const h = col.length * ROW; const y0 = Math.max(20, (700 - h) / 2);
  col.forEach((id, k) => { pos[id] = {x: 30 + c * COL, y: y0 + k * ROW}; }); });
stageNodes.forEach(n => {
  const p = pos[n.id]; n._x = p.x; n._y = p.y;
  const el = document.createElement('div');
  el.className = 'node ' + (n.kind || 'exec'); el.style.left = p.x + 'px'; el.style.top = p.y + 'px';
  el.innerHTML = '<div class="t">' + n.label + '</div><div class="s">' + (n.sub || '') + '</div>' +
                 '<div class="g">' + (n.group || '') + '</div>';
  el.onclick = e => { e.stopPropagation(); document.querySelectorAll('.node.sel').forEach(x=>x.classList.remove('sel'));
    el.classList.add('sel'); show(n); };
  cv.appendChild(el);
});
function anchor(id, side) { const n = byId[id]; if (!n || !n._x) return null;
  return {x: n._x + (side === 'r' ? NW : 0), y: n._y + NH / 2}; }
let svgBuf = '';
stageNodes.forEach(n => { const el = document.querySelectorAll('.node');
  const p = pos[n.id]; });
// 用内嵌 JSON 的链接画线（节点位置已知）
G.links.forEach(l => {
  const a = l.from.split('.')[0], b = l.to.split('.')[0];
  if (!byId[a] || !byId[b] || byId[a]._x === undefined || byId[b]._x === undefined) return;
  if (byId[a].group === '评测标尺（不自算）' || byId[b].group === '评测标尺（不自算）') return;
  const x1 = byId[a]._x + NW, y1 = byId[a]._y + NH / 2, x2 = byId[b]._x, y2 = byId[b]._y + NH / 2;
  const mx = (x1 + x2) / 2;
  svgBuf += '<path d="M' + x1 + ' ' + y1 + ' C' + mx + ' ' + y1 + ' ' + mx + ' ' + y2 + ' ' + x2 + ' ' + y2 +
            '" fill="none" stroke="#3c4859" stroke-width="2" marker-end="url(#ar)"/>';
  svgBuf += '<circle cx="' + x1 + '" cy="' + y1 + '" r="3" fill="#5cc8ff"/>';
});
svg.innerHTML = '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">' +
  '<path d="M0,0 L0,6 L7,3 z" fill="#3c4859"/></marker></defs>' + svgBuf;
let view = {k: .62, x: 0, y: 0}, drag = null;
function apply() { cv.style.transform = 'translate(' + view.x + 'px,' + view.y + 'px) scale(' + view.k + ')';
  svg.style.transform = cv.style.transform; }
wrap.addEventListener('mousedown', e => { if (e.target.closest('.node')) return; drag = {x: e.clientX - view.x, y: e.clientY - view.y}; wrap.style.cursor='grabbing'; });
window.addEventListener('mousemove', e => { if (!drag) return; view.x = e.clientX - drag.x; view.y = e.clientY - drag.y; apply(); });
window.addEventListener('mouseup', () => { drag = null; wrap.style.cursor='grab'; });
wrap.addEventListener('wheel', e => { e.preventDefault(); const s = e.deltaY < 0 ? 1.12 : .89;
  view.k *= s; apply(); }, {passive: false});
function fit() { const w = wrap.clientWidth; let maxx = 0; stageNodes.forEach(n => maxx = Math.max(maxx, n._x + NW));
  view.k = Math.min(.85, (w - 40) / maxx); view.x = 12; view.y = 12; apply(); }
apply();
function show(n) {
  let h = '<h2>' + n.label + '</h2><div class="sub">' + (n.sub || '') + '</div>';
  if (n.code) h += '<p><b>代码落点</b><br><span class="mono">' + n.code + '</span></p>';
  if (n.params && n.params.length) { h += '<p><b>关键参数与默认值</b></p><div class="kv">' +
    n.params.map(p => '<b>' + p[0] + '</b><span>' + p[1] + '</span>').join('') + '</div>'; }
  if (n.note) h += '<p><b>说明</b><br>' + n.note + '</p>';
  if (n.bullets && n.bullets.length) h += '<p><b>实测 / 教训</b></p><ul>' +
    n.bullets.map(b => '<li>' + b + '</li>').join('') + '</ul>';
  document.getElementById('detail').innerHTML = h;
}
// ── 主图：容差 vs tp
(function () {
  const c = X.curves, W = 520, H = 230, L = 46, R = 16, T = 16, B = 34;
  const xs = i => L + (i / (c.tol.length - 1)) * (W - L - R);
  const ymax = 900, ys = v => H - B - (v / ymax) * (H - T - B);
  let s = '<rect x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="' + (H - T - B) +
          '" fill="#0e1219" stroke="#2c3444"/>';
  for (let g = 0; g <= 800; g += 200) s += '<line x1="' + L + '" y1="' + ys(g) + '" x2="' + (W - R) +
    '" y2="' + ys(g) + '" stroke="#242c3a"/><text x="' + (L - 6) + '" y="' + (ys(g) + 4) +
    '" fill="#8b98ad" font-size="10" text-anchor="end">' + g + '</text>';
  const path = arr => c.tol.map((t, i) => (i ? 'L' : 'M') + xs(i) + ' ' + ys(arr[i])).join(' ');
  s += '<path d="' + path(c.tp) + '" fill="none" stroke="#5ad07a" stroke-width="2.5"/>';
  s += '<path d="' + path(c.fp) + '" fill="none" stroke="#ff6b6b" stroke-width="1.6" stroke-dasharray="4 3"/>';
  s += '<path d="' + path(c.fn) + '" fill="none" stroke="#f0b429" stroke-width="1.6" stroke-dasharray="4 3"/>';
  c.tol.forEach((t, i) => {
    s += '<circle cx="' + xs(i) + '" cy="' + ys(c.tp[i]) + '" r="4" fill="#5ad07a"/>';
    s += '<text x="' + xs(i) + '" y="' + (ys(c.tp[i]) - 9) + '" fill="#5ad07a" font-size="11" text-anchor="middle">' + c.tp[i] + '</text>';
    s += '<text x="' + xs(i) + '" y="' + (H - 12) + '" fill="#dfe6f2" font-size="11" text-anchor="middle">' + t + '</text>';
    s += '<text x="' + xs(i) + '" y="' + (H - 1) + '" fill="#8b98ad" font-size="9" text-anchor="middle">' +
         (c.tp[i] / (c.tp[i] + c.fn[i]) * 100).toFixed(0) + '%</text>';
  });
  s += '<line x1="' + xs(0) + '" y1="' + T + '" x2="' + xs(0) + '" y2="' + (H - B) +
       '" stroke="#5cc8ff" stroke-width="1" stroke-dasharray="3 3"/>';
  s += '<text x="' + (L + 4) + '" y="' + (T + 12) + '" fill="#8b98ad" font-size="10">绿=tp 红=fp 黄=fn（虚线）｜横轴 line_tol（官方默认 5）</text>';
  document.getElementById('curve').innerHTML = s;
  document.getElementById('curvetip').innerHTML =
    '<b>骨架在 20px 内找对了 ' + c.cover.tol20 + '% 的边（tol5 只有 ' + c.cover.tol5 +
    '%），但官方 <code>line_tol=5.0</code> 且只看两个端点坐标 ⇒ fn 405 里 <b class="hl">256 条 (63.2%)</b> 是几何误差。</b><br>' +
    'tol 5→10 回收 91 条（tp 148→239），tol 20 再回收 77 条（→316），<b class="hl">之后 207 条任何容差都救不回</b>（结构性）。<br>' +
    '<span class="mono">' + c.note + '</span>';
})();
// ── 网级配对口径
(function () {
  const g = X.gate;
  document.getElementById('gatecard').innerHTML =
    '<h2>★★ 最大的那一环：网级配对门禁</h2>' +
    '<table><tr><th>口径</th><th>tp</th><th>fp</th><th>fn</th><th>F1</th></tr>' +
    '<tr><td>现口径（先配网、网内再配边）</td><td>' + g.with_net.tp + '</td><td>' + g.with_net.fp +
    '</td><td>' + g.with_net.fn + '</td><td>' + g.with_net.f1 + '</td></tr>' +
    '<tr><td><b>取消网级门禁（全局边匈牙利）</b></td><td><b>' + g.no_net.tp + '</b></td><td>' + g.no_net.fp +
    '</td><td>' + g.no_net.fn + '</td><td><b>' + g.no_net.f1 + '</b></td></tr></table>' +
    '<div class="sub">' + g.note + '</div>';
})();
// ── 空网抢配对（单独一格）
(function () {
  const e = X.empty;
  const rows = Object.entries(X.six).map(([k, v]) =>
    '<tr><td>' + k + '</td><td>' + v.edges + '</td><td>' + v.verdict + '</td></tr>').join('');
  document.getElementById('emptycard').innerHTML =
    '<h2>⚠ 空网抢配对（口径问题，不是几何）</h2>' +
    '<p><code>metrics.py:477-494 match_nets</code> 只看引脚集合交叠：<b>单脚网只要有一个引脚撞上，就能把 GT 有边的整个网抢走</b>，' +
    '那些 GT 边于是全记 fn，而我们的有边网去配别的 GT 网、边又全记 fp。</p>' +
    '<div class="kv"><b>case 1</b><span>' + e.case1_net + ' / ' + e.case1_pred_net +
    ' 个预测网是 edges={}（净_4=(C1518.1)、net_6=(C152.1)、net_7=(C1515.1)…）</span>' +
    '<b>官方 9 张</b><span>145 个预测网里有 ' + e.gt9_empty_net + ' 个空边网</span>' +
    '<b>删掉空网</b><span>tp ' + e.tp_before + '→' + e.tp_after + '｜fp ' + e.fp_before + '→' + e.fp_after +
    '｜fn ' + e.fn_before + '→' + e.fn_after + '（都变好）</span>' +
    '<b>但 mean score</b><span>' + e.score_before + '→' + e.score_after + '（<b class="hl">−0.45</b>：case 4 −2.11 / 9 −3.30 / 5 −0.58 掉分，1 +0.20 / 7 +1.72 / 10 +0.79 涨分）</span></div>' +
    '<div class="sub">' + e.note + '</div>' +
    '<h2 style="margin-top:14px">派生 19 张里 6 张导线 F1=0 的真因</h2>' +
    '<table><tr><th>case</th><th>输出边</th><th>定性</th></tr>' + rows + '</table>';
})();
show(byId['skel'] || stageNodes[0]);
// ── 极简 Markdown 渲染（只吃本页用到的那几种语法）
function md2html(src) {
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const inline = s => esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/(^|[\s(])\*([^*]+)\*/g, '$1<i>$2</i>');
  const lines = src.split(String.fromCharCode(10));
  let out = [], i = 0, tbl = null;
  const flushTbl = () => { if (!tbl) return;
    out.push('<table>' + tbl.map((r, k) => '<tr>' + r.map(c =>
      (k === 0 ? '<th>' : '<td>') + inline(c) + (k === 0 ? '</th>' : '</td>')).join('') + '</tr>').join('') + '</table>');
    tbl = null; };
  while (i < lines.length) {
    const L = lines[i];
    if (/^\s*```/.test(L)) { i++; const buf = [];
      while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
      i++; flushTbl(); out.push('<pre class="mono" style="background:#0e1219;border:1px solid #2c3444;border-radius:6px;padding:8px;overflow:auto">' + esc(buf.join(String.fromCharCode(10))) + '</pre>'); continue; }
    if (/^\s*\|/.test(L) && /\|\s*$/.test(L)) {
      const cells = L.trim().replace(/^\||\|$/g, '').split('|').map(s => s.trim());
      if (/^[-:\s|]+$/.test(cells.join('|'))) { i++; continue; }
      (tbl = tbl || []).push(cells); i++; continue; }
    flushTbl();
    let m;
    if ((m = L.match(/^(#{1,4})\s+(.*)$/))) out.push('<h' + m[1].length + '>' + inline(m[2]) + '</h' + m[1].length + '>');
    else if (/^\s*>\s?/.test(L)) out.push('<blockquote style="border-left:3px solid #f0b429;margin:6px 0;padding:2px 10px;color:#b9c6d8">' + inline(L.replace(/^\s*>\s?/, '')) + '</blockquote>');
    else if (/^\s*[-*]\s+/.test(L)) { const buf = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) buf.push('<li>' + inline(lines[i++].replace(/^\s*[-*]\s+/, '')) + '</li>');
      out.push('<ul>' + buf.join('') + '</ul>'); continue; }
    else if (/^\s*$/.test(L)) out.push('');
    else out.push('<p>' + inline(L) + '</p>');
    i++;
  }
  flushTbl();
  return out.join(String.fromCharCode(10));
}
document.getElementById('docbody').innerHTML =
  md2html(JSON.parse(document.getElementById('docmd').textContent));
document.getElementById('recipebody').innerHTML =
  md2html(JSON.parse(document.getElementById('recipemd').textContent));
</script></body></html>
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    g = json.loads(DATA.read_text(encoding="utf-8"))
    if "--no-inspector" in sys.argv:
        g["nodes"] = [n for n in g["nodes"] if n["id"] != "s1json"]
    blob = json.dumps(g, ensure_ascii=False).replace("</", "<\\/")
    doc = json.dumps(DOC.read_text(encoding="utf-8") if DOC.exists() else "",
                     ensure_ascii=False).replace("</", "<\\/")
    rec = json.dumps(RECIPE.read_text(encoding="utf-8") if RECIPE.exists() else "",
                     ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.replace("__GRAPH__", blob).replace("__DOC__", doc).replace(
        "__RECIPE__", rec).replace("__EXTRA__", json.dumps(EXTRA, ensure_ascii=False))
    # 注意替换顺序：模板里 __DOC__/__RECIPE__ 必须在 __EXTRA__ 之前落位，
    # 否则内嵌正文里出现的占位符字样会被后一步替换污染（recipe 文档里踩过一次）。
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("写出 %s（%d 字节）" % (OUT, OUT.stat().st_size))
    if "--also-docs" in sys.argv or "--write-docs" in sys.argv:
        DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
        DOCS_OUT.write_text(html, encoding="utf-8")
        print("写出 %s（%d 字节）" % (DOCS_OUT, DOCS_OUT.stat().st_size))
    else:
        print("（未写 eda-pcb\\docs\\；父代理批准后加 --write-docs）")
    # 自检：JSON 能解回来、关键数字在位
    assert 'id="graphdata"' in html and "__EXTRA__" not in html and "const X = {" in html
    assert "__GRAPH__" + "</script>" not in html and "__DOC__" + "</script>" not in html
    assert "0.2426" in html and "0.1876" in html
    print("自检：graphdata JSON 已内嵌、主图数字在位 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
