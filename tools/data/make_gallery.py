"""把这次排查过程中产生的所有问题图整理成一个自包含的 HTML 图集。

做两件事：
  1. 重新渲染「导线提取 修前 / 修后」的对照图（0101、0107），用 stage1_all6（pskel 方向
     bug 修复前）和 stage1_all9（修复 + 文字遮挡后）两份产物，避免用过期图。
  2. 把所有 preview 图按「问题」分节，缩放到宽 <=1400 后 base64 内嵌进一个 HTML，
     这样单文件可离线打开，不依赖相对路径。

用法: python work\\make_gallery.py
"""
import base64
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from PIL import Image  # noqa: E402

PREV = r"Y:\workspace\EDA\work\preview"
WORK = r"Y:\workspace\EDA\work"
DOCS = r"Y:\workspace\EDA\docs"
OUT_HTML = os.path.join(DOCS, "问题图片集.html")
MAX_W = 1400
PNG_BUDGET = 340_000


# ------------------------------------------------------------------ 重渲染修前/修后
def render_pairs():
    """渲染导线提取的三个阶段，供「修前/修后」对照。

    stage1_all2  = 最早的全量跑（遮挡/过滤都还没做，pskel 方向 bug 也还在）
    stage1_all6  = 遮挡元件 + 去图纸边框 + 轴向过滤 + 大尺度连通过滤之后（方向 bug 仍在）
    stage1_all9  = 修掉 pskel 方向 bug + 涂掉文字 + 文字豁免之后
    """
    import _wire_look as WL
    from schematic_parser.data import discover, DEFAULT_DATA_ROOT
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    for cid in ("0101", "0107"):
        for out_dir, tag in (("stage1_all2", "orig"),
                             ("stage1_all6", "before"),
                             ("stage1_all9", "after")):
            WL.OUT = os.path.join(WORK, out_dir)
            WL.render(cid, cases, f"{cid}_{tag}")


# ------------------------------------------------------------------ 编码
def embed(path):
    if not os.path.exists(path):
        return None
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:                                     # noqa: BLE001
        print(f"  ! 打不开 {path}: {e}")
        return None
    w, h = im.size
    if w > MAX_W:
        im = im.resize((MAX_W, max(1, int(h * MAX_W / w))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    mime = "image/png"
    data = buf.getvalue()
    if len(data) > PNG_BUDGET:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=86, optimize=True)
        mime, data = "image/jpeg", buf.getvalue()
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii"), im.size, len(data)


# ------------------------------------------------------------------ 图集内容
def sections():
    P = PREV
    O = os.path.join(PREV, "official")
    W = WORK
    return [
        ("坐标系：JSON 里的 y 到底是朝上还是朝下",
         "这是整道题的地基，错了满盘皆输。两条独立证据：<br>"
         "① <b>统计证据</b>（<code>work\\axis_profile.py</code>）：对 220 个 case 比较「图像行/列墨迹剖面」"
         "与「component 密度剖面」的相关度 —— 假设 <code>y</code> 朝上 <b>mean 0.2102 / median 0.1895</b>，"
         "假设 <code>y</code> 与图像同向 <b>mean 0.1363 / median 0.1167</b>。<br>"
         "② <b>视觉证据（决定性）</b>：拿 <code>0001-KiCad</code> 的 <code>U1</code> bbox "
         "<code>[360,248,527,915]</code>，分别按两种假设裁同一块区域。",
         [(os.path.join(W, "crop_U1_up.png"),
           "<b>按 y 向上裁（y_json = H − y_img）</b>：画面正是 RP2350B 本体，看得见 IOVDD / "
           "QSPI_IOVDD / GPIO0..GPIO47 引脚名。<b>这就是决定性证据。</b>"),
          (os.path.join(W, "crop_U1_down.png"),
           "按 y 向下裁（与图像同原点）：不是器件本体。"),
          (os.path.join(P, "z_rp2350.png"), "同类裁剪抽查：RP2350"),
          (os.path.join(P, "z_ne555.png"), "同类裁剪抽查：NE555"),
          (os.path.join(P, "z_left.png"), "同类裁剪抽查：图纸左侧"),
          (os.path.join(P, "gt_zoom.png"), "GT 框放大对照")]),

        ("合成训练集保真度：一张图里 43 个元件有 24 个画错了位置/尺寸/朝向",
         "症状是符号重叠、图形扭曲。根因在 <code>eda-pcb/synth/scene.py</code> 的 "
         "<code>_measure</code>：它用 <code>rng.getstate()/setstate()</code> 做<i>预演</i>来量尺寸，"
         "但真实绘制走的是另一条随机流 ⇒ 量到的和画出来的不是一回事。"
         "改成每个元件一条独立流 <code>_comp_rng(seed, ctype, key)</code> 后 <b>0/43 不一致</b>。",
         [(os.path.join(P, "scene1.png"), "修复前的合成场景"),
          (os.path.join(P, "symtest.png"), "符号绘制自检"),
          (os.path.join(P, "symtest_zoom.png"), "自检放大：可见符号压在一起"),
          (os.path.join(P, "fixed1.png"), "<b>修复后</b>"),
          (os.path.join(P, "fixed2.png"), "<b>修复后</b>"),
          (os.path.join(P, "demo2", "montage.png"), "修复后批量采样")]),

        ("训练集标注核验（自建 763 张 PCB 训练集 / 训练集_v2 / v3）",
         "把每张图的标注框原样画回去，逐张确认无错位。",
         [(os.path.join(P, "label_contact_sheet.png"), "自建训练集 763 张的框叠加总表（24503 个框）"),
          (os.path.join(P, "check_10_GTcase-0059-Datasheet__full.png"), "抽查：0059-Datasheet（13 框全贴合）"),
          (os.path.join(P, "check_10GT-3__t00_00.png"), "抽查：10GT-3 切片（12 框全贴合）")]),

        ("GT 框比符号本体小：余量到底有多少",
         "对 220 个 case 量「框内墨迹占比」与「最大的墨迹连通域是否碰到框边界」。",
         [(os.path.join(P, "margin_gnd.png"), "gnd：ring_frac p50 0.0817，<b>58.7%</b> 的图里最大墨迹块碰到了框边界"),
          (os.path.join(P, "margin_v.png"), "v：0.1172 / <b>98.6%</b>"),
          (os.path.join(P, "margin_r.png"), "r：0.2281 / 92.8%"),
          (os.path.join(P, "margin_c.png"), "c：0.1111 / 99.6%"),
          (os.path.join(P, "fixbox_r.png"),
           "<b>形态学「把框长大」这条路的失败现场</b>：绿色是长大后的框，"
           "它把本体 + 引脚 + 导线一起吞了进去 ⇒ 形态学分不开「符号本体」和「引线」。")]),

        ("GND 不是一种形状：同一个 key，多种画法",
         "用户最初的判断（「GND 不是一个形状，是好几种」）被实测证实。"
         "2229 个 gnd 实例做 10 个 k-means 形状模式，<b>每个簇都混合了所有来源</b>"
         "（比如最大的簇里 jlc 315 / Datasheet 76 / Altium 32 / KiCad 6）"
         "⇒ 正确的切分轴是<b>形状</b>，不是厂家。",
         [(os.path.join(P, "gnd_forms.png"), "GT 里 gnd 裁片的各种形态"),
          (os.path.join(P, "gnd_glyphs.png"), "gnd 字形"),
          (os.path.join(P, "gnd_modes.png"), "10 个 k-means 形状模式（k=10 只解释了 35.6% 的形状方差）"),
          (os.path.join(P, "gnd_modes_margin.png"), "换裁剪余量后重聚：惯性 674.0 → 659.2（只差 2%）"),
          (os.path.join(P, "gnd_modes_tight.png"), "紧裁剪：簇内平均距离 25.22 → 25.11 ⇒ 模式不是裁剪假象"),
          (os.path.join(P, "gnd_grammar.png"),
           "<b>纯 CV 横线叠文法的检测叠加</b>：红框确实落在正确符号上（检测能力存在），"
           "但 183 张整图上 TP 1360 / FP 1365，精度只有 0.499。<br>"
           "根因是结构性的：最大的那个模式（439 个实例，占 19.7%）<b>字面上就是三条横杠</b>，"
           "它和任何一根水平导线都长得一样。")]),

        ("testpoint：一个 key 三种画法（CV 检测器的全部难点所在）",
         "官方 220 张里一共只有 <b>43 个</b> testpoint，集中在 6 张图上，全部是 20×20 图章。"
         "但它们的画法完全不同，单一特征覆盖不了。",
         [(os.path.join(P, "gt_0050_testpoint.png"),
           "<b>画法一：三层细圆环</b>（0050，20 个实例，4500×2850）。"
           "r_min≈14.1 / r_max≈20.1 / n_rings=3 ⇒ 可检测（TP 19/20）。"),
          (os.path.join(P, "gt_0011_testpoint.png"),
           "<b>画法二：粗圆环，且和导线粘连</b>（0011，12 个）。"
           "外轮廓因为连着导线不再是圆，只剩内孔是干净圆 ⇒ n_rings 只有 2，"
           "三层环判据打不着；要靠「粗圆环 + 孤立性」的环带门控救回来。"),
          (os.path.join(P, "gt_0102_testpoint.png"),
           "<b>画法三：成排的小圆点</b>（0102，8 个），周围是 IO8/IO12… 文字。"
           "门限只要放到能吃到 1 个真例，误报就是 <b>8497 个（38.6 张/图）</b> ⇒ "
           "圆点与字母碎片、连接器圆点、过孔同构，这条路判定为不可用，<code>ENABLE_DOT=False</code>。"),
          (os.path.join(P, "gt_0012_testpoint.png"), "孤例：0012（1 个）"),
          (os.path.join(P, "gt_0056_testpoint.png"), "孤例：0056（1 个）"),
          (os.path.join(P, "gt_0131_testpoint.png"), "孤例：0131（1 个）"),
          (os.path.join(P, "peek_testpoint.png"), "早期核验：GT 框确实压在真实的测试焊盘上，不是标注错")]),

        ("异常标注：GT 本身就有问题",
         "不是所有漏检都是模型的错。",
         [(os.path.join(P, "gt_0027_net_input.png"),
           "<code>0027-Datasheet</code> 的 <code>∅287</code>（type <code>net_input</code>）"
           "bbox 在图像坐标里是 <code>(436, −25, 446, 5)</code> —— <b>整个框在图纸外面</b>，"
           "框内墨迹占比 <b>0.0000</b>。这个元件不可能被检出，因为纸上没有它。"),
          (os.path.join(P, "gt_0050_r.png"),
           "<code>0050</code>（4500×2850）的 <code>r</code> 图章恒为 8×20px；"
           "整图缩到 imgsz=1024 时它只剩 <b>1.8×4.6 px</b>，比 YOLO 最小步长 P3=8 还小。"
           "光这一张图就贡献了 89 个 r 漏检里的 28 个。")]),

        ("官方口径（±20px）下的漏检 / 误检接触表",
         "判据换成赛题真正的口径：component 先按 key 匹配，再算四角逐项坐标差 ≤20px。"
         "<b>换了判据之后，gnd 的漏检从 100 掉到 3、r 从 89 掉到 10</b> —— "
         "框从来不是问题，判据才是。",
         [(os.path.join(O, "fnfp_all.png"), "14 格总表：上排漏检（FN），下排误检（FP）"),
          (os.path.join(O, "fp_gnd.png"),
           "<b>gnd 误检的典型</b>：大图被缩到 imgsz 1024 后，字母/数字/网格线糊成粗黑条，"
           "模型以 conf 0.72~0.78 开火；也有一批打在字面的 GND 文字上。"
           "（图中多数格子的「最近 GT 差 1e12px」是绘图脚本的已知小 bug，不影响统计。）"),
          (os.path.join(O, "fn_testpoint.png"), "testpoint 漏检 93/94（98.9%）—— 全库最惨的一类"),
          (os.path.join(O, "fn_v.png"), "v 漏检 30/369（8.1%）"),
          (os.path.join(O, "fp_v.png"),
           "v 误检 104 个 —— <b>testpoint 连一个误检都没有，所以官方目录里没有 fp_testpoint.png</b>"),
          (os.path.join(O, "fn_r.png"), "r 漏检 10/402（2.5%）"),
          (os.path.join(O, "fp_r.png"), "r 误检 74 个 —— 目视基本都压在真的电阻上"),
          (os.path.join(O, "fn_net_input.png"), "net_input 漏检 13/44（29.5%）"),
          (os.path.join(O, "fp_net_input.png"), "net_input 误检 17 个"),
          (os.path.join(P, "fnfp_all.png"),
           "<b>对照：旧判据（IoU≥0.5）的同一批图</b>。同一批数据、同一个模型，"
           "gnd 漏检 100、r 89、c 57、v 110 —— 换判据后分别变成 3、10、7、30。")]),

        ("导线提取：从「把每个字都提出来」到「只提走线」",
         "这是第一阶段最弱的一环。GT 的 <code>net_line</code> 是<b>真正画出来的走线</b>"
         "（1px 线宽栅格化后 94.1% 压在墨上，平移 (0,0) 就是最优）—— "
         "所以低分不是尺子坏了。难点在于：文字笔画和导线一样是 1px 细线。",
         [(os.path.join(P, "wire_0101_orig.png"),
           "<b>第 0 阶段 · 什么都还没做（0101，提取 1346 段）</b>：绿色覆盖了 IC 矩形边框、"
           "电容极板、每一个引脚名、参数值、标题栏文字和图纸边框 —— <b>它提的是「所有墨迹」，"
           "不是导线</b>。右侧红色才是 GT。"),
          (os.path.join(P, "wire_0101_before.png"),
           "<b>第 1 阶段 · 遮挡元件 + 去图纸边框 + 轴向过滤 + 大尺度连通过滤（289 段）</b>："
           "已经干净很多，但此时 <code>pskel.py:469</code> 的方向 bug 还在，"
           "结点处会瞬移出凭空多出的长线（轴对齐的变成图纸边框假线，斜的被 45° 吸附成假斜线）。"),
          (os.path.join(P, "wire_0101_after.png"),
           "<b>第 2 阶段 · 修掉方向 bug + 涂掉文字 + 文字豁免（148 段）</b>：文字不再被覆盖，"
           "假斜线归零。剩下的绿色主要是真走线，另有标题栏表格线与图纸边框（见第 8 节的归因）。"),
          (os.path.join(P, "wire_0107_orig.png"), "0107 第 0 阶段"),
          (os.path.join(P, "wire_0107_before.png"),
           "0107 第 1 阶段：<b>仍有 17 条 45° 假斜线</b>（最长 526px）"),
          (os.path.join(P, "wire_0107_after.png"), "0107 第 2 阶段：假斜线 <b>0 条</b>"),
          (os.path.join(P, "diag_0101_0.png"), "大尺度斜线的诊断叠图（1）"),
          (os.path.join(P, "diag_0101_1.png"), "大尺度斜线的诊断叠图（2）"),
          (os.path.join(P, "wire_kicad.png"), "KiCad 0001：引脚桩贴着 IC 矩形，框一涂掉就断了"),
          (os.path.join(P, "wire_altium.png"), "Altium 0050"),
          (os.path.join(P, "wire_good.png"), "表现最好的一张（0052）")]),

        ("文字定位（第一阶段的新交付物，纯 CV、不训练）",
         "关键判断：文字笔画和导线一样是 1px 细线，连通域形状分不开 —— "
         "必须先把「长直线」从墨迹里减掉，再做字形连通域 → 合并成文字行。"
         "自检用官方 GT 本身：每个 pin 都有 <code>pinname</code> 与绘制点，"
         "引脚名必然画在附近，于是「引脚名覆盖率」是官方数据上的召回下界。"
         "<b>@12px 0.8832 / @20px 0.9864 / @30px 1.0000</b>，平均 567 框/张，412 ms/张。",
         [(os.path.join(P, "textbox_0001.png"),
           "0001 KiCad：位号、参数、引脚名、网络标签、标题文字、标题栏、图框编号全部框住"),
          (os.path.join(P, "textbox_0101.png"), "0101 jlc"),
          (os.path.join(P, "textbox_0050.png"), "0050 Altium")]),

        ("stage1 端到端：元件 + 导线 + 附着 的叠加核验",
         "绿=YOLO 框、品红=CV 专类检测器、橙=待裁决、蓝=导线线段、黄点=附着点。",
         [(os.path.join(P, "stage1_0001.png"), "0001 KiCad"),
          (os.path.join(P, "stage1_0050.png"), "0050 Altium（20 个 testpoint 被 CV 全部抓到）"),
          (os.path.join(P, "stage1_0011.png"), "0011 Altium"),
          (os.path.join(P, "stage1_0027.png"), "0027 Datasheet（含那个图纸外的 ∅287）"),
          (os.path.join(P, "stage1_sheet.png"), "四张拼在一起的总表"),
          (os.path.join(P, "mindmap5.png"), "项目思路导图的另一版渲染")]),
    ]


def main():
    print("=== 1/2 重渲染导线 修前/修后 对照 ===")
    render_pairs()

    print("\n=== 2/2 生成 HTML 图集 ===")
    secs = sections()
    parts = ["""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>EDA 赛题六 · 问题图片集</title><style>
:root{--bg:#14171c;--card:#1c2027;--fg:#e6e9ee;--dim:#9aa4b2;--acc:#6fb1ff;--warn:#ffb454}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.7 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
header{padding:34px 40px 18px;border-bottom:1px solid #2a2f38;position:sticky;top:0;
 background:linear-gradient(180deg,#14171c 70%,rgba(20,23,28,.86));backdrop-filter:blur(6px);z-index:9}
h1{margin:0 0 6px;font-size:23px}
header p{margin:0;color:var(--dim);font-size:13.5px}
nav{padding:14px 40px 4px;display:flex;flex-wrap:wrap;gap:8px}
nav a{color:var(--acc);text-decoration:none;font-size:13px;
 border:1px solid #2f3540;border-radius:99px;padding:3px 11px}
nav a:hover{background:#222833}
main{padding:8px 40px 80px;max-width:1560px}
section{margin:44px 0 0;padding-top:22px;border-top:1px solid #262b34}
h2{font-size:19px;margin:0 0 8px;color:var(--warn)}
.lead{color:var(--dim);font-size:14px;margin:0 0 20px;max-width:1100px}
.lead b{color:var(--fg)}
figure{margin:0 0 26px;background:var(--card);border:1px solid #262b34;border-radius:10px;
 overflow:hidden}
figure img{display:block;width:100%;height:auto;background:#0d0f12}
figcaption{padding:11px 15px;font-size:13.5px;color:var(--dim);border-top:1px solid #262b34}
figcaption b{color:var(--fg)}
figcaption .fn{display:block;color:#6b7686;font-size:11.5px;margin-top:4px;
 font-family:ui-monospace,Consolas,monospace;word-break:break-all}
code{background:#222833;padding:1px 5px;border-radius:4px;font-size:12.5px;color:#9fd0ff}
</style></head><body>
<header><h1>EDA 赛题六 · 本次排查的问题图片集</h1>
<p>每一节都是「一个问题 → 现象 → 结论」。图片按问题分节，可离线单文件打开。</p></header>
<nav>"""]

    for i, (title, _, pics) in enumerate(secs, 1):
        parts.append(f'<a href="#s{i}">{i}. {title.split("：")[0][:20]}</a>')
    parts.append("</nav><main>")

    n_img = n_skip = 0
    for i, (title, lead, pics) in enumerate(secs, 1):
        parts.append(f'<section id="s{i}"><h2>{i}. {title}</h2><p class="lead">{lead}</p>')
        for path, cap in pics:
            r = embed(path)
            if r is None:
                n_skip += 1
                print(f"  ! 跳过（文件不存在或打不开）: {path}")
                continue
            data, size, nbytes = r
            n_img += 1
            rel = os.path.relpath(path, DOCS)
            parts.append(f'<figure><img src="{data}" alt="{os.path.basename(path)}">'
                         f'<figcaption>{cap}<span class="fn">{rel} · {size[0]}×{size[1]} · '
                         f'{nbytes//1024} KB</span></figcaption></figure>')
        parts.append("</section>")

    parts.append("</main></body></html>")
    html = "".join(parts)
    os.makedirs(DOCS, exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  {n_img} 张图内嵌，跳过 {n_skip} 张")
    print(f"  -> {OUT_HTML}   {len(html.encode('utf-8'))/1048576:.1f} MB")


if __name__ == "__main__":
    main()
