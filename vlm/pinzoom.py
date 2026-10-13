"""引脚名「框选重读」通道（用户 m07576 的提议在引脚名上的落地）。

设计（全部读数都是官方 10 张 / 220 池实测，见 `docs/RESULTS.md` §5.82）：

1. **选槽位**：只看我们自己的引脚名形态，不看语料 ⇒ 出厂可实现。
   `len(norm(name)) == 1` 或 `near_core(norm(name))`（与核心表编辑距离 ≤1）算可疑。
   官方 10 张选中 ≈21 槽（5 个真错全在，多选只花调用；漏选才是损失）。
2. **读一次**：沿引脚轴线朝元件体内切一条带（`LEN`×`2*HW`）放大 `SCALE` 倍，
   红叉标出脚点，问模型「这串字是什么」。prompt 里写明**宁可弃权不许猜**；
   模型若回 `UNCERTAIN: x1,y1,x2,y2`（整图坐标）就把那个框再切一次重问（最多 1 轮）。
3. **二选一强迫选择（2AFC）**：把「我们的读」与「模型自己的读」做成菜单，
   A/B 顺序按槽位固定伪随机；**多槽位拼成一张竖排图**（每格左上角画编号，
   格间蓝线分隔）一次问完 —— 实测每格没有编号时批量会串格（3 格只对 1 格），
   画了编号后与单槽单独问**逐格一致（3/3）**，于是调用数从 1 次/槽降到 1/`BATCH`。
4. **采信（pickd，净 +0.64 分）**：只在「2AFC 选中的正是模型那次自由读的答案」
   且「该答案没有被同一元件的另一个引脚用掉」且「答案不比我们现有的短」时改写。
   - 逐条读数（官方 10，基线 80.98）：naive（有读就写）−0.37｜保守（≥3 字不许变长、
     距离 ≤2）＋0.42｜两次独立读一致 −0.02~＋0.02｜pickd **＋0.64**｜完美读名 +0.94。
   - 残差：5 个真错里采到 3 个（5.U3.1 / 6.U4.4 / 7.U1.4），另有 1 次误采
     （10.U1.4 `VIA→EN`，−0.16）。
5. **预算**：`token 项 = 0.05·(m/M)·Score_Token`（上限 5 分），单次调用 ≈673 输入 token。
   本通道 ≈27 次调用（≈+11% token）⇒ token 项最坏损失 ≤0.55 分，小于净收益。
   仍然给硬上限：`MAX_SLOTS` 槽位、`MAX_CALLS` 次调用、`client.budget_left()` 兜底。

旋钮：`FUSION_PINZOOM`（**默认 0 = 关**）、`FUSION_PINZOOM_MAX`（槽位上限，默认 40）、
`FUSION_PINZOOM_BATCH`（每张图问几格，默认 4）、`FUSION_PINZOOM_CALLS`（调用上限，默认 80）、
`FUSION_PINZOOM_RETRY`（弃权重试上限，默认 2）、`FUSION_PINZOOM_IC_ONLY`（默认 1）。

★★ 为什么默认关（端到端 A/B，官方 10 张，`work/_fast_out_pz{off,on}`，2026-10）：

    臂          总分     pin F1   tp
    pzoff       80.98    0.7731   96
    pzon        80.01    0.7340   93     ← **−0.97 分**

    打开后共 4 次改写，**四次全是坏的**：case 3 84.92→80.75（pin 1.0000→0.8333，掉 2 个 tp）、
    case 4 79.39→77.12、case 7 83.64→80.31。case 3/4 本来**一个引脚名都没错**（pin F1 1.0/0.82），
    所以「多改」纯粹是损失。

    离线（用 `work/_pinzoomrule.py` 的 2AFC 缓存 + 直接改 `pinname` 复算）同一条 pickd 规则
    是 **+0.64**（5 个真错采到 3 个、1 次误采）。差别在**裁剪几何与问法**：离线是
    沿轴 100px + 按引脚间距给的半宽 + 「A/B 菜单」单槽问；落地版是 90px + 固定 13px 半宽 +
    竖排批量问。⇒ 这条规则对几何/问法**极其敏感**，别把离线的 +0.64 当成可交付读数。

    残差（若还要继续做）：①落地版与离线版对齐（pitch 半宽、B 视角 +8px）；②候选闸门加上
    「候选不得是同一元件其它脚已用的名字」之外的**几何核对**（模型读到的字框必须压在
    本条引脚的行上）；③把 3/4 这类「我们本来就全对」的案子用更保守的触发器挡掉。
"""

from __future__ import annotations

import os
import re
import zlib

import cv2
import numpy as np

#: 无论如何都算「像名字」的核心表（与 work/_pinzoomsel.py 的 CORE 逐字一致）
CORE = """GND VCC VDD VSS VIN VOUT EN IN OUT ADJ FB PP TAB NC NRST RESET PROG SWDIO SWCLK
TXD RXD SDA SCL L N P A K C B E GNDA GNDD AVDD DVDD VDDIO VBAT VREF AGND EP GND1
VDD1 VDD2 VIN1 IN1 OUT1 VOUT1 SS TRIG THRESH DISCH CONT COMP CS SCK MISO MOSI CLK
DATA D0 D1 D2 D3 D4 D5 D6 D7 A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15
PA0 PA1 PA2 PA3 PA4 PA5 PA6 PA7 PA8 PA9 PA10 PA11 PA12 PA13 PA14 PA15 PB0 PB1 PB2
""".split()

LEN = 90.0        # 沿轴朝体内切多长（px）
HW = 13.0         # 半宽（px）
SCALE = 5         # 放大倍数（客户端 upsample）
BATCH = 4         # 一张图问几格
MAX_SLOTS = 40
MAX_CALLS = 80


def _env_on(key: str, default: str) -> bool:
    return os.environ.get(key, default).strip().lower() not in ("0", "off", "false", "no", "")


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(os.environ.get(key, "") or default))
    except Exception:                                       # noqa: BLE001
        return default


ON = _env_on("FUSION_PINZOOM", "0")   # ★ 默认 0：端到端 A/B 实测**负分**（见下）

PROMPT_FREE = """这是一个电路板原理图的局部放大图（放大 %d 倍，红叉标出了要读的引脚脚点）。
问题：红叉所在的那根引脚，它旁边画的**引脚名文字**是什么？

规则（很重要）：
- 只按画面上真正画出来的那串字符回答，不要补全、不要纠正拼写、不要换成更常见的名字。
- 如果文字被画面边缘截断、或者你无法确定 ⇒ **不要猜**，改回一行：
  UNCERTAIN: x1,y1,x2,y2     （这四个数是**整幅原图**坐标系里、你认为必须包含进去的那个矩形）
- 能确定就只回那串字符本身，不要加引号、不要加解释。
"""

PROMPT_PICK = """这是一张把 %d 个「引脚局部」从上到下拼起来的图（每格放大 %d 倍，每格左上角的红字是格号，格与格之间有蓝线分隔）。

下面是每格的题目（**不要复述这些题目**）：
%s

对每一格 i（i=1..%d，从上到下）：判断该格红叉所在引脚上**真正画着的**那串字是 A 还是 B。
- 只按画面判断；不要补全、不要纠正拼写、不要挑更常见的名字。
- 该格文字被截断或无法确定 ⇒ 该格回 X。
- **只输出下面这种行，不要输出任何别的文字**：
1=B
2=A
3=X
"""


# --------------------------------------------------------------------- 触发
def norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", (s or "")).upper()


def _edit_dist(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 3:
        return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def suspicious(name: str):
    """(是否可疑, 原因)。只依赖名字形态与硬编码核心表 ⇒ 出厂可实现。

    ★ 只收**纯字母**名：`1`/`2` 这类「名字就是引脚号」的槽位在原理图上到处都是
    （电阻/电容/电感的两个脚），它们既不是本通道的目标，也把调用量抬了 5 倍
    （第一版实测：3 张图 166 次调用、19 次乱采）。目标错形是**截断/混淆的字母串**。
    """
    n = norm(name)
    if not n:
        return False, "空名（另有 rescure 通道处理）"
    if not n.isalpha():
        return False, "含数字/符号"
    if len(n) == 1:
        return True, "单字符"
    best, bw = 9, ""
    for w in CORE:
        if w == n:
            return False, "等于核心表 %s" % w
        d = _edit_dist(n, w)
        if d < best:
            best, bw = d, w
    if best <= 1:
        return True, "与核心表 %s 距离 %d" % (bw, best)
    return False, "不像核心表"


#: 本通道只碰这些元件类型 —— 官方引脚通道（`metrics.pin_f1`）也只统计它们，
#: 别把调用花在电阻电容的编号上（第一版就是因此 3 张图烧了 166 次调用）。
IC_TYPES = frozenset({"box", "ic", "boxic"})


# --------------------------------------------------------------------- 裁图
def _axis(p, c):
    """从脚点 p 指向元件中心 c 的单位向量，吸附到轴对齐（原理图引脚都是正交的）。"""
    dx, dy = c[0] - p[0], c[1] - p[1]
    if abs(dx) >= abs(dy):
        return (1.0 if dx >= 0 else -1.0), 0.0
    return 0.0, (1.0 if dy >= 0 else -1.0)


def band_rect(p, c, length=LEN, hw=HW):
    """沿轴向体内的带在 json 坐标下的矩形 (x1,y1,x2,y2)。"""
    ux, uy = _axis(p, c)
    x1, x2 = p[0], p[0] + ux * length
    y1, y2 = p[1], p[1] + uy * length
    if ux:
        y1, y2 = p[1] - hw, p[1] + hw
    else:
        x1, x2 = p[0] - hw, p[0] + hw
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def crop_json(img, H, rect, scale=SCALE, mark=None, pad=6):
    """按 json 矩形裁图（y_img = H − y_json）并放大；mark=(x,y) 画红叉。"""
    x1, y1, x2, y2 = rect
    xi1 = int(max(0, x1 - pad))
    xi2 = int(min(img.shape[1], x2 + pad))
    yi1 = int(max(0, H - y2 - pad))
    yi2 = int(min(img.shape[0], H - y1 + pad))
    if xi2 - xi1 < 4 or yi2 - yi1 < 4:
        return None
    sub = img[yi1:yi2, xi1:xi2].copy()
    if mark is not None:
        mx, my = int(mark[0]) - xi1, int(H - mark[1]) - yi1
        cv2.drawMarker(sub, (mx, my), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
    if scale != 1:
        sub = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return sub


def montage(imgs, sep=6):
    """竖排拼接 + 每格左上角红字编号 + 蓝线分隔（编号是批量不串格的关键）。"""
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    ws = max(im.shape[1] for im in imgs)
    hs = [im.shape[0] for im in imgs]
    out = np.full((sum(hs) + sep * (len(imgs) - 1), ws, 3), 255, np.uint8)
    y = 0
    for i, im in enumerate(imgs, 1):
        out[y:y + im.shape[0], 0:im.shape[1]] = im
        cv2.putText(out, str(i), (6, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                    cv2.LINE_AA)
        y += im.shape[0]
        if y < out.shape[0]:
            out[y:y + sep] = (255, 0, 0)
            y += sep
    return out


# --------------------------------------------------------------------- 解析
_RE_UNC = re.compile(r"UNCERTAIN\s*[::]?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)",
                     re.I)


def parse_free(txt: str):
    """返回 ('box', rect) | ('name', str) | ('none', '')。"""
    if not txt:
        return "none", ""
    m = _RE_UNC.search(txt)
    if m:
        try:
            x1, y1, x2, y2 = [float(v) for v in m.groups()]
            return "box", (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        except Exception:                                   # noqa: BLE001
            return "none", ""
    t = txt.strip().splitlines()[0].strip().strip("`'\"。 .")
    t = re.sub(r"^(答案|answer|name)\s*[::]\s*", "", t, flags=re.I)
    if not t or len(t) > 24 or len(re.findall(r"[A-Za-z0-9]", t)) < 1:
        return "none", ""
    if len(re.findall(r"[A-Za-z0-9(（）)\-+]", t)) / max(1.0, len(t)) < 0.7:
        return "none", ""
    return "name", t


def parse_pick(txt: str, n: int):
    """把 `1=B` 之类的行解析成 [0(选A)|1(选B)|None, ...]。"""
    out = [None] * n
    for ln in (txt or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[=:.)]\s*([ABXabx])", ln)
        if not m:
            continue
        i = int(m.group(1)) - 1
        if 0 <= i < n:
            ch = m.group(2).upper()
            out[i] = None if ch == "X" else (0 if ch == "A" else 1)
    return out


# --------------------------------------------------------------------- 主流程
def _ask(client, img, prompt):
    try:
        txt, _dt = client.ask(img, prompt)
    except Exception:                                       # noqa: BLE001
        return ""
    return txt or ""


def apply(img, doc, client=None, opts=None):
    """就地改写 doc 的引脚名。返回统计 dict（不抛异常）。"""
    opts = opts or {}
    st = {"slots": 0, "calls": 0, "adopted": 0, "uncertain": 0, "skipped": 0, "trace": []}
    if client is None:
        try:
            from vlm import client as _c                          # noqa: PLC0415
            client = _c
        except Exception as exc:                                  # noqa: BLE001
            st["error"] = "no client: %s" % exc
            return st
    # ★ 不要用 `client.available()` 当闸门：它只看 BASE/KEY 有没有配，KEY 为空时返回
    #   False，而本机端点（LM Studio 风格，无 KEY）实测可用。真正的判据是「问得动」，
    #   由 `_ask` 的 try/except + 预算兜底；问不动就一个字都改不动（安全）。
    if hasattr(client, "budget_exhausted") and client.budget_exhausted():
        st["error"] = "budget exhausted"
        return st
    H = float(img.shape[0])
    max_slots = int(opts.get("max_slots", _env_int("FUSION_PINZOOM_MAX", MAX_SLOTS)))
    batch = max(1, int(opts.get("batch", _env_int("FUSION_PINZOOM_BATCH", BATCH))))
    max_calls = int(opts.get("max_calls", _env_int("FUSION_PINZOOM_CALLS", MAX_CALLS)))
    ic_only = bool(opts.get("ic_only", _env_on("FUSION_PINZOOM_IC_ONLY", "1")))
    retry_cap = int(opts.get("retry", _env_int("FUSION_PINZOOM_RETRY", 2)))
    n_retry = 0

    pins = doc.get("pins") or {}
    comps = doc.get("components") or {}
    # 同一元件里其它引脚已用的名字（用于「不许抢别人的名字」）
    used = {}
    for cid, grp in pins.items():
        used[cid] = {norm(v.get("pinname")) for v in grp.values()
                     if isinstance(v, dict) and v.get("pinname")}

    todo = []
    for cid, grp in pins.items():
        comp = comps.get(cid) or {}
        box = comp.get("bbox") or None
        if not box or len(box) < 4:
            continue
        ctype = str(comp.get("type") or comp.get("canon_type") or "").lower()
        if ic_only and ctype and ctype not in IC_TYPES:
            continue
        c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        for pk, info in grp.items():
            if not isinstance(info, dict):
                continue
            name = info.get("pinname") or ""
            ok, why = suspicious(name)
            if not ok:
                continue
            pt = info.get("point") or {}
            p = (float(pt.get("x", 0.0)), float(pt.get("y", 0.0)))
            if not p[0] and not p[1]:
                continue
            todo.append({"cid": cid, "pk": pk, "info": info, "name": name, "p": p, "c": c,
                         "why": why})
    todo = todo[:max_slots]

    # 阶段 1：逐槽自由读（弃权时按模型给的整图框再切一次，最多 1 轮）
    kept = []
    for t in todo:
        if st["calls"] >= max_calls:
            st["skipped"] += 1
            continue
        sub = crop_json(img, H, band_rect(t["p"], t["c"]), mark=t["p"])
        txt = _ask(client, sub, PROMPT_FREE % SCALE)
        st["calls"] += 1
        kind, val = parse_free(txt)
        if kind == "box":
            st["uncertain"] += 1
            # ★ 重试要有上限：模型在「这一小格里确实没有字」时倾向回整页框，
            #   每次重试都是 +1 次调用（第一版 3 张图因此从 ~90 次涨到 166 次）。
            big = (val[2] - val[0]) > 0.8 * img.shape[1] or (val[3] - val[1]) > 0.8 * img.shape[0]
            if n_retry < retry_cap and not big and st["calls"] < max_calls:
                n_retry += 1
                sub2 = crop_json(img, H, val, mark=t["p"])
                if sub2 is not None:
                    txt2 = _ask(client, sub2, PROMPT_FREE % SCALE)
                    st["calls"] += 1
                    k2, v2 = parse_free(txt2)
                    if k2 == "name":
                        kind, val = k2, v2
        if kind != "name":
            continue
        cand = val.strip()
        if norm(cand) == norm(t["name"]):
            continue                     # 模型同意我们现在的读法
        t["cand"] = cand
        kept.append(t)

    st["slots"] = len(kept)
    if not kept:
        return st

    # 阶段 2：批量 2AFC
    picks = [None] * len(kept)
    for i in range(0, len(kept), batch):
        chunk = kept[i:i + batch]
        if st["calls"] >= max_calls:
            break
        cells, lines = [], []
        for j, t in enumerate(chunk):
            cells.append(crop_json(img, H, band_rect(t["p"], t["c"]), mark=t["p"]))
            # 顺序按槽位确定（不是随机的），保证可复现（crc32 而不是 hash()：后者随进程变）
            flip = (zlib.crc32(("%s.%s" % (t["cid"], t["pk"])).encode()) % 2) == 1
            a, b = (t["cand"], t["name"]) if flip else (t["name"], t["cand"])
            t["_model_side"] = 1 if flip else 0
            lines.append("格 %d）A) %s    B) %s" % (j + 1, a, b))
        mont = montage(cells)
        if mont is None:
            continue
        out = parse_pick(_ask(client, mont, PROMPT_PICK % (len(chunk), SCALE,
                                                           "\n".join(lines), len(chunk))),
                         len(chunk))
        st["calls"] += 1
        for j, pk in enumerate(out):
            picks[i + j] = pk

    # 阶段 3：采信（pickd）
    for t, pick in zip(kept, picks):
        ad = None
        if pick is None or pick != t["_model_side"]:
            ad = False
        else:
            cand = t["cand"]
            nc = norm(cand)
            if nc in used.get(t["cid"], set()):
                ad = None                             # 抢了同元件别的脚的名字
            elif len(nc) < len(norm(t["name"])):
                ad = None                             # 不许变短
            else:
                t["info"]["pinname"] = cand
                ad = True
                st["adopted"] += 1
        st["trace"].append({"cid": t["cid"], "pk": t["pk"], "ours": t["name"],
                            "cand": t["cand"], "pick": pick, "side": t["_model_side"],
                            "adopted": ad})
    return st


def apply_file(png_path, src_json, dst_json, opts=None):
    """从磁盘读图/文档、改写引脚名、写回 dst_json。返回统计或 None（读不了图就 None）。"""
    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    import json                                                # noqa: PLC0415
    from pathlib import Path                                   # noqa: PLC0415
    with open(src_json, encoding="utf-8") as f:
        doc = json.load(f)
    st = apply(img, doc, opts=opts)
    Path(dst_json).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return st
