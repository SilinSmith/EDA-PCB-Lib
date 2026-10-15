# -*- coding: utf-8 -*-
"""VLM（LM Studio 视觉）读文字 vs CRNN 的对照实验。

同一批**真值文字框**上比三条臂的 exact：
  ``crnn``     TextEngine 在真值紧框上读（这套 CRNN 的天花板 0.843）
  ``vlm``      VLM 在真值紧框（上采样）上读
  ``vlm_pad``  VLM 在真值框外扩 ``--pad``（默认 10px，带上下文）上读

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_vlmread.py --limit 1 --per 20
    python work\\tools\\diag\\_vlmread.py --limit 1 --per 20 --pad 10 --model qwen3.5-9b
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from stage2.text import TextEngine, imread_unicode  # noqa: E402
from _textseg_common import gt_of  # noqa: E402
from _textseg_eval import cases, iou_matrix  # noqa: E402
from cvparts import textseg  # noqa: E402

DET_CFG = dict(ink_mode="fixed", ink=225, min_dim=2, min_area=3, line_len=15,
               resid_min_h=4, resid_min_dim=1)

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
def _vlmkey():
    """§5.103：token 会被轮换 ⇒ 探针里**不要写死 key**。来源与 vlm/client.py 同序。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("LLM_API_KEY_FILE"),
             os.path.join(os.path.dirname(os.path.dirname(here)), ".lmstudio_key"),
             os.path.join(os.path.expanduser("~"), ".lmstudio", "api_key")]
    for _p in cands:
        if _p and os.path.exists(_p):
            _t = open(_p, encoding="utf-8").read().strip()
            if _t.startswith("LLM_API_KEY="):
                _t = _t.split("=", 1)[1].strip()
            if _t:
                return _t
    return ""


KEY = os.environ.get("LLM_API_KEY", "") or _vlmkey()
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b")
SCALE = int(os.environ.get("VLM_SCALE", "4"))
MAXTOK = int(os.environ.get("VLM_MAX_TOKENS", "2048"))
# 关思维链的**唯一**有效开关：OpenAI 兼容的 reasoning_effort=none。
# 实测（1/U1 492x891 引脚任务）：none -> reason_tok 0、2.5 s、有输出；
# low/medium、/no_think、chat_template_kwargs={"enable_thinking":False} 全部无效
# （reason_tok 吃满预算、finish_reason=length、content 为空）。
# 置空字符串可以关掉这一项，回到默认行为。
REASONING = os.environ.get("VLM_REASONING", "none")

PROMPT = ("这张小图取自一张电路原理图，里面印着一段文字。"
          "只回那段文字本身，不要引号、不要解释、不要标点。看不出来就回 ?")


def enc_png(bgr) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def crop(bgr, box, pad):
    h, w = bgr.shape[:2]
    x1 = max(0, int(round(box[0] - pad)))
    y1 = max(0, int(round(box[1] - pad)))
    x2 = min(w, int(round(box[2] + pad)))
    y2 = min(h, int(round(box[3] + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2]


def upscale(bgr, scale=SCALE):
    if scale <= 1:
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def strip_fence(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rstrip("`").strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


def _chat_once(img_bgr, prompt, model, max_tokens, timeout):
    import urllib.request

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + enc_png(img_bgr)}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if REASONING:
        payload["reasoning_effort"] = REASONING
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return strip_fence(msg.get("content") or ""), ch.get("finish_reason"), d.get("usage")


def ask_vlm(img_bgr, prompt=PROMPT, model=MODEL, timeout=600, max_tokens=None):
    """思维链兜底：正文空或 finish_reason==length 就把预算翻倍重试（上限 16384）。"""
    budget = int(max_tokens or MAXTOK)
    t0 = time.time()
    last = ""
    try:
        for _ in range(4):
            txt, fin, usage = _chat_once(img_bgr, prompt, model, budget, timeout)
            if txt and fin != "length":
                return txt, round(time.time() - t0, 1)
            last = txt
            if budget >= 16384:
                break
            budget = min(16384, budget * 2)
        return last, round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        return "ERR:%s" % repr(e)[:80], round(time.time() - t0, 1)


def collect(a):
    """收集样本：每个真值文字框一份紧框 / 外扩框 / 40px 上下文，以及与它配对的我们的检出框。"""
    rows = []
    for png, js in cases(a.limit):
        gts = gt_of(js)
        if not gts:
            continue
        bgr = imread_unicode(png)
        try:
            dets = [tuple(float(v) for v in q)
                    for q in textseg.detect_boxes(bgr, **DET_CFG)]
        except Exception:  # noqa: BLE001
            dets = []
        M = iou_matrix([g[0] for g in gts], dets) if dets else None
        for i, (b, txt, role) in enumerate(gts):
            if len(rows) >= a.per:
                break
            c = crop(bgr, b, 0)
            cp = crop(bgr, b, a.pad)
            ctx = crop(bgr, b, 40.0)
            if c is None or cp is None or ctx is None:
                continue
            ox = max(0, int(round(b[0] - 40.0)))
            oy = max(0, int(round(b[1] - 40.0)))
            loc = (b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy)
            # 跟我们自己的检出框配对（IoU 最大且 >= 0.05）
            dt = None
            if M is not None and M.shape[1]:
                row = M[i]
                j = int(np.argmax(row))
                if float(row[j]) >= 0.05:
                    dd = crop(bgr, dets[j], 0)
                    if dd is not None and dd.size:
                        dt = upscale(dd)
            rows.append(dict(case=png.parent.name, i=len(rows), box=b, txt=txt,
                             role=role, raw=c, tight=upscale(c),
                             pad=upscale(cp), ctx=ctx, loc=loc, det=dt))
        if len(rows) >= a.per:
            break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--pad", type=float, default=10.0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save", default="", help="把首个裁片存成 png 便于目视")
    a = ap.parse_args(argv)

    rr = TextEngine().recognizer()
    rows = collect(a)
    print("样本 %d 个（%d 张图）" % (len(rows), len({r['case'] for r in rows})))

    if a.save and rows:
        h = max(r["tight"].shape[0] for r in rows)
        cols = []
        for r in rows[:8]:
            im = r["tight"]
            pad = np.full((h - im.shape[0], im.shape[1], 3), 255, np.uint8)
            cols.append(np.vstack([im, pad]))
            cols.append(np.full((h, 8, 3), 200, np.uint8))
        cv2.imwrite(a.save, np.hstack(cols))
        print("saved:", a.save)

    # CRNN 在真值紧框上（40px 上下文裁片 + 平移后的框，复现 GT_raw 口径）
    t0 = time.time()
    crnn = {}
    for r in rows:
        out = rr.read_many(r["ctx"], [(tuple(r["loc"]), r["role"])])
        crnn[r["i"]] = (out[0][0] or "").strip() if out else ""
    t_crnn = time.time() - t0

    def job(r):
        t, dt = ask_vlm(r["tight"], model=a.model)
        p, dp = ask_vlm(r["pad"], model=a.model)
        d, dd = ask_vlm(r["det"], model=a.model) if r["det"] is not None else ("-", 0.0)
        return r["i"], t, dt, p, dp, d, dd

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        vres = list(ex.map(job, rows))
    t_vlm = time.time() - t0

    print("\n%-4s %-14s %-8s %-12s %-12s %-12s %-12s %s"
          % ("#", "GT", "role", "crnn(tight)", "vlm(tight)", "vlm(pad)",
             "vlm(det)", "T"))
    okc = okv = okp = okd = nd = 0
    for (i, t, dt, p, dp, d, dd), r in zip(vres, rows):
        g = r["txt"]
        c = crnn.get(i, "")
        okc += (c == g)
        okv += (t == g)
        okp += (p == g)
        if r["det"] is not None:
            nd += 1
            okd += (d == g)
        print("%-4d %-14s %-8s %-12s %-12s %-12s %-12s %.0f/%.0f"
              % (i, g[:14], r["role"], c[:12], t[:12], p[:12], d[:12], dt, dp))
    n = max(1, len(rows))
    print("\n=== exact（%d 个真值框） ===" % len(rows))
    print("  crnn(真值紧框) %5d/%d = %.4f   (%.1f s 总)"
          % (okc, len(rows), okc / n, t_crnn))
    print("  vlm (真值紧框) %5d/%d = %.4f   (%.1f s 总, %.1f s/次)"
          % (okv, len(rows), okv / n, t_vlm, t_vlm / n))
    print("  vlm (真值+pad) %5d/%d = %.4f" % (okp, len(rows), okp / n))
    print("  vlm (我们检出框) %5d/%d = %.4f   <- 关键：框不紧能不能照样读对"
          % (okd, max(1, nd), okd / max(1, nd)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
