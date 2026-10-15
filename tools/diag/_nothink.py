# -*- coding: utf-8 -*-
"""同一个视觉任务上试各种「关思维链」的写法，看哪个能真的把 thinking 压掉。

背景：``qwen3.5-9b`` 在「列出 IC 全部引脚」这类开放式任务上会把整个 token
预算烧在 reasoning 上、``content`` 永远为空（``finish_reason=length``）。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_nothink.py --case 1 --key U1 --margin 70
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from schematic_parser.data import discover, load_json  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402

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

P = ("这是电路原理图里的一个元器件（IC / 模块）。图里每个引脚旁边印着"
     "一个引脚编号（数字）和一个信号名。\n"
     "只回**一行**答案，格式是 `编号:名称` 之间用英文逗号分隔，例如：\n"
     "1:VIN,2:GND,3:EN\n"
     "不要 JSON、不要换行、不要解释、不要任何其它字符。")


def call(img_bgr, model, prompt, extra=None, budget=1024, timeout=600):
    ok, buf = cv2.imencode(".png", img_bgr)
    url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}}]}],
        "max_tokens": budget,
        "temperature": 0.0,
    }
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"dt": round(time.time() - t0, 1), "err": repr(e)[:160]}
    ch = (d.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    u = d.get("usage") or {}
    rt = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    return {"dt": round(time.time() - t0, 1), "finish": ch.get("finish_reason"),
            "content": (m.get("content") or "").strip()[:300],
            "reason": (m.get("reasoning_content") or "").strip()[:200],
            "reason_tok": rt, "completion_tok": u.get("completion_tokens"),
            "prompt_tok": u.get("prompt_tokens")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--key", default="U1")
    ap.add_argument("--margin", type=float, default=70.0)
    ap.add_argument("--budget", type=int, default=1024)
    a = ap.parse_args(argv)

    im = None
    for c in discover():
        if c.case_id != a.case or not c.json_path:
            continue
        gt = load_json(c.json_path)
        v = (gt.get("components") or {}).get(a.key)
        if not v:
            continue
        bgr = imread_unicode(c.png_path)
        H = bgr.shape[0]
        b = v["bbox"]
        x0 = max(0, int(b[0] - a.margin)); y0 = max(0, int(H - b[3] - a.margin))
        x1 = min(bgr.shape[1], int(b[2] + a.margin))
        y1 = min(bgr.shape[0], int(H - b[1] + a.margin))
        im = bgr[y0:y1, x0:x1]
        break
    if im is None:
        print("没找到"); return 1
    print("裁片 %dx%d，预算 %d" % (im.shape[1], im.shape[0], a.budget))

    variants = [
        ("baseline qwen3.5-9b", "qwen3.5-9b", P, None),
        ("+ /no_think", "qwen3.5-9b", P + "\n/no_think", None),
        ("+ enable_thinking=False", "qwen3.5-9b", P,
         {"chat_template_kwargs": {"enable_thinking": False}}),
        ("+ thinking=False", "qwen3.5-9b", P,
         {"chat_template_kwargs": {"thinking": False}}),
        ("+ reasoning_effort=none", "qwen3.5-9b", P, {"reasoning_effort": "none"}),
        ("+ reasoning_effort=low", "qwen3.5-9b", P, {"reasoning_effort": "low"}),
        ("qwen3.8-27b@iq3_s", "qwen3.8-27b@iq3_s", P, None),
        ("qwen3.8-27b@q4_k_m", "qwen3.8-27b@q4_k_m", P, None),
    ]
    for name, model, prompt, extra in variants:
        r = call(im, model, prompt, extra, a.budget)
        if "err" in r:
            print("%-30s ERR %s" % (name, r["err"]))
            continue
        print("%-30s %6.1fs reason_tok=%-6s comp_tok=%-6s finish=%-8s content=%r"
              % (name, r["dt"], r["reason_tok"], r["completion_tok"],
                 r["finish"], (r["content"] or "")[:90]))
        if r["reason_tok"] and r["reason_tok"] > 200:
            print("      reason: %r" % r["reason"][:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
