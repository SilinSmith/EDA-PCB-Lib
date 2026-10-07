# -*- coding: utf-8 -*-
"""探 LM Studio 的能力：关思维链、JSON 模式、批量多槽位。

    python work\tools\diag\_llmprobe.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "sk-lm-YPX0MZpY:YPu0ZenotNLwPeVYNJ3Q")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b")
URL = BASE + "/chat/completions"

SYS = ("你是 PCB 原理图理解专家。从已定候选里选最可能的那个。"
       "只输出一个 JSON 对象，不要解释，格式 "
       '{"choice": <候选 value 原样>, "confidence": 0-1, "why": "一句话"}。'
       '全都不可能就输出 {"choice": null}。')

USER = """判断点：元件 c0007（类型 r，框 [100,200,140,220]）的位号是哪一个？
候选（value / 置信度 / 来源）：
  - "R14" / 0.80 / ocr / 证据 {"文本": "R14", "框": [141,198,160,206], "离框": 12.0}
  - "Rl4" / 0.55 / ocr / 证据 {"文本": "Rl4", "框": [141,198,160,206], "离框": 12.0}
  - "R1" / 0.40 / ocr / 证据 {"文本": "R1", "框": [141,198,155,206], "离框": 14.0}"""


def post(payload, timeout=300):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + KEY} if KEY else {})})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return time.time() - t0, d


def show(tag, el, d):
    m = d["choices"][0]["message"]
    u = d.get("usage") or {}
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    print("[%s] %.1fs  finish=%s  out=%s reasoning=%s" % (
        tag, el, d["choices"][0].get("finish_reason"), u.get("completion_tokens"), rt))
    print("     content=[%s]" % (m.get("content") or "").replace("\n", " ")[:200])
    if m.get("reasoning_content"):
        print("     reasoning=[%s]" % m["reasoning_content"].replace("\n", " ")[:120])


def main():
    print("URL   =", URL)
    print("MODEL =", MODEL)
    print()

    # 1) 裸问
    el, d = post({"model": MODEL, "messages": [{"role": "system", "content": SYS},
                                               {"role": "user", "content": USER}],
                  "temperature": 0.0, "max_tokens": 2048})
    show("裸问 2048", el, d)

    # 2) chat_template_kwargs 关思维链
    for kw in ({"enable_thinking": False}, {"thinking": False}):
        try:
            el, d = post({"model": MODEL, "messages": [{"role": "system", "content": SYS},
                                                       {"role": "user", "content": USER}],
                          "temperature": 0.0, "max_tokens": 512,
                          "chat_template_kwargs": kw})
            show("chat_template_kwargs=%s" % json.dumps(kw), el, d)
        except Exception as e:
            print("[chat_template_kwargs=%s] ERR %s" % (json.dumps(kw), e))

    # 3) /no_think 软开关
    el, d = post({"model": MODEL, "messages": [{"role": "system", "content": SYS + "\n/no_think"},
                                               {"role": "user", "content": USER + "\n/no_think"}],
                  "temperature": 0.0, "max_tokens": 512})
    show("/no_think", el, d)

    # 4) JSON 模式
    try:
        el, d = post({"model": MODEL, "messages": [{"role": "system", "content": SYS},
                                                   {"role": "user", "content": USER}],
                      "temperature": 0.0, "max_tokens": 1024,
                      "response_format": {"type": "json_object"}})
        show("response_format json_object", el, d)
    except Exception as e:
        print("[response_format json_object] ERR %s" % e)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
