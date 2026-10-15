# -*- coding: utf-8 -*-
"""第二批探测：批量槽位、以及各模型的耗时对比。

    python work\tools\diag\_llmprobe2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

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
URL = BASE + "/chat/completions"

SYS = ("你是 PCB 原理图理解专家。从已定候选里选最可能的那个。"
       "只输出一个 JSON 对象，不要解释。")

SINGLE = """判断点：元件 c0007（类型 r）的位号是哪一个？
候选（value / 置信度 / 来源）：
  - "R14" / 0.80 / ocr / 证据 {"文本": "R14", "框": [141,198,160,206], "离框": 12.0}
  - "Rl4" / 0.55 / ocr / 证据 {"文本": "Rl4", "框": [141,198,160,206], "离框": 12.0}
  - "R1" / 0.40 / ocr / 证据 {"文本": "R1", "框": [141,198,155,206], "离框": 14.0}
答案格式：{"choice": <候选 value 原样>, "confidence": 0-1, "why": "一句话"}"""

BATCH = """下面有 5 个判断点，各自有候选。请对**每一个**都给出选择。
答案格式（严格 JSON 数组，长度必须是 5，顺序与判断点一致）：
[{"slot": "<槽位id>", "choice": <候选 value 原样或 null>, "confidence": 0-1}, ...]
不要解释。

判断点 1：
槽位 id：desig:c0007
元件类型：r
候选：
  - "R14" / 0.80 / ocr / {"文本": "R14", "离框": 12.0}
  - "Rl4" / 0.55 / ocr / {"文本": "Rl4", "离框": 12.0}
  - "R1" / 0.40 / ocr / {"文本": "R1", "离框": 14.0}

判断点 2：
槽位 id：desig:c0011
元件类型：c
候选：
  - "C3" / 0.72 / ocr / {"文本": "C3", "离框": 9.0}
  - "C8" / 0.60 / ocr / {"文本": "CB", "离框": 11.0}
  - "C9" / 0.31 / ocr / {"文本": "C9", "离框": 21.0}

判断点 3：
槽位 id：desig:c0020
元件类型：box
候选：
  - "U1" / 0.66 / ocr / {"文本": "U1", "离框": 30.0}
  - "U7" / 0.51 / ocr / {"文本": "U7", "离框": 33.0}

判断点 4：
槽位 id：desig:c0031
元件类型：r
候选：
  - "R5" / 0.44 / ocr / {"文本": "R5", "离框": 25.0}
  - "R6" / 0.43 / ocr / {"文本": "R6", "离框": 26.0}

判断点 5：
槽位 id：desig:c0044
元件类型：testpoint
候选：
  - "TP1" / 0.69 / ocr / {"文本": "TP1", "离框": 3.0}
  - "1" / 0.20 / ocr / {"文本": "1", "离框": 5.0}"""


def post(model, user, max_tokens, timeout=600):
    payload = {"model": model,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": user}],
               "temperature": 0.0, "max_tokens": max_tokens}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + KEY} if KEY else {})})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return time.time() - t0, None, str(e)
    return time.time() - t0, d, None


def show(tag, el, d, err):
    if err:
        print("[%s] %.1fs  ERR %s" % (tag, el, err))
        return
    m = d["choices"][0]["message"]
    u = d.get("usage") or {}
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    print("[%s] %.1fs  finish=%s out=%s reasoning=%s" % (
        tag, el, d["choices"][0].get("finish_reason"), u.get("completion_tokens"), rt))
    print("     content=[%s]" % (m.get("content") or "").replace("\n", " ")[:400])


def main():
    for model in ("qwen3.5-9b", "mistral-small-3.1-24b"):
        print("=" * 70)
        print("MODEL =", model)
        el, d, err = post(model, SINGLE, 4096)
        show("单槽位", el, d, err)
        el, d, err = post(model, BATCH, 4096)
        show("5槽位批量", el, d, err)
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
