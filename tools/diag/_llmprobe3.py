# -*- coding: utf-8 -*-
"""第三批探测：批量大小 vs 耗时、以及并发能力。

    python work\tools\diag\_llmprobe3.py [n_slots] [concurrency]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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
URL = BASE + "/chat/completions"

SYS = ("你是 PCB 原理图理解专家。下面有若干判断点，每个都已给出候选。"
       "请对每一个判断点，从它的候选里选出最可能正确的那一个。"
       "严格输出 JSON 数组，长度与判断点数相同，元素格式 "
       '{"slot": "<槽位id>", "choice": <候选value原样或null>, "confidence": 0-1}。'
       "不要解释，不要 markdown 代码块。")

TYPES = ["r", "c", "box", "d", "l", "testpoint", "switch", "led"]
REF = ["R", "C", "U", "D", "L", "TP", "SW", "LED"]


def make_batch(n):
    lines = ["共 %d 个判断点。" % n]
    for i in range(1, n + 1):
        t = TYPES[i % len(TYPES)]
        p = REF[i % len(REF)]
        lines.append("\n判断点 %d：" % i)
        lines.append("槽位 id：desig:c%04d" % i)
        lines.append("元件类型：%s" % t)
        lines.append("候选：")
        for j, sc in enumerate(("0.80", "0.62", "0.41")):
            lines.append('  - "%s%d" / %s / ocr / {"文本": "%s%d", "离框": %d.0}'
                         % (p, i, sc, p, i, 8 + 4 * j))
    return "\n".join(lines)


def post(user, max_tokens=8192, timeout=900):
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": user}],
               "temperature": 0.0, "max_tokens": max_tokens}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + KEY} if KEY else {})})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return time.time() - t0, d


def extract(txt):
    t = txt.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return [json.loads(t[i:j + 1])]
        except Exception:
            pass
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print("MODEL =", MODEL, " batch =", n, " concurrency =", conc)
    user = make_batch(n)

    def one(k):
        return post(user)

    if conc <= 1:
        el, d = one(0)
        u = d.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        arr = extract(d["choices"][0]["message"].get("content") or "")
        print("  一次调用 %.1fs  finish=%s out=%s reasoning=%s  解析出 %s 条"
              % (el, d["choices"][0].get("finish_reason"), u.get("completion_tokens"),
                 rt, len(arr) if arr is not None else "None"))
        if arr is not None:
            ok = sum(1 for a in arr if isinstance(a, dict) and a.get("choice"))
            print("  其中有 choice 的:", ok)
    else:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            res = list(ex.map(one, range(conc)))
        wall = time.time() - t0
        for i, (el, d) in enumerate(res):
            u = d.get("usage") or {}
            arr = extract(d["choices"][0]["message"].get("content") or "")
            print("  #%d %.1fs out=%s 解析 %s 条" % (
                i, el, u.get("completion_tokens"), len(arr) if arr is not None else "None"))
        print("  并发 %d 墙钟 %.1fs（串行约 %.1fs）" % (conc, wall, sum(r[0] for r in res)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
