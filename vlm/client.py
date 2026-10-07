# -*- coding: utf-8 -*-
"""VLM（OpenAI 兼容的本地视觉端点，例如 LM Studio）读图的极薄客户端。

存在的理由：本项目的 CRNN（`textrec`）在**真值紧框**上原样读也只有 0.843，
而我们的检出框系统性偏小（宽 0.77×、高 0.54×，两个方向比例不同，等比外扩补不回来），
所以「文字框切不紧」卡了很久。换成 VLM 之后这条路不再硬阻塞 —— 实测（40 个真值框）：

    CRNN 在真值紧框上   0.7000
    VLM  在真值紧框上   0.9500
    VLM  在真值框+10px  0.8250
    VLM  在**我们的检出框**上 0.7750   <-- 比 CRNN 读完美框还高

**关思维链的唯一有效开关是 `reasoning_effort: "none"`**（OpenAI 兼容字段）。
同一裁片（`1/U1` 492x891，预算 1024）的实测：`none` 2.5 s / reasoning 0 / 有输出，
而 `/no_think`、`chat_template_kwargs={"enable_thinking":False}`、`{"thinking":False}`、
`reasoning_effort="low"` 全部把预算吃满（reasoning 1024）、`finish_reason=length`、
`content` 为空。

环境变量：`LLM_BASE_URL`（默认 `http://127.0.0.1:1234/v1`）、`LLM_API_KEY`、
`LLM_MODEL`（默认 `qwen3.5-9b`）、`VLM_SCALE`（默认 4）、`VLM_MAX_TOKENS`（默认 2048）、
`VLM_REASONING`（默认 `none`，置空字符串关掉这一项）。
"""
from __future__ import annotations

import base64
import json
import os
import time

import cv2

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b")
SCALE = int(os.environ.get("VLM_SCALE", "4"))
MAXTOK = int(os.environ.get("VLM_MAX_TOKENS", "2048"))
REASONING = os.environ.get("VLM_REASONING", "none")
TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "600"))


def available() -> bool:
    """只回答「端点与密钥配好了没有」，**不代表端点真的能用**。

    真正的可用性判断是 `probe()`（它发一张 32x32 的小图问一句，验证对方收不收图）。
    2026-07 审计：本函数**全仓库无调用点**，而且原来的实现是 `bool(BASE)` ——
    `BASE` 有默认值，于是它**恒为 True**，特别容易被误读成「VLM 一定可用」。
    这里改成真的看配置，并在 docstring 里点明它不等于可用。
    """
    return bool(BASE) and bool(KEY)


def probe(timeout: float = 60.0) -> bool:
    """发一张 32x32 的小图问一句，判断这个端点**到底收不收图**。

    比赛容器只保证注入 `LLM_BASE_URL`/`LLM_MODEL`，不保证它是视觉模型；
    实测 `mistral-small-3.1-24b` / `qwen3-coder-next` 会回 HTTP 400
    「does not support image inputs」。所以 `-vlm auto` 先探一次。
    """
    import numpy as np
    try:
        img = np.full((32, 32, 3), 255, np.uint8)
        cv2.putText(img, "OK", (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        txt, _ = ask(img, "回这两个字。", timeout=timeout, max_tokens=32)
        return bool(txt) and not str(txt).startswith("ERR:")
    except Exception:  # noqa: BLE001
        return False


def preflight(strict: bool = True, timeout: float = 90.0):
    """启动前的**响亮**自检：返回 `(ok, detail)`，`strict` 时失败直接抛 `RuntimeError`。

    为什么必须有：`_chat_once` 抛的异常被 `ask()` 包成 `"ERR:..."` **字符串**返回（不抛），
    而 `enrich_dir` 只累加 `desig_calls/grid_calls/value_calls` —— 于是整条 VLM 链路可以
    **「1 秒跑完、产物与输入逐位相同」**地静默空转。已经栽过两次：

    1. `LLM_API_KEY` 为空 ⇒ 客户端发出空的 `Authorization: Bearer ` ⇒ 401；
    2. 端点换了模型、不支持图片 ⇒ HTTP 400「does not support image inputs」。

    两者在 `enrich` 的统计里长得**一模一样**（三个 set 全 0），所以必须在开跑前就问一次。
    """
    import numpy as np
    detail = ""
    try:
        img = np.full((32, 32, 3), 255, np.uint8)
        cv2.putText(img, "OK", (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        txt, _dt = ask(img, "回这两个字。", timeout=timeout, max_tokens=64)
        txt = str(txt or "")
        if txt.startswith("ERR:"):
            detail = txt
        elif not txt.strip():
            detail = "EMPTY:模型只产出 reasoning_content（预算被思维链吃满）"
        else:
            return True, txt.strip()
    except Exception as e:  # noqa: BLE001
        detail = repr(e)
    if strict:
        raise RuntimeError(
            "VLM 预检失败：%s\n  端点 %s｜模型 %s｜KEY %s\n"
            "  · 401 / 'header ... not in the expected form' ⇒ 设好 LLM_API_KEY（LM Studio "
            "Developer 页的 token，形如 sk-lm-XXXXXXXX:YYYYYYYYYYYYYYYYYYYY）\n"
            "  · 400 'does not support image inputs' ⇒ 换视觉模型（LLM_MODEL）\n"
            "  · EMPTY ⇒ 端点不吃 `reasoning_effort: none`，调大 VLM_MAX_TOKENS 或换模型"
            % (detail, BASE, MODEL, "已设" if KEY else "**未设**"))
    return False, detail


def enc_png(bgr) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def upscale(bgr, scale=SCALE):
    if scale <= 1:
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def strip_fence(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rstrip("`").strip()
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
    # KEY 为空时**不要**发这个头。发 `Authorization: Bearer `（空 token）时 LM Studio 会报
    # 「The 'Authorization' header is present but not in the expected form ... The provided
    # value cannot be used.」并回 401 —— 而**缺头**与**空头**是两条不同的诊断路径，后者会把
    # 人引向「key 写错了」的错误方向。实测（2026-10-06）：带上正确的 `LLM_API_KEY` 时
    # 同一个端点 `probe() == True`、`ask()` 0.2 s 读出 `C151`。
    hdr = {"Content-Type": "application/json"}
    if KEY:
        hdr["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return strip_fence(msg.get("content") or ""), ch.get("finish_reason"), d.get("usage")


def ask(img_bgr, prompt, model=None, timeout=None, max_tokens=None):
    """思维链兜底：正文空或 `finish_reason=="length"` 就把预算翻倍重试（上限 16384）。"""
    budget = int(max_tokens or MAXTOK)
    t0 = time.time()
    last = ""
    try:
        for _ in range(4):
            txt, fin, _usage = _chat_once(img_bgr, prompt, model or MODEL, budget,
                                          timeout or TIMEOUT)
            if txt and fin != "length":
                return txt, round(time.time() - t0, 1)
            last = txt
            if budget >= 16384:
                break
            budget = min(16384, budget * 2)
        return last, round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        return "ERR:%s" % repr(e)[:80], round(time.time() - t0, 1)
