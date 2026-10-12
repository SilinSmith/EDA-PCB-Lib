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
`VLM_REASONING`（默认 `none`，置空字符串关掉这一项）、`VLM_TIMEOUT`（默认 600 秒/次）、
`VLM_BUDGET_S`（默认 0 = 不限：**本进程**累计外部推理时间的上限，见 `ask()`）、
`VLM_RETRY`（默认 3 = 传输层最多再试 3 次，`0` 关掉 ⇒ 与旧行为逐字节相同）、
`VLM_RETRY_BASE`（默认 1.0 秒，指数退避基数，单次等待硬上限 8 秒，见 `_chat_with_retry`）。

★ 为什么要有 `VLM_BUDGET_S`（2026-10 加，为赛题「全部案例 < 2 小时」兜底）：
没有它时，**单次 `ask()` 的耗时没有上界**，最坏 `4 × VLM_TIMEOUT`（旧默认 600 ⇒
**2400 s = 40 min**）—— 那条路是「端点每次都**答得出来**、但正文空或被 `length` 截断」，
于是四次重试串起来等；黑洞端点（TCP 连得上但永不回包）便宜一点，但也要**整整一个
超时**才抛错返回。`wires.py` 的注释里记过这个黑洞（`0003` 单张 315.73 s）。交付口径下
一个 lane 要跑好几张图，若端点被限流/排队，每张图几十次调用能把「2 小时」整段吃掉，
而被判题机按墙钟硬杀时**没落进 `final_results/` 的案例是双罚**（分子少它、`m/M` 再折一次）。
所以这里给了两级保护，都只在设了 `VLM_BUDGET_S` 时生效（默认 0 ⇒ 与旧行为逐字节相同）：

1. **累计预算**：本进程花在外部推理上的时间（含失败的调用）超过 `VLM_BUDGET_S` 之后，
   `ask()` 立刻返回 `("", 0.0)` 而**不发请求** —— 优雅降级成「只有 CV 的那条链」，
   产物照出、只是少了 VLM 的补强；
2. **剩余预算夹住单次超时**：`eff = min(调用方超时, max(1 s, 剩余预算))`（每次重试重算），
   于是「一次调用卡满」最多只多吃掉一点点，不会出现 40 分钟的黑洞。

★ 传输层重试（`VLM_RETRY`，2026-10-13 加，为「8×A800 并列 lane + 外部端点」这一档）：
上面那两级保护解决的是「**耗时没有上界**」，这一级解决「**并发一高就丢读数**」——
单张图的 `ask()` 最坏耗时因此变成 `(1 + VLM_RETRY) × 4 × VLM_TIMEOUT`（默认 3 ⇒ 9600 s），
所以交付入口钉的 `VLM_BUDGET_S=1800`（每 lane）是**必需**的护栏，不是可选项；
退避本身也被剩余预算夹住（见 `_chat_with_retry`）。

统计用 `stats()` 读（`used_s / calls / refused / errors / retries / budget_s`），
`reset_budget()` 可清零（跑多案例时由调用方按需要决定「预算是整进程还是每案例」）。
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
# ★ 本进程累计外部推理时间上限（秒）；0 = 不限（默认，与旧行为逐字节相同）。
#   交付入口 `submission/run.py` 在**用户没设**的时候会给它一个默认值，见那里的注释。
BUDGET = float(os.environ.get("VLM_BUDGET_S", "0") or 0)

# ★ 传输层重试（2026-10-13 加，为「8×A800 并列 lane + 外部端点」这一档兜底）。
#   病：并发一高，端点会回 429 / 5xx，或干脆把连接挂成超时；而旧实现**一次失败就把它
#   当答案**（`ERR:...` 字符串，见 `ask()` 的 `except`），那个槽位的读数就永久丢了 ——
#   单卡串行时几乎见不到，8 条 lane 各带若干 worker 时是常态。
#   药：只对**传输层**错误（限流 / 网关 / 服务端 / 超时 / 连接）指数退避重试，
#   对 400 / 401 这类「问错了」不重试（那是配置病，重试只会浪费时间）。
#   `VLM_RETRY=0` ⇒ 一次就抛，路径与旧版逐字节相同；退避总时长仍被 `VLM_BUDGET_S` 夹住。
RETRY = int(os.environ.get("VLM_RETRY", "3") or 0)
RETRY_BASE = float(os.environ.get("VLM_RETRY_BASE", "1.0") or 0.0)
RETRY_CAP = 8.0   # 单次等待的硬上限（秒）：防止退避把「2 小时」吃掉一大块

_USED = 0.0      # 已花在外部推理上的时间（含失败/重试）
_CALLS = 0       # 真正发出去的 ask() 次数
_REFUSED = 0     # 因预算被拒的 ask() 次数
_ERRORS = 0      # 返回 "ERR:..." 的次数（只作日志用）
_RETRIES = 0     # 传输层重试**次数**（`VLM_RETRY` 触发，用来判断「今天端点是不是在限流」）
# ★ 外部推理的 **token 用量**（服务端 `usage` 块的累计）。赛题评分里
#   `FinalScore_Accuracy = 0.95·Score_Accuracy + 0.05·(m/M)·Score_Token` 而
#   `Score_Token` 是**消耗越少排名越高**（见 `docs/COMPETITION_RULES.md`）——
#   也就是说「少用外部模型」本身是 5 分制的排名项，所以用量必须可见、可核。
_TOK_IN = 0      # prompt tokens（含图片编码，图是 token 大户）
_TOK_OUT = 0     # completion tokens
_TOK_ALL = 0     # total_tokens（服务端给的，缺则用 in+out 兜）


def stats() -> dict:
    """预算、token 与调用统计（给日志/决策日志用）。"""
    return {"budget_s": round(BUDGET, 1), "used_s": round(_USED, 1),
            "calls": _CALLS, "refused": _REFUSED, "errors": _ERRORS,
            "retries": _RETRIES,
            "exhausted": budget_exhausted(), "base": BASE, "model": MODEL,
            "tokens": {"prompt": _TOK_IN, "completion": _TOK_OUT,
                       "total": _TOK_ALL}}


def reset_budget() -> None:
    """清零累计用量与会计数（预算语义由调用方决定：整进程 or 每案例）。"""
    global _USED, _CALLS, _REFUSED, _ERRORS, _RETRIES, _TOK_IN, _TOK_OUT, _TOK_ALL
    _USED = 0.0
    _CALLS = 0
    _REFUSED = 0
    _ERRORS = 0
    _RETRIES = 0
    _TOK_IN = 0
    _TOK_OUT = 0
    _TOK_ALL = 0


def budget_exhausted() -> bool:
    return BUDGET > 0 and _USED >= BUDGET


def budget_left() -> float:
    """剩余预算（秒）；`BUDGET=0` 时返回 `float("inf")` = 不限。"""
    if BUDGET <= 0:
        return float("inf")
    return max(0.0, BUDGET - _USED)


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


def _retryable(exc) -> bool:
    """这个异常「再问一次有可能成功」吗（传输层病 vs 配置病）。

    可重试：408/409/425/429 与全部 5xx（限流、网关、服务端抖动）、`URLError` /
    超时 / 连接类（`OSError` 家族，含 `socket.timeout`、`ConnectionResetError`）。
    不可重试：400 / 401 / 403 / 404（端点不认这个模型、key 错、图不收）—— 它们是
    **配置病**，重试只会把同一份错误再问几遍，白花时间与 token。
    """
    import urllib.error
    code = getattr(exc, "code", None)
    if isinstance(exc, urllib.error.HTTPError) or isinstance(code, int):
        try:
            c = int(code)
        except (TypeError, ValueError):
            return True
        return c in (408, 409, 425, 429) or 500 <= c <= 599
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))


def _chat_with_retry(img_bgr, prompt, model, max_tokens, timeout):
    """`_chat_once` + 传输层重试：`VLM_RETRY` 次，指数退避 `VLM_RETRY_BASE·2^i`（上限 8 s）。

    为什么放在这一层：`ask()` 的四次循环是**语义**重试（正文空 / 被 `length` 截断 ⇒
    把 `max_tokens` 翻倍再问），而并发一高时真正丢读数的是**传输**失败。两者独立，
    所以这里只补传输层，且成功后**立刻返回**，不改动任何既有语义。

    预算语义完全不变：剩余预算 ≤ 0 就**不再重试**（把错误抛出去，`ask()` 收成 `ERR:`），
    单次等待与整个重试循环都被剩余预算夹住 ⇒ `VLM_BUDGET_S` 依旧夹住总时长。
    ★ 为什么这里要自己算一个 `deadline`：`_USED` 是 `ask()` 在**出口**才累加的，
    所以「同一次 `ask()` 内部的退避」光看 `budget_left()` 是**看不见**的 ——
    不自己记账的话，`VLM_BUDGET_S=20` 也会被退避拖成 `(1+RETRY)×超时`。
    `VLM_RETRY=0` ⇒ 一次就抛，路径与旧版逐字节相同。
    """
    global _RETRIES
    n = max(0, int(RETRY))
    left0 = budget_left()
    deadline = None if left0 == float("inf") else time.monotonic() + left0
    for i in range(n + 1):
        try:
            return _chat_once(img_bgr, prompt, model, max_tokens, timeout)
        except Exception as e:                                      # noqa: BLE001
            if i >= n or not _retryable(e):
                raise
            wait = min(RETRY_BASE * (2 ** i), RETRY_CAP)
            if deadline is not None:
                rest = deadline - time.monotonic()
                if rest <= 0:
                    raise
                wait = min(wait, rest)
            time.sleep(max(0.0, wait))
            _RETRIES += 1


def ask(img_bgr, prompt, model=None, timeout=None, max_tokens=None):
    """思维链兜底：正文空或 `finish_reason=="length"` 就把预算翻倍重试（上限 16384）。

    带 `VLM_BUDGET_S` 时另做两件事（默认 0 = 不做，行为与旧版逐字节相同）：
    ① 累计用量已到预算 ⇒ **不发请求**，立刻返回 `("", 0.0)`（记进 `stats()["refused"]`）；
    ② 单次超时被**剩余预算**夹住（下限 1 s），于是「一次调用卡满」最多只能多吃几秒。

    传输层失败（429 / 5xx / 超时 / 连接）另由 `_chat_with_retry` 退避重试 `VLM_RETRY` 次；
    重试次数累进 `stats()["retries"]`（用来判断「今天端点是不是在限流」），
    `VLM_RETRY=0` 时与旧版逐字节相同。

    另：每次响应里的 `usage` 都累进 `stats()["tokens"]` —— 赛题的 `Score_Token` 是
    **排名项**（`0.05·(m/M)·Score_Token`，消耗越少越高），用量必须可见。
    """
    global _USED, _CALLS, _REFUSED, _ERRORS
    if budget_exhausted():
        _REFUSED += 1
        return "", 0.0
    budget = int(max_tokens or MAXTOK)
    t0 = time.time()
    last = ""
    base_to = float(timeout or TIMEOUT)
    _CALLS += 1
    try:
        for _ in range(4):
            # ★ 剩余预算**每次重试都重算**：否则第一次调用就把预算吃光之后，
            #   后面三次重试还各自按满超时等下去（最坏多等 3 × 超时）。
            left = budget_left()
            eff = base_to if left == float("inf") else min(base_to, max(1.0, left))
            txt, fin, usage = _chat_with_retry(img_bgr, prompt, model or MODEL, budget, eff)
            _acc_tokens(usage)
            if txt and fin != "length":
                return _close(t0, txt)
            last = txt
            if budget >= 16384 or budget_exhausted():
                break
            budget = min(16384, budget * 2)
        return _close(t0, last)
    except Exception as e:  # noqa: BLE001
        _ERRORS += 1
        return _close(t0, "ERR:%s" % repr(e)[:80])


def _acc_tokens(usage) -> None:
    """把一次响应的 `usage` 累进 token 计数（缺字段就跳过；端点不给 `usage` 也不报错）。"""
    global _TOK_IN, _TOK_OUT, _TOK_ALL
    if not isinstance(usage, dict):
        return
    try:
        pi = int(usage.get("prompt_tokens") or 0)
        co = int(usage.get("completion_tokens") or 0)
        tt = int(usage.get("total_tokens") or 0) or (pi + co)
    except (TypeError, ValueError):
        return
    _TOK_IN += pi
    _TOK_OUT += co
    _TOK_ALL += tt


def _close(t0: float, txt: str):
    """收口：把这次耗时累进 `_USED`，返回 `(文本, 秒)`（与旧版 ask 的返回值同形）。"""
    global _USED
    dt = time.time() - t0
    _USED += dt
    return txt, round(dt, 1)
