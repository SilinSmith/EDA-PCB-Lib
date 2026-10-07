# -*- coding: utf-8 -*-
"""LLM 裁决器 —— 只干一件事：在**已经给出的候选**里选一个。

用户 m03069 的机制：「如果不能判断，那就并行保存几种情况……最后 LLM 会参与识别，
只需要让 LLM 结合上下文选正确的就行了。」

所以这里**不让模型自由生成**，只让它做**多选**（外加 ``"none"`` 表示全不对），
理由：
- 自由生成会引入候选之外的值，而官方按 **key** 匹配 —— 多一个错值就是多一个错误元件；
- 多选可以把答案空间钉死在 CV/规则已经算出来的东西上，可复现、可审计；
- 输出直接回填 `Slot.chosen` 与 `decided_by="llm"`，写进 `decision_log.json`。

运行时注入的环境变量（赛题 Docker 镜像会提供）
--------------------------------------------
`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。缺失时本模块**不报错**，
只把所有槽位标成 `PENDING`，让流水线照常产出中间结果。

2025-10-04 实机接本地 LM Studio（`qwen3.5-9b`）时改掉了三处硬伤
----------------------------------------------------------------
1. **`max_tokens` 写死 200 会被思维链吃光。** `qwen3.5-9b` 是带思维链的模型
   （`message` 里有 `reasoning_content`），实测问「只回两个字符 OK」都要烧掉
   392 token 其中 313 是 reasoning；一句真正的多选要 **2400+ reasoning token**。
   于是 `max_tokens=200` 时 `finish_reason="length"`、`content=""` —— 解析必然失败，
   而且失败得**静默**（当 ERROR 计数）。现在默认 `LLM_MAX_TOKENS=4096`，并且
   **遇到 `finish_reason=="length"` 或 `content` 为空会自动翻倍重试**（上限 `LLM_MAX_TOKENS_CAP`）。
2. **一次只问一个槽位太贵。** 思维链是**每次调用**的固定开销：实测单槽位 38.8 s
   （2442 token）、5 槽位 52.6 s、20 槽位 119.6 s ⇒ 单槽位 38.8 s → 批量 20 时 **6.0 s/槽位**。
   所以加了 `ask_batch`：**一次调用裁决一批槽位**。
3. **并发。** LM Studio 实测能吃 4 路并发（4×20 槽位 171.8 s，串行要 590.5 s）。
   默认 `LLM_WORKERS=4`。

另外：**不是所有模型都支持 `response_format`**（LM Studio 上 `json_object` 直接 400），
所以解析走「剥 markdown 围栏 + 找最外层 `[...]`/`{...}`」，比依赖服务端约束稳。
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))
# `available()` 的 `/v1/models` 探测超时。短一点：这条路每次跑图都要过一次，
# 端点不在时不应该让整条管线等 180s。可用 `LLM_PROBE_TIMEOUT` 覆盖。
PROBE_TIMEOUT = float(os.environ.get("LLM_PROBE_TIMEOUT", "5"))
MAX_ALTS = 6                     # 每个槽位最多喂给模型几个候选
BATCH = int(os.environ.get("LLM_BATCH", "8"))          # 一次调用裁决几个槽位
WORKERS_DEFAULT = int(os.environ.get("LLM_WORKERS", "4"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
MAX_TOKENS_CAP = int(os.environ.get("LLM_MAX_TOKENS_CAP", "16384"))
RETRY = int(os.environ.get("LLM_RETRY", "1"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

SYSTEM = (
    "你是 PCB 原理图理解专家。下面给你**已经定好候选**的判断点，"
    "请结合给出的几何与上下文证据，从候选里选出最可能正确的那一个。"
    "只允许输出一个 JSON 对象，不要任何解释性文字，格式："
    '{"choice": <候选的 value 原样>, "confidence": 0-1 的小数, "why": "一句话"}。'
    "如果所有候选都不可能，输出 {\"choice\": null}。"
)

SYSTEM_BATCH = (
    "你是 PCB 原理图理解专家。下面有若干**已经定好候选**的判断点，"
    "请对**每一个**判断点，从它自己的候选里选出最可能正确的那一个。"
    "严格输出一个 JSON 数组，长度与判断点数相同，顺序一致，元素格式："
    '{"slot": "<槽位id>", "choice": <候选的 value 原样，都不可能时为 null>, '
    '"confidence": 0-1 的小数}。'
    "不要输出任何解释、不要 markdown 代码块、不要多余字段。"
)


# ---------------------------------------------------------------- 环境

def base_url() -> str:
    """默认值与 `eda-pcb\vlm\client.py:32` 一致 —— 指向同一个本机 OpenAI 兼容端点。

    历史 bug（owner m25661 点名的「LLM 根本没放进去」）：这里原来写的是
    `os.environ.get("LLM_BASE_URL","")`，**没有默认值**，于是 `available()`
    恒为 False ⇒ 阶段三的 LLM 仲裁与融合层的 LLM 轮**从来没有跑过**，
    而同一个端点在 `vlm\client.py` 里因为带默认值一直好好地服务 VLM。
    实测：`-mode llm` 的日志会打「LLM 未配置（LLM_BASE_URL / LLM_MODEL 缺失）」，
    融合层日志打「LLM 轮 关」。这行默认值就是那个分叉点。
    """
    return (os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")


def model() -> str:
    return os.environ.get("LLM_MODEL") or "qwen3.5-9b"


def available() -> bool:
    """端点**可达**才算可用。

    只要求 `LLM_BASE_URL`/`LLM_MODEL` 存在是不够的（本地这两个变量本来就不设）：
    本机部署时走默认地址，但要真发一次 `/v1/models` 确认服务在。比赛容器会注入
    这两个变量、端点也一定在，探测照样通过；端点不在时返回 False，行为与改动前
    的「没有 LLM」一致，不会把管线拖死。
    """
    if not (os.environ.get("LLM_BASE_URL") or os.environ.get("LLM_MODEL")):
        pass                             # 没配就走默认地址 + 探测
    try:
        req = urllib.request.Request(base_url() + "/models",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as r:
            json.loads(r.read().decode("utf-8"))
        return True
    except Exception:                    # noqa: BLE001
        return False


def describe() -> str:
    if not available():
        return "未配置（%s/models 探不通 / LLM_BASE_URL 缺失）" % base_url()
    return "model=%s url=%s batch=%d workers=%d max_tokens=%d" % (
        model(), base_url(), BATCH, WORKERS_DEFAULT, MAX_TOKENS)


# ---------------------------------------------------------------- HTTP

def _post(url: str, payload: dict, key: str, timeout: float = TIMEOUT):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _chat(messages, max_tokens: int, timeout: float = TIMEOUT):
    """发一次对话，返回 `(正文, finish_reason, usage)`。

    **思维链兜底**：如果正文为空而 `reasoning_content` 非空、或者
    `finish_reason == "length"`，说明预算被思维链吃光了 —— 翻倍重试
    （直到 `MAX_TOKENS_CAP`）。这是接本地推理模型时最容易踩的坑。
    """
    base = base_url()
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    key = os.environ.get("LLM_API_KEY", "")
    budget = max_tokens
    last = ("", "?", {})
    for _ in range(RETRY + 2):
        payload = {"model": model(), "messages": messages,
                   "temperature": TEMPERATURE, "max_tokens": budget}
        d = _post(url, payload, key, timeout)
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        txt = (msg.get("content") or "").strip()
        fin = ch.get("finish_reason")
        last = (txt, fin, d.get("usage") or {})
        if txt and fin != "length":
            return last
        if budget >= MAX_TOKENS_CAP:
            return last
        budget = min(budget * 2, MAX_TOKENS_CAP)
    return last


# ---------------------------------------------------------------- 解析

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")


def strip_fence(t: str) -> str:
    return _FENCE.sub("", t.strip()).strip()


def loads_tolerant(txt: str):
    """从模型输出里抠出 JSON。返回解析出来的对象，失败返回 `None`。

    不用 `response_format`：LM Studio 上 `json_object` 直接 400，
    而 `mistral-small` 会把答案包在 ```json 围栏里。所以这里自己剥。
    先试数组（批量），再试对象（单槽位）。
    """
    t = strip_fence(txt)
    if not t:
        return None
    for op, cl in (("[", "]"), ("{", "}")):
        i, j = t.find(op), t.rfind(cl)
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------- 提示词

def _fmt_alt(a):
    return "  - %s / %.2f / %s / 证据 %s" % (
        json.dumps(a.value, ensure_ascii=False), a.score, a.source,
        json.dumps(a.evidence, ensure_ascii=False))


def _slot_block(slot, idx: int) -> str:
    lines = ["判断点 %d：" % idx, "槽位 id：%s" % slot.slot_id]
    if slot.question:
        lines.append("说明：%s" % slot.question)
    lines.append("候选（value / 置信度 / 来源 / 证据）：")
    for a in slot.alts[:MAX_ALTS]:
        lines.append(_fmt_alt(a))
    if slot.context:
        lines.append("上下文：" + json.dumps(slot.context, ensure_ascii=False)[:1200])
    return "\n".join(lines)


def _prompt(slot) -> str:
    return _slot_block(slot, 1)


def _prompt_batch(slots) -> str:
    head = "共 %d 个判断点，请逐个裁决。\n\n" % len(slots)
    return head + "\n\n".join(_slot_block(s, i) for i, s in enumerate(slots, 1))


# ---------------------------------------------------------------- 单次问答

def ask_one(slot, timeout: float = TIMEOUT):
    """裁决单个槽位（保留旧接口）。"""
    try:
        txt, fin, _u = _chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": _prompt(slot)}], MAX_TOKENS, timeout)
        obj = loads_tolerant(txt)
        if isinstance(obj, list):
            obj = obj[0] if obj else None
        if not isinstance(obj, dict):
            return slot.slot_id, None, 0.0, "ERROR 解析不出 JSON：%r" % txt[:120]
        return (slot.slot_id, obj.get("choice"),
                float(obj.get("confidence") or 0.0), str(obj.get("why") or ""))
    except (urllib.error.URLError, KeyError, ValueError, IndexError, OSError) as e:
        return slot.slot_id, None, 0.0, f"ERROR {type(e).__name__}: {e}"


def ask_batch(slots, timeout: float = TIMEOUT):
    """一次调用裁决一批槽位。返回 `{slot_id: (choice, conf, why)}`。

    批量是把「每次调用的思维链固定开销」摊到多个槽位上 —— 这是本机
    `qwen3.5-9b` 唯一可行的用法（单槽 38.8 s → 20 槽 6.0 s/槽）。
    """
    out = {}
    if not slots:
        return out
    try:
        txt, fin, _u = _chat(
            [{"role": "system", "content": SYSTEM_BATCH},
             {"role": "user", "content": _prompt_batch(slots)}], MAX_TOKENS, timeout)
        obj = loads_tolerant(txt)
    except (urllib.error.URLError, KeyError, ValueError, IndexError, OSError) as e:
        why = "ERROR %s: %s" % (type(e).__name__, e)
        return {s.slot_id: (None, 0.0, why) for s in slots}

    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        why = "ERROR 解析不出 JSON：%r" % (txt or "")[:160]
        return {s.slot_id: (None, 0.0, why) for s in slots}

    by_id = {}
    rest = []
    for it in obj:
        if not isinstance(it, dict):
            continue
        sid = it.get("slot")
        if sid:
            by_id[str(sid)] = it
        else:
            rest.append(it)
    # 模型有时不写 slot，靠顺序兜底
    for s, it in zip(slots, rest):
        by_id.setdefault(s.slot_id, it)

    for s in slots:
        it = by_id.get(s.slot_id)
        if not it:
            out[s.slot_id] = (None, 0.0, "ERROR 该槽位没有出现在回答里")
            continue
        try:
            conf = float(it.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out[s.slot_id] = (it.get("choice"), conf, str(it.get("why") or ""))
    return out


# ---------------------------------------------------------------- 批量裁决

def arbitrate(hyps, slots=None, workers: int = WORKERS_DEFAULT,
              timeout: float = TIMEOUT, min_confidence: float = 0.0,
              batch: int = BATCH, max_slots: int = 0, log=print):
    """并发 + 分批裁决槽位，把选中的值写回 `hyps`。返回统计 dict。

    `max_slots` > 0 时只裁决分数最高的那些（控预算；比赛限时 2 小时）。
    """
    todo = list(slots if slots is not None else hyps.unresolved())
    st = {"asked": len(todo), "answered": 0, "resolved": 0, "failed": 0,
          "skipped": 0, "n_calls": 0, "batched": 0}
    if not todo:
        return st
    if not available():
        log(f"[llm] 端点 {base_url()} 探不通 —— {len(todo)} 个槽位保持 PENDING（这是允许的）")
        st["skipped"] = len(todo)
        return st

    if max_slots and len(todo) > max_slots:
        todo.sort(key=lambda s: -max((a.score for a in s.alts), default=0.0))
        dropped = len(todo) - max_slots
        todo = todo[:max_slots]
        st["skipped"] = dropped

    batch = max(1, batch)
    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    st["batched"] = len(groups)
    log(f"[llm] {len(todo)} 个槽位 -> {len(groups)} 次调用（每批 {batch}，并发 {workers}，"
        f"模型 {model()}）")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(ask_batch, g, timeout): g for g in groups}
        for f in as_completed(futs):
            try:
                res = f.result()
            except Exception as e:                       # noqa: BLE001
                res = {s.slot_id: (None, 0.0, "ERROR %s" % e) for s in futs[f]}
            st["n_calls"] += 1
            for sid, (choice, conf, why) in res.items():
                sl = hyps.slots.get(sid)
                if sl is None:
                    continue
                if choice is None or conf < min_confidence:
                    if why.startswith("ERROR"):
                        st["failed"] += 1
                    else:
                        st["answered"] += 1
                    continue
                st["answered"] += 1
                # 只在候选集合内接受；不在候选里就仍不采纳（防止模型编值）
                ok = any(a.value == choice or a.key() == json.dumps(
                    choice, sort_keys=True, ensure_ascii=False) for a in sl.alts)
                if ok:
                    hyps.resolve(sid, choice, by=f"llm:{conf:.2f}")
                    sl.context.setdefault("llm_why", why)
                    st["resolved"] += 1
                else:
                    st["failed"] += 1
    return st


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("LLM_BASE_URL:", os.environ.get("LLM_BASE_URL", "(未设置)"))
    print("LLM_MODEL   :", os.environ.get("LLM_MODEL", "(未设置)"))
    print("available() :", available())
    print("describe    :", describe())
