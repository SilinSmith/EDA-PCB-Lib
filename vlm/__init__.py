# -*- coding: utf-8 -*-
"""VLM 补强：用 OpenAI 兼容的本地视觉端点读 IC 位号与引脚号/名。

`client` 是极薄的 HTTP 客户端（含 `reasoning_effort=none` 这个关思维链的唯一有效开关），
`enrich` 是把它接到 stage2 产物上的应用层。
"""
from . import client, enrich  # noqa: F401

__all__ = ["client", "enrich"]
