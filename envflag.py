# -*- coding: utf-8 -*-
"""环境开关的「假值词表」唯一实现。

为什么要有这个文件
------------------
2026-07 全仓静态审计发现：同一个语义（`STAGE1_XXX=0` 关掉某功能）在 **六个文件里
有六套互不相同的词表**，于是「关不掉」或「关成了别的意思」：

    wires.py:442    ("0","false","False")      ← 不认 off / none
    pipeline.py:47  ("0","off","false")        ← 不认 False / OFF（没 .lower()）
    labels.py:71    ("0","off","false")        ← 同上
    labels.py:107   ("0","off","false")        ← 同上
    detect.py:69    ("off","none","0","false") ← 已经 .lower()，是当时唯一对的
    pins.py:117     最宽松

行为变化（本次统一带来的、有意为之的改变）
------------------------------------------
以「旧词表不认、旧行为=真」为基准，下列写法以前会被当成 **开**，现在会被当成 **关**：

    STAGE1_SPLIT_JUNCTIONS=off / none      （旧表只有 "0"/"false"/"False"）
    STAGE1_BIGIC=False / OFF               （旧表没有 .lower()，认不出 "False"）
    STAGE1_WIRE_EXT_NEAREST=False          （同上）
    STAGE1_WIRE_INK=False                  （同上）

也就是说：**以前写 `False` 却仍然是开的人，现在会真的关掉**。仓库里现有的脚本与
README 一律用 `0` / `1`，所以这条改变不影响任何已记录的命令行。

用法
----
    from envflag import env_on
    ON = env_on("STAGE1_DO_SOMETHING", "1")        # 默认开
    OFF = env_on("STAGE1_DO_SOMETHING", "0")       # 默认关
"""
from __future__ import annotations

import os

#: 视为「假」的取值（比较前统一 `.strip().lower()`）。空串也算假。
FALSE_WORDS = frozenset(("", "0", "off", "false", "none", "no", "disable", "disabled"))

#: 视为「真」的取值。**其余任何值都算真**（保持历史行为：写个没见过的词不会静默关掉功能）。
TRUE_WORDS = frozenset(("1", "on", "true", "yes", "enable", "enabled"))


def env_raw(name: str, default="1") -> str:
    """取原始字符串（去首尾空白）。未设置时返回 `default`。"""
    v = os.environ.get(name)
    if v is None:
        return str(default)
    return str(v).strip()


def env_on(name: str, default="1") -> bool:
    """`name` 是否表示「开」。假值词表见 `FALSE_WORDS`，其余一律为真。"""
    return env_raw(name, default).lower() not in FALSE_WORDS


def env_off(name: str, default="0") -> bool:
    """`name` 是否表示「关」。`env_off(n, d) is not env_on(n, d)`。"""
    return not env_on(name, default)


def env_str(name: str, default: str = "") -> str:
    """取一个**模式字符串**（如 `mixed` / `cap` / `sparse`）；假值词统一归一成 `""`。

    这个是为 `STAGE2_BOX_PIN_DROP_UNREAD=mixed|cap|0` 这类「多档开关」准备的：
    调用方只要判断 `env_str(...)` 是不是空串就知道功能开没开。
    """
    v = env_raw(name, default)
    return "" if v.lower() in FALSE_WORDS else v


def _selftest() -> None:
    import os as _os
    cases = [
        ("1", True), ("0", False), ("on", True), ("off", False),
        ("OFF", False), ("  False ", False), ("None", False), ("none", False),
        ("no", False), ("yes", True), ("", False), ("mixed", True),
        ("sparse", True), ("cap", True), ("2", True), ("TRUE", True),
    ]
    saved = {}
    try:
        for val, want in cases:
            _os.environ["EDA_ENVFLAG_PROBE"] = val
            got = env_on("EDA_ENVFLAG_PROBE", "1")
            assert got is want, "env_on(%r) = %r, 期望 %r" % (val, got, want)
            assert env_off("EDA_ENVFLAG_PROBE", "1") is (not want)
        # 未设置时按 default
        _os.environ.pop("EDA_ENVFLAG_PROBE", None)
        assert env_on("EDA_ENVFLAG_PROBE", "1") is True
        assert env_on("EDA_ENVFLAG_PROBE", "0") is False
        assert env_str("EDA_ENVFLAG_PROBE", "mixed") == "mixed"
        _os.environ["EDA_ENVFLAG_PROBE"] = "0"
        assert env_str("EDA_ENVFLAG_PROBE", "mixed") == ""
        _os.environ["EDA_ENVFLAG_PROBE"] = "cap"
        assert env_str("EDA_ENVFLAG_PROBE", "mixed") == "cap"
    finally:
        for k in saved:
            save = saved[k]
            if save is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = save
        _os.environ.pop("EDA_ENVFLAG_PROBE", None)
    print("envflag 自检 1 项：全部通过")


if __name__ == "__main__":
    _selftest()
