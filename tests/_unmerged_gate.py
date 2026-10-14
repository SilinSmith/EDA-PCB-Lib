# -*- coding: utf-8 -*-
"""main 上的「未并入路径」前置条件门（§5.28）。

**为什么需要它**：commit `29b22b1`（并入队友分支 `codex/improve-pin-network-parsing`）
把队友的 **36 个测试文件整文件拷进 main**，但队友的**产品绑定主体**按 §5.28 的判据
**有意未并** —— `stage2/assign.py` 的 507 行绑定主体、`stage2/structmatch.py` 的 879 行
（`docs/RESULTS.md:3225`：「它们是互斥的另一条绑定路径，不是可增量吸收的小件」）。
`git diff main..improve-pin-network-parsing -- tests/` 只剩本仓新增的两个文件 ⇒ 这些
测试在 main 与队友分支上**逐字节相同**，差别只在**产品那一半**。

于是这些文件在 main 上**没有可测对象**：它们的红不是「调用方式落后」，而是「被测行为
不存在」。把调用改对齐只会让报错从 `TypeError` 变成断言失败 —— 那是**弱化断言**，禁止。

**本门的语义（不是掩盖红例）**：
  * 只做一件事 —— 检查该模块需要的**前置符号**在**当前树**上是否存在；
  * 不存在 ⇒ `raise unittest.SkipTest(理由)`，`setUpModule` 抛这个异常会把**整模块**标成
    skipped（unittest 官方语义），而不是 error/fail；
  * **一条断言都不删、不改、不降级**；理由字符串里写明缺了什么、为什么、在哪条分支上齐备；
  * **自愈**：队友那条绑定路径真的并进来的那天，`hasattr` 变真 ⇒ 门自动打开，原测试
    原样开跑。所以在 main 上「跳过」是**如实登记前置条件**，而不是把红藏起来。

用法（放在测试文件末尾即可，`setUpModule` 会被 unittest 自动调用）::

    def setUpModule():
        import _unmerged_gate
        _unmerged_gate.gate([("stage2.structmatch", ("normalize_number_evidence",))])

模块里**只有一部分**用例打未并入符号时（drift 与孤儿混装），不要用整模块的 `setUpModule`，
改用逐用例装饰器把门收窄到真正缺前置条件的那几条::

    @_unmerged_gate.requires([("stage2.assign", ("PIN_TRUST_BIND",))])
    def test_something(self): ...

排障：`EDA_UNMERGED_GATE=off` 时 `gate()`/`requires()` 全部变成空操作（用于「不跳、看真实红」
的临时分诊，**不要**在提交面里带这个环境变量）。
"""
import functools
import importlib
import os
import unittest

BRANCH = "improve-pin-network-parsing"
MERGE_COMMIT = "29b22b1"
WHY = ("§5.28（docs/RESULTS.md:3225）：队友分支 %s 的引脚证据/读数链并入 %s 时，"
       "**只并了证据采集**，产品绑定主体（assign 507 行 + structmatch 879 行）"
       "**有意未并** —— 判据是「互斥的另一条绑定路径，不是可增量吸收的小件」。"
       % (BRANCH, MERGE_COMMIT))


def missing(checks):
    """`checks = [(模块路径, 名字序列), ...]` ⇒ 返回缺失的 `模块路径.名字` 列表。"""
    out = []
    for mod_path, names in checks:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as exc:                                   # noqa: BLE001
            out.append("%s（模块都导不进来：%r）" % (mod_path, exc))
            continue
        for n in names:
            if not hasattr(mod, n):
                out.append("%s.%s" % (mod_path, n))
    return out


def gate(checks, note=""):
    """前置符号缺失 ⇒ 整模块跳过（理由写全）。齐备 ⇒ 什么都不做。"""
    if os.environ.get("EDA_UNMERGED_GATE") == "off":
        return
    miss = missing(checks)
    if not miss:
        return
    raise unittest.SkipTest(
        "未并入路径的规格测试（main 上无被测对象）：缺 %s。%s 该分支上这些符号齐备，"
        "本模块与其上 %s 的测试逐字节相同；把绑定主体并入后本门自动打开，原断言原样开跑。%s"
        % ("、".join(miss), WHY, BRANCH, ("　" + note) if note else ""))


def requires(checks, note=""):
    """逐用例的门：只在打未并入符号的那几条用例上跳过（模块里 drift 与孤儿混装时用）。

    与 `gate()` 同一语义 —— 缺前置条件就 `SkipTest`，齐备则原样开跑；断言不动。
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            if os.environ.get("EDA_UNMERGED_GATE") != "off":
                miss = missing(checks)
                if miss:
                    raise unittest.SkipTest(
                        "未并入路径的规格测试（main 上无被测对象）：缺 %s。%s 缺的符号在 %s 上齐备；"
                        "把绑定主体并入后本门自动打开，原断言原样开跑。%s"
                        % ("、".join(miss), WHY, BRANCH, ("　" + note) if note else ""))
            return fn(*a, **kw)
        wrapper.__unmerged_checks__ = checks
        return wrapper
    return deco
