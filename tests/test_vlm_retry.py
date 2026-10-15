# -*- coding: utf-8 -*-
"""传输层重试（`VLM_RETRY`）必须「并发丢读数」能救回来、且默认路径逐字节不变。

背景（RESULTS §5.60）：8×A800 那一天每条 lane 都各自带若干 VLM worker，对外部端点
是**并发**请求；端点回 429 / 503 或把连接挂成超时时，`vlm/client.py` 从前**一次失败
就把它当成答案**（`"ERR:..."` 字符串，`ask()` 的 `except` 吞掉异常），那个槽位的读数
就永久丢了。这里用一个本地假端点把四种情形钉死：

1. 429 → 503 → 200：**能读出来**，`stats()["retries"] == 2`；
2. `VLM_RETRY=0`：与旧版一致 —— 一次就返回 `ERR:`，只发一次请求；
3. 400（配置病）：**不重试**，立刻返回 `ERR:`（重试只会白花时间）；
4. 一直 429：重试到上限就放弃，且 `VLM_BUDGET_S` 仍夹住总时长。
"""
import importlib
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm import client as vlm_client                                   # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    """按 `plan` 逐请求回状态码；`plan` 用尽即回一个合法补全。"""

    plan = []
    seen = []

    def do_POST(self):                                                 # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        try:
            _Handler.seen.append(json.loads(body.decode("utf-8")))
        except Exception:                                              # noqa: BLE001
            _Handler.seen.append({})
        code = _Handler.plan.pop(0) if _Handler.plan else 200
        if code != 200:
            raw = b"{}"
        else:
            raw = json.dumps({
                "choices": [{"message": {"content": "C151"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3,
                          "total_tokens": 14},
            }).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):                                         # noqa: A003
        pass


def _serve(plan):
    _Handler.plan = list(plan)
    _Handler.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/v1" % srv.server_address[1]


def _reload(**env):
    """按给定环境变量重新加载 `vlm.client`（常量在 import 时读环境）。"""
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "VLM_RETRY", "VLM_RETRY_BASE",
              "VLM_BUDGET_S", "VLM_TIMEOUT"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    mod = importlib.reload(vlm_client)
    mod.reset_budget()
    return mod


IMG = np.zeros((32, 32, 3), np.uint8)


class VlmRetryTests(unittest.TestCase):
    def setUp(self):
        self._srv = []

    def tearDown(self):
        for s in self._srv:
            s.shutdown()
            s.server_close()

    def _endpoint(self, plan):
        srv, base = _serve(plan)
        self._srv.append(srv)
        return base

    # ---------------------------------------------------------------- 1
    def test_retry_recovers_from_throttle(self):
        base = self._endpoint([429, 503, 200])
        c = _reload(LLM_BASE_URL=base, VLM_RETRY=3, VLM_RETRY_BASE=0.01)
        txt, dt = c.ask(IMG, "读这个位号。", max_tokens=8)
        self.assertEqual(txt, "C151")
        st = c.stats()
        self.assertEqual(st["retries"], 2)
        self.assertEqual(st["errors"], 0)
        self.assertEqual(len(_Handler.seen), 3)
        self.assertEqual(st["tokens"]["total"], 14)     # 只有成功那次的 usage 记账
        self.assertGreaterEqual(dt, 0.0)

    # ---------------------------------------------------------------- 2
    def test_retry_off_matches_old_behaviour(self):
        base = self._endpoint([429, 200])
        c = _reload(LLM_BASE_URL=base, VLM_RETRY=0, VLM_RETRY_BASE=0.01)
        txt, _dt = c.ask(IMG, "读这个位号。", max_tokens=8)
        self.assertTrue(str(txt).startswith("ERR:"), txt)
        st = c.stats()
        self.assertEqual(st["retries"], 0)
        self.assertEqual(st["errors"], 1)
        self.assertEqual(len(_Handler.seen), 1)         # 一次就放弃（旧行为）

    # ---------------------------------------------------------------- 3
    def test_config_error_is_not_retried(self):
        base = self._endpoint([400, 200])
        c = _reload(LLM_BASE_URL=base, VLM_RETRY=3, VLM_RETRY_BASE=0.01)
        txt, _dt = c.ask(IMG, "读这个位号。", max_tokens=8)
        self.assertTrue(str(txt).startswith("ERR:"), txt)
        self.assertIn("400", str(txt))
        st = c.stats()
        self.assertEqual(st["retries"], 0)
        self.assertEqual(len(_Handler.seen), 1)

    # ---------------------------------------------------------------- 4
    def test_retry_gives_up_at_limit(self):
        base = self._endpoint([429] * 10)
        c = _reload(LLM_BASE_URL=base, VLM_RETRY=2, VLM_RETRY_BASE=0.01)
        t0 = time.time()
        txt, _dt = c.ask(IMG, "读这个位号。", max_tokens=8)
        el = time.time() - t0
        self.assertTrue(str(txt).startswith("ERR:"), txt)
        self.assertEqual(c.stats()["retries"], 2)
        self.assertEqual(len(_Handler.seen), 3)         # 1 次首发 + 2 次重试
        self.assertLess(el, 5.0)

    # ---------------------------------------------------------------- 5
    def test_budget_still_clamps_retries(self):
        base = self._endpoint([429] * 10)
        c = _reload(LLM_BASE_URL=base, VLM_RETRY=5, VLM_RETRY_BASE=0.02,
                    VLM_BUDGET_S=0.02)
        t0 = time.time()
        txt, _dt = c.ask(IMG, "读这个位号。", max_tokens=8)
        el = time.time() - t0
        self.assertTrue(str(txt).startswith("ERR:"), txt)
        self.assertLess(c.stats()["retries"], 5)        # 预算夹住了退避
        self.assertLess(el, 5.0)

    # ---------------------------------------------------------------- 6
    def test_retryable_predicate(self):
        c = _reload(VLM_RETRY=0)
        mk = lambda code: urllib.error.HTTPError("http://x/v1", code, "t", {}, None)
        for code, want in ((429, True), (408, True), (500, True), (503, True),
                           (400, False), (401, False), (403, False), (404, False)):
            self.assertIs(c._retryable(mk(code)), want, code)
        self.assertTrue(c._retryable(urllib.error.URLError("boom")))
        self.assertTrue(c._retryable(TimeoutError("slow")))
        self.assertTrue(c._retryable(ConnectionResetError("reset")))
        self.assertFalse(c._retryable(ValueError("bad json")))

    # ---------------------------------------------------------------- 7
    def test_happy_path_is_untouched(self):
        base = self._endpoint([200])
        c = _reload(LLM_BASE_URL=base, VLM_RETRY=3, VLM_RETRY_BASE=0.01)
        txt, _dt = c.ask(IMG, "读这个位号。", max_tokens=8)
        self.assertEqual(txt, "C151")
        self.assertEqual(c.stats()["retries"], 0)
        self.assertEqual(len(_Handler.seen), 1)


if __name__ == "__main__":
    unittest.main()
