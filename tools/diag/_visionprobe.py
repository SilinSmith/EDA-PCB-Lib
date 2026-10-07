"""探 LM Studio 的视觉能力：逐个模型试 OpenAI 风格的 image_url 部件。

用法：
  python work\tools\diag\_visionprobe.py                # 全部模型 x 一张图
  python work\tools\diag\_visionprobe.py --models qwen3.5-9b
  python work\tools\diag\_visionprobe.py --img <png> --prompt "..."

判据：HTTP 200 且 content 里出现图上真有的字 => 这个模型吃图。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "sk-lm-YPX0MZpY:YPu0ZenotNLwPeVYNJ3Q")

DEF_IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "preview", "_zoom_0003_U1.png")
DEF_PROMPT = "这张图里印的文字是什么？只回文字本身，不要解释。"


def _req(path: str, payload=None, timeout=600):
    url = BASE + path
    headers = {"Authorization": "Bearer " + KEY}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models():
    return [m["id"] for m in _req("/models", timeout=30)["data"]]


def b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def try_vision(model: str, img_path: str, prompt: str, image_first: bool):
    url = "data:image/png;base64," + b64(img_path)
    parts = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    if image_first:
        parts = parts[::-1]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    t0 = time.time()
    try:
        r = _req("/chat/completions", payload)
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        return {
            "ok": True,
            "dt": round(time.time() - t0, 1),
            "content": (msg.get("content") or "").strip()[:400],
            "reasoning": (msg.get("reasoning_content") or "").strip()[:200],
            "finish": ch.get("finish_reason"),
            "usage": r.get("usage"),
        }
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        return {"ok": False, "dt": round(time.time() - t0, 1),
                "err": "HTTP %s" % e.code, "body": body}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "dt": round(time.time() - t0, 1),
                "err": repr(e)[:300], "body": ""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="逗号分隔；空 = 全部")
    ap.add_argument("--img", default=DEF_IMG)
    ap.add_argument("--prompt", default=DEF_PROMPT)
    ap.add_argument("--image-first", action="store_true")
    ap.add_argument("--skip-embed", action="store_true", default=True)
    a = ap.parse_args(argv)

    ids = list_models()
    print("models:", ", ".join(ids))
    if a.models:
        want = [m.strip() for m in a.models.split(",") if m.strip()]
    else:
        want = [m for m in ids if "embed" not in m.lower()]
    if not os.path.exists(a.img):
        print("!! 图片不存在:", a.img)
        return 1
    print("img:", a.img, os.path.getsize(a.img), "B")
    print("-" * 78)
    for m in want:
        r = try_vision(m, a.img, a.prompt, a.image_first)
        if r["ok"]:
            print("[OK ] %-34s %5.1fs finish=%s" % (m, r["dt"], r["finish"]))
            print("      content  :", repr(r["content"][:200]))
            if r["reasoning"]:
                print("      reason   :", repr(r["reasoning"][:120]))
            u = r.get("usage") or {}
            print("      usage    :", {k: u.get(k) for k in
                                      ("prompt_tokens", "completion_tokens")})
        else:
            print("[ERR] %-34s %5.1fs %s" % (m, r["dt"], r["err"]))
            if r.get("body"):
                print("      body:", r["body"][:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
