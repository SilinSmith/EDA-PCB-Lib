# 本地开发环境（**不要提交到比赛包里**）：在本机跑 VLM 通道前先 dot-source 这个文件。
#
#     . Y:\workspace\EDA\work\tools\dev_env.ps1
#     python eda-pcb\submission\run.py -input work\_submit_in220 -output work\_out -mode off -vlm on
#
# 为什么需要它：`eda-pcb\vlm\client.py` 的 `KEY = os.environ.get("LLM_API_KEY", "")`，
# 而本机 LM Studio 的 HTTP 服务器**开着鉴权**。不设这个变量时有两条错误路径，而它们的
# 统计表现完全一样（`enrich_dir` 的三个 set 全归零、10 张图 1 秒跑完）：
#   · KEY 为空 → 客户端仍发 `Authorization: Bearer `（空 token）→ 401；
#     LM Studio 日志的原话是 "The 'Authorization' header is present but not in the
#     expected form 'Authorization: Bearer $LM_API_TOKEN'. The provided value cannot be used."
#     —— 注意它是说**形式不对**，不是缺头，很容易被误读成「key 写错了」。
#   · 端点换了不支持图片的模型 → HTTP 400 "does not support image inputs"。
# `client.preflight()`（2026-10-06 起 `enrich_dir` 默认调用）会在开跑前把这两种情况
# 变成一句响亮的 RuntimeError。
#
# 端点/模型的权威事实（2026-10-06 实测）：
#   · LM Studio HTTP 服务器 = 127.0.0.1:1234，`lms server status` → "The server is running on port 1234."
#   · 已加载模型 qwen3.5-9b（8.38 GB，ctx 32768，4 parallel）
#   · 带对 key：`GET /v1/models` 与 `POST /v1/chat/completions` 都返回 200；不带：401
#   · token 形如 `sk-lm-XXXXXXXX:YYYYYYYYYYYYYYYYYYYY`（SDK 正则见
#     `~\.lmstudio\extensions\plugins\lmstudio\*\node_modules\@lmstudio\sdk\dist\index.mjs`）

$env:PYTHONPATH = 'Y:\workspace\EDA\eda-pcb'
$env:PYTHONIOENCODING = 'utf-8'

$env:LLM_BASE_URL = 'http://127.0.0.1:1234/v1'
$env:LLM_MODEL    = 'qwen3.5-9b'
$env:VLM_SCALE    = '4'

# 本机 LM Studio Developer 页生成的 API token（随时可在那里重新生成并替换这一行）。
# ★ 2026-10-27 轮换：旧 token `sk-lm-YPX0MZpY:...` 已失效（端点回 401）。
#   从这一版起 `vlm/client.py` 支持**文件来源**（§5.103）：`eda-pcb\.lmstudio_key`
#   （已 gitignore）或 `~\.lmstudio\api_key`，所以这个脚本不设 `LLM_API_KEY` 也能跑。
$env:LLM_API_KEY  = 'sk-lm-L74IBwPJ:3xLxFOiO83R74q480gFy'

Write-Host "[dev_env] LLM_BASE_URL=$env:LLM_BASE_URL  LLM_MODEL=$env:LLM_MODEL  LLM_API_KEY=$(if ($env:LLM_API_KEY) { '已设' } else { '未设' })"
