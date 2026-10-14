#!/bin/sh
# 赛题容器入口。官方契约是 /submission/agent_system -input /input -output /output，
# 但示例的 ENTRYPOINT 统一是 /submission/run.sh，所以这一层两个约定都满足：
# 解析参数 -> 转交给队友自己的 agent_system -> 透传退出码。
set -u

INPUT_DIR=""
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    -input)   INPUT_DIR="${2:-}";  shift 2 ;;
    -output)  OUTPUT_DIR="${2:-}"; shift 2 ;;
    --input)  INPUT_DIR="${2:-}";  shift 2 ;;
    --output) OUTPUT_DIR="${2:-}"; shift 2 ;;
    *)        shift ;;
  esac
done

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "usage: run.sh -input <dir> -output <dir>" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/final_results" "$OUTPUT_DIR/runs"

# ---------------------------------------------------------------------------
# 密钥别名兜底：官方示例注入的是 `DASHSCOPE_API_KEY`，而本仓库代码读的是
# `LLM_API_KEY`（`vlm/client.py:76`、`stage1/llm.py:141`）。平台只会给其中一个名字，
# 所以这里映射一次，否则会出现「平台给了 key 但代码读不到」→ 所有 VLM 通道静默降级
# （官方 10 张 84.95 → ~70，日志里只有一行「跳过」）。
# 只在 `LLM_API_KEY` 为空时映射，**不覆盖**调用方显式设置的值。
# ---------------------------------------------------------------------------
if [ -z "${LLM_API_KEY:-}" ] && [ -n "${DASHSCOPE_API_KEY:-}" ]; then
  LLM_API_KEY="$DASHSCOPE_API_KEY"
  export LLM_API_KEY
  echo "LLM_API_KEY <- DASHSCOPE_API_KEY（别名兜底）"
fi

echo "=== run.sh ==="
echo "INPUT_DIR  = $INPUT_DIR"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "LLM_BASE_URL      = ${LLM_BASE_URL:-<unset>}"
echo "LLM_MODEL         = ${LLM_MODEL:-<unset>}"
if [ -n "${LLM_API_KEY:-}" ]; then echo "LLM_API_KEY       = set"; else echo "LLM_API_KEY       = <unset>"; fi
if [ -n "${DASHSCOPE_API_KEY:-}" ]; then echo "DASHSCOPE_API_KEY = set"; else echo "DASHSCOPE_API_KEY = <unset>"; fi
echo "NVIDIA_VISIBLE_DEVICES = ${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "python3 = $(command -v python3)"
# 比赛日档位（`docs/RESULTS.md` §5.51）：平台不给我们档位开关，所以档位钉在
# `agent_system` 里（`EDA_TIER_ARGS`，默认 `-lean`）。这里只把它**打出来**，
# 免得判题机日志里看不出这一轮跑的是哪个档；改档就改这个变量（或改 Dockerfile）。
echo "EDA_TIER_ARGS (档位) = ${EDA_TIER_ARGS:-<unset，用 agent_system 的默认 -lean>}"
echo "EDA_PUBLISH_S (增量发布) = ${EDA_PUBLISH_S:-<unset，默认 15>}"
echo "input files:"; ls -1 "$INPUT_DIR" | head -20

# ---------------------------------------------------------------------------
# 环境自检。这个脚本是 2026-10 为了「镜像里到底有没有 ultralytics」这一类问题
# 写的：第一次提交的 decision_log.json 里 10 张图全是同一条
#     ModuleNotFoundError: No module named 'ultralytics'
# 而当时日志里看不到任何线索。
#
# ⚠️ 放在**运行期**而不是 build 期：build 期一旦自检报错，docker build 整个失败，
#    一次提交机会就白扔了；运行期自检最坏也只是多打几行日志。
# ⚠️ 结果必须出现在**平台收集的 stdout** 上，所以不加 `|| true` 吞掉退出码，
#    只用 `; true` 保证它**不影响**后面真正的流程。
# ---------------------------------------------------------------------------
echo ""
echo "=== 环境自检（verify_imports.py）==="
PYTHONPATH="/submission/eda-pcb-main:/submission" \
  python3 /submission/verify_imports.py 2>&1 || true
echo "=== 自检结束，进入正式流程 ==="
echo ""

# stdout 直接进平台日志；stderr 先落 run.log 再原样转出来。
# 不用管道 —— dash(/bin/sh) 下管道会让 $? 变成 tee 的退出码，
# python 崩了却报 exit 0，平台就看不到失败原因。
# `-u` 关掉 python 缓冲，否则平台日志可能一直空到进程结束。
LOG="$OUTPUT_DIR/final_results/run.log"
python3 -u /submission/eda-pcb-main/submission/agent_system \
  -input "$INPUT_DIR" -output "$OUTPUT_DIR" 2> "$LOG"
EXIT_CODE=$?
cat "$LOG" >&2

if [ ! -s "$LOG" ]; then
  echo "(run.log 为空)" > "$LOG"
fi

echo "=== run.sh exit $EXIT_CODE ==="
exit $EXIT_CODE
