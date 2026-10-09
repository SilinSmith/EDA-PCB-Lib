FROM edarace-agent-base:ubuntu24

# ============================================================================
# 方式 3「Dockerfile-env-api」：API 端点/密钥/模型写在 ENV 里，代码从环境变量读。
#
# 队友代码实际读的变量（不是官方示例里的 DASHSCOPE_API_KEY）：
#     vlm/client.py:32/33/34    BASE / KEY / MODEL
#     stage1/llm.py:87/91/141   同三个变量
# 两个密钥变量名都写上，避免官方变量名与示例变量名不一致导致读不到。
#
# ⚠️ 代码用的是 os.environ.get(...)，所以**平台运行时注入的值会覆盖这里的 ENV**。
#    这里的 ENV 是兜底。想让模型名一定生效，必须改代码默认值（见 提交包说明.md 位置B）：
#        eda-pcb-main/vlm/client.py:34    ← 注意这里用的是 os.environ.get(K, 默认)，
#                                            变量存在但为空串时会返回空串，要改成 `or` 形式
#        eda-pcb-main/stage1/llm.py:91    ← 这个已经是 `or` 形式，安全
# ============================================================================

ENV LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
ENV LLM_MODEL="qwen3.7-plus"
ENV LLM_API_KEY="REDACTED-DASHSCOPE-KEY"
ENV DASHSCOPE_API_KEY="REDACTED-DASHSCOPE-KEY"

# ----------------------------------------------------------------------------
# ★ 这一行是必须的，别删。
#
# vlm/client.py:37  REASONING = os.environ.get("VLM_REASONING", "none")
# vlm/client.py:141-142
#     if REASONING:
#         payload["reasoning_effort"] = REASONING
#
# 也就是**不设这个变量时，默认就会带 `reasoning_effort: "none"`**。
# 而阿里云百炼的 OpenAI 兼容接口对未知字段是严格的，很可能直接 400
# ⇒ `probe()` 失败 ⇒ VLM 整条链路被静默跳过（白丢约 3 分且不报错）。
#
# 置成空串后 `if REASONING:` 为假，这个字段就不会被发出去。
ENV VLM_REASONING=""

WORKDIR /submission

# ============================================================================
# ★★★ 依赖安装 —— 第一次提交就是死在这一步。
#
# 第一次提交（2026-10-09 09:53）的 decision_log.json 里，10 张图全是同一条：
#     "error": "ModuleNotFoundError: No module named 'ultralytics'"
# 基础镜像 edarace-agent-base:ubuntu24 **不含 ultralytics**，必须自己装。
# 官方示例 Dockerfile-env-api 本来就有一行 `RUN pip3 install ...`，
# 只是它装的是它自己用到的 openai 包。
#
# 要装什么是由**基础镜像实际有什么**倒推的，不是拍脑袋。
# 我把 edarace-agent-base_ubuntu24.tar（OCI 格式）逐层扫了一遍
# /usr/local/lib/python3.12/dist-packages/，拿到全部 44 个已装包，结论：
#
#   ultralytics   ❌ 缺 —— YOLO 检测器，stage1 命门
#   scipy         ❌ 缺 —— linear_sum_assignment / cKDTree，
#                          stage1 与 stage2 都用到
#   cv2           ✅ 已有（opencv_python_headless-4.10.0.84），不用装
#   pillow        ✅ 已有（12.3.0）
#   PyYAML        ✅ 已有（6.0.2）
#   psutil        ✅ 已有（7.2.2）
#   requests      ✅ 已有（2.32.3）
#   numpy         ✅ 已有（2.5.3）
#   torch         ✅ 已有（2.5.1+cu121，CUDA 版）
#   torchvision   ✅ 已有（0.20.1+cu121）
#
# ⇒ 真正要下的：ultralytics 本体（--no-deps，1.42 MB）
#   + scipy / matplotlib / nvidia-ml-py / ultralytics-thop / ultralytics-platform
#     （约 51 MB，含它们自己那几个小依赖）≈ **52 MB**。
#   之前那一版装了 opencv-python（62.5 MB）/ polars（54.5 MB）/ pillow /
#   PyYAML / psutil / requests / httpx，全是白下的，已删。
#
# ⚠️ 四个坑：
#   0) **必须换国内镜像源**。第二次提交（2026-10-09）build 进到 Step 11 炸了：
#        Downloading scipy-1.18.1-...whl (35.3 MB)
#          7.3/35.3 MB 26.2 kB/s eta 0:17:50
#        pip._vendor.urllib3.exceptions.ReadTimeoutError:
#          HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
#      平台直连 files.pythonhosted.org 实测只有 14~60 kB/s（本机实测
#      pypi.org 索引页 3 kB/s）。而阿里云镜像实测 **287 kB/s**（快 6 倍）、
#      清华 791 kB/s ⇒ 现在这 52 MB 实测约 2~4 分钟下完（第三次提交日志里
#      阿里云实测 309 kB/s 下 scipy、187 kB/s 下 opencv）。
#      阿里云用 **HTTP**（http://mirrors.aliyun.com/pypi/simple/），
#      要配 --trusted-host；清华走 HTTPS 不需要。三个源依次兜底写在
#      install_deps.py 的 MIRRORS 里。
#   1) `--break-system-packages` 是必须的：ubuntu24 是 PEP668 externally-managed
#      环境，不加这个 flag pip 会直接拒绝安装（官方示例也是这么写的）。
#   2) **绝不能让 pip 去动 torch**。基础镜像里的 torch 是 CUDA 版（阶段二已经
#      成功在 cuda:0 上加载了 CRNN 权重），被换成 PyPI 的 CPU 版就会让整条链路
#      掉到 CPU 上跑，20 分钟预算直接爆。ultralytics 声明依赖 torch，所以这里
#      **从基础镜像现装版本生成约束文件**（不手写版本号），再用 PIP_CONSTRAINT
#      钉住。`===` 是 pip 的「任意相等」，能把 `2.13.0+cu130` 这种带本地版本号
#      的写法原样钉死（`==` 对局部版本号的匹配规则不完全一致）。
#   3) 约束是**全局**的：PIP_CONSTRAINT 走环境变量而不是命令行 flag，
#      这样 build 期和运行期新起的 pip 都吃得到。
# ============================================================================
# ---------------------------------------------------------------------------
# ★★ 装依赖：交给 install_deps.py 三档自动降级（脚本里有完整理由）。
#
# 为什么不写成 Dockerfile 里一行 pip：
#   第二次提交 build 到 Step 11 炸在 pip 下载超时（平台直连
#   files.pythonhosted.org 只有 14~60 kB/s）。换国内镜像解决带宽，
#   但如果同时把「下载问题」换成「约束问题」，仍然是白扔一次提交。
#   脚本会依次试：严格约束(===) → 宽松约束(区间) → 无约束，
#   每换一个源再试一遍，只有全失败才让 build 红。
#   约束那四个版本号是脚本从**基础镜像现装版本**现推的，不是手写的。
#
# ★★★★ 第五次改动的核心：不再让 pip 解析 ultralytics 的依赖树。
#
#   问题：ultralytics 8.4.174 的 METADATA（我下载 wheel 读出来的原文）声明
#       Requires-Dist: opencv-python!=4.13.0.90,>=4.7.0
#       Requires-Dist: polars>=0.20.0
#       Requires-Dist: matplotlib>=3.3.0
#   而基础镜像装的是 opencv_python_headless-4.10.0.84（36 MB 的 wheel）。
#   pip **不认为 headless 能顶替 opencv-python**，于是它真去下了 62.5 MB 的
#   普通版；polars 又拖来 54.5 MB 的 polars-runtime-32。这两样加上 scipy，
#   就是 170 MB 的下载量 —— 第四次提交 30 分钟 build 超时的直接来源。
#
#   三次提交的血泪史（每一次都是实测，不是推断）：
#     第二次：平台直连 files.pythonhosted.org 只有 14~60 kB/s，
#             scipy 下到 7.3/35.3 MB 就 ReadTimeoutError，build 炸。
#     第三次：换阿里云源成功（309 kB/s），但 pip 装上了**普通版** opencv-python，
#             dist-packages/cv2/ 被换成需要 libGL.so.1 的那份，于是
#                 !!! 致命: ultralytics/scipy 导入失败: libGL.so.1:
#                     cannot open shared object file: No such file or directory
#             （我上一版于是加了 apt 装 libGL —— 那是治标，也治错了。）
#     第四次：apt 段花掉 800+ 秒，撞 30 分钟 build 上限被杀。
#
#   ⇒ 正确解法：**用 `--no-deps` 装 ultralytics 本体，再把真正需要的
#     依赖一个一个显式列出来**（清单见 install_deps.py）。这样 pip 不会去
#     碰 opencv-python 和 polars，基础镜像里那份 headless cv2 原封不动，
#     不需要 libGL，也就不需要 apt —— 下载量从 170 MB 降到约 52 MB。
#
#   为什么"显式列依赖"是对的（不是偷懒）——用 AST 把 ultralytics 全部
#   231 个 py 文件按"导入期是否执行"分成两类（模块级 try/if 也算执行，
#   所以不能靠缩进猜），得到上界：
#       cv2       29 个文件在导入期 / 4 个在函数内   ⇒ **必须要**（headless 可顶）
#       torch    129/4  numpy 85/4  PIL 17/1         ⇒ 基础镜像全有
#       polars     0 个在导入期 / 5 个在函数内       ⇒ **导入期不需要**
#       matplotlib 0 个在导入期 / 9 个在函数内       ⇒ **导入期不需要**
#       scipy      0/1   pandas 0/0   yaml 0/1       ⇒ scipy 由我们自己代码要
#   并且 `ultralytics/__init__.py` 对模型是**懒加载**（`def __getattr__`
#   里才 `importlib.import_module("ultralytics.models")`），
#   `from ultralytics_platform import ...` 只在 `if TYPE_CHECKING:` 下，
#   全文没有 `checks()` 调用 ⇒ `import ultralytics` 的导入期闭包只有 8 个模块。
#   还核过全部 `check_requirements(...)` 调用点：**没有任何一处要求
#   opencv-python**（唯一提到 opencv 的是 utils/export/ncnn.py:37，
#   作者还特意加了 `cmds="--no-deps"` 并注释 "no deps to avoid installing
#   opencv-python"）⇒ 不存在"偷偷自动装 opencv-python"的路径。
#
# ★ 绝不能让 pip 去动 torch：基础镜像里是 CUDA 版 2.5.1+cu121（第三次提交实测
#   torch.cuda.is_available() 为 True），换成 PyPI 的 CPU 版会让整条链路掉到
#   CPU 上跑。约束文件从**基础镜像现装版本**现推（不手写版本号），
#   用 PIP_CONSTRAINT 走环境变量全局生效。`===` 是 pip 的"任意相等"，
#   能把 `2.5.1+cu121` 这种带本地版本号的原样钉死。
#
# ★ 依赖冲突已离线验证过（下载 ultralytics wheel 读 METADATA +
#   逐条对照基础镜像 44 个已装包）：与 torch 2.5.1+cu121 /
#   torchvision 0.20.1+cu121 / numpy 2.5.3 无冲突。
#   第三次提交实测：全部装成功，torch.cuda.is_available() 仍为 True。
# ---------------------------------------------------------------------------
# ★★★★ 第五次改动：apt 段**整段删除**。根因是第四次提交 build 超时。
#
#   第四次提交（2026-10-09 19:51，平台 ID 4702）错误日志只有一行：
#       [Ingestion] ERROR: Image build timeout (30 minutes)
#   平台 metadata 原文：
#       ingestionElapsedTime = 1801.4480867385864 s（= 30.02 分钟，撞上限被杀）
#       elapsedTime          = 301.19566321372986
#   对比**第三次提交**（没有 apt 那一段）：
#       ingestionElapsedTime = 986.7940814495087 s = 16.4 分钟，按时走完
#   ⇒ apt 那一段自己就花了 800+ 秒，把 30 分钟预算顶穿；
#     而 pip 那一段单独就吃掉 16.4 分钟（预算的 55%），两处都得治。
#
#   治 apt 的办法不是"想办法让它快"，而是**根本不需要它**：
#     headless 版 cv2 不链接 libGL。我把基础镜像第 87.3 MB 那层里的
#       usr/local/lib/python3.12/dist-packages/cv2/cv2.abi3.so  (65.7 MB)
#     抽出来，扫它引用的全部 lib*.so，结果原文：
#       引用的 lib*.so: libOpenCL.so, libavcodec-<hash>.so.59.37.100,
#           libavformat-<hash>.so.59.27.100, libavutil-<hash>.so.57.28.100,
#           libc.so.6, libdl.so.2, libgcc_s.so.1, libittnotify.so, libm.so.6,
#           libopenblas-<hash>.so, libpng16-<hash>.so.16.43.0, libpthread.so.0,
#           librt.so.1, libstdc++.so.6, libswscale-<hash>.so.6.7.100, libz.so.1
#       含 libGL.so.1 ?          False   ← 决定性的两行
#       含 libgthread-2.0.so.0 ? False
#     它自带的 ffmpeg / openblas / png / z 都是带 hash 后缀的私有 .so，全在同一个
#     dist-packages/cv2/ 目录里，不依赖任何系统库。
#     ⇒ 只要让 pip **别把 headless 换成普通版**，就完全不需要 apt。
#       办法是 install_deps.py 里的 `--no-deps`（不是"再补装 libGL"）。
#
#   顺带两条实测结论（都是从基础镜像里扫出来的，不是推断）：
#     · var/lib/apt/lists 是个**空目录**（0 B）⇒ 将来真要用 apt，
#       `apt-get update` 是省不掉的，躲不过那个网络往返。
#     · 镜像里 libGL.so / libGLX.so / libGLdispatch / libglib-2.0.so /
#       libgthread-2.0.so 命中数**全是 0** ⇒ 早就确认它们确实缺，
#       只是现在知道"缺了也不影响"，因为不该走这条路。
#
#   （历史留档：我最早那一版注释写的是"把 opencv-python 钉成基础镜像里那个
#     headless 的同一版本号，覆盖文件逐字节一致，几乎零成本"——
#     **这个推理是错的**。同版本号的普通版比 headless 版多一条对 libGL 的
#     动态依赖，版本号对齐保证不了二进制一致。第三次提交就是死在这句话上。）

COPY install_deps.py /submission/install_deps.py
ENV PIP_CONSTRAINT=/submission/constraints.txt
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
RUN python3 /submission/install_deps.py
# 装完立刻自检 —— 但放在**运行期**（run.sh 里），不放在 build 期。
# 理由：build 期自检一旦报错，docker build 整个失败，一次提交机会就白扔了；
# 运行期自检最坏也只是多打几行日志，而且结果会出现在平台收集的日志里。
# 脚本本身仍然要 COPY 进去（run.sh 会调用它）。
COPY verify_imports.py /submission/verify_imports.py

# 整棵目录树一起拷进去：assets.py:24 用 Path(__file__).resolve().parent 定位
#   weights/pcb43_v2_from_best.pt   (YOLO 元件检测)
#   weights/textrec_best.pt         (CRNN 文字识别)
# 目录结构一错两个权重都找不到，stage1 第一行就 FileNotFoundError，整批全灭。
# 所以不要摊平、不要改名。
COPY eda-pcb-main /submission/eda-pcb-main
COPY run.sh /submission/run.sh
RUN chmod +x /submission/run.sh

# 把真正的 agent 也放到官方文档指定的那个绝对路径上。
#
# 基础镜像自带一个**占位**脚本 /submission/agent_system（65 行 bash，
# 末尾直接 exit 10），它的注释原话就是
#   "Participants must replace /submission/agent_system with their own agent."
# 而官方 README.md:33 / :76 说的是
#   "You must replace /submission/agent_system with your own agent implementation
#    (keep the same CLI flags and output layout)."
#   "Provide /submission/agent_system as the entrypoint, accepting
#    -input <dir> -output <dir>."
#
# 我们实际走的是 ENTRYPOINT run.sh（它再调 agent_system），功能上等价；
# 但既然官方两处都点名了这个路径，就把它补上——镜像里用
#   docker run <img> -input /input -output /output
# 手测时也会走对地方，不会撞上那个 exit 10 的占位脚本。
#
# ★ 用 cp 而不是第二个 COPY：我们的容器里必须且只能有一个 agent_system
#   副本，绝不能出现"更新了一个、跑的是另一个"。这个文件是 1236 B 的
#   启动壳（把 HERE 和它的父目录塞进 sys.path 后 from run import main），
#   两份内容原本就一样。
RUN cp /submission/eda-pcb-main/submission/agent_system /submission/agent_system \
 && chmod +x /submission/agent_system \
 && ls -l /submission/agent_system

# vLLM 这次不加：容器里自起推理服务会和 YOLO/CRNN 抢显存
# （submission/run.py:162 注释 + 默认 -gpu-frac 0.45），且会让镜像胖 3 倍。
# 平台若已提供 LLM 端点，自起服务是降级。

ENTRYPOINT ["/submission/run.sh"]
