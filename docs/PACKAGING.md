# 打包与交付（怎么上传 / 怎么把判题机单独给队友）

> 面向两件事：**把仓库推上 GitHub**，以及**只把判题机交给队友自评**。
> 判题机的用法细节见 `docs/JUDGE.md`；分数轨迹见 `docs/RESULTS.md`。
> 本文写于 2026-10-14，仓库状态：`main` 领先 `origin/main` **178** 个提交，远端 `https://github.com/SilinSmith/eda-pcb.git`（**private**）。

---

## 0. 三条交付线，先选一条

| 交付线 | 适合 | 产物 | 耗时 |
|---|---|---|---|
| ① 推 GitHub | 队友要一起改代码 | 整个仓库（含 220 个 GT 标注、3 个权重） | 首次 ~170 MB 上行 |
| ② 判题机单独交付 | 队友只想自己打分 / 复核我们的分数 | 几十 KB 纯 Python + 11.5 MB GT json | 秒级 |
| ③ Docker | 平台提交 / 复现推理环境 | `eda-pcb-main` 目录 + `run.sh`（或镜像） | 视基础镜像 |

---

## 1. 推 GitHub（推荐路径）

```powershell
cd Y:\workspace\EDA\eda-pcb

# ① 先确认没有脏东西（见 §2 的 4 条自检）
git status -sb                  # 期望：## main...origin/main [ahead 178]，且无 ?? / M

# ② 提交本轮改动（GT 标注、docs/JUDGE.md、Dockerfile 去密钥…）
git add -A
git commit -m "chore: 交付判题机与 GT 标注，清理硬编码密钥"

# ③ 推送：local 领先、远端 0 behind ⇒ 纯 fast-forward，不会产生 merge
git push origin main
```

- **凭据**：本机已配 `credential.helper=manager`（Windows 凭据管理器里存了 token），`git ls-remote origin` 匿名也能跑。换机器请用 PAT（`git remote set-url origin https://<token>@github.com/SilinSmith/eda-pcb.git` 或直接在弹出的窗口里登录）。
- **仓库是 private**：匿名访问 API 返回 404，队友必须先被 owner 加进仓库（Settings → Collaborators），否则 clone 会 404。
- 推完建议打个 tag，方便队友对齐"84.95 那一版"：

```powershell
git tag -a v0.1-official10 -m "官方 10GT 交付口径 84.95（确定性口径 80.48）"
git push origin v0.1-official10
```

- **没有可用外网 / 只想拷 U 盘**时，两种离线打包：

```powershell
# 带完整历史的单文件（队友 git clone eda-pcb.bundle eda-pcb 即可）
git bundle create ..\work\eda-pcb.bundle --all

# 只要一份当前源码快照（无历史、无 .git，最适合"给我一份代码"）
git archive --format=zip -o ..\work\eda-pcb-src.zip HEAD
```

---

## 2. 上传前自检（4 条，照抄）

```powershell
cd Y:\workspace\EDA\eda-pcb

# ① 密钥：必须无命中（历史里的旧值见 §3）
git grep -I -n -E "sk-ws|sk-[A-Za-z0-9]{20,}" -- . | Select-Object -First 20

# ② 数据集：未跟踪文件里只应有 GT 的 json + type_list.md（不该有 png）
git status --porcelain -uall | Where-Object { $_ -match '^\?\?' } |
  ForEach-Object { $_.Substring(3) } | Group-Object { ($_ -split '\.')[-1] } |
  Format-Table Count, Name -AutoSize

# ③ 判题机自检：220 例全部跑过一遍口径
python -X utf8 -m schematic_parser.evaluate --self-test

# ④ 口径复核：这一条必须复现 RESULTS.md 里记录的分数
python -X utf8 -m schematic_parser.evaluate --pred ..\work\_fast_out_sjndef\final_results --split 10GT
```

② 的期望值：`json` 220 个、`md` 1 个（`10GT/type_list.md`）；**出现 `png` 就说明 `.gitignore` 被改坏了**（`.gitignore` 里 GT 白名单的三条例外就是为这个留的）。

④ 的期望值（交付口径，`-vlm-pin off` 的 `sjndef` 那一版）：

```
component_f1 1.000000   pin_f1 0.816599   net_f1(micro) 0.820979   line_f1 0.579579
mean_score   84.945
```

---

## 3. ⚠️ 密钥红线（**push 前必须先做掉**）

**事实**：`packaging/Dockerfile:20-21` 曾**硬编码**一枚 DashScope key
（`ENV LLM_API_KEY="sk-ws-…"` / `ENV DASHSCOPE_API_KEY="sk-ws-…"`），由**未推送**提交
`a6cac45`（"fix: 打包链本机可修项…"）带进来。2026-10-14 已从工作树清掉（改成空默认值 + 注释），
`git grep sk-ws origin/main` 为空 ⇒ **远端树从未出现过密钥**。

**旧值仍活着（实测，不是推测）**：`work/_keyprobe.py` 把 `a6cac45` 里的 key 抽出来打
`GET https://dashscope.aliyuncs.com/compatible-mode/v1/models` ⇒ **HTTP 200**；
对照组 `work/_keyprobe_control.py` 用伪造 key ⇒ **401** `Incorrect API key provided`，
空 key ⇒ 401 `You didn't provide an API key` ⇒ 端点确实鉴权，泄漏的那枚确实可用。

**它还在 178 个未推送提交里**：一旦 `git push`，旧值就永久进入远端历史（仓库是 private，
但协作者可见、且日后可能公开）。两条路，**先做掉再推**：

1. **作废并重发该 key**（推荐，唯一真正的补救）：百炼控制台
   <https://bailian.console.aliyun.com/cn-beijing/model/settings/api-key> ——
   注意**右上角要选对地域**（每个地域的 API Key 与 endpoint 域名独立、不能混用），
   并确认 key 的**归属**（主账号 / RAM 用户、哪个业务空间）；删除**立即失效且不可恢复**，
   API Key 本身没有有效期；单账号+单地域上限 50 个。也可用 OpenAPI
   `DeleteApiKey`（ModelStudio `2026-02-10`，
   <https://api.aliyun.com/meta/v1/products/ModelStudio/versions/2026-02-10/apis/DeleteApiKey/api>），
   但需要阿里云 **AK/SK**（本机没有：无 `aliyun` CLI、`ALIBABA_CLOUD_ACCESS_KEY_ID` 为空、
   `~/.aliyun/config.json` 与 `~/.alibabacloud/credentials` 均不存在 ⇒ **只能由有控制台权限的人做**）。
   作废后直接 `git push`，历史里留着旧字符串也无害。
2. **首次 push 前重写历史**（只有在该 key 无法作废时才考虑）：`git filter-repo --replace-text
   <规则文件>` —— 代价：`git rev-list --count a6cac45..HEAD` = **146**，连同 `a6cac45` 共
   **147 个 SHA 会变**，其中 **18 个被文档以 7 位 SHA 引用**（`docs/RESULTS.md` §5.x、
   `docs/reports/官方评测机_第5次提交归因.md:108` 与 `:192` 直接引用 `a6cac45`），要逐条改文档；
   而且 **`git filter-repo` 本机未安装**（既非 git 子命令也无 python 模块），得先 `pip install
   git-filter-repo`，或退回内置 `git filter-branch`（更慢更易错）。**不推荐。**

**顺带**：`.lmstudio_key`、`*.key`、`.env` 均未被跟踪；全树（排除 `docs/RESULTS.md` 与 GT 数据）
用 `(api[_-]?key|token|password|secret)\s*[:=]\s*["'][A-Za-z0-9_\-\.]{16,}` 扫过，零命中。
**队友自填项**见 `README.md` §3.1；容器注入时注意官方示例给的是 `DASHSCOPE_API_KEY` 而本仓库代码读
`LLM_API_KEY`，`packaging/run.sh` 已加别名兜底（只在 `LLM_API_KEY` 为空时映射，不覆盖显式值）。

---

## 4. 判题机单独交付（队友只想打分）

判题机 = `schematic_parser/`，**只依赖 numpy + scipy**（`schematic_parser/metrics.py:42` 用 `scipy.optimize.linear_sum_assignment` 做 bbox / pin / line 三处匈牙利匹配），不 import torch / cv2 / ultralytics，所以可以脱离主链路单独打包。

```powershell
cd Y:\workspace\EDA\eda-pcb
$dst = "..\work\_judge_pkg"
Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dst | Out-Null

Copy-Item -Recurse -Force schematic_parser $dst          # 0.19 MB
Copy-Item -Force docs\JUDGE.md $dst

# ⚠️ 只拷 *.json：数据集目录里同时躺着 PNG（200_train_cases 光 png 就 44 MB），
#    整目录 Copy-Item 会把它们一起打进交付包。robocopy /S 保持用例目录结构。
$src = "yolo_part\Dataset(waitforpreprocess)"
foreach ($s in @("10GT","10_GTcase","200_train_cases")) {
  robocopy "$src\$s" "$dst\$s" *.json /S /NFL /NDL /NJH /NJS /NP | Out-Null
}
Copy-Item -Force "$src\10GT\type_list.md" "$dst\10GT\"

# 顺手清掉字节码缓存，别打进包
Get-ChildItem -Recurse -Directory -Filter __pycache__ $dst | Remove-Item -Recurse -Force

Compress-Archive -Path "$dst\*" -DestinationPath ..\work\eda_judge.zip -Force
# ⇒ 产物 ≈1.0 MB（220 个 GT json 未压缩 11.45 MB，json 压得很狠，最大单文件 212,777 B）
# ⇒ 解压后：10GT 11 文件 / 10_GTcase 10 / 200_train_cases 200 / schematic_parser 6 / JUDGE.md，
#   **0 张 png**；用 84.95 那版预测从解压包跑 `--data-root <解压目录>` 复现出同样的 84.945。
```

**队友侧三步跑通：**

```powershell
pip install "numpy>=1.26" "scipy>=1.11"

# 自检（220 例，会把 GT 与自身比一遍，用来确认环境没问题）
python -X utf8 -m schematic_parser.evaluate --self-test

# 打分：--data-root 指向 GT 根目录（默认指仓库内的 yolo_part/Dataset(waitforpreprocess)）
python -X utf8 -m schematic_parser.evaluate --data-root . --pred <预测目录> --split 10GT
```

- `--pred` 目录里放每个用例一个 json：文件名 `NNNN_Pred.json`（也兼容 `NNNN.json`、去零写法），细节见 `docs/JUDGE.md`。
- **给队友一份"标准答案"预测**，让他先复现我们的 84.95 再信这个工具：把 `work\_fast_out_sjndef\final_results` 一并给出（或任何一份 `NNNN_Pred.json` 目录）。
- 判题机的默认口径（`schematic_parser/metrics.py:84-129` 的 `EvalConfig`）：`bbox_tol=20.0`、`pin_tol=5.0`、`line_tol=5.0`、`bbox_mode='corners'`、`net_mode='micro'`、`net_soft=True`、`unkeyed_match='gt'`、`coord_gate=40.0`、`pin_box_only=True`、`net_pin_collapse_nonbox=False`。这些都是**按官方判分器逐位校准过**的，别随手改。
- 已知性质：**空预测仍有 25.0000 分**——因为 GT 里 7 个 `10_GTcase`/`200_train_cases` 派生用例没有 `box` 元件，此时 `pin_f1` 退化恒等 1.0（官方 10GT 不受影响）。看到"乱交也有 25 分"不要以为是 bug，解释在 `docs/JUDGE.md`。

---

## 5. Docker（要提交平台或复现环境时）

`packaging/` 里四个文件：`Dockerfile`(16.2 KB)、`install_deps.py`(15.2 KB)、`run.sh`(3.5 KB)、`verify_imports.py`(4.9 KB)。

- 镜像链：`FROM edarace-agent-base:ubuntu24` → `WORKDIR /submission` → `COPY install_deps.py` → `ENV PIP_CONSTRAINT=/submission/constraints.txt` → `COPY verify_imports.py` → **`COPY eda-pcb-main /submission/eda-pcb-main`** → `COPY run.sh` → `ENTRYPOINT ["/submission/run.sh"]`。
- ⚠️ `edarace-agent-base:ubuntu24` 是**自建基础镜像**，队友本地没有就没法直接 `docker build`：要么先拿到 base，要么跳过 Docker 用 `pip install -r requirements.txt` 直接跑。
- ⚠️ 构建上下文里仓库目录**必须叫 `eda-pcb-main`**（Dockerfile 就是这么 COPY 的）：
  ```powershell
  # 在仓库父目录下，把 eda-pcb 拷/软链成 eda-pcb-main 再 build
  docker build -f eda-pcb-main\packaging\Dockerfile -t eda-submit .
  ```
- 用法：`run.sh -input <输入目录> -output <输出目录>`，内部 `PYTHONPATH=/submission/eda-pcb-main:/submission python3 -u /submission/eda-pcb-main/submission/agent_system`。
- 容器坑（Dockerfile 注释里已列）：scipy 必须自己装；缺 `libGL.so.1` 会让 cv2 直接炸；交付路径只用标准库 `urllib.request`，别指望容器里有 `httpx`。

---

## 6. 体积与文件事实（别被 4.7 GB 工作树吓到）

- 仓库工作树 4,775.9 MB，但 `.git` 只有 **171.2 MB**；**只有入库文件**会被推送（`work/`、`outputs/`、`submissions/`、数据集与 PNG 全部被 `.gitignore` 挡掉）。
- 入库 1040 个文件；其中 3 个 >20 MB 的权重：`weights/pcb43_v2_from_best.pt` 38.8 MB、`weights/textrec_best.pt` 29.6 MB、`yolo_part/weights/yolo11m.pt` 38.8 MB——均低于 GitHub 100 MB 单文件硬上限，未用 Git LFS。
- 本轮新增入库：`yolo_part/Dataset(waitforpreprocess)/{10GT,10_GTcase,200_train_cases}/**/*.json` 共 **220 个 / 11.5 MB / 最大单文件 212,777 B**，外加 `10GT/type_list.md`（34 类）。**GT 的 PNG 一律不入库**（判题只用 json）。
- 为了让"只有 json 的目录也算一个用例"，`schematic_parser/data.py:discover()` 已放宽：`if not pngs and not jsons: continue`，并把 `png_path` 置为 `None`。已用只含 json 的 GT 树（221 个文件、0 张 png）复现出同样的 84.945，确认队友只拿 json 也能判分。

---

## 7. 一句话清单（要发给队友时抄这段）

```
仓库：https://github.com/SilinSmith/eda-pcb （private，需要 owner 加你为 collaborator）
自评判分：
  pip install "numpy>=1.26" "scipy>=1.11"
  python -X utf8 -m schematic_parser.evaluate --pred <你的预测目录> --split 10GT
判题机说明书：docs/JUDGE.md      默认口径表、pred 文件命名、已知性质
我这版的参考分：84.945（= 1.0000 元件 / 0.8166 引脚 / 0.8210 网络 / 0.5796 导线）
```
