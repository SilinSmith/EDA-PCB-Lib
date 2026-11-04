# 2026-11-04 仓库历史重写说明（务必先读再 pull）

本轮对 `main` 做了一次**历史重写 + 强制推送**。旧提交哈希全部作废，远端 `main` 已替换。
本文记录原因、做了什么、影响了谁、怎么恢复。

---

## 一、为什么重写

| 事实 | 数值 |
|---|---|
| `origin/main` 上原有最新提交 | `77b9f93`（2026-10-18） |
| 本地 `main` 超前 | **275 个 commit**、1842 个对象 |
| 待推送的新增 blob | 约 **970 MB** |
| 实测上行速度 | **~50 KiB/s**（偶发 0.6–1 MiB/s） |

体积集中在三处：

1. `docs/RESULTS.md` 有 **407 个历史版本**（原始累计 744 MB，单次最新约 3 MB）；
2. 历次训练权重：`weights/textrec_best.pt` 就有 **5 个版本 = 148 MB**，另有 `textrec_h64`、`sym16_1280_best`、`yolo11s`、`yolo11m`、`pcb43_v2_from_best` 等；
3. `docs/media/*.gif|mp4`（`loss_case1.gif` 7.5 MB 等）。

按原样推送需要数小时，且中途断网会整包重传（实际发生过一次：110.68 MiB 传完、
最后一步 `fatal: the remote end hung up unexpectedly`，远端未更新）。

---

## 二、做了什么

```bash
# 1) 安全备份（在仓库目录之外，不入库）
git bundle create Y:\workspace\EDA\eda-pcb-full-20261104.bundle --all   # 332 MB

# 2) 历史重写：删掉所有 >4MB 的 blob（权重、gif/mp4、glyph_gallery.pkl、旧 RESULTS 大版本…）
"Y" | git filter-repo --strip-blobs-bigger-than 4M --force

# 3) 只补回当前运行必需的三个二进制，单独一个 commit
git add weights/pcb43_v2_from_best.pt weights/textrec_best.pt pipeline/glyph_gallery.pkl
git commit -m "assets: keep only current runtime binaries after history strip (>4MB blobs removed from history)"

# 4) 强制推送
git push --force origin main
#   + 77b9f93...d95bcdd main -> main (forced update)
```

结果：

| | 重写前 | 重写后 |
|---|---|---|
| `.git` 体积 | 354.8 MB | **164.7 MB** |
| commit 数 | 687（解析）/ 669（`main`） | **669** |
| 推送体积 | ~970 MB | **110.68 MiB** |
| 远端 `main` | `77b9f93` | **`d95bcdd`** |

提交对应关系（重写会改哈希）：

| 旧 | 新 | 说明 |
|---|---|---|
| `427e0f8` | `48c8aa3` | 45 个默认开开关反扫，只留 `STAGE2_PIN_NUMBER_REPAIR=0` |
| —（新增） | `d95bcdd` | 重写后补回三个运行时二进制的资产 commit |

---

## 三、谁受影响、怎么做

**任何手上还留着旧 clone 的人，都不要直接 `git push`** —— 那会把 970 MB 的旧历史整包带回远端。

三种处理方式，任选其一：

```bash
# A. 最省事：重新 clone
git clone git@github.com:SilinSmith/eda-pcb.git

# B. 保留工作区改动：扔掉旧历史，对齐新 main
git fetch origin
git reset --hard origin/main          # ⚠️ 会丢弃未提交改动，先 git stash 或备份

# C. 想留下旧历史做参考（本地只读分支，别推）
git branch archive/pre-20261104 77b9f93
```

已经提交、还没推的本地分支，同样会与远端分叉；需要时用 `git rebase --onto origin/main <old-base>` 或直接重建分支。

---

## 四、现在仓库里有什么 / 没有什么

**保留（`d95bcdd` 的树）**

- 全部代码：`stage1/ stage2/ fusion/ pipeline/ submission/ schematic_parser/ tools/ tests/`
- 全部文档：`docs/*.md`（`RESULTS.md` 只保留**最新一版**，历史版本不再可 diff）、`README.md`
- 全部 GT 标注 `*.json`（11.5 MB）
- 三个运行时二进制：
  - `weights/pcb43_v2_from_best.pt`（38.8 MB，检测器）
  - `weights/textrec_best.pt`（29.6 MB，CRNN 识别器）
  - `pipeline/glyph_gallery.pkl`（11 MB，字形库）
- `packaging/`（`Dockerfile`、`run.sh`、`install_deps.py`、`verify_imports.py`）

**不在仓库里（本地保留即可，不要补推）**

- `yolo_part/weights/*.pt`（`yolo11m.pt` 等，本来就在 `.gitignore`）
- `weights/textrec_h64.pt`、`weights/sym16_1280_best.pt`、`yolo_part/weights/yolo11s.pt`
- `docs/media/*.gif|mp4`、`pipeline/glyph_gallery.pkl` 的历史版本
- GT 的 PNG、训练数据、`work/` 下的探针与跑批产物

---

## 五、如何恢复旧历史

完整旧仓库（含所有旧哈希、所有大文件版本）在：

```
Y:\workspace\EDA\eda-pcb-full-20261104.bundle
```

```bash
git clone Y:\workspace\EDA\eda-pcb-full-20261104.bundle eda-pcb-old
```

该 bundle 只在本机，**不入库、不上传**。确认新历史无问题后可自行删除。

---

## 六、以后怎么避免再出现 970 MB 待推

- `docs/RESULTS.md` 每次追加都会产生一个新版本；批量写文档后**先推一次**，别攒 275 个 commit。
- 大文件（`.pt`、`.pkl`、`.gif`、`.mp4`）如果必须入库，走 Git LFS；不入库就写进 `.gitignore` 并用打包/网盘传递。
- 推送前先量体积再决定：

```bash
git rev-list --objects origin/main..main | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob"{s+=$2} END{printf "new blobs: %.1f MB\n", s/1048576}'
```
