# WSL 开发与实验环境

新实验使用局域网 WSL。源码通过 `git@github.com:BabyDrangoner/duet-vln.git` 管理；数据集、模型权重、运行输出和本机凭据保存在 Git 之外。历史 Colab 结果保持其原有实验身份。

## 目录和连接

当前部署约定：

- SSH：`ssh -p 2222 xxl@192.168.0.100`
- 源码：`/home/xxl/projects/duet-vln`
- 持久备份：`/home/xxl/vln-backups`
- 数据：仓库下 `datasets/`；环境与运行记录：`.venv/`、`outputs/runtime-wsl/`。

2026-10-06 连接检查发现 Ubuntu 24.04、RTX 4060 Laptop **8 GiB 显存**。后续实验按实测显存规划。代码和数据建议放在 WSL 的 Linux 文件系统中。

## 准备环境

已具有 Python 3.11、编译器及系统依赖时，在仓库目录运行：

```bash
bash scripts/setup_wsl.sh --python python3.11 --smoke
```

当前机器可使用无需 sudo 的路径：

```bash
cd /home/xxl/projects/duet-vln
bash scripts/setup_wsl.sh --bootstrap-python --user-native-deps --smoke
```

`--bootstrap-python` 在 `outputs/runtime-wsl/bootstrap/` 创建临时 Python 环境，通过 PyPI 安装 `uv==0.12.5`，再安装 Python 3.11.16。它需要系统 `python3-venv`，不会修改系统 Python。`--user-native-deps` 使用系统 APT 源下载缺失的 GLM、OSMesa 及 APT 解析的依赖，解包至 `outputs/runtime-wsl/native/`；不会安装系统包。当前机器已有编译器、CMake、JSONCPP 和 OpenCV。

如果系统依赖需要管理员安装，显式指定 `--install-system` 才会执行 APT 安装；脚本列出的依赖为 `build-essential cmake pkg-config git libjsoncpp-dev libglm-dev libosmesa6-dev libopencv-dev`。安装 Python 3.11 的方式也可以自行指定为 `--python /path/to/python3.11`。

准备脚本固定 PyTorch 2.5.1 CUDA 12.1、NumPy 1.26.4、Transformers 4.30.2；DUET 使用 `configs/upstream.json` 锁定的源码和补丁。MatterSim 使用提交 `589d091b111333f9e9f9d6cfd021b2eb68435925`，pybind11 使用 v2.13.6 提交 `a2e59f0e7065404b44dfe92a28aca47ba1378dc4`。完整 Python 包版本写入 `outputs/runtime-wsl/requirements-freeze.txt`。[PyTorch 官方旧版本安装表](https://pytorch.org/get-started/previous-versions/)、[pybind11 v2.13.6](https://github.com/pybind/pybind11/releases/tag/v2.13.6)。

WSL 的 GPU 驱动由 Windows 提供。脚本不安装 Linux NVIDIA 驱动，也不需要为当前预编译 PyTorch 安装完整 CUDA Toolkit。[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

网络需要能够访问 PyPI、PyTorch wheel 源、GitHub、Dropbox；首次 DUET 启动还会读取 Hugging Face 的 BERT 配置。安装脚本失败后可重跑；已有源码若存在未预期修改会停止并保留文件。

本次连接中 WSL 的外部域名解析超时，使用 Mac 已有的 `127.0.0.1:7897` 代理，通过 SSH 转发解决。需要时在 Mac 保持以下连接：

```bash
ssh -p 2222 -N -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:17897:127.0.0.1:7897 xxl@192.168.0.100
```

在另一个 WSL 终端为下载命令设置 `HTTP_PROXY=http://127.0.0.1:17897`、`HTTPS_PROXY=http://127.0.0.1:17897` 和小写 `http_proxy`、`https_proxy`。端口仅绑定 WSL 回环地址，转发依赖 Mac 代理和 SSH 连接；准备好的本地训练无需它。没有修改系统 DNS 或全局 Git 网络配置。

## 资源与校验

`configs/assets-manifest.json` 来自之前已验证的实验资产清单：预训练 DUET 约 2.17 GB、ViT 特征约 3.14 GB、R2R 标注及连接图，总计约 5.3 GB。SHA-256 和字节数保持不变，便于跨机器比较。

```bash
# 单独下载或接着下载
.venv/bin/python scripts/download_assets.py
# 完整校验已有资产，不下载
.venv/bin/python scripts/download_assets.py --verify-only
```

下载器只在字节数和 SHA-256 正确后把 `.downloading` 文件原子重命名为正式文件；支持服务器允许的 HTTP Range 续传。已有正式文件哈希不符会停止，避免静默覆盖。文件锁避免两个下载器同时操作同一目录。校验记录保存在 `outputs/runtime-wsl/assets-verification.json`。`--skip-assets` 可让准备脚本只配置环境；之后再单独下载。

## GPU 冒烟

`setup_wsl.sh --smoke` 会检查 CUDA 张量运算、MatterSim 导入和两个 `train_fit` 指令的零残差轨迹一致性；不会启动正式训练或访问 `val_unseen`。也可以在已准备的环境中单独运行：

```bash
source outputs/runtime-wsl/activate.sh
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/run_duet.py \
  --mode identity --split train_fit --limit 2 \
  --output "outputs/runtime-wsl/identity-$(date -u +%Y%m%dT%H%M%SZ).json"
```

激活文件会设置虚拟环境、MatterSim 模块路径、用户态 OSMesa 动态库路径和实验所需线程变量。开启新的 shell 后重新 `source` 即可。冒烟通过只说明运行环境和接入一致性通过，不能作为导航提升结论。

## 备份、续训与断线

`configs/pipeline_wsl.json` 是训练模板：先采集 `train_fit` 缓存，将 `cache` 改为实际缓存目录，并为新实验设置唯一 `run_id`。下面的八条指令冒烟给出缓存采集示例；该小缓存只用于工程验收，不能作为正式研究训练集。

配置好缓存后，明确创建持久目录并启动：

```bash
mkdir -p "$HOME/vln-backups"
bash scripts/train_wsl.sh --config configs/pipeline_wsl.json
```

`pipeline_wsl.json` 使用 `backup_backend: "filesystem"`、`backup_root: "~/vln-backups"`，关闭 Colab 运行时寿命上限。已有检查点策略继续保存模型、优化器、随机数和采样游标；先确认备份成功再清理本地旧检查点。WSL 的持久目录应位于实际磁盘，临时内存文件系统会被拒绝。同机目录能应对进程中断和本地运行目录清理，不能应对整块磁盘损坏；需要防此类故障时，把备份根目录换成已挂载的另一块盘或网络存储。

此入口运行现有残差训练流水线；E3 尚未完成可用训练头，不能把运行这个入口视为训练了 E3 新方法。E3 采集器接受 `--backup-backend filesystem --backup-dir /明确的持久目录`，该目录必须预先创建。迁移导致源码身份变化，应使用新运行目录；不能把新结果写入旧冻结实验的缓存。

先用小型训练缓存做续训检查：

```bash
source outputs/runtime-wsl/activate.sh
.venv/bin/python scripts/run_duet.py --mode collect --split train_fit --limit 8 \
  --cache outputs/wsl-cache-fit8 --output outputs/wsl-cache-fit8-report.json
.venv/bin/python scripts/smoke_resume.py \
  --run-id "wsl-resume-$(date -u +%Y%m%dT%H%M%SZ)" \
  --backup-backend filesystem --backup-root "$HOME/vln-backups" \
  --cache outputs/wsl-cache-fit8
```

长任务建议在 `tmux` 会话中启动，SSH 断开后会继续运行。Windows 关机或 WSL 被停止仍会终止进程；恢复时使用相同运行配置和备份目录，由检查点恢复。

同一个备份 run 只允许一个写入进程，不要从不同本地目录同时运行同一个 run。需要严格续训时使用 `bash scripts/train_wsl.sh --config <原配置> --require-resume`；找不到有效状态会报错。

## 代码同步

开发机提交后推送 Git，WSL 在没有运行任务且工作区干净时执行 `git pull --ff-only`。每次正式运行保存提交号、配置和依赖清单。数据和 checkpoint 不随代码推送；评估结论需要另行带上结果文件、资源哈希及划分记录。历史 E2/E3 结果见仓库研究报告，当前还没有达到 SR 增加 5 个百分点且 SPL 不下降的目标。
