# Colab T4 冒烟记录

日期：2026-10-03。远端会话：`vln-t4`。工程目录：`/content/project/vln`。

这次检查用于确认现有 DUET 实验工程能够连接 GPU、导入模拟器、执行模型和训练接口。论文中拟研究的多来源证据与到达监督方法尚未实现。

## 当前验证状态

| 检查 | 状态与范围 |
| --- | --- |
| GPU 环境 | Python 3.11.16、PyTorch 2.5.1+cu121、Tesla T4，约 15 GiB 显存 |
| 工程测试 | 64 项测试、48 项子测试通过 |
| GPU 模型接口 | 小随机 DUET 模型的语言编码、全景编码、导航评分、残差头前向和反向通过 |
| MatterSim | 使用真实 connectivity，完成离散视角转向和实际节点移动；关闭 RGB 渲染 |
| 数据与权重 | 已取得官方 R2R 标注、ViT 特征和 DUET 检查点，下载共 5,322,416,266 字节，约 5.322 GB |
| 8 条开发任务 | 原始策略与零初始化残差头的轨迹逐条完全一致 |
| 8 条训练任务 | 采集得到 44 条决策记录，CUDA 训练 1 个 epoch 通过，该阶段约 5 秒 |
| 重载与推理 | 训练后检查点重载成功，同一组 8 条开发任务推理及配对指标比较通过；SR/SPL 与基线相同 |

全部阶段已通过。小随机模型检查只能验证接口；本次真实任务仅来自训练集内部的 `train_fit` / `train_dev`，使用 1 个种子、训练 1 个 epoch，未评测官方验证集，不能据此报告论文中的导航收益。

这批开发任务的基线推理峰值 CUDA 已分配显存约 0.68677 GiB，加入残差头后约 0.68751 GiB。它们是本次小批次的 PyTorch 显存记录，不包含全部进程显存开销，也不能外推为完整数据、更多候选节点或新方法的训练需求。T4 的成功运行尚不能证明全部训练配置适合用户报告的 12 GB 显存机器。

## 从本机连接

先确认已有会话仍是 T4，再使用本次专用 SSH 配置连接：

```bash
~/.local/bin/colab --auth oauth2 status -s vln-t4
ssh -F '<project-root>/outputs/colab-t4-20261003T035623Z/ssh_config' vln-t4-ssh
```

配置文件中的 `ProxyCommand` 使用本机 Colab CLI 的 OAuth2 认证，并带有 `-s vln-t4`。它引用现有 SSH 私钥路径；不需要把私钥复制到远端。专用配置和 `known_hosts` 位于本次输出目录，连接命令依赖这些本机文件。

连接后激活环境：

```bash
cd /content/project/vln
source .venv/bin/activate
export PYTHONPATH="/content/project/vln/third_party/Matterport3DSimulator/build${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
nvidia-smi
python scripts/preflight.py
```

`preflight.py` 返回 0 表示所检查的源码、CUDA、依赖与必要文件齐全；返回 2 时阅读输出中的缺失项。它不代替真实 rollout。

Colab 的 `/content` 随运行时结束可能清空，重要输出应复制回本机。
后续已接通 Google Drive 并完成训练状态、best 模型和训练缓存的备份恢复演练，
详见 [训练与续训流程](training_pipeline.md)。当前专用 SSH 配置通过隔离的 `uv run`
使用 Colab CLI 0.7.4 与 `jupyter-kernel-client<1`，避免旧 CLI 的依赖兼容问题；未修改全局 CLI 安装。
此版本 CLI 的 SSH 命令可能在命名会话不存在时自动创建运行时，因此连接前要检查状态。
专用配置显式带 `--gpu T4`，避免失效后自动连接到默认 CPU 实例。

## 准备脚本与用途

本次脚本保存在本机目录：

```text
<project-root>/outputs/colab-t4-20261003T035623Z/
```

| 文件 | 用途 |
| --- | --- |
| `setup_gpu.sh` | 在远端工程目录解包 `source.tar.gz`，建立 Python 3.11 环境，安装 CUDA 12.1 版 Torch 与工程依赖，校验 DUET 源码，运行测试和 GPU 模型接口冒烟 |
| `build_mattersim.sh` | 安装编译依赖，构建固定提交的 MatterSim Python 扩展，并保存兼容补丁 |
| `download_assets.py` | 下载官方 R2R 标注、预计算特征和检查点，复制 connectivity，记录下载来源、大小与本地计算的 SHA-256 |
| `sim_smoke.py` | 验证 MatterSim 的真实图连通、转向、节点移动以及 BERT 配置缓存 |
| `run_real_smoke.py` | 顺序执行预检、8 条开发任务的零头一致性、8 条训练任务采集、GPU 训练 1 个 epoch、开发任务推理和指标比较 |
| `ssh_config`、`known_hosts` | 本次会话的专用连接配置与主机指纹 |

GPU 模型接口脚本的本机来源：

```text
<project-root>/outputs/colab-smoke-20261003T034609Z/full_model_smoke.py
```

该脚本上传至远端工程根目录后，可运行：

```bash
python full_model_smoke.py --device cuda
```

准备脚本按首次安装流程编写：依赖已上传的源码包，且 clone、复制目录等步骤不保证重复执行安全。已有会话优先直接运行验证命令，避免重复下载数 GB 数据或覆盖已有环境。

## MatterSim 的三个兼容改动

使用上游仓库 `peteanderson80/Matterport3DSimulator` 的固定提交：

```text
589d091b111333f9e9f9d6cfd021b2eb68435925
```

本次构建采用以下兼容调整：

1. **Python 绑定依赖**：使用 `pybind11 v2.13.6`，配合 Python 3.11 构建扩展。
2. **OpenCV 常量**：在 `src/lib/NavGraph.cpp` 中将 `CV_LOAD_IMAGE_ANYDEPTH` 改为 `cv::IMREAD_ANYDEPTH`，适配当前 OpenCV。
3. **CMake 最低版本声明**：将 `CMakeLists.txt` 中的 `cmake_minimum_required(VERSION 2.8)` 改为 `VERSION 3.10`，适配当前 CMake。

构建使用 `-DOSMESA_RENDERING=ON`、`-DCMAKE_BUILD_TYPE=Release`，并显式指定 `/content/project/vln/.venv/bin/python`；目标为 `MatterSimPython`，并行度为 2。上述两处源码差异保存在远端 `artifacts/mattersim-compat.patch`；pybind11 版本由构建脚本单独记录。

模拟器检查关闭渲染、预加载和深度，导航视觉输入采用已下载的预计算特征。该流程不验证 RGB 渲染，也没有下载完整 Matterport 图像。

## 日志与结果位置

远端工程下的主要记录：

```text
/content/project/vln/artifacts/pytest.log
/content/project/vln/artifacts/pytest.xml
/content/project/vln/artifacts/gpu-model-smoke.json
/content/project/vln/artifacts/simulator-smoke.json
/content/project/vln/artifacts/assets-manifest.json
/content/project/vln/artifacts/mattersim-compat.patch
/content/project/vln/artifacts/real-smoke-status.json
/content/project/vln/artifacts/real-<阶段名>.log
/content/project/vln/outputs/identity-dev8.json
/content/project/vln/outputs/collect-fit8.json
/content/project/vln/outputs/cache-fit8/
/content/project/vln/outputs/smoke-head.pt
/content/project/vln/outputs/head-dev8.json
/content/project/vln/outputs/smoke-comparison.json
```

`real-smoke-status.json` 逐阶段记录命令、运行状态、退出码和耗时。全部日志、轻量训练头和决策缓存已回收到本机，主要入口如下：

- [汇总报告](../outputs/colab-t4-20261003T035623Z/summary.json)
- [逐阶段运行状态](../outputs/colab-t4-20261003T035623Z/results/artifacts/real-smoke-status.json)
- [配对指标比较](../outputs/colab-t4-20261003T035623Z/results/outputs/smoke-comparison.json)
- [训练得到的轻量头](../outputs/colab-t4-20261003T035623Z/results/outputs/smoke-head.pt)

截至本次汇总，远端运行时仍保留。完整数据和骨干检查点位于远端；本机 `results` 保存的是报告、脚本、日志、决策缓存和轻量头。
