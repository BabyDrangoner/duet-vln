# Colab 训练、云盘备份与续训

入口是 `bash scripts/train_colab.sh --config configs/pipeline.json`。
该入口训练当前的冻结 DUET 残差头，每个 epoch 后执行真实导航评测。
小论文拟研究的新模块仍需接入这一训练接口。当前入口固定使用 `train_dev`，用于新增适配的开发与回归；新研究的官方验证访问预算另见 [协议审计](protocol_audit_20261003.md)，尚未改为每个 epoch 自动运行官方验证。

## 保存与清理规则

| 项目 | 默认行为 |
| --- | --- |
| 最近状态 latest | 每 100 次参数更新或 300 秒保存一次，以先达到者为准；在完整优化器步结束后检查 |
| 当前最优 best | 每个 epoch 在完整 `train_dev` 导航；SR 优先、SPL 打破平局，出现新最优立即同步备份 |
| 退出保护 | SIGINT / SIGTERM 在当前更新结束后保存；评测前先保存待评测状态，重启会先重评；Linux 训练父进程消失时，内核同时结束其评估进程 |
| 本地保留 | 最近 2 份，加上受保护的 best；只有对应备份校验成功才删除旧文件 |
| 云盘保留 | 最近 5 份，加上受保护的 best |
| 运行预算 | 每次进程最多 10 小时，VM 开机年龄最多 10.5 小时；评测前至少预留 30 分钟 |
| 云盘故障 | 停止后续训练，保留本地恢复点和已有备份；不会将失败当作备份成功 |

300 秒是保存检查间隔，不是对任意故障的严格损失上限：一个更新和一次上传也需要时间。
突然销毁 VM 时只能恢复已完成持久化的状态；正常信号退出还会额外保存。
清理 checkpoint 控制磁盘占用，不能延长 Colab 的运行时限制。
Colab 的运行时间和资源可用性会变化，也可能提前结束；见 [官方 FAQ](https://research.google.com/colaboratory/faq.html)。

## 首次启动

1. 在 Colab 挂载 Google Drive，确保 `/content/drive/MyDrive` 可读写。
2. 按 `docs/colab_t4.md` 准备 Python、CUDA、MatterSim、DUET、R2R 特征与基座权重。
3. 采集 `train_fit` 缓存，随后编辑 `configs/pipeline.json` 的 `cache`、`run_id` 和训练参数。
4. 运行：

```bash
cd /content/project/vln
bash scripts/train_colab.sh --config configs/pipeline.json
```

默认持久化目录：

```text
/content/drive/MyDrive/VLN-Research/runs/<run_id>/
  STORE.json                运行存储身份
  refs.json                 原子提交的 latest 与 best 引用
  latest.json / best.json   兼容镜像；以 refs.json 为准
  snapshots/<id>/
    state.pt                完整训练与流程状态
    head.pt                 可直接用于导航推理的模型
    manifest.json           文件大小、SHA-256、步数与评测指标
    COMMITTED               提交标记
  inputs/                   训练缓存、内容索引和校验信息
  reproducibility/          源码归档、环境清单、资产清单及补丁（存在时）
  config.json / status.json
  eval-*.json               完整逐任务导航结果
```

完整状态包含模型、Adam 一二阶矩与步数、epoch、记录位置、累计 loss、历史指标，以及
Python / NumPy / PyTorch CPU / CUDA 随机状态。本训练器未使用 scheduler 和 AMP scaler；
状态显式记录它们为 `None`，不能冒充兼容未来加入的训练器。

备份顺序为：本地临时文件 → 校验并提交 → 云盘临时复制 → SHA-256 读回 → 提交云盘引用 → 清理。
`refs.json` 一次更新 latest 和 best，避免进程在两次独立引用写入之间退出后发生混用。
程序会检查真实的 Drive FUSE 挂载，拒绝把同名普通目录当云盘。
校验等级记录为 `drive-mount-readback-sha256`：这是挂载盘读回校验，未取得独立 Drive API 服务端持久化回执。

## 原 VM 消失后续训

重新连接用户选择的运行时、挂载同一个云盘，并准备相同版本的依赖和导航资产。
从 `reproducibility/` 恢复代码或使用本地已拉回的工程，保留原 `run_id` 和训练配置：

```bash
cd /content/project/vln
bash scripts/train_colab.sh --config configs/pipeline.json --require-resume
```

`--require-resume` 要求找到有效状态，避免误从第 0 步开始。训练缓存若已丢失，会从云盘
`inputs/` 恢复到本地运行目录。完整 R2R 特征、基座权重和 MatterSim 仍按资产清单重建，
不会自动当作训练 checkpoint 重复复制。

恢复会严格检查训练超参、实际缓存内容、源代码和验证协议。允许本地路径改变；改变
学习率、batch size、目标 epoch 数、数据内容或方法代码需要新 `run_id`。
同一个 `run_id` 同时只运行一个 VM；文件锁保护同机并发，不是跨 VM 的分布式锁。
两台 VM 并行实验应使用不同的 `run_id`。

正常预算暂停退出码为 **75**，完成为 **0**，其他非零码代表失败。不要将所有非零码都
解释为已安全暂停。若突然退出留下 `.pending-*` 或 `.inputs-*.tmp`，程序保守保留；
检查有效 COMMITTED 备份后再处理残留，勿删除整个运行目录。

如果云端备份损坏，会尝试其他已校验的完整快照；所有已有快照均损坏时显式失败，
不会静默新建实验。若备份失败而原 VM 尚在，本地最新状态仍保留，默认优先恢复云端
最后确认的状态，可能重做尚未成功备份的若干步。

## 导出最优模型

```bash
.venv/bin/python scripts/export_checkpoint.py \
  --config configs/pipeline.json --kind best --output outputs/best-head.pt

.venv/bin/python scripts/run_duet.py --mode eval --split train_dev \
  --head outputs/best-head.pt --output outputs/best-dev.json
```

best 根据真实导航 SR/SPL 选择。离线 loss 只用于观察训练，不能代替导航指标。
`research` 运行禁止用 `--limit` 子集选择 best。新研究的 `val_unseen` 属于有访问登记的有限开发集；不要将当前残差底座每 epoch 的评测循环直接切到官方验证后进行无预算搜索。

## 故障演练

```bash
bash scripts/train_colab.sh --config configs/pipeline.json --stop-after-steps 10

PYTHONPATH=third_party/Matterport3DSimulator/build .venv/bin/python scripts/smoke_resume.py \
  --run-id resume-smoke-<唯一编号> --cache outputs/cache-fit8
```

`smoke_resume.py` 使用 CUDA 和 8 条真实开发任务：第 3 步正常暂停；从全新的本地目录
恢复后，在评测前已备份时 SIGKILL；再从另一空目录与云端缓存恢复，完成两个 epoch。
最终逐项比较模型、Adam 和训练统计是否与不中断运行完全相同，并检查 best 哈希和保留数量。
它只终止自身创建的进程组，不会停止或重新创建 Colab VM。

`--allow-local-backup-for-tests` 仅用于无云盘环境的故障测试，报告明确标记
`local-filesystem-test`。这种测试不能证明 Google Drive 备份成功。

### 2026-10-03 已完成验证

- 本机和 Colab T4：172 项测试、48 项子测试通过。
- T4 真实 Google Drive 演练：第 3 步暂停，第 6 步备份后 SIGKILL，空目录恢复并完成第 12 步。
- 最终 CUDA 模型参数、Adam 状态和训练统计与连续训练逐项完全一致。
- 两次真实导航评测完成；本地保留 3 份（最近 2 份和 best），本次 Drive 演练保留 4 份（最近 3 份和 best）。
- 已确认 `fuse.drive` 挂载、写入与读回 SHA-256 校验；训练缓存、latest 和 best 的备份与恢复通过。
- 本次通过进程骤停和全新本地目录模拟恢复，没有销毁 Colab VM；未取得独立 Drive API 服务端回执。

云盘运行目录：`/content/drive/MyDrive/VLN-Research/runs/resume-drive-20261003`。
演练报告：`outputs/pipeline-20261003/results/drive-acceptance/summary.json`。
此前普通文件系统演练保留在 `outputs/pipeline-20261003/results/acceptance/summary.json`。
其中 8 条任务的 SR/SPL 仅用于选优流程的工程检查，不是论文效果证据。

### 评估子进程保护补充（2026-10-03）

`child_guard.py` 在 Linux 设置父进程死亡信号并检查启动竞态，再执行评估程序。训练进程被强制结束时，其评估进程不会留在后台占用 GPU；正常中断仍先保存训练状态。Colab 上 34 项相关测试已通过，包括真实结束测试父进程、确认评估结束且独立哨兵继续运行。该保护没有终止实际 Colab VM。
