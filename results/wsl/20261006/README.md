# WSL 迁移验收：2026-10-06

**验收通过。** 测试代码为 `d031a8d1692aa0ce36d651a0b70843d5051b290c`。后续提交补充本记录和手册，运行代码未改。

运行位置：`xxl@192.168.0.100:2222`，仓库 `/home/xxl/projects/duet-vln`。GPU 实测为 RTX 4060 Laptop，8 GiB 显存。Python 3.11.16、PyTorch 2.5.1+cu121、MatterSim 已运行成功。

| 检查 | 结果 | 证据 |
|---|---|---|
| WSL 全量测试 | 881 项测试、48 项子测试通过 | [pytest.log](pytest.log) |
| CUDA 与模拟器 | 真实张量运算、MatterSim 实例创建通过 | [cuda-smoke.json](cuda-smoke.json) |
| 固定资产 | 5 份资产、92 个连接目录文件（含 90 份导航图） SHA 校验通过 | [assets-verification.json](assets-verification.json) |
| 离线导航一致性 | 2 条 train_fit，零残差轨迹完全相同 | [wsl-identity-smoke.json](wsl-identity-smoke.json) |
| 训练缓存 | 8 条 train_fit 采集完成 | [wsl-cache-fit8-report.json](wsl-cache-fit8-report.json) |
| GPU 续训与备份 | 第 3 步暂停、第 6 步强制终止，空目录恢复到第 12 步 | [resume-acceptance.json](resume-acceptance.json) |

恢复后模型、优化器、训练历史、步数、epoch、采样游标和累计量与连续训练精确一致。两次八条 train_dev 导航验证、最优 checkpoint 读回校验和快照保留检查均通过。导航与续训作业设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 并清除代理环境变量；实际命令保存在 [acceptance-job.sh](acceptance-job.sh)。该文件记录已经运行过的唯一 run_id，复验时应使用新的目录和 run_id。

备份位于 `/home/xxl/vln-backups`，校验后才清理旧快照。它是同一 WSL 磁盘上的持久副本，没有配置自动云端同步。历史 E2/E3 代码和结果归档也已迁入其 `history/` 子目录，并核对原 SHA。

本次只做工程验收。小样本中的 SR/SPL 不能当作研究成绩，没有新增 val_unseen 导航访问；原访问账本仍为 18 行，SHA 未变。E3 预测模块仍待开发训练，SR 提高 5 个百分点且 SPL 不降的目标尚未达到。

- [机器可读汇总](migration-summary.json)与[证据 SHA 清单](manifest.json)
- [运行就绪检查](preflight.json)、[BERT 离线缓存检查](bert-config-cache.json)、[阶段退出码](acceptance-stages.jsonl)
- [依赖实录](requirements-freeze.txt)、[依赖检查](pip-check.log)、[官方 wheel SHA 验证记录](verified-wheel-manifest.json)。依赖实录含当时本地 wheel 路径，用于审计；安装入口为 `scripts/setup_wsl.sh`。
- [WSL 运行手册](../../../docs/wsl.md)

本目录没有数据集、权重、训练缓存或凭据。完整运行状态保留在 WSL 的 `outputs/wsl-migration-resume-20261006-acceptance/` 和备份目录中。
