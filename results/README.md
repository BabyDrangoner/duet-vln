# 随 Git 保存的实验证据

- [WSL 迁移验收](wsl/20261006/README.md)：环境、离线导航、GPU 强制终止后的精确续训和持久目录备份均通过。
- [E2 完整闭环报告](e2/loop-report.json)：训练、选模与完整导航结果，未超过 DUET。
- [E3 完成决定](e3/completion-decision.json)、[机会统计](e3/opportunity-feasibility.json)、[目标定义](e3/target-feasibility.json)：128 条训练侧原指令的分支实验，尚未训练 E3 模型。
- E3 [fit 汇总](e3/probe-fit-v1/probe-summary.json)、[dev 汇总](e3/probe-dev-v1/probe-summary.json)及同目录 `audit-v2.json`：保留原实验 JSON 字节。
- [验证访问账本](../outputs/study-20261003/val-unseen-access-ledger.jsonl)：原路径、原内容，不因换机器重新计数。
- [封存产物清单](manifest.json)：完整轨迹、原源码和交付归档的 SHA、大小及本地位置。二进制归档、数据集和模型权重不纳入 Git。

原实验的 Drive 路径和备份校验类型是历史事实，保留在 JSON 中。它们不代表新 WSL 实验仍需 Colab。新的文件系统备份是单机持久化副本；要防整机或磁盘损坏，还需要另一个物理位置的副本。

审计入口已提升到 [audit_continuation_probe.py](../scripts/audit_continuation_probe.py) 和 [summarize_continuation_probe.py](../scripts/summarize_continuation_probe.py)。迁移改变了脚本位置和存储入口，原实验源码 SHA 仍以封存清单为准。恢复旧实验必须使用其对应的封存源码；新实验使用新目录，不能绕过身份检查混用缓存。

`docs/e3/` 保存实验前的设计建议，实际执行规模和完成结论以 [E3 结果报告](../docs/e3_continuation_probe_results.md)为准。
