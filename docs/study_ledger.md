# 官方验证访问登记

`scripts/register_study_access.py` 提供可单独调用的登记器。当前没有接入训练或评测入口，也不执行 GPU 任务。它只记录 `val_unseen` 的评测与新的结果导向分析；`train_fit`、`train_dev` 诊断使用各自的实验记录，不写入此日志。

## 使用顺序

所有路径相对于当前工作目录。先在项目根目录设置 `PYTHONPATH=src`，或安装项目。评测或查看新诊断之前登记，例如：

```sh
PYTHONPATH=src .venv/bin/python scripts/register_study_access.py register \
  --access-id P0001 --category pilot --variant-id method-v1 \
  --config configs/method-v1.json \
  --checkpoint-sha256 <实际检查点SHA256> --code-sha256 <实际代码快照SHA256> \
  --purpose '预先指定的第一个检查点完整导航评测' \
  --seed 0 --expected-episodes 2349
```

方法配置必须完整描述方法、目标和超参数，不能只传所有变体共用的 R2R 数据配置。种子和待评测检查点另行登记；同一变体跨种子应使用同一份固定方法配置。修改方法配置须分配新变体 ID；也不能给完全相同的配置换名字来重置限额。

执行完成后，准备两个 JSON 对象文件，例如 `metrics.json` 为 `{"sr":71.5,"spl":60.4}`，`resources.json` 为 `{"rollout_seconds":393,"cuda_peak_allocated_bytes":745100800}`，再追加结果：

```sh
PYTHONPATH=src .venv/bin/python scripts/register_study_access.py complete \
  --access-id P0001 --metrics outputs/metrics.json --resources outputs/resources.json \
  --report outputs/navigation-report.json --decision '记录预先规定的选模决定'

PYTHONPATH=src .venv/bin/python scripts/register_study_access.py status
```

CLI 计算配置文件和结果文件的 SHA；检查点/代码 SHA 由调用者提供，便于登记远端文件。登记器不验证指标是否与报告内容一致，也不证明调用者实际运行过评测；评测器仍需验证轨迹、完整性及指标。登记成功不表示授权使用验证标签训练。

失败也追加记录，例如 `fail --access-id P0001 --resources outputs/resources.json --decision '需要新访问 ID 重试' --error 'runtime lost'`。已有部分结果时同时传 `--metrics` 和 `--report`；未知成本明确写 `null`，不要编造 0。所有已登记的尝试立即占用预算，包括失败和中断。真正重跑需要新访问 ID；同 ID、同参数重试登记或写入结果是幂等操作，已终止访问不能改写成另一个结果。CLI 返回当前状态，调用者不能把幂等返回误认为需要再次启动评测。

## 预算与分析类别

- `baseline`、`pilot` 共用研究配置中的先导上限：最多 12 个变体，每变体 3 次访问，默认 seed 0。基线占一个变体；不同检查点、失败重试、子集评测都计次。
- `diagnostic_analysis` 必须传 `--label-use analysis --budget <预算文件>`。新的错误分型即使只读取已有报告，也要登记。该类别单独计次，不能用其结果冒充一次未记录的完整评测。
- `confirmatory` 必须显式提供预算文件，并固定 `variant_id -> config_sha256` 清单。只接受完整 split。预算应在三种子确认开始前冻结。
- 子集需提供 `--subset-ids <JSON指令ID数组>` 和相同数量的 `--expected-episodes`；登记时固定完整 ID 清单，避免事后改变样本。

显式预算的示例格式如下（这是文档示例，不是已批准的实验预算；SHA 必须替换为真实值）：

```json
{
  "schema_version": 1,
  "study_id": "arrival-evidence-20261003",
  "category": "confirmatory",
  "budget_id": "frozen-method-v1",
  "reason": "主方法与消融已冻结，报告预定三个种子",
  "max_variants": 1,
  "max_accesses": 3,
  "max_accesses_per_variant": 3,
  "seeds": [0, 1, 2],
  "splits": ["val_unseen"],
  "variants": {"method-v1": "<方法配置文件的64位SHA256>"}
}
```

诊断预算使用 `category: diagnostic_analysis`，无需 `variants` 映射，其余预算字段相同。每次登记保存当时预算内容与文件 SHA。预算上调须事前写明原因；计数按类别累计，换预算文件或预算 ID 不会重置已用次数。修改先导上限则须依照研究协议修订研究配置，不能删除历史访问。

## 历史兼容与持久化范围

现有手工 `V0001` 两行保持原样读取，登记器不重写、补填或删除原记录。按协议“12 个变体包含基线”，用量视图把它计为 **1 个变体、1 次访问**，尽管原行含 `variant_budget_charge: 0`；`status.legacy_notes` 明确展示这一解释。未知手工格式报错，需要先明确导入规则。

JSONL 只追加，并使用同目录 `.lock` 的本地 `flock` 将读记录、预算检查与追加放在同一个事务中，写入后 `fsync`。截断或损坏行会阻止后续写入，不会静默丢弃记录。这只保护共享同一个本地文件系统的调用者，**不提供多机器或 Google Drive 分布式锁**。使用单一权威本地日志，并由外部同步流程备份；当前模块尚未自动备份到 Drive。

模块没有训练或归一化接口，只允许登记 `parameter_fitting_split=train_fit` 与 `label_use=evaluation/analysis`。数据是否真的隔离，仍由采集和训练代码负责；日志不能替代这些检查。
