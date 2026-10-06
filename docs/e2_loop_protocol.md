# E2 第一轮：从训练到完整导航的固定实验

日期：2026-10-06。本文在本轮新采集、训练与导航结果出现前写定；执行前由主进程冻结源码、配置、ID 清单和资产身份。本文描述计划，完成状态由实际运行报告确认。

本轮回答两个问题：完整执行收益监督能否训练出有用的终点干预；在相同输入、容量和预算下，相对收益学习是否优于绝对效用回归。单轮不能确认论文创新成立，也不替代其他强对照和三种子验证。

## 1. 先固定数据与计算预算

唯一主配置是 [`configs/e2_loop_v1.json`](../configs/e2_loop_v1.json)。正式运行前从它产生 `relative` 与 `absolute` 两个完整方法配置；除目标及必要输出处理外保持一致。两个方法配置各自计算 SHA，不能让两个变体共用同一个配置 SHA。

| 项目 | 固定值 |
| --- | --- |
| 基座 | 原 frozen DUET，ViT-B/16 特征，dynamic fusion，argmax，最多 15 次决策 |
| 适配训练集 | 原 `train_fit` 49 个场景中，按场景均衡轮转固定 4,096 条自然指令 |
| 训练条件 | 上述 4,096 条自然历史；相同 ID 各 1 条固定动作扰动历史，共 8,192 条 |
| 适配开发集 | 完整 2,890 条自然 `train_dev`，12 个场景 |
| 训练 | 两臂各 seed 0、20 epoch、batch 64、AdamW、LR 1e-4、weight decay 0.01 |
| 更新预算 | 每臂 2,560 次更新；不按结果加轮次或重采样 |
| 监测 | epoch 2、4、…、20，共 10 次完整自然开发缓存评估 |
| 官方导航 | 两个选定训练 checkpoint，各 1 次完整 `val_unseen` 2,349 条 |

场景划分继续使用 seed 20261003 和 dev fraction 0.2。采集 seed 为 0：场景按 `object_sha256([seed,scan])` 排序，每场景指令按 `object_sha256([seed,instr_id])` 排序，按深度跨场景轮转取前 4,096 条。该哈希使用排序键、紧凑分隔符且拒绝非有限数的 JSON UTF-8 编码。运行前保存完整 ID 数组与 SHA。选 ID 不读取任务成败、目标距离或任何新验证结果。自然与扰动条件必须覆盖同一指令清单。

两个条件均保留全部样本，包括没有实际触发扰动的历史。不按标签丢弃失败、重复采样少数正例或改变自然/扰动比例。训练每 epoch 对全部样本进行可恢复的固定种子洗牌。由同一原指令派生的历史不得进入不同 fit/dev 划分。

### 一次可观测动作扰动

在零起算第 2 次决策，利用此时策略可见的全局候选，从有限 logit、未访问、非 STOP、不同于原 argmax 的合法动作中按固定哈希选 1 个。随后恢复原冻结策略。到第 2 次决策前已结束或没有替代动作时不修改，并记录 `applied=false`。若第 2 次原 argmax 为 STOP 且有合法替代，允许该唯一覆盖；原始 STOP 概率仍照实保留。

将合法替代 ID 按字符串排序后，取 `int(object_sha256([instr_id,seed,'perturb_step2',sorted_alternative_vpids]),16) % len(sorted_alternative_vpids)` 所指动作。扰动选择不能查看目标、专家动作或参考路径。它只用于 `train_fit`；开发集与官方验证的在线探索保持原策略。该增强是训练支持工具，不能独立算作创新。它也会造成分布变化，因此自然与扰动的支持量、训练损失和路径长度须分别报告。

### 支持量是解释边界

本轮事前要求：自然与扰动合并后，至少有 **40 条去重原指令、覆盖 8 个 fit 场景**，存在真实正 ΔSPL 的非原终点候选。同一指令在两个条件中出现只能计 1 条。指令 ID 去重不等于统计独立。另报自然集、扰动集、自然开发集的实际分母及支持，不临时设开发集筛选条件。

若未达到上述值，仍完成用户要求的同一个训练和导航闭环，但明确称为“正收益支持不足的工程/探索实验”。不按结果继续采集，也不因训练完成就称方法得到充分支持。该门槛是有限预算下的事前描述性标准，不是统计功效保证。

## 2. 两臂唯一主差别

两臂使用同一最终上下文、候选集合、原终点参照、已执行长度和发现图返回成本，均只训练小头。骨干权重、候选 token、数据、参数量、优化器和更新预算相同。

- **Relative：** 监督每个候选的实际 ΔSR、ΔSPL；原终点输出固定为零，原终点与 padding 不进入回归损失。
- **Absolute：** 监督每个候选的实际 SR、SPL，原终点也需要学习；推理时减去该模型预测的原终点效用，得到预测相对收益。

两臂采用相同的逐 episode 双输出 Smooth L1 回归，不启用额外有害干预惩罚。两臂都仅考虑预测 ΔSR ≥ 0 且 ΔSPL > 0 的候选，选择唯一最大 ΔSPL；并列或无候选则保留原终点。KEEP 包含基线的原返回路线。没有验证集阈值搜索，也没有实际收益安全保证。

标签计算对每个合法历史终点构造实际返回路径，再计算完整前缀加返回成本。参考长度按当前评估器的参考路径逐段最短距离求和。完整评测图和目标标注只进入监督及评价，不能传入模型输入和动作选择。

## 3. 选模与负结果的预定处理

自然 `train_dev` 上逐候选的完整返回指标可从固定历史精确计算，因为方法仅修改最终返回。缓存评估使用 batch 1，与实际导航推理一致，训练 batch 仍为 64。缓存评估须验证与实际导航输出一致；它不允许改变基座探索。

1. 只考虑已训练的 epoch 2、4、…、20；epoch 0 不参与选模。
2. 先筛选实际 SR 与 SPL 均不低于同历史原基线的 checkpoint。
3. 在通过者中按 SR 降序、SPL 降序、epoch 升序选择。
4. 若没有通过者，固定使用最终 **epoch 20** 做一次官方导航，事前标为开发门槛未通过的探索/负结果。
5. 若通过者与 KEEP 完全相同，仍可选择该已训练模型，但结论为无提升；不得将保留基线写成学习收益。

数值比较容差最多 1e-12（分数单位）；报告原始数值。开发集将近饱和，且属于基座训练场景，所以门槛本身不保证未见房屋表现。上述 fallback 在看到新训练结果前写定，是为完成这次闭环而透明呈现失败，不是结果不佳后改选其他模型。

两个选定 checkpoint 均在首次新 `val_unseen` 导航前冻结。验证结果不反向更换 checkpoint、阈值、种子或数据比例。若要下一轮调整，使用新配置、ID 和访问记录。

## 4. 身份、恢复与有限资源

每个缓存记录并核验：骨干 SHA、输入特征和标注身份、连接图身份、原始数据划分、源码快照 SHA、采集配置及指令清单 SHA、每个 episode 文件 SHA。没有新 token 的 E1 旧缓存不能冒充 E2 缓存。

恢复状态包括模型、AdamW、CPU/CUDA/Python/NumPy RNG、epoch、洗牌顺序与下一批游标、损失历史、开发历史和 best 选择。恢复时匹配数据、配置和源码身份。正式 trainer 上须完成一次中途保存后恢复与连续运行的等价性检查；既有小批 smoke 只证明原型，不替代正式 trainer 验收。

每 300 秒及每次开发监测备份，best 与 latest 分开。Drive 写入并读回 SHA 校验通过后才能清理本地旧 checkpoint。本地保留数 2、云端普通历史保留数 5；best/latest 受保护。单进程上限 10 小时、VM 年龄 10.5 小时，并预留 30 分钟评测。中断仅允许恢复已固定实验，不能借恢复延长训练总更新数。

训练、采集和测试时记录实际时间、峰值已分配显存、缓存大小、是否恢复。未知测量写 null 和原因。所有代码、日志、报告与小型清单拉回本地；数据集与特征缓存继续留在云端。

## 5. 官方验证登记与剩余预算

本轮开始时权威日志是 [`val-unseen-access-ledger.jsonl`](../outputs/study-20261003/val-unseen-access-ledger.jsonl)，14 行，SHA `26b26492a904edd4cdebbc3e7a48ccbd12ce8a8adafe09f6b8211279d0463a89`。已有 5 个 pilot 变体、5 次 pilot 访问；上限分别为 12 个、36 次，每变体最多 3 次。S0001 与 S0002 已占用独立诊断预算 2/3。本轮不申请新的事后错误分型访问。

本轮计划新增 V0006、V0007，各 1 次，因此成功执行后用量为 7/12 变体、7/36 次访问。所有失败、中断也消耗访问；真正重跑必须新 ID，并在重跑前说明原因。训练与 `train_dev` 不写 val 日志，但仍有独立运行报告。一个机器作为日志唯一写入端，写后同步另端和 Drive，不能在两台机器并发追加。

每个方法配置都应包含完整数据、目标、决策规则和超参数。以 relative 为例，以下是模板，SHA 与路径必须由真实文件替换，**本文没有执行登记**：

```sh
PYTHONPATH=src .venv/bin/python scripts/register_study_access.py \
  --ledger outputs/study-20261003/val-unseen-access-ledger.jsonl register \
  --access-id V0006 --category pilot --variant-id E2-loop-v1-relative \
  --config outputs/e2-loop-v1/frozen-relative.json \
  --checkpoint-sha256 ACTUAL_SELECTED_CHECKPOINT_SHA256 \
  --code-sha256 ACTUAL_SOURCE_SNAPSHOT_SHA256 \
  --purpose 'Fixed E2 relative-utility checkpoint full navigation; report even if train_dev gate fails' \
  --split val_unseen --seed 0 --expected-episodes 2349
```

absolute 对应 `V0007`、`E2-loop-v1-absolute` 及其独立配置和 checkpoint SHA。启动器在运行前核对登记的这些身份，必须确认 ID 尚未完成，且没有已产生结果的重复执行。

结果核验并保存后追加：

```sh
PYTHONPATH=src .venv/bin/python scripts/register_study_access.py \
  --ledger outputs/study-20261003/val-unseen-access-ledger.jsonl complete \
  --access-id V0006 --metrics outputs/e2-loop-v1/relative-metrics.json \
  --resources outputs/e2-loop-v1/relative-resources.json \
  --report outputs/e2-loop-v1/relative-navigation.json \
  --decision 'Record the frozen loop outcome; do not change the selected checkpoint from this validation'
```

失败则使用 `fail`，保留错误、部分结果及实际成本。日志 `status` 为只读，可随时检查；不存在登记成功就已完成评测的含义。

## 6. 运行前验收与本轮预定比较

正式训练前必须通过：

- 自然采集不改变基线逐条路径、STOP 分数、候选 ID 和指标。
- 扰动只在规定时刻修改一次合法动作，所有原始 STOP 分数照实保留；没有目标依赖的选择。
- 所有候选标签与原评估器的完整实际路径指标一致；KEEP 返回与原基线相同。
- 输入构造不访问目标、参考路径、oracle 或监督字段；只允许实际观察节点。
- 两臂参数数、训练数据、批次顺序、更新数匹配；absolute 先回归自身 anchor，再相减，并应用相同门控。
- padding、原 anchor、唯一候选、预测并列、恰好 3 米边界与返回路径计费处理正确。
- 训练中断恢复完全一致；正式选定 checkpoint 再加载能复现开发选择与指标。

预定报告 relative−DUET、absolute−DUET、relative−absolute 的 SR、SPL、nDTW、SDTW、CLS、导航误差、路径长度，以及配对救回数、损害数、双方成功/失败数和改动终点比例。提供完整 2,349 条、11 个场景的指标；配对场景 bootstrap 使用固定 seed 20261006、10,000 次重采样和逐项 95% 区间，每次保留被抽中场景的全部指令及重复次数。不能把 2,349 条当成独立环境，也不把逐项区间当多比较校正后的确认检验。上述比较属于本次事前固定的评测报告，不是事后新增错误分型；本轮不增加阈值切片分析。基线复用原报告，其 SHA 为 `32e4b21422a86c2b8a7e81eb1b423b903ac19cf25972d450c003bdd33d57cd61`，须核对同 ID 和骨干/动作协议。训练及正式导航都失败时照实报告已完成边界。

单 seed、两个学习臂只能回答第一轮可行性和有限比较。官方 `val_unseen` 是已有开发暴露的验证集；本轮不进行隐藏测试提交，不把验证收益直接称为最终论文结论。几何/STOP-only、概率成本、同类 node token 时间对照，以及后续多 seed，仍是研究贡献成立所需的工作。
