# 低算力视觉语言导航：小论文研究与实验工程

## 开发与运行

源码仓库：[BabyDrangoner/duet-vln](https://github.com/BabyDrangoner/duet-vln)。从 2026-10-06 起，实验迁移到局域网机器的 WSL，环境准备和启动命令见 [WSL 运行手册](docs/wsl.md)。后续运行使用 Git 提交标识，数据和模型按固定 SHA 下载到被忽略的 `datasets/`。

WSL 迁移验收已通过：881 项测试、48 项子测试，以及离线导航和 GPU 断点恢复。实测显卡为 RTX 4060 Laptop 8 GiB；[验收证据](results/wsl/20261006/README.md)已随 Git 保存。

最新实验汇总收录在 [results/](results/README.md)，设计与审计代码随源码提交。完整轨迹、训练缓存和检查点保留为独立实验产物；历史文档中指向 `outputs/` 的链接通常需要本地归档。验证访问账本仍保留原路径并纳入 Git，迁移不重置已使用的访问预算。

## 当前目标与状态

**2026-10-06：目标已明确为相对原 DUET 的 SR 至少提高 5 个百分点，同时 SPL 不下降。尚未达到。**

在相同 2,349 条 R2R `val_unseen` 指令上，要求 SR ≥76.5198、SPL ≥60.40985；离散成功数至少为 1,798 条，相比原来的 1,680 条净增至少 118 条。`val_unseen` 是已暴露的开发验证集，尚无官方盲测成绩。

E2 已完成训练、选模和完整导航闭环，不能再按“尚未训练”理解：

| 方法 | SR | SPL | 相对 DUET 的 ΔSR / ΔSPL（百分点） |
|---|---:|---:|---:|
| 原 DUET | 71.5198 | 60.40985 | — |
| E2 相对完整执行收益，选中 epoch 8 | 71.1792 | 60.34137 | −0.3406 / −0.0685 |
| E2 同容量绝对效用，固定回退 epoch 20 | 71.0941 | 60.24439 | −0.4257 / −0.1655 |

见 [E2 完整结果与证据边界](docs/e2_loop_results.md)。原实验代码、选中 checkpoint、报告和 Drive 备份保持封存。

**E3 当前检验的方向是在线动作干预后的完整续跑收益。** 在同一已执行前缀上改变一次合法动作，随后恢复冻结 DUET；监督实际执行到结束的 SR/SPL，而不是只在终止后改排历史终点。先完成 64 条 `train_fit` 与 64 条 `train_dev` 的自然/固定扰动机会检查，再决定是否扩大训练。仅有候选理想选择的机会不能证明模型学得会，也不能外推正式验证必涨 5 个百分点。

真实 T4 固定 probe 已完成：128 条原指令、256 条参考轨迹、2,394 条完整分支；683 次原动作 anchor 全部复现，前缀 logits 最大差异为 0。几何审计和 Drive 空目录恢复检查通过。
扰动 dev 的 5 条失败均存在成功替代动作，且原完整路径从未进入目标成功半径，支持把干预提前的动机；但 fit 去重后只有 3 条成功救回正例，当前不直接晋级正式训练。下一版先补可恢复失败的训练数据，再训练在线比较器与强对照。尚未训练 E3 预测模块，不能将理想选择上界算作导航提升。

见 [E3 完整结果与设计决定](docs/e3_continuation_probe_results.md)。代码和结果已保存本地，WSL 的数据资产由固定清单重建。

- [冻结 probe 配置](configs/e3_continuation_probe_v1.json)、[采集入口](scripts/probe_continuation_actions.py)
- [训练可行性与必要对照](docs/e3/learning-feasibility.md)、[接口与真实路径计费](docs/e3/integration-feasibility.md)
- [一手文献核验及创新边界](docs/e3/prior-art-feasibility.md)、[协议审查](docs/e3/probe-protocol-review.md)

## 历史记录

以下保留各阶段当时的状态；当前结论以本文开头和对应完成报告为准。

**2026-10-06 后续：E1 失败原因已按 S0002 完成逐条诊断，开始设计 E2。**
M 的误改损害和成功任务额外回走共同导致 SPL 下降；旧目标没有监督相对原终点的实际执行收益。新方向是用完整历史学习何时值得替换原决定，明确保留原终点，并监督真实返回后的 ΔSR/ΔSPL。共同历史重评、门控与距离惩罚都有先行工作，不能分别包装成首创。
已实现 362,370 参数核心原型；T4 的 8 条 `train_fit` /44 候选冒烟通过，包括完整轨迹不变、候选指标对齐、20 步训练与 Drive 空目录恢复一致。尚未完成正式 E2 训练或导航评测。
旧自然 fit 的支持量审计发现，1,024 条任务中 990 条基线已成功，仅 10 条存在正 ΔSPL 候选；新版先扩大自然覆盖，必要时补充受控困难历史，不能直接依赖旧池训练收益模型。
见 [失败原因](docs/e2_failure_diagnosis.md)、[调整后的研究设计与强对照](docs/e2_research_design.md)和[查新边界](docs/e2_prior_art_check.md)。

**2026-10-06 E1 评测结果：T4 与 Drive 已恢复，四臂完整导航评测完成，未验证出超过 DUET 的导航提升。**
使用冻结的 seed 0、20 epoch / 1280 updates final checkpoint，每臂评测完整 `val_unseen` 的 2,349 条指令、11 个场景；V0002–V0005 均已登记、完成并备份。原封存代码、四个 head 和数据/权重 SHA 一致，709 项测试、48 项子测试通过。

| 固定 final，seed 0 | SR↑ | SPL↑ | nDTW↑ |
|---|---:|---:|---:|
| DUET 基线 | 71.5198 | 60.4098 | 67.0205 |
| C1 自然 BCE | 71.1792 | 59.2246 | 66.2924 |
| C2 已知增强 BCE | 71.2218 | 58.5269 | 65.7286 |
| C3 配对数据 BCE | 70.8812 | 58.1286 | 65.6664 |
| M 配对数据 + 排序 | 71.3921 | 58.5615 | 65.7932 |

M 相对 DUET 的 SR 为 −0.1277 pp、SPL 为 −1.8484 pp；相对 C3 有小幅正增量，但尚未达到课题的导航提升目标。
完整轨迹已独立从导航图重算，报告、执行记录和访问账本均保存在本地与 Drive；本地不含数据集。
见 [完整结果、场景置信区间与返回成本](docs/e1_full_navigation_results.md)和[本轮记录](outputs/study-20261005/)。
这是开发验证集上的单种子结果，尚无官方盲测 `test` 分数。以下保留早期实验记录，以本段及完整结果报告为最新状态。

目标是在相同输入、检查点和评测协议下，提高 R2R 完整导航的 **SR 与 SPL**，
并形成一篇小论文所需的明确问题、方法贡献、机制证据和迁移实验。
+1 个百分点仅是预研投入参考；“小头涨点”不构成课题完成标准。

首个候选 **用到达后的观测校准未访问候选的间接视觉证据** 已完成自然样本和一次预先固定的分支补采；证据不足以晋级训练，现已停止该 arrival-margin 监督方案。
[研究方案与负结果](docs/research_plan.md)保留设计及停止理由；停止决策机会诊断与路线融合诊断也已完成。当前推进同一物理历史下的指令条件终点辨别，方法收益和论文贡献仍待验证。

- STOP：完整 2,349 条验证指令逐条复现基线；212 条失败指令曾观测到成功历史位置，但简单重评分最多净救回 2 条。存在研究空间，尚未证明新的训练方法有效。见 [结果审计](docs/stopping_result_audit.md)。
- 路线融合：训练场景保留集 598 个有效状态仅改变 1 次选择，代价更高，停止当前固定首跳融合候选。见 [结果与边界](docs/route_fusion_result_audit.md)。
- 普通终点分类器 E0：小样本自然 BCE 训练后，完整 train_dev 的 SR 为 97.7163、SPL 为 96.1221，均未超过基线，暂不占用官方验证访问。见 [实现与实测](docs/endpoint_probe_engineering.md)。
- 新候选 D3：完整 512/128 对配对缓存已在 T4 采集并封存，共 2,560 条强制轨迹、26,304 个状态，非文本输入、朝向和路径严格一致，已备份 Drive。新版自然／越过目标对照的 2/1 对真实冒烟通过。四臂分别为自然 BCE、越过目标增强 BCE、相同配对数据 BCE、配对数据加排序损失；[训练协议](docs/endpoint_training_protocol_draft.md)和 `configs/endpoint_group_*.json` 已固定。

**2026-10-03 恢复核验：** 原 T4 会话丢失后，现已通过 `vscode-colab` 的新 CPU 会话重新挂载 Drive。
云盘显示完整对照数据和四臂 seed 0 实际已在旧 T4 完成；四臂均为固定 final 20 epoch / 1280 updates。
取回的 70 个模型和记录文件已逐项校验，本地加载实际 head 后复算全部对照，与原云端结果一致。
新 E1 的真实 CUDA 恢复验收也已核实：连续和空目录恢复后的完整模型、优化器、RNG、游标、历史完全相同。
四份完整特征缓存也已在新 CPU 会话严格重载通过，512/128 组身份与原锁一致，见
[完整缓存核验](outputs/study-20261003/recovered-feature-cache-verification.json)。
见 [独立恢复核验](outputs/study-20261003/e1-recovery-independent-verification.json)和
[本地复算](outputs/study-20261003/e1-seed0-local-reanalysis.json)。

| 固定 final，seed 0 | 自然 BCE↓ | C2 增强 BCE↓ | 配对 BCE↓ | 两种访问顺序均正确↑ |
|---|---:|---:|---:|---:|
| C1 自然 BCE | 0.26177 | 0.36071 | 0.43050 | 82.03% |
| C2 已知增强 BCE | 0.24751 | 0.28656 | 0.33359 | 83.59% |
| C3 配对数据 BCE | 0.25073 | 0.32135 | 0.24544 | 89.84% |
| M 配对数据 + 排序 | 0.25076 | 0.32268 | 0.24986 | 89.06% |

这些是 128 对、10 个训练房屋的开发压力面板，**不是导航 SR/SPL**。
C3−C2 的两顺序正确率为 +6.25 个百分点（场景 bootstrap 95% 区间 +3.01 至 +9.78）；
同时 C2 面板 BCE 恶化 0.03480，必须保留这一代价。M 未显示相对 C3 的额外贡献，不调 λ 或换 best 来改写结果。
当时保留四臂 final，完整导航与返回成本尚待 GPU 复核；该次恢复工作未增加官方 val_unseen 访问。
当时申请 T4 返回 503；此阻塞已在 2026-10-06 解除，完整评测结果见本文开头。恢复流程见
[运行手册](docs/endpoint_group_runbook.md)，不能重跑已完成的训练队列。

**同日补充：256 条自然导航缓存的离线终点回放已完成，未验证出导航提升。**
使用原固定 128 对中的全部 256 条自然轨迹、四个 seed 0 final head 和原始已探索地图的回退规则，完整计入回走成本。
全部基线轨迹、SR/SPL、距离和步数逐条复现；nDTW/SDTW/CLS 最大末位差异为 4 ULP，已逐条记录。

| 固定自然轨迹子集 | SR↑ | SPL↑ | 相对 DUET 的 SPL 变化 |
|---|---:|---:|---:|
| 原始 DUET | 98.4375 | 97.2466 | — |
| C1 自然 BCE | 98.4375 | 96.7784 | −0.4681 pp |
| C2 已知增强 BCE | 96.8750 | 94.6722 | −2.5743 pp |
| C3 配对数据 BCE | 97.6563 | 94.9193 | −2.3272 pp |
| M 配对数据 + 排序 | 98.0469 | 95.6244 | −1.6222 pp |

C3 相对 DUET 救回 1 条、伤害 3 条；另外 29 条仍成功的任务改变终点，新增回走使这些任务对整体 SPL 的贡献下降 1.3024 pp。
C3 的整体 SPL 差值场景 bootstrap 95% 区间为 [−3.9316, −0.9309] pp。
C3 对 C2、M 对 C3 的 SR/SPL 差值区间都跨 0，不能据此声称数据构造或排序损失带来导航增量。
这说明压力面板的辨别改善尚未转化为这批自然导航的收益；当前替换终点评分的方案不据此晋级官方验证。
结果仅覆盖固定训练场景子集、单种子和已存特征的 CPU 回放；候选缓存与完整 GPU 运行不同，仍不能替代 2,890 条完整导航。
见 [逐条回放与统计](outputs/study-20261003/e1-natural-subset-navigation-v3.json)和 [Drive 发布校验](outputs/study-20261003/navigation-subset-publication.json)。

**2026-10-04 未见场景评测重试：** 用户要求继续检验未见场景表现，下一轮优先完整 `val_unseen` 的 2,349 条指令，保留 C1/C2/C3/M 四个固定 seed 0 final。
再次连接 T4 时，两版 CLI 均返回 503；读取响应体后确认 `outcome=2`，官方定义为 `QUOTA_EXCEEDED_USAGE_TIME`（使用时长配额已耗尽）。
账号计算单元余额为 0；此前 CPU 会话也已过期，服务器当前没有运行中的实例。尚未启动新评测或新增正式验证访问。
见 [最新连接诊断](outputs/study-20261004/t4-reconnect-diagnostic.json)、[四臂完整评测计划](outputs/study-20261004/e1-val-unseen-plan.json)和[首次重试记录](outputs/study-20261004/colab-unseen-preflight.json)。
`val_unseen` 仍属于已暴露的开发验证集，不能写成官方盲测 `test`；256 条自然子集负结果也不能外推到它。

最新已下载的 Colab 代码归档含 528 个文件，位于 `outputs/study-20261003/code-snapshot-e1-d3/`；不含数据集，逐文件 SHA 已校验。
后续本地新增的恢复检查入口及[先行工作补核](docs/endpoint_prior_art_delta.md)保留在当前源码目录。
本轮恢复脚本与报告另封存在 `outputs/study-20261003/recovery-code-records.tar.gz`，与原源码快照配套使用。
四份特征缓存已额外封成 348 MB 恢复包并通过 Drive 读回校验，包含 3,852 个文件；缓存包留在云端。
发布身份见 [缓存包记录](outputs/study-20261003/e1-full-cache-bundle.publication.json)。

已在 Colab T4 完成真实 R2R 冒烟：8 条开发任务的零残差轨迹完全一致；
8 条训练任务采集出 44 个决策，完成 CUDA 小头训练、重载和再次导航。
这批冒烟任务中小头与基线的 SR/SPL 相同。正式基线随后已在 T4 完成：

| 数据划分 | 任务数 | SR | SPL | nDTW |
|---|---:|---:|---:|---:|
| `train_dev`（仅对新增适配隔离，基座已见） | 2,890 | 97.8201 | 96.4168 | 97.2351 |
| 官方 `val_unseen`（开发验证） | 2,349 | 71.5198 | 60.4098 | 67.0205 |

**尚未证明新方法提高导航指标。** 分支补采覆盖 2,061 条拟合候选和 1,087 条保留候选。来源统计没有稳定增量；预测到达量对 P0 的微小改善未获 P0+P1 比较及辅助 regret 支持。[独立审计](docs/branch_analysis_audit.md)给出完整判读。T4 全套工程检查为 382 passed、48 subtests passed（后续新诊断模块另有测试）。结果与访问记录保存在 `outputs/study-20261003/` 并备份到 Drive。
连接、环境和完整记录见 [Colab T4 冒烟记录](docs/colab_t4.md)。

已增加 [Colab 训练与续训流程](docs/training_pipeline.md)：定期保存完整训练状态，
按开发集导航 SR/SPL 立即备份新最优模型，验证备份后清理旧 checkpoint，支持丢失本地目录后的恢复。
使用 `configs/pipeline.json` 和 `scripts/train_colab.sh` 启动。
真实 Google Drive 上的 T4 备份与恢复演练已通过：第 6 步强制终止后，从空目录恢复到第 12 步，
最终参数、Adam 和训练统计与连续 CUDA 训练完全一致。

## 已实现的实验底座：困难动作的轻量纠偏

以下代码保留为基线、接口检查和普通可训练头对照；它不再承担论文的主创新。

冻结 DUET 主干，对每个合法全局动作（含 STOP）添加有界残差：

```text
新分数 = DUET 融合分数 + max_delta × tanh(小型 MLP 的输出)
```

输入来自 DUET 已有的全局候选表征、按节点 ID 对齐的局部表征、两分支及融合分数、
候选数量、位置特征。位置特征只涉及已探索地图的几何信息；目标位置、真实路径、
到目标的距离不进入小模块。约 20 万个可训练参数（默认特征维度 1549、隐藏宽度 128）。

末层零初始化，初始策略与原始 DUET 相同。保持原有动作掩码、动态融合、路径执行、
步数限制和历史停止位置回退。所有回走长度继续计入官方指标。

训练流程：

1. 基座在 `train_fit` 自己选择动作，保存遇到的状态。
2. 沿用 DUET 的 SPL 伪交互教师，为训练状态提供标签。
3. 训练残差头：困难状态加权交叉熵，加上限制策略偏移的 KL 项。
4. 让新策略完整运行；可在 `train_fit` 重新采集一次，然后合并两轮缓存重训。
5. 原残差底座用 `train_dev` 做接口和适配回归检查；新研究按登记预算在官方 `val_unseen` 开发选模，完整报告访问和消融。

困难状态目前定义为“基座 argmax 与教师动作不同”。它只是可实现的第一版选择规则。
离轨教师标签可能偏离原指令顺序，需抽查失败轨迹，并同时观察 nDTW。
若冻结表征不能区分错误分岔，或重排收益不超过均匀训练对照，应更换切口。

### 创新边界

“残差融合”“错误轨迹再训练”本身已有工作，不能直接当作创新。
后续方法仍需独立的问题证据、与既有方法的实质区别和严格对照；停止当前候选后，不再把原到达监督设计当作已成立的贡献。

- [DUET：动态双尺度融合与伪交互教师](https://arxiv.org/abs/2202.11742)
- [HSPR：相关残差融合工作](https://arxiv.org/abs/2403.11541)
- [CorrectNav：错误轨迹与纠错训练](https://correctnav.github.io/)
- [BudVLN：离轨状态与指令一致性](https://arxiv.org/abs/2602.06356)

## 现有底座的对照与工程检查

下表用于验证已有训练通路。小论文的正式实验矩阵以研究方案为准。

| 实验 | 设置 | 回答的问题 |
|---|---|---|
| B0 | 官方 fine-tuned DUET 检查点，argmax | 真实基线表现 |
| Z0 | 同一检查点 + 零初始化残差头 | 逐条轨迹是否完全一致 |
| M0 | 残差头，`hard-weight=1`，`kl-weight=0` | 单纯增加小头训练是否有效 |
| M1 | 残差头，`hard-weight=3`，`kl-weight=0` | 困难状态加权是否有效 |
| M2 | 残差头，`hard-weight=3`，`kl-weight=0.1` | 限制策略偏移是否有效 |
| M3 | M2 + 一轮自身轨迹重新采集 | 新策略遇到的状态是否需要补充训练 |

先在相同缓存、训练轮数和开发任务上筛选 M0–M2。M3 是单独的数据采集消融，
其增加的轨迹数、训练样本曝光次数和卡时要单独报告；正式研究还应加入等额采集基线轨迹的对照。
额外训练原有动作头也是后续正式论文的重要对照，当前首版尚未实现。

主结果报告 SR、SPL；辅助报告 NE、nDTW、轨迹长度、推理时间、显存峰值。
最终方法训练种子使用 0、1、2，完整报告每个种子和均值/标准差。
单个 head 与基线还可按场景做配对 bootstrap；该区间不包括训练种子方差。
若区间跨 0 或种子间不稳定，应明确证据不足。

### 数据与评估边界

- 将官方训练集按房屋固定划为 `train_fit`（约 80%）和 `train_dev`（约 20%）。
- `train_dev` 仅隔离新增适配的拟合；官方基座配方已使用全部训练房屋。
- `val_unseen` 用于有预算、有访问记录的开发选模，不能用于参数拟合或训练缓存；详见 [协议审计](docs/protocol_audit_20261003.md)。
- 官方代码按 `val_unseen` SPL 选 `best_val_unseen`，因此该结果应称开发验证成绩。
  最终独立测试需按官方 test server 协议另行提交，本工程暂不实现自动提交。
- 所有方法使用同一基座检查点和特征；不把不同论文、不同输入配置的数字直接相减。
- `--limit` 仅作烟雾检查，输出明确记录子集身份；不能作为完整基准成绩。

## 文件结构

```text
configs/r2r.json             数据路径、模型与固定划分配置
configs/upstream.json        DUET 提交与源码哈希
docs/research_plan.md        小论文主线、查新、方法设计与实验规格
scripts/prepare_duet.py      获取固定源码，加入 hook 和 NumPy 兼容修正
scripts/preflight.py         数据、依赖、GPU 可用性检查
scripts/run_duet.py          baseline / identity / collect / eval
scripts/compare_metrics.py   协议校验、配对 SR/SPL 差值与场景 bootstrap
scripts/collect_diagnostics.py  原策略来源证据和自然到达采集
scripts/replay_diagnostics.py   固定状态的自然观测干预
scripts/replay_branches.py      独立合法分支补采与受控重放
scripts/analyze_diagnostics.py  固定探针、房屋 bootstrap、匹配对照
configs/research_study.json     实验阶段、评估协议和预算
configs/branch_extension.json   一次性补采的事先规则
src/vln_improve/features.py  按节点 ID 对齐表征
src/vln_improve/head.py      有界残差头与检查点
src/vln_improve/capture.py   只在训练集采集监督
src/vln_improve/train.py     分片缓存上的离线训练
src/vln_improve/resumable.py 模型、优化器、游标和随机状态的完整续训
src/vln_improve/checkpoint_store.py  校验、原子提交与保留策略
src/vln_improve/pipeline.py  定期备份、导航选优和中断恢复
configs/pipeline.json        正式训练配置
scripts/train_colab.sh       Colab 训练入口
scripts/smoke_resume.py      CUDA 暂停、骤停与空目录恢复演练
tests/                      功能、协议和真实上游模型接口测试
```

## 环境与数据准备

建议 GPU 端采用 Linux + **Python 3.11**。旧 tokenizers 在 Python 3.12 上可能转为源码编译；
已在本机安装 DUET Python 依赖，并在 Colab T4 验证 Python 3.11.16、Torch 2.5.1+cu121
及 MatterSim 禁渲染导航；构建兼容调整见 Colab 记录。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按机器 CUDA 驱动安装兼容的 PyTorch 2.5.x，再安装本工程。
python -m pip install -e '.[test,duet]'
python scripts/prepare_duet.py
python -m pytest -q
```

`prepare_duet.py` 固定 DUET 提交 `93e8b233164bc079a6db48b8a0a78d123ec8de41`，
校验源码并拒绝覆盖未知修改。修改仅有：两处 `np.bool` 替换为 `np.bool_`；
在 logits 输出后、softmax 前插入可选 hook。评测器保持上游内容并校验哈希。
`third_party/` 不加入本工程版本库，可以从固定来源重建。

按 [Matterport3DSimulator 官方说明](https://github.com/peteanderson80/Matterport3DSimulator)
安装 MatterSim，并把 build 目录加入 `PYTHONPATH`。安装应采用适配当前驱动的工具链，
不要直接照搬旧 CUDA 10.1 配置。数据及衍生特征按提供方的访问条件获取。

从 [DUET 官方资源入口](https://github.com/cshizhe/VLN-DUET) 准备：

```text
datasets/R2R/
  annotations/R2R_train_enc.json
  annotations/R2R_val_unseen_enc.json
  annotations/R2R_val_seen_enc.json       # 只在评测 val_seen 时需要
  connectivity/*_connectivity.json
  features/pth_vit_base_patch16_224_imagenet.hdf5
  trained_models/best_val_unseen         # 必须是 fine-tuned checkpoint
```

完整 connectivity 也包含 angle-feature 初始化所用的 `ZMojNkEp431` 场景。
可以在 `configs/r2r.json` 改数据目录及权重路径，相对路径以工程根目录解释。
首次构建基座还会读取 `bert-base-uncased` 的公开配置，可提前缓存；不需要重新预训练 BERT。

```bash
python scripts/preflight.py
```

未就绪时返回码为 2，报告缺失项与实际 GPU 显存；不会自动租卡或购买服务。

## 接入 GPU 后按顺序运行

以下命令在工程根目录执行，采用激活的 Python 3.11 环境。
默认 `batch_size=1`；先记录资源，再决定是否增大。更改公共评测配置后，基线也要重新运行。

### 1. 零残差一致性与资源检查

```bash
python scripts/run_duet.py --mode identity --split train_dev --limit 32 \
  --output outputs/identity-smoke.json
```

这个模式先运行原始策略，再运行零残差策略，逐条比较轨迹；不一致直接报错。
小规模检查通过后，应去掉 `--limit` 再验证一次。

### 2. 采集训练缓存

```bash
python scripts/run_duet.py --mode collect --split train_fit \
  --cache outputs/cache-round0 --output outputs/collect-round0.json
```

缓存每片最多 128 个决策；表征以 FP16 存盘、分数以 FP32 存盘。
基座始终 eval/no_grad。推理过程中只输入合法观察，教师只用于生成训练标签。
缓存及 head 记录基座、特征、训练标注、图、源码与划分指纹；每轮另记采集策略身份。

### 3. 训练一个候选，并在开发集比较

Colab 长任务使用受保护的训练入口。先挂载云盘，并为每个实验设置唯一 `run_id`：

```bash
bash scripts/train_colab.sh --config configs/pipeline.json
# 新 VM 准备好依赖、数据和相同云盘后：
bash scripts/train_colab.sh --config configs/pipeline.json --require-resume
```

它在每个 epoch 后评测 `train_dev`，自动选 best。备份位置、清理规则和导出命令见
[训练流程说明](docs/training_pipeline.md)。下面的旧入口适合一次性短实验，不具备上述续训保护：

```bash
python -m vln_improve.train --cache outputs/cache-round0 \
  --output outputs/head-m2-seed0.pt --device cuda --seed 0 \
  --epochs 3 --batch-size 32 --hard-weight 3 --kl-weight 0.1

python scripts/run_duet.py --mode baseline --split train_dev \
  --output outputs/baseline-dev.json
python scripts/run_duet.py --mode eval --split train_dev \
  --head outputs/head-m2-seed0.pt --output outputs/m2-dev.json
python scripts/compare_metrics.py --baseline outputs/baseline-dev.json \
  --method outputs/m2-dev.json --output outputs/m2-dev-comparison.json
```

训练报告只有离线 loss。SR/SPL 必须来自 `run_duet.py` 的完整 rollout。
按上面的 M0/M1 参数重复训练和开发评测，限制首轮搜索范围。

### 4. 可选：补一轮新策略数据

```bash
python scripts/run_duet.py --mode collect --split train_fit \
  --head outputs/head-m2-seed0.pt --cache outputs/cache-round1 \
  --output outputs/collect-round1.json
python -m vln_improve.train --cache outputs/cache-round0 --cache outputs/cache-round1 \
  --output outputs/head-m3-seed0.pt --device cuda --seed 0
```

该命令在聚合数据上从零初始化重训小头，保留原始 DUET 基座。

### 5. 锁定方案后的正式评测

```bash
python scripts/run_duet.py --mode baseline --split val_unseen \
  --output outputs/baseline-unseen.json
python scripts/run_duet.py --mode eval --split val_unseen \
  --head outputs/head-m2-seed0.pt --output outputs/m2-unseen-seed0.json
python scripts/compare_metrics.py --baseline outputs/baseline-unseen.json \
  --method outputs/m2-unseen-seed0.json --output outputs/m2-unseen-comparison-seed0.json
```

将训练种子改为 1、2 重复最终方案，评测 seed 保持默认 0。所有输出使用新的文件名。
比较工具拒绝样本缺失、重复 ID、错误指标范围或协议不一致；SR/SPL 输入为 [0,1]，
输出 `delta_pp` 和 `ci95_pp` 为百分点。少于两个场景时不生成跨场景置信区间。

## 资源和下一步

笔记本主要承担冻结基座推理、小头训练及开发评测。若采集/评测吞吐量太低，
再把这些作业集中放到短租单卡；具体卡时以首轮测量为准。
本次 8 条开发任务的基线 rollout 约 2.51 秒，PyTorch 峰值已分配显存约 0.69 GiB；
该记录仅对应这批短任务，不包含全部进程显存，也不代表新方法或完整训练的需求。

下一阶段使用已封存的 E1 四臂 final 完成 2,890 条 train_dev 导航复核，比较 SR、SPL 和返回成本。
C3 的配对辨别结果支持继续检验数据构造，但 M 的排序损失尚无额外贡献。
这轮完整导航通过后，再按预先登记的预算决定后续验证与多种子实验；保留跨面板代价和负结果。
