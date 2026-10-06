# D2 分支诊断判读审计

2026-10-03，在 D1 分支报告产生前完成。只读审查 `analyze_diagnostics.py`、`probes.py`、`replay_branches.py` 和 `configs/branch_extension.json`；没有改变运行代码、采样、拟合器或门槛。D0 仅作接口验证，不加入统计。

## 1. 已登记规则及结论边界

以 `configs/branch_extension.json` 为准：唯一一次补采，固定 D1 历史状态、`chosen_candidates(seed=0)`、alpha=1、训练房屋 5 折、seed=0 的 1000 次整房屋配对 bootstrap。主指标为 teacher-optimal log loss；regret MAE、朝向和覆盖分层是辅助诊断。

登记的晋级条件是定性的：既要有当前可见证据的增量，也要有匹配控制不能解释的、与目标观测有关的辅助信号。配置没有登记最小 log-loss 改善幅度、校正后的显著性水平或自动判定公式；不能看结果后补设数值标准并称其预注册。点估计变好但区间仍容许明显变差，不应表述为方向已确定。

若本次仍证据不足，停止这条 arrival-margin 监督方案，不再换样本、模型、阈值或 delta 定义寻求通过。离线探针通过也只支持进一步研究，不能宣称 SR/SPL 改善、因果可靠性或论文规模已满足。

## 2. 报告完整性核对

| 项目 | train_fit | train_dev |
|---|---:|---:|
| D1 collection identity | `c7b1a8ce3940f3bc29958a173ddc01eb8630287ed906092b4b9256722d4a1223` | `019b63425176ee7e545543137e134e476b15b59823b26e7ea1a6c627d98f68fa` |
| episodes / scans | 193 / 49 | 96 / 12 |
| 全部历史 panorama 检查 | 1139 | 599 |
| eligible navigation identity 回放 | 1139 | 598 |
| `selected_candidate_states` / `paired_rows` | 2061 / 2061 | 1087 / 1087 |
| `natural_chosen_parity_checks` | 946 | 503 |
| `selected_without_natural_arrival` | 1113 | 576 |

具体检查位置：

- `REPLAY.json`：`schema=duet_simulated_branch_replay_v1`；collection SHA 对应上表；`analysis_cohort=simulated_branch_all_preselected_candidates`；seed=0；9 项 controls 与登记一致。核对 runtime 的模型配置、权重 SHA、上游锁和 torch 版本，及 `simulator_sha256`、implementation 文件 SHA。
- `summary.json`：`status=complete`、split/usage 正确、rows/episodes/scans 及 `counters` 对应上表；`max_replay_error`、`max_natural_feature_error` 报告实际数值，现有代码容差和 argmax 检查均已通过。不能只用 D0 的零误差代替 D1 验证。
- 每 episode JSON：内容 SHA、输入 manifest SHA、replay identity、association 通过 `read_branch_result`。`selections` 与原 D1 同 seed 预选列表一致；row key `(scan,instr,step,target)` 无重复、无缺失；主样本不因 donor 缺失被删。
- `rows.jsonl`：所有行 `arrival_kind=simulated_branch`，没有虚构的原轨迹 `arrival_step`。natural 报告和 D0 smoke 不合并。与每 episode 的 rows 完整拼接对应，并核对云盘读回 SHA。
- 分析报告：`input_files` 的 SHA 与最终 JSONL 对应；analysis provenance 对应封版 analyzer/probes；`protocol` 中 alpha=1、seed=0、bootstrap=1000、arrival kind 正确；训练与开发房屋不交叠。

注意：全景检查覆盖所有历史状态；完整 navigation forward 只覆盖 eligible 状态。开发集有 1 个不可执行末状态，因此不能写“599 个状态均重放了导航 logits”。Drive 挂载读回校验不等于独立服务端持久化回执。

## 3. 固定判读顺序

1. **可用错误量与样本组成。** 检查 `data.selection_coverage` 的 chosen/unchosen × single/multi-source 四格及每房屋正负标签。对所有固定行描述，不据此重新筛样本或加权。只有少数房屋含错误时，需明确这限制了证据强度。
2. **当前可见信息。** 主看 `paired_loss_comparisons.teacher_optimal.P0_P1`，然后同时看 `P3_predicted_arrival_delta` 和 `P3_vs_P0_P1`。差值为模型 loss 减参考 loss，负值更好；同时报告原始 log loss、差值和区间。不能只凭 P3 胜 P0 跳过它与 P0+P1 的直接比较。
3. **目标观测是否有特异性。** 同样本查看 `shuffled_control` 的 `true_arrival_vs_shuffled`，以及 `normmatched_shuffled_control` 的 `true_vs_shuffled`。必须使用这些子集内重新拟合的 P0/true/shuffled，不与全样本 P0 数值拼接比较。`normmatched_arrival_vs_shuffled`（raw donor）不是对称范数控制，不据此排除 scale 解释。
4. **朝向及扰动解释。** 查看 `heading_control`、`headingmatched_shuffled_control` 的对称比较，以及主表 `normmatched_vs_noise`。原 arrival 和朝向控制均完整报告，不选择其中结果最好的一项替代预定主结果。`arrival_headingmatched_normmatched` 仅被校验/记录，没有新增统计假设。
5. **统一结论。** 当前可见增量与目标观测特异性必须一起评估；辅助 MAE/AUROC/AUPRC 的偶然改善不推翻主判读。teacher/execution 若标签相同，只是一组分类证据。同一样本的自然/分支结果也不是两次独立复现。

## 4. 可能造成正向误判的边界

- **P3 不增加推理时的信息。** 第一阶段是 P0+P1 的线性 ridge，dev 上 P3 仍在同一线性特征空间中；差异可以来自辅助目标和正则化。需与 P0+P1 比较，不能称为新观测信息。
- **多个相关比较。** analyzer 为多目标、多组输出未校正的 95% percentile 区间；不是家族错误率控制后的显著性检验。不能挑一个排除零的辅助结果宣布通过。bootstrap 固定已拟合模型，只重采开发房屋，未覆盖训练种子/拟合不确定性。
- **房屋样本少且权重不等。** 12 个 dev 房屋是重采样单位，但最终 loss 仍按候选行加权，轨迹长、候选较多的房屋权重更大；不是等权房屋平均。基座已见这些 R2R train 房屋，只能称适配器保留诊断。
- **候选任务和导航任务不同。** chosen+hash-other 不均匀覆盖所有动作；增加错误的未选候选使分类更有区分度，但不自动说明能纠正实际动作。teacher/execution regret 由路径成本定义，不等于完整指令遵循、导航最终成功或 STOP 质量。
- **特权观测及分布变化。** 目标 panorama 包含新邻居的 nav_types/角度；未来全景均值替换历史候选代理也可能落到模型分布之外。只替换一个节点的离线 loss 不能证明实际策略使用它会变好。
- **donor 并非完全匹配。** donor 仅按当前图距离差、来源数差、固定哈希选择；未严格匹配语义、方向、局部拓扑。需从 `row.shuffled_donor.distance_difference/source_count_difference` 描述全部匹配误差及缺失覆盖，不能事后删除差匹配行。当前 analyzer 不自动汇总这些差值。
- **JSON 校验不是来源校验。** analyzer 只接收数值 JSON，允许任意一致的 p0/p1 列名，也不读取 COLLECTION/REPLAY、不强制 2061/1087 行或 9 控制齐全。上面的来源、完整性、控制同样本核验必须在解释结果前完成；不能单凭 `status=complete` 宣布协议合格。

## 5. 标签隔离审查结果

现有实现中，P0/P1 仅从各自数值块构造；标准化只用对应训练折。P3 的训练 delta 预测按训练房屋交叉拟合，dev delta 预测仅用 train_fit 拟合的 P0+P1；dev 真 delta 只用于明确的特权参考及误差报告，dev oracle 标签只用于评价。未发现这条数据流中的直接 dev-label 泄漏。

这一结论依赖上游 `observable_features` 白名单和不可变 replay 来源。任意外部 JSON 把标签藏进名为 p0/p1 的列，analyzer 本身不能识别；应核对封版程序及实际列名，而非仅检查数值类型。

## 6. 完成后独立判读：不晋级训练

已核对 `outputs/study-20261003/d1-branch-probe-analysis.json`，其 SHA-256 为 `092bb14feffb0229c43d8d372bd384cc530b304c8607a82058f4c9f885e697eb`。本节只解释固定报告，没有重新拟合、改变样本或增加统计检验。

**结论：本次唯一补采完成后，证据仍不足以支持将 arrival-margin 监督晋级为主方法训练。** 停止这一候选监督方案，保留其工程、数据与负向诊断结果。此结论限于已登记的样本和探针，不能扩大为“所有多来源方法均无效”。

### 完整性与可用样本

- 本地 train/dev 的 REPLAY、rows 和 summary 文件 SHA 均与 artifact audit 一致；分析输入 SHA、运行实现及 analyzer/probes 源码 SHA 也吻合。两份 replay identity 与 summary 一致。
- 2061/1087 行完整且唯一，split/kind 正确，没有伪造自然 `arrival_step`。汇总记录的 1737 次 eligible navigation 回放、1738 次历史 panorama 检查、1449 次实际 chosen-next 检查均通过；记录的最大 navigation/自然特征误差为 0。云盘验证依据既有 artifact audit 的挂载读回记录，本次没有重新连接云端。
- 用本地 JSONL 标签和报告保存的预测独立重算全部主分类 log loss 与回归 MAE：最大差异为 `1.4e-17`。未重新拟合模型。
- train/dev 的 teacher-optimal 正/负数为 917/1144、483/604；每个 dev 房屋都有正负样本。两侧 teacher/execution 分类标签逐行完全相同，只计一组分类证据。
- 当前预选候选中，多来源样本为 train 357、dev 178；其中实际 chosen 仅 11、17。dev 未选多来源 161 行中只有 3 行 teacher-optimal；因此补采明显增加了负候选覆盖，仍不能把候选判别表现视为实际动作修复。

### 主分类结果

下表均为 teacher-optimal log loss。差值为候选模型减相应参考，负值更好。P0 的 log loss 为 **0.103629152**。

| 比较 | loss 差 | 95% 房屋 bootstrap 区间 |
|---|---:|---:|
| P0+P1 − P0 | +0.000283513 | [−0.004626954, +0.004820818] |
| P3 − P0 | −0.000096348 | [−0.000191833, −0.000015379] |
| P3 − P0+P1 | −0.000379861 | [−0.004993741, +0.004560090] |
| P0+true arrival − P0 | −0.000260764 | [−0.000771347, +0.000103316] |
| P0+normmatched arrival − P0 | −0.000570393 | [−0.001721678, +0.000192993] |
| P0+headingmatched arrival − P0 | −0.000446017 | [−0.001379245, +0.000255003] |

P3 相对 P0 有一个很小、当前未校正区间不跨零的改善，应如实保留。但 P1 没有清楚的主指标增量，P3 对同样可见的 P0+P1 比较不确定，且 P3 本身仍是这些输入的线性投影。不能挑出 P3 单项，绕过既定的整体判读条件。P1 的 AUROC/AUPRC 略高也不改变这一结论；其 log loss 和 Brier 均略差。

### 对称控制：差异主要来自 shuffled 变差

三个 shuffled 比较都用固定同样本子集：train 2037、dev 1073；缺失 donor 的 24/14 行仍保留在主分析中。

| 同样本分类比较 | true − P0（95% 区间） | shuffled − P0（95% 区间） | true − shuffled（95% 区间） |
|---|---|---|---|
| 实际到达朝向 | −0.000270 [−0.000796,+0.000107] | +0.001694 [+0.000185,+0.003621] | −0.001964 [−0.004381,−0.000177] |
| 对称范数匹配 | −0.000590 [−0.001780,+0.000200] | +0.002435 [+0.000538,+0.005177] | −0.003025 [−0.006864,−0.000580] |
| 对称历史朝向 | −0.000458 [−0.001417,+0.000260] | +0.001972 [+0.000236,+0.004158] | −0.002430 [−0.005489,−0.000182] |

真实目标与错配目标的响应确有区别，但主要由 shuffled 相对 P0 变差体现；真实目标相对强 P0 的三个区间均跨零。全样本 normmatched arrival 对 normmatched noise 的分类差为 −0.000822，区间 [−0.002470,+0.000348]，也没有形成清楚的额外优势。故这些控制尚不能支撑“具有实际增量的辅助监督信号”结论。

donor 距离并非精确匹配：dev 距离差中位数 0.348 m、95 分位 2.320 m、最大 12.828 m；来源数完全相同的比例为 62.44%。train 相应为 0.417/2.606/11.053 m、60.73%。这些是全部固定匹配行的描述，没有删除差匹配样本。

### MAE 辅助结果没有补强主结论

P0 的 teacher/execution regret MAE 分别为 1.068110/0.972577 m。

| 模型相对 P0 | teacher MAE 差（95% 区间） | execution MAE 差（95% 区间） |
|---|---|---|
| P0+P1 | −0.000426 [−0.004115,+0.003716] | +0.000936 [−0.003966,+0.006128] |
| P3 | +0.000024 [−0.001784,+0.002228] | +0.000534 [−0.001907,+0.003678] |
| P0+true arrival | −0.000936 [−0.003215,+0.001560] | −0.000176 [−0.001757,+0.001562] |
| P0+normmatched arrival | +0.000784 [−0.002381,+0.004168] | +0.001247 [−0.000978,+0.003442] |

P3 相对 P0+P1、历史朝向控制以及真实目标与各对称 shuffled 的 MAE 比较也全部跨零。相反，normmatched shuffled 在其固定子集上相对 P0 的 teacher/execution MAE 差为 −0.001236/−0.001350 m，对应区间均为负；真实 normmatched 目标没有同样表现。这进一步说明不能从少数分类对照的有利结果推导强机制，更不能将本轮结果写成导航指标提升。
