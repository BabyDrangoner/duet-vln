# 多来源证据与到达配对：诊断采集实现规格

更新：2026-10-03。本文件保留诊断设计及其理由；D0、D1 自然观测采集与回放已完成，尚无导航改善结论。执行中的一次补充分支观察以 `configs/branch_extension.json` 为准：使用原 D1 状态和预先选定候选，标记 `simulated_branch`，不等同于下文早期设想的仅邻接 `shadow_adjacent`。当前固定分析器使用 ridge/logistic，未实施下文建议的小 MLP；不应把设计建议写成已完成实验。
基于 DUET 固定提交 `93e8b233164bc079a6db48b8a0a78d123ec8de41`，当前 `configs/r2r.json`：
`fusion=dynamic`、`enc_full_graph=true`、`graph_sprels=true`、`act_visited_nodes=false`、
`num_pano_layers=2`、`max_action_len=15`、`batch_size=1`。

## 1. 优先确认的代码事实

1. **当前配置基本不产生同一来源的重复观察。** 已访问节点被动作 mask 排除；每次全局动作选择一个未访问终点。途中经过的已知节点写入轨迹，但不会逐节点执行全景编码。一次 episode 内，作为决策当前位置的 source 通常只出现一次。因此同一 `(source, target)` 的重复写入应为零，先用采集断言验证。不能以“机器人反复看同一门口导致过计数”为默认 DUET 的主要问题。不同已访 source 对同一未访 target 的多来源写入仍然可能发生。
2. **代理证据已经包含全景上下文。** `pano_embeds[i,j]` 经过图像投影、相对角度、导航类型嵌入以及两层全景 self-attention；它不是原始单方向图像特征，也尚未与指令交互。可以称“朝候选方向的全景上下文表征”，不能声称它仅看见一个视角。
3. **图记忆没有保存过时的指令分数。** 每一步图和局部分支都与同一份文本编码重新交互。课题不能叙述为“更新旧指令相关性”。
4. **到达全景也不是无偏真值。** `avg_pano_embeds` 是当时候选行与其余视图行的掩码平均，包含当前朝向及导航类型。候选数可能大于独立方向数，平均中部分原始方向可能重复出现。把它替换到未访节点会造成表示分布和坐标语义变化。
5. **图内替换会影响其他节点及融合权重。** 图 self-attention 改变整张图，包括 STOP 表征；动态融合使用 global/local STOP 表征。因此即使局部分支输入不变，输出 `local_logits` 仍会因融合权重改变。不能只重算某一候选的标量 head，也不能假设其他 logits 固定。

上述事实决定第一轮任务：测量“不同来源的代理表征压缩后是否丢失了有用信息”，再判断到达监督是否能补充普通聚合；同来源去重暂只作为健壮性测试。

源码定位：

- [agent.py：全景输入构造](../third_party/VLN-DUET/map_nav_src/r2r/agent.py:51)
- [agent.py：动作及 rollout](../third_party/VLN-DUET/map_nav_src/r2r/agent.py:287)
- [graph_utils.py：写入与平均](../third_party/VLN-DUET/map_nav_src/models/graph_utils.py:114)
- [vilmodel.py：全景与导航前向](../third_party/VLN-DUET/map_nav_src/models/vilmodel.py:704)

## 2. 当前证据的准确来源

记 `B` 为 batch，`D=768`，`A=4`，`C_i` 为第 i 个位置的邻居候选数，
`P_i=C_i+36-|unique(pointId)|`，`N_i` 为包含 STOP 的图节点数，`T_i` 为文本长度。
`P_i` 不保证等于 36；多个候选可能共享原始视觉方向，但相对角度或上下文编码仍不同。

| 字段 | 合法来源及形状 | 用途与注意事项 |
|---|---|---|
| 原始方向视觉 | `obs[i]['candidate'][j]['feature'][:768]`，`[D]` | 可选保存，用于区分原始视觉差异与全景编码差异 |
| 原始候选角度 | 同一 `feature[768:]`，`[A]` | 相对当前 heading/elevation；不是全局坐标 |
| 核心证据 | `pano_embeds[i,j]`，`[D]`，`j<C_i` | 必须在图累加前克隆；这才是实际写入候选图节点的值 |
| 候选身份 | `pano_inputs['cand_vpids'][i][j]` | 与 `obs[i]['candidate'][j]['viewpointId']` 核对，不用 `pointId` 索引 `pano_embeds` |
| 来源身份 | `obs[i]['viewpoint']` | 仅作关联、计数、去重；不把房屋或节点 ID 学成 embedding |
| 来源几何 | 当前与候选 `position`、heading、elevation、pointId | 已暴露的邻接观测；距离用位置计算。候选的 `distance` 在首次构造时是角度误差，且缓存路径可能没有此键 |
| 采集时间 | rollout 的 `t`、每候选 first_seen/last_seen | 未访节点的 `gmap_step_ids` 固定为 0，不能用它代替发现年龄 |
| 当前聚合 | `nav_inputs['gmap_img_embeds'][i,k]`，`[D]` | 应等于该未访候选截至当前的全部写入值之均值 |
| 当前直接观察 | `avg_pano_embeds[i]`，`[D]` | 当前位置到达表征；只为较早的待配对状态提供标签 |

所有采集张量采用 `detach().cpu().clone()`。`GraphMap` 首次写入保存张量引用，后续 `+=` 原地累加；只保存引用或仅 `detach()` 会让历史“单条证据”随图累计一起改变。

第一轮不设置每候选 4 条记录的截断：先保存实际完整来源及计数，测清分布。方法实现时再选择有界容量，并报告丢弃比例。最多 15 个决策步，诊断规模下没有必要先牺牲证据可追溯性。

**禁止混入当前策略输入：** `obs['distance']`、`gt_path`、真实目标 ID、完整环境 shortest path/distances、到达后的新邻居、新图位置、新指令状态、未来图 embedding。环境对象本身含有这些量，不能整体序列化 `obs` 作为模型输入。

## 3. 采集插入点及调用约定

新增可选 `diagnostic_observer`，只记录、不得返回修改后的策略张量。现有 `DecisionHook` 是输出后接口，无法恢复平均前的来源；旧动作缓存 schema 保持不变。

### A. episode 初始化

在 `agent.py::rollout` 创建 `gmaps` 后初始化各 episode 的独立 observer 状态。
键使用 `(collection_id, scan_id, instr_id, episode_occurrence)`，其中 occurrence 区分数据尾部 batch 回绕。正式输出每个 instruction 只接收第一次完整 occurrence，重复 rollout 不重复计算统计。

保存 episode 级 `txt_embeds [1,T,D]`、`txt_masks [1,T]`，以及指令 token 的校验值。文本编码每条 episode 一份，状态记录通过 ID 引用。

### B. 全景输出后、图写入前

准确位置：当前 `agent.py` 第 358 行后、第 360 行前。

1. 对 `ended[i]==False` 的 slot，记录当前实际观察；若当前位置对应之前未访候选的待配对请求，在这里完成第一份自然到达标签。
2. 只对 `not gmap.graph.visited(candidate_vpid)` 的候选记录代理写入事件，和原始更新循环使用同一个条件。
3. 事件保存克隆后的 `[D]` 表征及上一节的来源元数据。
4. 允许此时为前一步完成配对，即使当前随后选择 STOP；不能为已结束 batch slot 再生成新证据。

时序陷阱：上一轮动作后的 `_get_obs()` 随即调用 `gmap.update_graph(ob)`，它已经把新终点加入 `graph._visited`。因此这里的当前 v 即使首次到达，`graph.visited(v)` 也已经为真。**到达判断用 observer 独立的 `direct_observed` 集合和 `pending[v]`，不能用 `not graph.visited(current_vp)`。** 记录本次直接观察、闭合旧 pending 后，再把 v 加入 `direct_observed`。对邻居候选是否继续添加代理证据则仍使用 `not graph.visited(candidate)`，和基线写入条件严格相同。

不要改 `GraphMap.update_node_embed` 的计算顺序、求和 dtype 或 inplace 行为。observer 从旁记录，图策略仍按原实现运行。

### C. 导航输入构造完成、原始前向输出后

准确位置：`nav_outs = self.vln_bert('navigation', nav_inputs)` 后，现有修改 logits 的 `decision_hook` 前。

- 保存同一步的完整冻结输入与原始输出。初版仅采集原始 DUET，禁止加载残差头。
- 保存各合法未访节点对应的证据 event ID 集合；节点顺序以当前 `gmap_vpids` 为准。
- 标记 `ended`、`no_vp_left`、`forced_last_step`。这些状态可用于关闭旧配对，但不生成新的动作错误训练样本。
- 保存 logits 的 argmax；若还需要真实执行动作，增加动作转换后的只读事件，区分“模型选中”与“因步数上限强制 STOP”。

### D. rollout 完成

将仍未到达的候选标成 `arrival_available=false`，不能记为负例；写明 censor 原因：episode STOP、步数上限、未被选择。
自然配对必须满足 `arrival_step > state_step`，两者相同 episode/scan/target。最后回退到历史最大 STOP 节点只会追加路径，不能据此捏造该节点的新到达图像。

新增 hook 要通过 `prepare_duet.py` 的确定性补丁与 prepared hash 管理，不能直接修改上游而绕开 `verify()`。新采集入口独立 `scripts/collect_diagnostics.py` 较清楚；复用初始化逻辑，但使用新 schema 和输入白名单。

## 4. 冻结状态反事实前向：必须保存什么

每个选中状态保存以下完整输入。初版 batch=1、FP32；不要先压成 FP16。空回放先通过后再评估量化误差。

| `navigation` 输入键 | 每条状态形状/类型 |
|---|---|
| `txt_embeds`, `txt_masks` | `[1,T,D]` float32，`[1,T]` bool；可引用 episode 级存储 |
| `gmap_img_embeds` | `[1,N,D]` float32，含 STOP 行 |
| `gmap_step_ids` | `[1,N]` int64 |
| `gmap_pos_fts` | `[1,N,7]` float32 |
| `gmap_masks`, `gmap_visited_masks` | `[1,N]` bool |
| `gmap_pair_dists` | `[1,N,N]` float32；保持原始尺度，不自行除以 30 |
| `gmap_vpids` | 一层 batch list，长度 N，首项 `None` |
| `vp_img_embeds` | `[1,P+1,D]` float32，含 STOP 行及所有非候选视图 |
| `vp_pos_fts` | `[1,P+1,14]` float32 |
| `vp_masks`, `vp_nav_masks` | `[1,P+1]` bool |
| `vp_cand_vpids` | `[[None]+候选ID]`，长度 `C+1`，不等于 P+1 |
| `vp_obj_masks` | 显式 `None`，当前 R2R 无 object 分支 |

另存 `no_vp_left` 和动作有效 mask、原始 global/local/fused logits、当前 viewpoint/heading/elevation、t、来源关联、基础 checkpoint SHA、源码 hash、配置、特征 hash、划分及软件环境。
`gmap_embeds` / `vp_embeds` 是前向输出，不能代替上述前向输入。

### 替换定义

对状态 t 的合法未访候选 v，取后续首次直接观察得到的 `arrival_avg[v]`：

```python
with torch.no_grad():
    baseline = model('navigation', clone(frozen_inputs))
    alternative = clone(frozen_inputs)
    alternative['gmap_img_embeds'][0, candidate_index] = arrival_avg
    changed = model('navigation', alternative)
```

只替换一个候选的 `gmap_img_embeds`；不更新图、step ID、位置、mask、文本、局部视图或候选顺序。整个 `navigation` 正常重算，包括动态融合与所有动作。不要把候选改成 visited，否则合法动作直接被 mask 掉。

令合法动作集合为 L，fused logits 为 z，使用
`m(v,z) = z[v] - logsumexp(z[a] for a in L if a != v)`，
`delta_margin = m(v,z_changed)-m(v,z_baseline)`。
保存完整变化向量、argmax 是否变化、global 分支变化与 STOP 变化。另一可解释输出是相对于“替换前最高分的其他动作”的固定参照 margin。遇到无其他合法动作时记缺失。

**delta_margin 叫表示替换敏感性，不叫可靠性真值或因果导航收益。** 将后续观察硬插入历史状态是离线干预，未必来自自然可实现的状态分布。

### 必做的有效性控制

- identity：用原 `gmap_img_embeds[v]` 替换自身，FP32 同设备回放的 mask、argmax 与 logits 应匹配；先按 `atol=1e-6, rtol=1e-5` 检查并报告实测最大误差，不能因失败自行放宽到掩盖排序变化。
- same-mean：按保存的全部来源重建均值，检查与原图 embedding 一致；原顺序重建，避免浮点求和顺序混淆。
- mean/last/uniform-source/small attention：检验普通聚合能否解释全部变化。
- shuffled-arrival：同房屋、相近几何/步数的错误目标配对，检验变化是否仅由“任意到达全景”造成。只用于负控制，不参与真实策略输入。
- norm control：记录替换前后 embedding 范数，增加范数匹配版本；若任意同范数噪声具有同样关联，停止可靠性解释。
- orientation control：记录来源与到达的相对角度；抽取一小部分到达点，在独立离线实例按固定朝向重新编码。若结论主要由朝向或候选行重数变化驱动，不能解释为语义证据被证实。
- dynamic-fusion 分解：先按原模型完整重算；随后仅作诊断，分别记录未乘融合权重的两分支分数及融合权重变化，定位效应。固定融合权重的对照必须明确标注已改变原计算路径。

## 5. 到达观察仅作离线标签

按文件及加载 API 隔离：`evidence/` 与 `states/` 只含当前可用输入，`arrivals/`、`labels/` 独立存后续观测、oracle 和干预结果。
policy adapter 的函数签名不接收 arrivals/labels/environment。label builder 单独进程读冻结状态与后续观察，不反向修改采集时的事件。

初版用自然到达配对：前面每个状态均保留候选快照引用，第一次在 v 执行全景编码时闭合 `state_step < arrival_step` 的请求。v 被发现但未到达的样本仍计入覆盖率分母。
自然到达偏向基线认为较好的节点；不能把配对样本上的关联外推到所有候选。

第二批仅在需要时添加“已知局部相邻边的补充观察”：

1. 固定原始 baseline rollout，不修改它的动作或累计路径。
2. 从当前合法局部候选按基础 rank、来源数、几何距离分层抽样，记录抽样概率。
3. 独立 simulator/environment 从当前 source 沿已暴露的合法邻接边到达 v，编码 v 的全景作为离线补充标签。不得复用并改变在线 simulator 状态。
4. 补充样本标记 `arrival_kind=shadow_adjacent`；自然样本为 `natural`。两个来源分开报告；补充观察开销进入训练成本，不进入基线导航轨迹。

隔离测试：打乱或替换 future arrival、目标位置和真实路径后，之前保存的策略输入 hash、基线 logits、实际轨迹完全不变；缺失 labels 时策略推理仍能运行。替换 GT 的测试保持合法当前 observations 不变，只修改标签接口，不能重新选择起点任务后要求轨迹相同。

## 6. 覆盖率、重复与冲突统计

第一份报告先包含所有未结束、非强制停止的状态，而非仅有 arrival 的样本。至少按 scan、episode、state、candidate 四层计数，提供分母：

- `num_writes`、`num_unique_source_vps`、`num_unique_source_pointids`、first/last seen、发现年龄。
- 候选级 `unique_source>=2` 比例、状态级“至少一个多来源候选”比例、被选动作中的比例，以及对应 episode 的覆盖率。
- 真正重复键 `(episode, source_vp, target_vp)` 次数。当前配置预计为 0；出现时先排查尾部 episode 回绕、ended slot、采集重复或启用了 act_visited_nodes。
- 不同来源的相似特征单列：来源视觉的 cosine 分布、上下文表征的 cosine 分布、源位置间距与角度覆盖。不要把语义相似当作同一观测。
- 同 source 同 pointId 对多个 target 的原始视觉重用单列，它属于跨候选歧义，不等于同 target 重复来源。
- 自然配对率按基础动作 rank、是否被选、来源数、距离和步数分层；报告未配对的 censor 原因。
- 来源差异先用连续量（平均成对 cosine、到均值距离、有效秩）描述，不根据最终方法获益回头定义“冲突”阈值。阈值只能在 train_fit 固定，再到保留房屋验证。

复制干预的数学预期：只有一个来源，或所有来源相同，均值复制应不变；把全部来源等比例复制也应不变。只复制多来源集合中的其中一条会把均值拉向它，其 logit 变化只能证明均值按写入数加权，不能证明真实轨迹中发生了过计数。保留这一检查，但不要把它当主结果。

## 7. 控制几何和基础分数后，如何判断额外价值

### 标签与“错误”的名称

在 label builder 中计算训练期 oracle 代价，不能传给 adapter：

- `teacher_cost(v)=d_full(current,v)+d_full(v,goal)`，复现当前 `expert_policy=spl` 的教师规则。
- 另报 `execution_cost(v)=d_discovered_graph(current,v)+d_full(v,goal)`，因为实际全局动作执行已发现图路径，教师的完整图最短路可能经过尚未发现的捷径。
- 非 STOP 候选使用相对最小代价的 regret，允许浮点容差内多个同优动作。不要把教师按顺序选出的一个 index 当成唯一正确动作。
- STOP 单独分析。精确到目标才 STOP 的教师规则与评测成功半径不同；teacher disagreement 不是导航失败，更不是指令遵循真值。
- 最终 episode SR/SPL/nDTW 可作描述性分层，但同 episode 多状态不是独立的失败样本，也不能由离线 regret 改善直接宣布 SR 提高。

### 固定 probe 对照

所有 probe 使用相同样本、相同 scan 划分和相同调参预算，标准化仅拟合训练房屋。

| 对照 | 特征 | 解释 |
|---|---|---|
| P0 | fused/global/local 分数与 rank、top margin、STOP 概率、候选数、是否 local、t、发现年龄、7维图位置特征、来源距离/角度汇总 | 足够强的现有分数＋几何基线 |
| P1 | P0＋当前来源数、视觉/上下文差异及可观测的来源集合摘要 | 来源信息是否有额外可用信号 |
| P2 | P0＋真实 `delta_margin` 与替换后变化 | 仅衡量离线后续观察的关联上限；明确不可部署 |
| P3 | P0＋只从当前证据预测得到的 `predicted_delta` | 检验辅助信号是否可提前预测并有实际信息增量 |

先采用正则 logistic/ridge 与一个相同容量的小 MLP 两种简单 probe，避免只挑一个拟合器讲故事。任务分别为候选是否低 regret、状态 teacher disagreement，以及连续 regret。报告 held-out log loss/Brier、AUROC（两类都存在时）、AUPRC、MAE，并给出按房屋聚类的配对 bootstrap 区间。

P3 的训练预测必须 cross-fit 或使用独立训练子折，不能先在同一行拟合 delta 再当“可部署预测”送入误差 probe。P2/P3 只在同一批有标签样本上成对比较；自然和补充配对各报一份，不用配对缺失本身当可靠性标签。

先在 `train_fit` 房屋内拟合/调参，最终预研结论在预先保留的 `train_dev` 房屋检查。DUET 原始检查点可能已经用过这些训练房屋，所以它们只是本项目新模块的保留集，不冒充基座完全未见环境。不要用官方 `val_unseen` 反复筛选假设。

## 8. 可落地的最小版本及停止条件

### M0：只读采集与覆盖率

先 16 条跨房屋任务验证 hook 开/关逐轨迹一致、来源均值重建及 episode 隔离。随后固定抽样约 1,000 个决策状态，跨多个训练房屋，保存 FP32 状态和全部来源。每状态存一次，不为每个候选复制整份张量。

文本按 episode 共享；状态按短 shard 顺序保存；完成一个 shard 就生成大小/SHA-256/COMMITTED 并备份。诊断数据用独立 schema，不得冒用只接受旧动作缓存的 `prepare_inputs` 校验。保存已完成 episode 清单，断线时只重跑未完成 episode；不要宣称模拟器能在任意半步精确续接。

预算先由 16 条实测估算，粗略单状态 FP32 张量字节数为
`4 * (D*(N+P+1) + N*N + 7*N + 14*(P+1))`，另加共享文本与事件表。典型几十节点约数百 KB/状态，1,000 状态可能数百 MB；这只是容量公式，最终以实测为准。不要给每候选重复储存文本/全图。

### M1：标签生成与便宜的证伪

先处理自然到达的每状态至多两个候选（基线选中、另一个固定随机候选），保留原始抽样概率与全部覆盖统计。每个状态 baseline 重放一次；每个标签单独多一次 navigation 前向。其余诊断只在较小固定子样本执行。未选中候选不足时再增加相邻边补充观察。

最低交付：覆盖报告、可重放状态、无泄漏检查、替换敏感性/几何控制 probe、正负案例，以及每步耗时和存储量。此阶段不训练新导航方法。

### 进入方法实现的条件

- 多来源情形在正常轨迹里有足够覆盖；若只能靠人为重复生成，应撤下重复证据的主叙事。
- P2 在保留房屋上相对 P0 有可复核增量，且不被朝向、范数、任意全景替换或配对选择偏差解释。
- P3 或直接 P1 有当前可用的预测信号；只有事后信息有效、提前无法预测，不能支撑部署方法。
- 普通聚合不足以解释拟议监督的全部价值；若 last/attention 与带到达监督等效，优先接受简单解释。

若配对不足、置信区间宽或标签由几何控制解释，结论记为“不足以支持”，有限扩大采样一次或停止该候选。不得用更多模块代替问题成立的证据。真正方法成立仍需同预算强对照与完整导航收益；本规格的离线测试不能替代论文实验。

## 9. 基线协议核对与批量提速门槛

固定提交的 [官方 R2R 启动脚本](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/scripts/run_r2r.sh) 把同一组 flag 用于训练与 `--test --submit`：`max_action_len=15`、`max_instr_len=200`、`batch_size=8`，并使用 dynamic fusion、完整图、图关系与 SPL expert。官方 [README](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/README.md) 的评测入口指向这一类脚本。

因此当前动作上限 15 与官方一致；当前 batch=1 是本项目的保守实现选择。15 是高层决策循环次数，最后一次还会强制结束，不是最多走 15 条物理边；一次全局动作的已发现图路径和最终 STOP 回退都可能增加轨迹边数。任何方法都不能通过减少路径计费制造 SPL 改善。

本轮先保持 `max_action_len=15/batch_size=1` 跑完整 `train_dev` 基线，不修改既有配置。若后续提速到 batch=4，先做以下独立协议检查：

1. 固定相同的 32 条指令、权重、特征、seed、动作上限与代码，唯一变化为 batch size；使用两个输出目录，不能覆盖原基线。
2. 按 instr_id 比较完整轨迹（包括中间路径、停止回退）、每任务 NE/SR/SPL/nDTW、最终指标和覆盖的 instruction 集合，而非只比较平均 SR。
3. 记录两种设置的 wall time、peak allocated/reserved 显存，以及任何最先分叉的状态及 top-two margin。eval 模式、mask 和 sample 独立性按实现应支持批处理，但 CUDA 浮点计算及不同 padding 尺寸可能影响近平局动作；不能预先保证一致。
4. 轨迹全相同时才将 batch=4 视为本数据子集通过的提速候选，并在正式基线/方法中统一该设置。32 条通过不是任意输入的数学保证；完整实验仍记录 batch size。
5. 现有 `protocol_sha256` 包含 batch size，因此两次报告的 hash 预期不同，不能用“完全相同协议”的严格比较器直接报错后跳过检查。专用 parity 检查仅允许 batch size 字段差异，其他协议项逐一相等，再比较轨迹。

完整 `train_dev` 是本项目选模与预研基线，不是官方论文 `val_unseen` 成绩；当前自定义 train/dev 房屋划分不能与论文表格直接对齐。
