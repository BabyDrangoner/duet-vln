# 普通终点 sigmoid 强对照：工程接入方案

日期：2026-10-03。目标是检验冻结表征能否支持普通终点分类；不把该对照作为论文创新。本文只读取现有代码和训练集合摘要，没有重新读取验证集明细或运行 GPU。

> 执行补充：随后已完成 T4 特征重放与 E0 训练。fit 为 193 回合 / 1,139 状态 / 434 正例，dev 为 96 回合 / 599 状态 / 224 正例；全部状态的三分支 logits 精确复现。缓存提取分别约 46.78、23.99 秒，PyTorch 分配显存峰值均约 701 MiB。固定 20 epoch、140 次更新完成；第 17 epoch 的 dev BCE 最低，为 0.268408。该 head 已备份 Drive 并下载本地，SHA `5d0d2d737799046b6e66239caeacb82334a210ef2efe38b682b569d57a6f5b6b`。16 条无 head 评测接入检查逐轨迹一致；完整导航结果另行记录，不把 BCE 当导航收益。

> 特征缓存版本：这次全新生成使用 `e0-source.tar.gz` 中的 v1 导出器，完成的 manifest 与数据 SHA 已校验并固定。审计随后补强了 v2 恢复流程：按旧 COMMITTED / manifest 验证现存字节，部分缓存增加逐回合 seal，不重新签入已改写文件。v2 代码身份不同，后续采集使用新目录；既有完成缓存仍由严格训练加载器读取。训练 checkpoint 的恢复与清理由 CheckpointStore 独立管理。

## 1. 最小特征已经存在

固定源码的 `forward_navigation_per_step` 直接返回：

| 输出 | 形状 | 语义 |
|---|---|---|
| `gmap_embeds` | `[B,N,768]` | 图节点经过语言交叉注意力与图自注意力后的表征；第 0 项是全局 STOP token。 |
| `vp_embeds` | `[B,P,768]` | 当前全景经过语言交叉注意力与视觉自注意力后的表征；第 0 项是局部 STOP token。 |
| 三套 logits | 全局 `[B,N]`、局部 `[B,P]` | 已乘融合权重并应用相应动作 mask 的策略分数。 |

建议固定特征为 `cat(gmap_embeds[:,0], vp_embeds[:,0])`，得到 `[B,1536]` FP32。两个 token 都已经与当前视觉和语言交互，不需要再跑语言编码、全景编码或新增 PyTorch forward hook。读取时 `detach` 并复制，原 `nav_outs` 和策略 logits 保持原值。

这两个 token 不是当前位置的原始图像特征，而是含全景、语言、历史图上下文的停止表征。`nav_inputs.gmap_img_embeds/vp_img_embeds` 与 panorama forward 输出在该步尚未进行语言交互，不能与上述表征混称。全局当前节点也有跨模态输出，但首轮固定使用 1536 维，不追加特征搜索。

源码入口：归档 [vilmodel.py](../outputs/study-20261003/source/third_party/VLN-DUET/map_nav_src/models/vilmodel.py:750)、[agent.py](../outputs/study-20261003/source/third_party/VLN-DUET/map_nav_src/r2r/agent.py:382)。`VLNBert` 包装器原样返回该输出字典。

## 2. 优先重放 D1，避免重新走环境

现有 D1 完整缓存保存了真实决策时的 15 项 `nav_inputs`、三套原始 logits 和逐步 STOP 距离标签。最低成本做法是：

1. 使用原 `load_episode` / `validate_episode` 验证 episode 及其关联，严格加载相同冻结模型、上游源码、特征与运行时身份。
2. 对每个真实状态执行一次完整 navigation forward，核对原始三套 logits，然后提取两个 STOP token。
3. 保留全部状态，包括动作上限和无合法移动的结束状态；不沿用 arrival 的 eligible 筛选，不读取 arrival feature，不改任何历史输入。
4. 标签仅在独立字段保存：`y = 1[distance_to_goal < 3 m]`。特征构建函数不接收 `obs`、目标、标签或未来状态。

D1 train_fit 为 193 回合、1,139 状态，train_dev 为 96 回合、599 状态。原始缓存位于远端/云盘，本地摘要不能替代缓存。已有轻量 STOP trace 不含视觉/文本输入，不能恢复跨模态特征。

旧 action training cache 虽然 `features[0,:1536]` 包含两个 STOP token，但不保存 step、current viewpoint 或 3 米标签，还过滤强制结束状态。`teacher target==0` 表示精确目标，不等于 `<3m`；不能把旧 action target 直接转换成终点监督，也不能靠回合内顺序猜回缺失状态。

## 3. 缓存接口

`manifest.json` 使用 `schema='duet_endpoint_features_v1'`，包括：

- `feature_dim=1536`、`split=train_fit/train_dev`、`usage=training/analysis_only`；
- 完整 `identity` 与 `identity_sha256`，原 collection 身份及源码身份；
- `runtime`：模型配置、上游锁、权重 SHA、PyTorch 版本；
- `common_provenance`：权重、视觉特征、训练注释、connectivity、model、upstream_lock、partition_seed、dev_fraction、torch_version，fit/dev 应一致；
- `files`：每文件的 name、SHA、association、num_states、positives；以及 summary。

每个 episode 的 `.pt` 字典保存 `schema`、`identity_sha256`、`association={episode_id,scan_id,instr_id}`、`features:FP32[T,1536]`、`labels:FP32[T]`、`steps:int64[T]`、`viewpoints:list[str]`、`base_stop_probability:FP32[T]`、`distance_to_goal:FP64[T]`、`input_manifest_sha256`。距离用 FP64 保留 3 米边界。训练只从 features 读取输入，从 labels 读取监督；概率和距离仅校验/诊断。

每个完整 episode 原子提交、校验并同步云盘，不能把半回合当完成。完整缓存使用 `COMMITTED.json` 绑定 `manifest.json` 的 SHA，`IDENTITY.json` 保存相同身份；训练加载器逐文件校验 SHA、全部状态关联、距离标签和三分支严格 parity 声明，拒绝混合模型与数据版本。fit/dev 房屋必须不相交。训练数据身份绑定 identity 与有序文件清单，资源耗时摘要不参与该身份，避免完整缓存重新检查时的计时变化阻断续训。

## 4. 最小训练与资源

模型固定 `1536 → 128 → ReLU → 1`，输出 logit，训练用 `BCEWithLogitsLoss`。每回合先平均其全部状态损失，再对回合平均，使长轨迹不会因状态更多而获得额外总权重；不做正负类别重加权。工程 pilot 固定 seed 0、AdamW lr `1e-3`、weight decay `1e-4`、20 epochs，每个 minibatch 32 回合。

- 单状态特征为 6,144 字节；D1 fit+dev 的 1,738 状态原始特征共约 **10.18 MiB**。
- 10 万状态约 **585.94 MiB**，这是 CPU/磁盘特征大小；不需要一次全放显存。
- 小头共 **196,865 参数**。FP32 参数、梯度和 AdamW 两个矩合计约 **3.00 MiB**，另有小量 activation 和框架内存。
- 每批最多 32×15=480 状态的 FP32 输入约 **2.81 MiB**；训练可直接使用 CPU。缓存生成维持 batch 1，冻结骨干 `eval/no_grad`，每状态转 CPU 后释放输出，不累计 GPU tensors。

因此 12GB 显存足够该方案的新增开销；不把估算当成整套 CUDA 峰值保证。无需原始 RGB、额外大模型或全骨干反向传播。

现有 `CheckpointStore.save(state, head_payload, step, is_best, metrics)` 和 `restore()` 可直接复用其原子快照、云盘 SHA 读回、latest/best 保护和清理。**现有 `ResumableTrainer` 不能直接复用**：其数据和损失绑定 action CE+KL / bounded residual head，需独立 endpoint trainer。

endpoint 状态须保存模型、AdamW、Python/NumPy/torch CPU/CUDA RNG、epoch、回合采样游标、global step、待评估标志、每 epoch 历史，以及严格数据/代码/训练身份；在完整 optimizer step 边界保存。每 epoch 记录 train_dev 的回合平均 BCE。按此值保存的 best 必须称 **best_dev_bce**，不是最佳导航 checkpoint；官方验证结果不得参与选择。

## 5. 接入最终选点，保留原在线策略

首次普通对照只替换结束时历史位置的排序，保留原 `fused_logits`、移动 argmax、在线 STOP 和动作上限。在每个实际决策点缓存一个 endpoint logit；结束时从同一历史位置集合选最高分，按同一已发现图回到该位置，计入完整原前缀及返回路。

上游 `node_stop_scores` 当前保存 `{'stop':原始动作STOP概率}`，随后尾部回退读取它。独立 `EndpointReranker` 保存 endpoint logit，并包装 `make_equiv_action`：先完成原调用，仅当该步动作已为 `None` 时，使用原分数重建完整 baseline 路径并与参考逐条比较；通过后才把历史排序值换成 endpoint logit，返回路仍由上游加入。比较 logit 可避免 sigmoid 饱和产生人为平局，数学上与 sigmoid 排序等价；平局仍保留最早位置。这个适配器不修改冻结 agent 源码，训练模块也不接收导航目标。

不能把 q 写入 `fused_logits[:,0]`：那会改变在线停止与后续路线。也不能在现有 decision_hook 中直接修改 `gmaps`：它不在参数中，且上游稍后会覆盖该位置的 stop 字典。只对返回轨迹任意删尾也不可靠，必须区分真实移动前缀与原末尾返回段，并由真实已发现图重建新返回路。

完整导航复核应检查两个层次：无 endpoint head 时与原基线逐指令轨迹/官方指标完全一致；启用 head 时，全部原始决策位置、移动动作、在线停止原因保持一致，只有最后返回段允许变化。再用官方 evaluator 计算 SR/SPL/NE/完整路径长度，不能用 BCE 代替导航成绩。

## 6. 自然训练正负是否足够

**最终成功率高，不代表状态二分类缺少负样本。** 一条正常成功路径的前段通常距离目标超过 3 米，是负样本；后段进入范围的状态是正样本。首次 fit 普通 BCE 不要求额外制造失败，应先使用 D1 全部真实前缀，报告标签数量、每房屋正负支持和高 STOP 分数错误终点数量。当前摘要没有逐状态标签计数，不能在未读取缓存前声称正负已足够。

这批数据不足以保证学习到困难终点错误：容易的起点/途中负样本可能让 BCE 很好看，实际错误终点仍不可分。若 D1 太小，首先按结果无关的固定规则扩大自然 train_fit 路径覆盖；不要从已看的验证错误抽样，也不要直接把 val_unseen 状态送入训练。

只有训练数据自己的审计确认困难负样本不足时，才另行固定最小扰动方案：每条选定训练路径最多一次，在预定决策步选择一个合法非 STOP 替代动作，然后恢复冻结 DUET argmax，保持原动作预算和真实 graph/history；目标只用于训练标签，不参与扰动位置或候选选择。没有合法替代则记录跳过，不据结果重抽。扰动生成的是单独训练分布，应与自然样本保留来源和固定混合权重，不能冒称原始基线轨迹。

不要直接启用上游 `feedback='sample'` 作为最小扰动：代码在该模式下按 GT 精确 goal 判定停止，改变了结束语义。`teacher` 模式也依赖 GT 路径并需要 ML targets。独立合法动作扰动比复用这些训练模式更容易清楚控制协议。

train_dev 能检测拟合和训练房屋内部迁移，但基础 DUET 已见过这些房屋。只有在方法与 checkpoint 固定后另行登记一次完整 val_unseen 导航评测，才能估计当前对照对未见房屋的表现；这是开发验证，不能称新的盲测或创新成立。

## 7. 已实施的 CPU 验证

`scripts/train_endpoint_probe.py --train-cache FIT --dev-cache DEV --local-run LOCAL --backup-run DRIVE --device cpu` 使用上述固定 pilot。`cuda` 是可选设备；本轮未运行 GPU 或导航验证。

2026-10-03 的相关 CPU 回归共 **117 项通过**，其中 endpoint 模块 22 项，另含缓存导出、终点适配器、checkpoint store、旧 resumable trainer 与 pipeline 测试。恢复测试覆盖 epoch 中途、完成训练但尚未监测 dev 两种状态；删除本地模拟 VM 后恢复，最终 head、AdamW 状态、RNG 和监测历史与连续训练严格一致。另验证缓存未提交、字节损坏、错误标签/状态/关联、模型与数据身份改变、全部备份损坏均拒绝加载，不能悄悄重新开始。

这些测试证明合成 CPU 工程行为；尚不能证明 CUDA 的逐位恢复一致或导航指标改善。真实云盘的提交和读回由现有 `CheckpointStore` 与挂载检查执行。

## 8. T4 工程小试与导航结果

2026-10-03 已完成真实 T4 特征提取、20 epoch / 140 step 训练和完整 train_dev 导航评测。训练有 193 回合 / 1,139 状态 / 434 正例，监测缓存有 96 回合 / 599 状态 / 224 正例；全部缓存状态三分支 logits 与原采集逐位一致。训练按预定开发 BCE 选择第 17 epoch / step 119，BCE 为 0.2684077869。该模型是 best_dev_bce，不是按导航成绩选择。

完整 train_dev 共 2,890 条指令 / 12 个场景，基座 DUET 已见这些训练场景。无 head 的 16 条接入检查轨迹完全一致；启用 head 后全量原在线移动与停止一致，终点改变 80 条，救回 2 条、损害 5 条。结果如下：

| 指标 | DUET | E0 自然 BCE | 差值 |
|---|---:|---:|---:|
| SR (%) | 97.8201 | 97.7163 | −0.1038 pp |
| SPL (%) | 96.4168 | 96.1221 | −0.2947 pp |
| nDTW (%) | 97.2351 | 97.1132 | −0.1219 pp |
| 完整路程 (m) | 9.6134 | 9.6478 | +0.0344 |

成对场景 bootstrap（10,000 次、seed 0）的 SR 差值区间为 [−0.2766, +0.0641] pp，SPL 为 [−0.4603, −0.1644] pp；不含训练种子变异。评测耗时 428.95 秒，CUDA 已分配显存峰值 745,191,936 字节。best head 的 SHA256 为 `5d0d2d737799046b6e66239caeacb82334a210ef2efe38b682b569d57a6f5b6b`，已从 Drive 校验恢复并导出到本地。实际保留为本地 2 个最近快照加 best、云盘 5 个最近快照加 best。

本小样本普通分类器未改善导航，暂不占用官方 val_unseen 访问；它不能否定完整训练数据下的 C1，也不能作为后续配对方法的最终强对照。配对可行性与同数据 BCE 比较独立推进。报告和决策保存在 `outputs/study-20261003/e0-endpoint-train-dev.json`、`e0-endpoint-train-dev-comparison.json` 和 `e0-decision.json`。
