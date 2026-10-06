# E3 continuation probe 协议审查

审查日期：2026-10-06。范围：配置、probe 实现、固定 DUET 调用链与现有测试的只读审查；未执行导航、训练或新增 val-unseen 访问，未修改被审源码。

## 结论

**当前静态实现符合“训练场景中单动作干预机会探测”的语义，可以进入 T4 工程验证。** 本次发现的完成态云盘恢复错误和子集标记错误已由实现者修复；最终源码中未发现剩余的单次干预或 oracle 聚合阻断问题。

这不是方法效果审查。probe 不训练选择器，oracle 使用结果标签挑选分支，不能称为 SR 提升实验或预测 val-unseen 会涨 5 pp。当前实现已在 rollout 前强制核对四项冻结资产 SHA；仍需真实 T4 冒烟确认锚点、前缀和恢复检查通过，源码与合成测试不能替代运行证据。

## 审查版本

| 文件 | SHA256 |
|---|---|
| `configs/e3_continuation_probe_v1.json` | `b7ad17809ca03948a12931c9120bc52fc62dc1a2505372450a725c00b5b1a966` |
| `src/vln_improve/continuation_probe.py` | `54fae5cbbd3e0b89f61049bad5209d86c223185ffcca005ac73f7cf5675d1ff1` |
| `scripts/probe_continuation_actions.py` | `2d60ac2e7e89aa992fa9e235f07154d3c4ea10e79b73a634945674b440801efe` |
| `tests/test_continuation_probe.py` | `80396a5d64185a1668e75a6df8a783f54b308f8cfa2a827a7e0504f38d7b214e` |
| `tests/test_continuation_probe_integration_audit.py` | `cc1c0428e71555875202b34fcdf07b7fd2b339af2a3a826f50ec1f72bfbdb571` |

如源码随后有变化，应重新核对相关结论；本表不包含运行资产的 SHA。

## 1. 原先发现的阻断问题与修正

| 问题 | 风险 | 当前状态 |
|---|---|---|
| 完成的 `probe-summary.json` 仅在 Drive，而新 VM 本地为空时，脚本重算含新耗时字段的摘要再做不可变复制 | 内容不同导致 SHA 冲突，已经完成的任务无法正常恢复收尾 | **已修正**：先从 Drive 恢复原摘要字节，再核对 complete、provenance、selection、opportunity 和 task-manifest SHA；不重写原运行资源字段 |
| 传入 `run_duet` 的 `limit=None` 使 64 条 natural 原始报告标成 `subset=false` | 下游可能误认完整 train-dev 评测 | **已修正**：传 `limit=args.limit`；选择后仍是同一固定集合，原始报告明确标记 subset |
| 写入云端任务后、更新 manifest 前中断 | 未登记的完整任务可能被重复计算，且运行耗时变化造成不可变冲突 | **已处理**：恢复时识别原子写完且 task/identity 匹配的云端任务文件，校验内部 `payload_sha256` 后登记复用；失败记录不进入完成任务表 |

## 2. 状态、候选与续跑语义

- CLI 仅允许 `train_fit/train_dev`；选择在 scene-level partition 之后，固定 scene-stratified 指令集合。正式单次 invocation 为一个 split 的 64 条，smoke 明确为 2 条。配置与实际硬编码状态步、条件、动作预算、候选上限有 fail-closed 校验。
- natural 的状态是实际存在的 `0/3/terminal`；perturb 是实际存在的 `3/6/terminal`。后者均在零起算 step2 扰动之后。早停导致不存在的状态不会被补造，未施加扰动的指令也不会被丢弃。
- 候选为原执行动作、按原 logits 排序的两个有限合法非 STOP 替代动作和 STOP，去重。排序只读合法 mask、已访问 mask、候选 ID 和原分数；没有读取目标距离、GT 路线或分支结果。
- terminal 是原策略在当前状态本来会结束的决策点，属于可观测触发；它不能在未来学习模块中被偷换成事先知道原轨迹最终步数。
- 分支从原起点重放相同真实前缀，逐步校验 viewpoint、朝向、候选 ID、mask、完整路径前缀、原 logits 与原 STOP 概率。在选定状态只强制一个新动作，之后使用原 DUET。
- perturb 的 step2 扰动属于所有分支共享的前缀条件。分支不会在干预之后再施加新扰动；“一次干预”指相对该条件参考轨迹的一次修改，而不是声称扰动轨迹从未经历外源动作。
- 原动作 anchor 会完成整条续跑，并核对路径、全部决策状态、动作和完整指标。step14 或没有剩余 frontier 时只允许 STOP；分支没有从干预点重新获得 15 步。

## 3. STOP、完整路径与指标单位

hook 在每次强制动作前保留原 STOP 概率，并在原 `make_equiv_action` 后恢复图中的证据，再由原 DUET 做历史终点返回。强制 STOP 因而仍可能返回历史节点，不能解读成必然停在当前 viewpoint。

标签只在完整 rollout 与历史返回结束后，从固定评测器取得。`R2RNavBatch._eval_item` 拼接所有路径 segment，累计全部边的行走距离，使用完整轨迹计算 SPL；当前实现没有裁剪到有利前缀或免费回到旧位置。项目此前的固定评测器与图路径约定仍是本轮依赖。

原始 `success` 是 0/1，`spl` 是 [0,1]，脚本和摘要函数均校验该范围。摘要乘 100 后输出 `*_percent`；两摘要百分数相减才是百分点差。`actual_perturbation_rate` 仍是 [0,1] 比例，不应在文字中直接附 `%`。

## 4. Oracle 聚合正确，但解释必须受限

`opportunity_summary` 对一个 split 内的每个 condition 分开处理。每条原指令的备选集合是原始轨迹加上各个采样状态下独立完成的分支；它选择**一条**候选完整轨迹，没有把不同状态的有利分支拼成多次干预。

候选先满足该指令 `SPL >= 自身条件参考 SPL`，然后按 success、SPL 排序。保留原始轨迹保证总能选择 KEEP。这个逐指令约束比“全体平均 SPL 不下降”更严格；报告只能称为**当前采样候选集合中、逐指令 SPL 不降条件下的 hindsight oracle**，不能称为全部合法策略空间的最大值。

- `rescuable_failed_instructions` 对原指令去重；同一指令多个救回分支仅计一条。
- `rescue_branches_nonindependent` 是相关分支数量，明确不是独立成功任务数。
- harmful coverage 表示存在伤害原成功任务的候选，不是学习策略实际误伤数。
- 自然与扰动条件必须各报基线、oracle 和实际扰动比例。不能把两个条件相加扩充有效样本量，也不能把扰动恢复当成自然导航涨点。
- 自然完整 train-dev 的既有基线约 97.82%，可提升天花板仅约 2.18 pp；本次 64 条子集的基线需独立报告。无论自然或扰动 oracle 是否超过 +5，都不能据此判定 val-unseen 目标能否实现。

## 5. 样本数核对

| 层级 | 正式一个 split | 正式两个 split 合计 |
|---|---:|---:|
| 原始指令 | 64 | **128** |
| natural/perturb 参考 rollout | 128 | **256** |
| 最多选中状态 | 384 | 768 |
| 最多分支 rollout，包含 anchor | 1,536 | **3,072** |

实际数量会因早停、状态去重、候选不足下降。每个 split 的 `unique_original_instructions_union_conditions` 应为 64；两个 scene-disjoint split 合计才是 128。smoke 与正式 probe 不是额外独立样本；若采样重叠，不能相加报告。目标评测的 **2,349** 条任务及 **118** 条净新增成功属于后续验证目标，与 probe 的 128 条指令不同。

## 6. Drive 和 resume 身份

- `validate_backup_root` 要求目的目录位于实际 `fuse.drivefs/fuse.drive` 挂载，CLI 没有允许本地目录伪装 Drive 的开关。
- 每个完整分支先写本地，再不可变复制到 Drive 并回读 SHA；云端 manifest 更新并回读核对后才作为完成任务复用。每个任务边界重新验证挂载。
- identity 包含 checkpoint、特征、训练注释、connectivity、原始 DUET 源码锁、probe 及依赖源码、配置、PyTorch 版本、partition 参数、split、seed、scope、固定指令 ID 和场景集合。变化会拒绝混合续跑。
- `validate_asset_pins(provenance, spec)` 在首次 rollout 前核对 checkpoint、特征、训练注释、connectivity 的实际 SHA 与配置中的四项 `asset_pins`；缺失或不匹配直接报错。配置中的值来自负责人冻结的恢复清单，资产身份不再仅靠本次运行记录。
- 原子云端任务文件可恢复；完整记录内有 `payload_sha256`，已登记和未登记的孤立任务均验证内容摘要。已登记文件还核对 manifest 的文件 SHA；校验失败报错，不静默重算替换。失败记录独立保存。结束前要求实际 task 集合与预期集合完全一致。
- SIGINT/SIGTERM 设置停止标记，在完整任务持久化边界退出；极端断电最多重做没有完成持久化的当前任务。无删除已确认任务的逻辑。

### 已落实的资产保护与待取得的运行证据

1. **资产绑定已加入强校验。** 已只读核对 `validate_asset_pins` 的四字段检查及其在 `main` 中、导入并运行 agent 前的调用。当前配置与上表 SHA 一致，运行资产必须匹配冻结恢复清单中的 pin 才能开始采集。此前“脚本只记录资产 SHA、没有强制绑定”的保留项已关闭；本审查没有重新读取旧 val 结果。
2. **真实运行一致性。** T4 smoke 应出现全部 anchor、prefix 通过，且至少覆盖强制移动、强制 STOP 的实际执行；恢复演练应在同一冻结版本上从空本地、现存 Drive 任务恢复。合成测试覆盖了相关逻辑，但不能代替 CUDA 与模拟器验证。

负责人报告本地 24 项测试通过，已封版并启动 T4 `smoke-fit-v1`；本次修订不把“已启动”记为冒烟通过。以上核验不需要新增 val 访问。当前 probe 只负责确认可恢复机会、样本支持与工程成本，是否值得训练应结合这些量判断；最终论文价值仍需强对照和完整导航实验支持。
