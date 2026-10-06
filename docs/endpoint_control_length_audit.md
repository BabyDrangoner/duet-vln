# 控制采集的两套距离口径

## 首次真实 smoke 的失败

首个 fit pair 尚未提交时，v1 controls collector 因路径长度检查失败而中止。检查把两个不同坐标来源的长度当作同一数值比较：

- 冻结 `forced_rollout` 累加 DUET 已发现图的距离。图中的 `ob.position` 和候选 `position` 来自 MatterSim。
- 官方评估图由 `utils.data.load_nav_graphs` 直接读取 connectivity JSON 的原始 pose 坐标。

没有修改失败目录、历史路线或冻结的 `endpoint_pairs.py`。

## 真实缓存的定量诊断

纯 CPU 读取已提交配对缓存的 `A_then_B/A`，截取目标 A 首次实际观察时的前缀。该 7 状态前缀与第一条 C2 reference 的观察顺序完全一致。来源为 pair `007adac0b72eae5158796fb47c478aaa3cb0d39b7eca7e8a4cb5e786c4e863ba`，指令 `5902_2`。

| 测量 | 米 |
|---|---:|
| 冻结 rollout 保存的累计距离 | 8.206755407959456 |
| 同一实际路线，按保存的真实观察坐标逐边求和 | 8.206755407959456 |
| 同一实际路线，按官方 connectivity 图逐边求和 | 8.206755271671916 |
| 前两项与官方图的差 | 0.00000013628753947614314 |

全部观察坐标精确等于原始 JSON 坐标先转 float32 再转 Python float 的结果；最大坐标差为 `3.692626950879685e-07`。这定位了坐标精度来源，未通过放宽误差阈值让检查通过。

诊断报告 SHA-256：`b8594fc15c9d3b7382af4873582786b3205441b55356f0aefd42a02efc1cb533`。可用 `scripts/diagnose_endpoint_control_lengths.py` 从真实已提交缓存重新检查；脚本不构建 GPU runtime。

## v2 的记录规则

`duet_endpoint_controls_v2` 保留同一实际 `trajectory` 和全部真实状态：

- `actual_length_m` 与每个 `prefix_length_m`：统一按官方 connectivity 图遍历实际完整边序列计算。
- C2 的 `execution_graph_length_m` 与 `execution_graph_prefix_length_m`：原样保留冻结 rollout 保存的数字。
- 自然 rollout 没有输出执行图累计量，这两个字段为 `null`，不冒称已采集。
- `execution_position_edge_sum_m`：按已保存真实观察坐标对实际路线逐边计算；与上面的执行图原始累计量分开记录。
- `length_audit.edges`：保存每条实际边的两个距离，验证全部边、总长与所有前缀。返回路线中的中间节点仍只贡献路程，不补造观察或 feature。

原始执行图累计值和按同一套已保存 MatterSim 坐标逐边求和的值，逐总长及逐前缀使用 `rel_tol=1e-10, abs_tol=1e-8` 检查加法分组的舍入差。该检查只在执行坐标口径内部进行；不会用这个阈值把官方 JSON 坐标与执行坐标判为相同。真实诊断的 `1.3628753947614314e-7 m` 跨口径差仍完整保留。

新版本使用独立 smoke 输出目录。CPU 回归测试覆盖不同坐标口径、原始执行图数字保留、实际路线不变，以及遗漏边或更改累计量的拒绝。真实 v2 GPU smoke 结果另记。
