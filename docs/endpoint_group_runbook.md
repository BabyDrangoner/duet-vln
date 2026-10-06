# E1 四臂训练与恢复运行手册

日期：2026-10-03。本文保留固定执行与恢复流程。

**2026-10-06 更新：四臂完整导航评测已完成并备份，本轮未验证出超过 DUET 的导航提升。**
T4、Drive、Python 3.11.16、Torch 2.5.1+cu121、模拟器与资产均已恢复；709 项测试和 48 项子测试通过，真实骨干严格加载与 CUDA 前向检查通过。四臂 head、23 个评测源码文件和基线报告 SHA 与原计划一致。
授权后已核对原云端账本，顺序完成 V0002/C1、V0003/C2、V0004/C3、V0005/M，每臂完整 2,349 条 `val_unseen` 指令。M 的 SR 71.3921、SPL 58.5615，基线为 71.5198、60.4098；全图独立指标审计通过。见 [完整结果](e1_full_navigation_results.md)。
报告、账本、claim/outcome、日志和统计已下载本地，并通过 Drive 读回 SHA 校验。本轮目录是 `outputs/study-20261005/`，冻结计划和四份报告仍沿用 `outputs/study-20261004/`；UTC 完成时间为 2026-10-05 16:57:01（北京时间 2026-10-06 00:57:01）。
[四臂执行辅助脚本](../outputs/study-20261005/run_four_arm_navigation.py)已经执行完毕，不可直接重跑。它在任何既有登记/claim/报告或账本分歧下停止，不能以删文件方式绕过。再次分析已有轨迹无需新增导航访问；再次执行模型须使用新的明确实验计划与登记。
环境检查证据在 `outputs/study-20261005/remote-readiness/`；其中 backbone 预检查仅用合成输入。实际导航与独立全图审计记录另存于本轮目录。以下为历史恢复记录和固定流程，以此更新为最新状态。

2026-10-04 更新：用户要求进一步测试未见场景，准备优先完整 `val_unseen` 的四臂 seed 0 final 评测，按第 8 节先同步账本、核对预算和注册访问。
这项请求不要求重新训练、修改 checkpoint 或挑选模型；此前 train_dev 子集负结果不作为禁止此次验证的审批条件。
新 T4 分配返回 503，进一步读取响应体确认 `outcome=2`，对应官方 `QUOTA_EXCEEDED_USAGE_TIME`（使用时长配额已耗尽）。账号计算单元余额为 0，服务器当前无运行实例。见 [连接诊断](../outputs/study-20261004/t4-reconnect-diagnostic.json)。当前没有新正式访问或运行结果。入口单文件 SHA 与 23 个依赖文件的聚合 code SHA 是不同字段，注册须使用入口 `--print-code-sha256` 的聚合值。

**已完成的恢复工作：** 曾从新的 `vscode-colab` CPU 会话挂载 Drive，取回并核验旧 T4 上完成的 seed 0 四臂 final、工程 best 和真实 CUDA 恢复验收；该 CPU 会话现已过期。
四臂均完成 20 epoch / 1280 updates，本地分析器复算与原云端数值一致；当时尚未进行 E1 完整导航评测。
见 [恢复进度](../outputs/study-20261003/e1-recovery-progress.json)、[独立 checkpoint 核验](../outputs/study-20261003/e1-recovery-independent-verification.json)和[复算结果](../outputs/study-20261003/e1-seed0-local-reanalysis.json)。
当时恢复时跳过已经验证完成的采集、验收和 seed 0 训练，优先第 7 节四臂导航复核；T4 分配返回 503。该阻塞现已解除，第 7 节导航复核已完成。

**新增自然子集回放：** `scripts/replay_endpoint_natural_subset.py` 已在 CPU Colab 对固定 256 条自然缓存完成四臂 final 终点回放。
全部基线路径、SR/SPL、非指数指标逐条一致，nDTW/SDTW/CLS 只存在最多 4 ULP 的已记录末位差异。
四臂均未超过原始 DUET 的 SPL；C3 为 SR −0.78125 pp、SPL −2.32723 pp（SPL 场景 95% 区间 [−3.93165, −0.93093]）。
报告 `outputs/study-20261003/e1-natural-subset-navigation-v3.json` 已备份 Drive 的 `navigation-subset-20261003/`。
这是已存自然特征的离线回放，不能替代第 7 节完整导航；不据此晋级官方验证，也不在该子集上调阈值、挑 head 或删样本。
前两次执行因真实字段名和指数函数末位差异中止，失败日志与三个执行脚本均保留。最终版本 24 项 CPU 测试通过，评分公式和固定样本未变。

四份缓存已严格重载通过，另外打包为 `recovery-20261003/e1-full-cache-bundle.tar.gz`（348,359,983 bytes，3,852 files），并通过 Drive 读回 SHA 校验。
包内每个成员也已重新读出并验证；原缓存目录保留。归档 SHA 为 `cd01c0da0872a1f93d221636fd91f2b70b81542a17a21ab111b4a364a26e5102`。
新 VM 使用时，先核对同目录的 publication 与 manifest SHA，再解压到新的空暂存目录，逐文件核对 manifest，最后用第 2 节严格 loader 检验原数据锁后才发布目录。数据包没有下载到 Mac。

本地本轮新增代码和报告保存在 `outputs/study-20261003/recovery-code-records.tar.gz`；它是原 `code-snapshot-e1-d3/vln-research-code-with-upstream.tar.gz` 的补充包，包含 SHA 清单，不含数据集或特征张量。
旧 CLI 的缓存代理凭据可能在一小时后失效：出现 404/401 时，先检查服务端 assignment 是否仍存在。本轮通过 `refresh_existing_colab_proxy.py` 从已授权 assignment 刷新凭据后恢复连接，原 CPU 会话和 Drive 文件均在，未重建或重置运行时。该脚本绑定本轮 endpoint，后续会话不可直接套用。

协议见 [训练协议](endpoint_training_protocol_draft.md) 和 [结果解释预案](endpoint_group_interpretation_plan.md)。所有训练只用 `train_fit`；`train_dev` 用于监测与机制分析。完整导航评测是另一步。

## 1. 固定目录和环境

以下命令在已有 Colab 工程执行；同一时间只运行一个 GPU 作业。

```bash
cd /content/project/vln
export PYTHONPATH=src:third_party/Matterport3DSimulator/build
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export E1_STUDY=/content/project/vln/outputs/study-20261003
export E1_CLOUD=/content/drive/MyDrive/VLN-Research/studies/arrival-evidence-20261003
export E1_PY=/content/project/vln/.venv/bin/python
```

| 内容 | 本地与 Drive 下的同名子目录 | 完整数量 |
| --- | --- | ---: |
| fit 配对缓存 | `endpoint-pair-full-train-fit` | 512 pairs / 2048 rollouts |
| dev 配对缓存 | `endpoint-pair-full-train-dev` | 128 pairs / 512 rollouts |
| fit 自然 + C2 缓存 | `endpoint-controls-full-v2-train-fit` | 512 groups / 3072 rollouts |
| dev 自然 + C2 缓存 | `endpoint-controls-full-v2-train-dev` | 128 groups / 768 rollouts |

不得拿 pilot、smoke 或 controls v1 目录替代。controls v2 的 `actual_length_m` / `prefix_length_m` 使用官方图边长，同时保留模拟器执行图长度审计；两种坐标口径不可直接要求浮点数相等。

## 2. 全量 seal、SHA 与来源审计，然后锁定数据

等待四个缓存全部封存。`COLLECTION.json`、根 `manifest.json` / `COMMITTED.json`、每个 pair 的 manifest / commit / tensor 文件都必须齐全。不要对不完整目录手写 seal，也不要重新给损坏文件签 SHA。

唯一审计入口是 `outputs/study-20261003/audit_full_endpoint_groups.py`，唯一锁文件是 **`e1-full-data-lock.json`**。它要求 `full-pairs-status.json.state=complete`、`controls-full-v2-status.json.state=collected`，读取本地和 Drive 的完整缓存，验证 512/128 清单、split 来源与全部封存字节，并逐条核对 dev 池 256 条自然轨迹与完整基线一致。它同时记录四臂配置和训练源码 SHA。

初次审计执行一次；锁文件任一副本已存在时脚本拒绝覆盖，应进入后面的只读核查，不能生成第二份锁或另换 schema：

```bash
"$E1_PY" outputs/study-20261003/audit_full_endpoint_groups.py
```

实际 schema 为 `duet_endpoint_full_group_data_lock_v1`，`usage=fixed_before_E1_training`。`splits.train_fit/train_dev` 各包含 `groups`、`data_sha256`、`source_identity`、`support`、`drive_cache_strict_load_equal`、`paired_manifest_sha256` 和 `controls_manifest_sha256`；顶层另外保存自然基线 parity、训练源码/四臂配置 SHA、零训练/验证计数、审计耗时和 `content_sha256`。耗时属于这一次审计记录，不可通过重跑审计覆写原文件。

首次审计之后，以及新 VM 恢复缓存之后，都执行下面的**只读**核查。它使用同一锁文件重新验证本地完整缓存和当前训练配置；不写新锁：

```bash
"$E1_PY" - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from train_endpoint_groups import validate_experiment
from vln_improve.endpoint_group_training import load_endpoint_group_cache, training_code_identity
from vln_improve.pipeline import validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256

local, cloud = Path(os.environ['E1_STUDY']), Path(os.environ['E1_CLOUD'])
validate_backup_root(cloud)
lock = json.loads((local/'e1-full-data-lock.json').read_text())
assert lock['schema'] == 'duet_endpoint_full_group_data_lock_v1'
assert lock['usage'] == 'fixed_before_E1_training'
assert lock['content_sha256'] == object_sha256({k:v for k,v in lock.items() if k != 'content_sha256'})
assert file_sha256(local/'e1-full-data-lock.json') == file_sha256(cloud/'e1-full-data-lock.json')
assert lock['training_updates'] == lock['official_validation_accesses'] == 0
assert lock['training_code_identity'] == training_code_identity()
parity = lock['train_dev_natural_baseline_parity']
assert parity['instructions'] == 256 and parity['all_full_trajectories_exact'] is True
assert parity['baseline_sha256'] == file_sha256(local/'baseline-train-dev.json')
caches = {}
for split, suffix in [('train_fit', 'train-fit'), ('train_dev', 'train-dev')]:
    pair = 'endpoint-pair-full-' + suffix
    control = 'endpoint-controls-full-v2-' + suffix
    row = lock['splits'][split]
    assert row['drive_cache_strict_load_equal'] is True
    a = load_endpoint_group_cache(local/pair, local/control, split, expected_data_sha256=row['data_sha256'])
    assert len(a.groups) == row['groups'] and a.source_identity == row['source_identity']
    assert file_sha256(local/pair/'manifest.json') == row['paired_manifest_sha256']
    assert file_sha256(local/control/'manifest.json') == row['controls_manifest_sha256']
    caches[split] = a
for arm in ('C1', 'C2', 'C3', 'M'):
    config = Path(f'configs/endpoint_group_{arm}.json')
    assert file_sha256(config) == lock['experiment_sha256'][arm]
    spec = json.loads(config.read_text())
    validate_experiment(spec, train=caches['train_fit'], dev=caches['train_dev'])
print(json.dumps({k: v['data_sha256'] for k,v in lock['splits'].items()}, indent=2))
PY
```

loader 已校验全部 tensor 字节与配对映射；上面的根文件比较另外确认两份封存清单一致。组数据 SHA 绑定 collection 身份及有序 pair 的 manifest/data 字节；资源计时不作为训练数据身份。Drive 证据是挂载路径读回校验，并非独立服务器回执。

保存本次源码快照和配置，确认包含后面要用的验收、训练、分析及评测脚本。训练恢复会校验源码 SHA、配置、数据身份与运行参数；新 VM 应恢复原版本，不混用之后修改的文件。

从已锁定文件读取参数，后面始终复用：

```bash
export E1_FIT_SHA=$("$E1_PY" -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["E1_STUDY"])/"e1-full-data-lock.json").read_text())["splits"]["train_fit"]["data_sha256"])')
export E1_DEV_SHA=$("$E1_PY" -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["E1_STUDY"])/"e1-full-data-lock.json").read_text())["splits"]["train_dev"]["data_sha256"])')
```

## 3. 先运行 CUDA / Drive 恢复验收

完整缓存通过上一步之后执行。验收根必须是专用的新目录或空目录；失败证据保留，重新验收应使用新名称。

```bash
"$E1_PY" scripts/verify_endpoint_group_resume.py \
  --train-pairs "$E1_STUDY/endpoint-pair-full-train-fit" \
  --train-controls "$E1_STUDY/endpoint-controls-full-v2-train-fit" \
  --dev-pairs "$E1_STUDY/endpoint-pair-full-train-dev" \
  --dev-controls "$E1_STUDY/endpoint-controls-full-v2-train-dev" \
  --expected-train-data-sha256 "$E1_FIT_SHA" \
  --expected-dev-data-sha256 "$E1_DEV_SHA" \
  --local-root "$E1_STUDY/e1-resume-acceptance-v1" \
  --backup-root "$E1_CLOUD/e1-resume-acceptance-v1" \
  --experiment configs/endpoint_group_M.json --device cuda
```

脚本先严格加载完整 512/128 来源，再明确取前 8/4 组，重新计算子集身份、支持量，并记录 `subset_of` 与有序 pair IDs。M / seed 0 / batch 2 / 3 epochs 共 12 updates；对比连续训练与“暂停 2 updates → 直接验证 Drive 快照 → 删除专属本地 interrupted 目录 → 空目录恢复”。验收保留策略为本地 2、Drive 3，受保护的 latest/best 另保留。

通过条件是 `resume-acceptance.json` 的 `status=passed`、全部 checks 为 true、对应 `COMMITTED.json` 及两侧报告 SHA 一致。最终 head、AdamW、RNG、游标与历史逐项一致。这是 `acceptance_only`，不是正式 512 组训练或导航成绩。**该次真实 T4 验收的报告和最终 payload 已取回，独立加载后的完整状态一致性也已通过；两份 state.pt 的序列化文件字节可以不同，各自仍必须匹配自身 manifest。**

## 4. 四臂 seed 0 固定训练

真实验收通过后，逐臂运行以下循环。发现错误就停止后续臂，不用较早 checkpoint 替代失败的 final。

```bash
for e1_arm in C1 C2 C3 M; do
  "$E1_PY" scripts/train_endpoint_groups.py \
    --experiment "configs/endpoint_group_${e1_arm}.json" --seed 0 --device cuda \
    --train-pairs "$E1_STUDY/endpoint-pair-full-train-fit" \
    --train-controls "$E1_STUDY/endpoint-controls-full-v2-train-fit" \
    --dev-pairs "$E1_STUDY/endpoint-pair-full-train-dev" \
    --dev-controls "$E1_STUDY/endpoint-controls-full-v2-train-dev" \
    --expected-train-data-sha256 "$E1_FIT_SHA" \
    --expected-dev-data-sha256 "$E1_DEV_SHA" \
    --local-run "$E1_STUDY/e1-seed0-${e1_arm}" \
    --backup-run "$E1_CLOUD/e1-seed0-${e1_arm}" || break
done
```

- 四臂同 seed 的初始化和组顺序相同；C3/M 的数据也逐项相同。其他臂的真实状态数可能不同，读取 summary 中的预算和跨面板去重计数。
- 固定 `1536→128→ReLU→1`，AdamW `lr=0.001, wd=0.0001`，batch 8，20 epochs / **1280 updates**。主结果只用 final。
- epoch 5/10/15/20 记录同一自然、C2、paired dev 面板。**工程 best** 按三个面板 episode-normalized BCE 的等权均值保存；它不是导航 SR/SPL 最优 checkpoint，也不能替代主 final。
- `training-summary.json.final_checkpoint.head_relative_path` 指定主 head；同时要求 `status=complete`、epoch 20、step 1280、`pending_dev=false`。不要根据文件名或 `best.json` 猜主模型。
- 默认每 10 updates 或 60 秒备份，监测前后及最终也提交。每次包含模型、完整 AdamW、RNG、组游标、pending-dev 与历史；备份校验通过后才清理。本地保留最近 2、Drive 最近 5，并额外保护 latest/best，故目录数可以略多于 2/5。
- `dev-final.json` 保存每个原始 pair/scene 的三个 BCE、ranking 和两顺序正确性。它与 summary 一同写入 Drive 并读回 SHA。

中断后**用完全相同的配置、seed、数据 SHA 与同一对 run 目录重跑对应命令**。SIGINT/SIGTERM 在完整 optimizer 边界保存；断电或硬杀从最近已提交快照恢复，可能重做少量尚未提交的 updates。`CheckpointStore` 自动从有效副本恢复 checkpoint；本地目录完全丢失也可从同名 Drive run 恢复。它不会自动从头覆盖全损坏状态。

## 5. 新 VM 上先恢复特征缓存，再续训

**当前组训练 CLI 不自动恢复特征缓存、源码或 DUET 资产。** checkpoint 在云盘不代表四个 feature cache 也已在新 VM 就绪。先恢复冻结源码、相同 PyTorch/CUDA 环境、原 `e1-full-data-lock.json` 和 `baseline-train-dev.json`，重新挂载 Drive，再处理四个缓存。

安全顺序：

1. 用本手册第 2 步中的 strict loader 读取 Drive 四个完整缓存，核对原先锁定的组 SHA。
2. 仅对完全缺失的本地缓存，用 `shutil.copytree` 复制到新 staging 目录；禁止 `dirs_exist_ok=True` 合并新旧文件。
3. 在 staging 上运行对应 strict loader，并与 Drive 的 `data_sha256` 比较；通过后才将 staging 原子重命名为原目录名。
4. 对已经存在但不完整/损坏的本地目录先保留证据、明确隔离，再按上一步恢复；不要重写 manifest/COMMITTED 去接受坏文件。Drive 也损坏时停止，不能重新采一份后沿用旧训练身份。
5. 对恢复后的完整四个本地目录运行第 2 步的只读核查并要求与原锁一致，再重跑同一个训练命令。不要重跑生成锁的审计入口。组身份不依赖本地绝对路径。

例如，仅恢复一个确认缺失的配对缓存；controls 应改用 `load_control_cache(..., expected_split=...)` 验证。这里只复制缓存，不重新采集：

```bash
"$E1_PY" - <<'PY'
import os, shutil, uuid
from pathlib import Path
from vln_improve.endpoint_pair_training import load_endpoint_pair_cache
from vln_improve.pipeline import validate_backup_root
local, cloud = Path(os.environ['E1_STUDY']), Path(os.environ['E1_CLOUD'])
validate_backup_root(cloud)
name = 'endpoint-pair-full-train-fit'
source, target = cloud/name, local/name
assert not target.exists() and not target.is_symlink(), 'preserve existing cache for diagnosis'
reference = load_endpoint_pair_cache(source, 'train_fit')
stage = local/('.restore-' + name + '-' + uuid.uuid4().hex)
shutil.copytree(source, stage, symlinks=True)
restored = load_endpoint_pair_cache(stage, 'train_fit')
assert restored.data_sha256 == reference.data_sha256
assert not target.exists()
stage.rename(target)
PY
```

恢复数据之前已按第 1 项绑定完整 group SHA；单个复制例子不替代这项来源检查。strict loader 会拒绝符号链接和未封存/损坏文件。

## 6. 下载小 head、报告与冻结源码，再分析

四臂完成后只导出 final head 和小报告；无需把 feature cache 或原始数据集下载到本机。下面保留 head 在 run 内的相对路径，让分析器能自动找到它：

```bash
"$E1_PY" - <<'PY'
import json, os, tarfile
from pathlib import Path
from vln_improve.protocol import file_sha256
study = Path(os.environ['E1_STUDY'])
archive = study/'e1-seed0-results.tar.gz'
assert not archive.exists(), 'preserve previous exports'
with tarfile.open(archive, 'x:gz') as tar:
    tar.add(study/'e1-full-data-lock.json', arcname='e1-full-data-lock.json')
    for arm in ('C1', 'C2', 'C3', 'M'):
        name = f'e1-seed0-{arm}'; run = study/name
        summary = json.loads((run/'training-summary.json').read_text())
        assert summary['status'] == 'complete' and summary['global_step'] == 1280
        assert summary['completed_epochs'] == 20 and not summary['pending_dev']
        final = summary['final_checkpoint']; head = run/final['head_relative_path']
        assert file_sha256(head) == final['head_sha256']
        assert file_sha256(run/'dev-final.json') == summary['final_dev_report']['sha256']
        for rel in ('training-summary.json', 'dev-final.json', final['head_relative_path']):
            tar.add(run/rel, arcname=f'{name}/{rel}')
print(file_sha256(archive), archive)
PY
```

本机使用当前已验证的 Colab SSH 会话下载。例如，已有 [T4 连接记录](colab_t4.md) 中的专用配置仍有效时：

```bash
scp -F 'outputs/colab-t4-20261003T035623Z/ssh_config' \
  vln-t4-ssh:/content/project/vln/outputs/study-20261003/e1-seed0-results.tar.gz \
  outputs/study-20261003/
shasum -a 256 outputs/study-20261003/e1-seed0-results.tar.gz
mkdir outputs/study-20261003/e1-seed0-results
tar -xzf outputs/study-20261003/e1-seed0-results.tar.gz \
  -C outputs/study-20261003/e1-seed0-results
```

先比较远端打印的 archive SHA；本机解压目录必须新建。会话失效或 SSH 429 时使用已授权的现有 Colab 接口取文件，不为下载重新创建运行时。另取回并校验本次冻结源码快照，确保分析器和 head 绑定的训练代码/配置一致。运行分析前把需要的冻结版本放在当前项目中；不要重签训练源码 SHA。

```bash
.venv-duet/bin/python scripts/analyze_endpoint_group_results.py \
  --c1 outputs/study-20261003/e1-seed0-results/e1-seed0-C1/dev-final.json \
  --c2 outputs/study-20261003/e1-seed0-results/e1-seed0-C2/dev-final.json \
  --c3 outputs/study-20261003/e1-seed0-results/e1-seed0-C3/dev-final.json \
  --method outputs/study-20261003/e1-seed0-results/e1-seed0-M/dev-final.json \
  --output outputs/study-20261003/e1-seed0-comparison.json
```

分析保留全部 128 pairs，按 scene 聚类配对 bootstrap，不能把四条 rollout 当四个独立样本。单 seed 的区间不含重新训练的种子不确定性；压力面板正确性不是导航成功率。按既定解释预案报告所有控制比较，不从多个指标中只挑有利项。

## 7. 完整 train_dev 导航复核

使用冻结 final head，对原完整 `baseline-train-dev.json` 的 **2890 条指令**复核。以下示例为 M；其余臂只改 arm 和对应输出名。不要加 `--limit`，不要传官方验证 ledger 参数。

```bash
export E1_NAV_ARM=M
export E1_HEAD_REL=$("$E1_PY" -c 'import json,os; from pathlib import Path; p=Path(os.environ["E1_STUDY"])/("e1-seed0-"+os.environ["E1_NAV_ARM"])/"training-summary.json"; print(json.loads(p.read_text())["final_checkpoint"]["head_relative_path"])')
"$E1_PY" scripts/evaluate_endpoint_groups.py \
  --head "$E1_STUDY/e1-seed0-$E1_NAV_ARM/$E1_HEAD_REL" \
  --experiment "configs/endpoint_group_${E1_NAV_ARM}.json" \
  --config configs/r2r.json --split train_dev --seed 0 \
  --baseline-report "$E1_STUDY/baseline-train-dev.json" \
  --output "$E1_STUDY/e1-seed0-${E1_NAV_ARM}-train-dev.json" \
  --backup "$E1_CLOUD/e1-navigation"
```

评测只在最终历史终点选择使用小头，原移动和在线终止保持基线轨迹；回到历史节点的完整路径计入导航长度。评测器会校验原轨迹、完整指令清单、final 身份和模型/数据来源。导航随机种子始终为 0；以后 head seed 1/2 仅是训练种子及 ledger 标记，不改变导航环境顺序。

评测输出拒绝覆盖，尚无按 episode 自动续跑功能。失败后保存日志和中间文件，在新的输出路径完整重跑；这与训练“同目录恢复”不同。

## 8. 官方 val_unseen 必须另登记

train_dev 或机制分析通过不自动启动官方验证。先按研究协议检查剩余预算、固定待测 arm/seed/final head、评测代码和配置 SHA，再使用现有 `StudyLedger` 注册单次完整访问。账本为 `outputs/study-20261003/val-unseen-access-ledger.jsonl`；保持单个本地 writer，并同步备份账本。

登记需要：新的 access ID、`pilot` 或已批准预算文件对应的 `confirmatory`、variant ID、实验配置 SHA、final head SHA、下列命令的评测 code SHA、head 训练 seed、完整 2349 episodes、`subset=false`、`label_use=evaluation`、`parameter_fitting_split=train_fit`。confirmatory 另需冻结预算文件。失败/未完成访问也消耗预算；实际重跑必须新 access ID。

```bash
"$E1_PY" scripts/evaluate_endpoint_groups.py --print-code-sha256
```

在 `StudyLedger.register(...)` 成功且身份逐项核对后，才执行下面模板；`E1_ACCESS_ID` / `E1_VAL_CATEGORY` 必须来自已经存在的登记，不在此手册中预设或自动生成：

```bash
: "${E1_ACCESS_ID:?use an existing registered access ID}"
: "${E1_VAL_CATEGORY:?use its registered pilot or confirmatory category}"
"$E1_PY" scripts/evaluate_endpoint_groups.py \
  --head "$E1_STUDY/e1-seed0-$E1_NAV_ARM/$E1_HEAD_REL" \
  --experiment "configs/endpoint_group_${E1_NAV_ARM}.json" \
  --config configs/r2r.json --split val_unseen --seed 0 \
  --baseline-report "$E1_STUDY/baseline-val-unseen.json" \
  --output "$E1_STUDY/${E1_ACCESS_ID}-endpoint-val-unseen.json" \
  --backup "$E1_CLOUD/e1-navigation" \
  --study configs/research_study.json \
  --ledger "$E1_STUDY/val-unseen-access-ledger.jsonl" \
  --execution-backup-root "$E1_CLOUD/validation-executions" \
  --access-id "$E1_ACCESS_ID" --category "$E1_VAL_CATEGORY"
```

`--execution-backup-root` 必须始终指向同一个 study 级 Drive 目录，本手册统一为 `$E1_CLOUD/validation-executions`；不能按 arm、输出名或新 VM 换目录。评测开始前会持久化一次性 claim、账本快照，完成/失败时追加独立 outcome。已有 claim 即阻止再次执行同一个 access，包括换输出名或换 VM；`started` 而没有 outcome 也不代表可以重跑。保留这些文件，不删除 claim 绕过访问计数。

执行 sidecar 不替代账本结项：评测器**不会自动调用 `StudyLedger.finish`**。完成或失败后均需通过现有账本结项入口追加 outcome：实际指标、报告 SHA、耗时/显存或明确 unknown、结论与失败原因，并将账本同步到 Drive。本手册未登记任何新验证访问，也未执行上述官方评测。
