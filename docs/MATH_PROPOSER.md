# Qwen3-4B 数学出题模型：SFT → GRPO / MGPO

本配置只训练出题模型；无需执行下游 `solver-grpo`。Teacher 和 Verifier 均使用
EnvMux `qwen3.8-flash`。API key 从环境变量或 Git 忽略的本地凭据文件读取。

## 已补充的训练资产

- `configs/math_skills/atomic/`：40 个带独立可执行计算器的数学构造 Skills，
  覆盖线性方程、整数根、模约束、数字串计数、格点路径、递推、条件概率、
  整数优化、最短路和有理数求和，并扩展到排列、函数计数、数论、坐标几何、
  有限状态概率和期望；另有 10 个检查/修订指导 Skills。完整目录见
  `configs/math_skills/README.md`。
- `configs/proposer_seeds.math.jsonl`：40 类 × 6 种出题目标，共 240 个 seeds。
- `configs/math_skills/validation_report.json`：80 个手算答案的数值重放记录。
- `configs/proposer_training.math.json`：采集与 SFT 配置；本批选择 136 条合格轨迹训练。
- `configs/sft_dataset.math.json`：136 条轨迹、1,543 个动作的数量、来源和 SHA-256。

这些是本项目新编写的有界数学试验库，不是论文官方 Skills。40 个计算器的
`verified` 状态只指报告中的数值样例校验；10 个过程指导 Skills 标为 `curated`。
生成题的题面有效性还需要模型委员会独立核查。数值正确不代表题目已达到竞赛难度。

## API 采集

在仓库根目录运行：

```bash
python3 -m venv .venv-proposer
.venv-proposer/bin/python -m pip install -e '.[dev]'
python3 scripts/prepare_math_training.py
.venv-proposer/bin/python scripts/collect_math_sft.py
```

本机已按用户要求将 key 保存到 `research-workspace/envmux.local.json`（权限 0600，
Git 忽略，迁移包不包含）。采集命令自动读取该文件；也可设置 `ENVMUX_API_KEY`
覆盖本地 key。迁移到其他机器后，没有本地文件或环境变量时才提示输入，输入不回显。
`configs/envmux.models.json` 提供 `qwen3.5-flash`、`qwen3.6-flash`、
`deepseek-v4-flash-0731`、`qwen3.7-plus` 和 `qwen3.8-flash` 配置；修改训练配置中的
`teacher.model` / `verifier.model` 即可选择。正式采集切换模型时保留逐条模型来源，
并使用检查点指纹迁移记录延续同一采集目标。
`ENVMUX_BASE_URL` 可覆盖配置中的地址。若 macOS 的 Python
缺少系统 CA 配置，可使用：

```bash
SSL_CERT_FILE=/etc/ssl/cert.pem .venv-proposer/bin/python scripts/collect_math_sft.py
```

不要关闭 TLS 证书验证。Linux 通常直接使用默认 CA 配置即可。

采集目标为 40 类均衡分配，共 500 条**通过验证且去重的真实轨迹**：
按类别名字排序的前 20 类各 13 条，其余 20 类各 12 条。最多启动 1500 次
尝试。默认 8 个 API worker；若服务限流，在 JSON 中降低 `collection_workers`。
重新执行相同命令会从已有数据继续；不覆盖合格轨迹。

每条轨迹必须实际加载 Skill、执行最终参数对应的 `skill_python`、接收工具结果、
通过精确数值检查、完成同一候选的 self-check，并通过 3 次同模型 privileged votes
和 1 次 blind vote（必要时 second audit）。盲评不会收到答案、解法和参数 witness。
privileged 评审获得完整候选及最多 12000 字符的近期可观察轨迹。
同模型多次投票不是多模型独立集成，不能据此宣称论文中的验证准确率。

工具失败、真实检查失败和修订都会保留在最终合格轨迹中；不会人为制造失败步骤。
按规范化题面和最终数学构造参数去重，尚未实施通用语义去重或外部题库查重。

## 输出和断点

```text
research-workspace/math-proposer/training/
  sft_trajectories.jsonl           合格的多轮轨迹，SFT 直接读取
  sft_trajectories.failed.jsonl    含动作和工具结果的失败/无效轨迹，仅作诊断
  sft_trajectories.rejected.jsonl  拒绝原因及 run_id
  sft_trajectories.progress.json  数量、类别配额、状态及 API 使用量
research-workspace/math-proposer/runs/
  <run_id>/                      可审计的动作、检查和最终产物
```

启动时会核对已有采集配置指纹，避免将不同 seeds/验证配置的数据混在一起。
从旧 12-Skill 库升级时，先运行 `python3 scripts/upgrade_math_workspace.py`。
升级会归档旧 Skill 包和进度、重新检查已有轨迹的精确答案与原始 API 通过报告，
保留数据 JSONL 字节不变，并显式更新类别配额和配置指纹。本工作区已完成此升级；
重复运行升级命令不会再次修改数据。迁移证据存于
`sft_trajectories.library-v2-migration.json`。
网络/API 连续错误会停止启动新任务、保存已完成记录并报告阻塞。
失败或无效尝试从本次采集器版本起写入 `sft_trajectories.failed.jsonl`，保留已观测
动作、工具返回、已形成的候选题和失败原因；API 错误导致的半截轨迹也会保存。
该文件不供正例 SFT 直接训练，避免把无效题目或不完整动作当作正确示范。
之前版本已经记录但未保存轨迹正文的失败尝试无法从拒绝日志中还原。
当前用户已结束采集并选择现有 136 条轨迹训练，`sft.min_accepted_for_training=136`
明确覆盖原始采集目标的训练门槛；文件少于 136 条时仍会拒绝训练。
历史 `target_accepted=500` 仅保留为采集目标，不再要求继续补采。
试采集配置为独立 workspace，不自动混入正式数据；若经过重新校验导入，记录会
带 `collection_origin` 来源字段。进度中的 API 使用量仅统计已汇总的调用，不等于账单。

## 在 CUDA 训练机训练

先将配置中 `sft.base_model` 改为已有 Qwen3-4B 的**实际本地路径**。
默认示例为非 thinking 的 `Qwen/Qwen3-4B-Instruct-2507`；普通 Qwen3-4B 可以使用，
但必须保留其对应 tokenizer，并用训练机小规模检查动作协议和 chat template。
出题模型的 SFT/RL prompt 会关闭 thinking，监督简洁的可观察动作 JSON。

若冻结 Prober 也使用本地权重，同时修改 `grpo.prober_model` / `mgpo.prober_model`。
当前批次数据已确定为 136 条。传入训练机本地模型路径后运行：

```bash
bash scripts/train_sft.sh --base-model /models/Qwen3-4B
bash scripts/train_grpo.sh --prober-model /models/Qwen3-4B
bash scripts/train_mgpo.sh --prober-model /models/Qwen3-4B
```

GRPO 与 MGPO 分别从同一 SFT checkpoint 开始，均在线生成轨迹；GRPO 仅用终局奖励，
MGPO 另用过程奖励。两者需要可访问的 Verifier API 和本地或环境变量中的 API 凭据。
用 `--dry-run` 检查输入、`--sft-checkpoint` 指定已有 SFT 模型、`--resume-from` 续跑。
完整命令见 [训练脚本说明](../scripts/TRAINING_SCRIPTS.md)。
Prober 最大解题长度已设为 4096 tokens，要求输出 boxed 标量答案；应先检查实际
解题样例、截断率和判分，避免把输出格式问题当成难度。

当前 proposer GRPO/MGPO 是单设备、全参数更新实现，同时加载训练策略、冻结参考策略和冻结
Prober，不具备 LoRA/FSDP/DeepSpeed 模型分片。GPU 显存和端到端 CUDA 训练尚需
在训练机验证。基础配置的 16 次更新与上述专用脚本的 100 次更新均是试跑预算，不能据此声称充分训练。

## 100 个在线 GRPO/MGPO 出题目标

`configs/proposer_seeds.math.grpo100.jsonl` 是出题模型的 RL prompt 池，共 100 行，
覆盖全部 40 个构造 family：20 类各 3 个目标，另 20 类各 2 个。每行包含
`id`、`category`、`skill_ids`、`objective` 和可追溯的 `source_seed_id`，不预填
最终题目、答案或奖励。它从现有 240 个目标中确定性选取，和 SFT 共享任务池，
不是独立测试集。可重新生成：

```bash
.venv-proposer/bin/python scripts/prepare_math_grpo_seeds.py
```

完成 SFT 后，在 CUDA 训练机运行：

```bash
bash scripts/run_proposer_training.sh mgpo --config configs/proposer_training.math.grpo100.json
```

专用配置保留原 SFT checkpoint 路径，GRPO 与 MGPO 各设 100 次更新、每组 8 条轨迹。
输出目录分别为 workspace 下的 `training/grpo-proposer100/` 和 `training/mgpo-grpo100/`。
每组针对一个 seed 在线采样，验证最终题目、调用冻结 Prober 测量 pass rate，
再计算相应算法的奖励。100 个 seed 不等于 100 条已生成的 rollout；当前 curriculum
按类别和目标随机抽样，100 次更新最多启动 800 条轨迹，不保证每个 seed 都被访问。
无效和失败轨迹也参与 RL 分组，不能将 800 条视为 800 道合格题目。

## 当前采集记录

2026-10-05：API 冒烟调用成功，两次试采集分别产出 1 条合格轨迹。经精确答案、
实际工具执行、skill lineage 和模型委员会报告复核，2 条已导入正式数据，保留原始
workspace 来源与运行记录。之后曾因 EnvMux 域名 DNS 解析失败停止。
同日重新测试，三个指定模型均返回成功，已恢复正式采集；恢复时为 **2/500，未完成**。
以 `research-workspace/math-proposer/training/sft_trajectories.progress.json` 的实时值为准。

同日：技能库升级为 v2，40 个构造计算器分别绑定对应 family，拒绝跨 family 输入。
旧 12 个 Skill 包保存在 `skills/deprecated/*@1.0.0`。新版先做离线数值验证，尚未逐类
进行 API 合成重放。需要小规模 API 测试时使用
`configs/proposer_training.math.validation.json`，目标为 40 类各 1 条，数据保存在独立
workspace，不自动混入 500 条正式数据。

2026-10-07：正式轨迹采集 Teacher 从 `qwen3.5-flash` 切换到 `qwen3.7-plus`，
Verifier 仍为 `qwen3.5-flash`。2026-10-08：Teacher 与 Verifier 均切换为
`qwen3.8-flash`。切换保留已有接受轨迹与采集断点；每条记录保留实际模型来源。
指纹迁移凭据为 `research-workspace/math-proposer/training/teacher-model-switch.json`。
