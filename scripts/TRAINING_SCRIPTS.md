# 出题模型训练入口

三个脚本均训练出题模型，自动切换到仓库根目录：

| 脚本 | 输入 | 更新目标 |
|---|---|---|
| train_sft.sh | 已接受的 136 条轨迹、1,543 个动作 | 监督学习模型动作 |
| train_grpo.sh | SFT checkpoint + 100 条 prompt | 同一 prompt 内终局奖励标准化，token PPO clipping 与参考模型 KL |
| train_mgpo.sh | SFT checkpoint + 100 条 prompt | 终局与过程奖励、阶段优势和 MGPO 更新 |

在 CUDA 机器运行，模型路径替换为自己的路径：

    bash scripts/train_sft.sh --base-model /models/Qwen3-4B
    export ENVMUX_API_KEY='YOUR_KEY'
    bash scripts/train_grpo.sh --prober-model /models/Qwen3-4B
    bash scripts/train_mgpo.sh --prober-model /models/Qwen3-4B

GRPO 与 MGPO 分别从相同 SFT checkpoint 开始，互相不依赖。Verifier 默认使用
qwen3.8-flash。两者在线采样，默认各 100 次更新、每组 8 条轨迹，冻结 Prober
对有效题采样 16 次计算答对比例。GRPO 不调用反思过程奖励评分。

默认 SFT 输出为 research-workspace/math-proposer/training/sft_checkpoint/final，
GRPO 输出为 training/grpo-proposer100，MGPO 输出为 training/mgpo-grpo100。
后两个路径均位于同一 math-proposer workspace 内，保存各自的 rollout 和 checkpoint。
RL 每 10 次更新及最后一次保存 checkpoint，默认保留最近 2 个（save_steps/save_total_limit）；
原 SFT 参考模型与指定的续跑来源不会自动删除。每轮 rollout 均写入日志。
单设备全参数训练会同时加载训练策略、参考策略和 Prober；当前 RL 不支持 torchrun/DDP。

常用选项：

    bash scripts/train_sft.sh --dry-run
    bash scripts/train_grpo.sh --dry-run
    bash scripts/train_mgpo.sh --dry-run
    bash scripts/train_mgpo.sh --sft-checkpoint /checkpoints/sft/final --prober-model /models/Qwen3-4B --iterations 10
    bash scripts/train_grpo.sh --resume-from /checkpoints/grpo/checkpoint-0010 --prober-model /models/Qwen3-4B

--dry-run 仅读取配置和数据，不下载模型、不调用 API。它报告 SFT checkpoint 是否存在；
启动实际 RL 前必须先训练 SFT 或通过 --sft-checkpoint 提供已有模型。
--config 可替换配置文件；--output-dir 可替换当前阶段输出目录。
续跑须保持原参考模型、prompt、Verifier 模型和 RL 参数设置；增加 iterations 可继续更新。
已有 rollout 或 checkpoint 的目录禁止直接重跑，需使用 --resume-from 或新的 --output-dir。

脚本默认建立 .venv-proposer 并安装 training 依赖；使用已有环境时设置
PROPOSER_PYTHON=/path/to/python，已安装依赖时再设置 SKIP_INSTALL=1。
SFT 最低轨迹数明确设为 136，原 500 条采集目标不再阻止本批数据训练。
版本清单及校验哈希见 configs/sft_dataset.math.json。
SFT/RL 上下文预算为 32,768 tokens；本批数据中 87 个动作的固定状态加输出超过原 16K
预算，最长固定状态加输出为 26,519 tokens，因此不能直接降回 16K。
使用默认 Qwen tokenizer 全量检查：1,543 个动作均可编码，57 个会裁去部分早期历史，
监督动作完整保留。训练机可用对应模型的 tokenizer 离线复核：

    python scripts/check_sft_lengths.py --tokenizer /models/Qwen3-4B


下游解题模型的 Solver GRPO 是独立的可选入口：

    bash scripts/run_proposer_training.sh solver-grpo --config configs/proposer_training.math.grpo100.json

该入口需要 MGPO 产出的有效题目/答案，不属于上述出题模型 GRPO 基线。
权重、API 密钥、采集失败日志和训练日志不纳入 Git。
