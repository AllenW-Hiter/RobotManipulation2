# Meanflow Policy BC 跑通计划

## 目标

先在当前 manipulation 项目中跑通一个纯 BC 的 MeanFlow Policy 基线：能训练、保存、加载，并在 Can 等任务上通过 `eval_checkpoint.py` 完成动作 chunk 决策评估。该阶段不引入 DAMP、ADN 或在线强化学习，只验证 MeanFlow policy 本身的监督学习和推理闭环。

## 范围

本计划只覆盖 BC 预训练和 checkpoint 评估。核心文件包括 `src/meanflow_model.py`、`src/meanflow_model_config.py`、`src/flow_net_mlp.py`、`pretrain_flow_bc.py` 和 `eval_checkpoint.py`。`finetune_online_rl.py` 暂不接入 DAMP，只在 MeanFlow BC checkpoint 可用后再考虑在线强化微调。

## 当前状态核对

`MeanFlowConfig` 已通过 `@PreTrainedConfig.register_subclass("meanflow")` 注册，`MeanFlowPolicy.name = "meanflow"` 已存在。`src/meanflow_model.py` 已有 `predict_action_chunk()`、`get_meanflow_loss()` 和 FPO replay loss 雏形。`src/flow_net_mlp.py` 已支持 `meanflow_separate_time_encoders` 和 `meanflow_encode_t_minus_r`；其中 `meanflow_encode_t_minus_r=True` 的语义是额外加入区间编码 `[t, r, t-r]`，不是用 `t-r` 替换 `r`。

原主要缺口是训练入口：`pretrain_flow_bc.py` 只处理 `cfg.policy == "flowmatching"`。阶段 1 已补齐 `--policy meanflow` 的创建、恢复和保存路径。

## 阶段 1：补齐 MeanFlow BC 训练入口

在 `pretrain_flow_bc.py` 中增加 `meanflow` policy 分支：

- 导入 `MeanFlowConfig` 和 `MeanFlowPolicy`。
- 允许 CLI 使用 `--policy meanflow`。
- 按 FlowMatching 分支相同方式解析 `horizon`、`n_action_steps`、`sampling_steps`、`vision_backbone`、`learning_rate`、`lr_backbone`、`weight_decay`、`network_architecture` 和 `mlp_dims`。
- 构造 `MeanFlowConfig`，默认使用 `network_architecture="mlp"`、`cfm_loss_mode="u"`、`meanflow_time_sampling="logit_normal_pair"`、`meanflow_flow_ratio=0.5`。
- 保持 dataset stats、feature shape、normalization、EMA 保存逻辑与 FlowMatching 一致。

验收标准：能运行到创建 policy 和 dataloader，不因 policy 类型报错。

实施记录（2026-06-02）：

- `pretrain_flow_bc.py` 已导入 `MeanFlowConfig` 和 `MeanFlowPolicy`。
- `TrainFlowBCConfig.policy` 已支持 `meanflow`。
- 新增 MeanFlow BC CLI 参数：`meanflow_flow_ratio`、`meanflow_time_sampling`、`meanflow_logit_mu`、`meanflow_logit_sigma`、`meanflow_use_adaptive_loss`、`meanflow_adaptive_gamma`、`meanflow_adaptive_c`、`meanflow_separate_time_encoders`、`meanflow_encode_t_minus_r`。
- `pretrain_flow_bc.py` 的 policy config 构造已支持 FlowMatching/MeanFlow 双分支；MeanFlow 当前强制 `network_architecture="mlp"`。
- checkpoint 恢复和新建 policy 的逻辑已改成按 `cfg.policy` 选择 config class 和 policy class。
- W&B 初始化已取消默认 `far-wandb`，只有显式传入 `--wandb-entity` 时才传 entity；若从 W&B artifact 恢复 checkpoint，仍要求显式提供 entity。
- 已用项目 conda 解释器确认 CLI 暴露 `--policy {flowmatching,meanflow}` 以及全部 `--meanflow-*` 参数；本地检查命令需设置 `NUMBA_DISABLE_JIT=1`。

## 阶段 2：验证 MeanFlow 前向训练 loss

检查 `MeanFlowPolicy.forward()` 是否与 `pretrain_flow_bc.py` 的训练循环输出约定一致：

- 输入 batch 经过 `normalize_inputs`。
- action target 经过 `normalize_targets`。
- `get_meanflow_loss()` 返回可反传的标量 loss。
- loss dict 至少包含 `loss` 或训练循环实际读取的字段。
- `step_ema()` 在训练后正常更新。

验收标准：用很小步数完成一次 smoke train，例如 100 到 500 step，不出现 shape、device、dtype 或 autograd JVP 错误。

实施记录（2026-06-02）：

- 修复 `src/meanflow_model.py` 中被 FlowMatching 版本 `forward_fpo()` 覆盖 MeanFlow 实现的问题，保留 MeanFlow replay 版本 `forward_fpo()`。
- 将 `MeanFlowPolicy.get_cfm_loss()` 改为 BC 训练循环兼容包装，实际转发到 `get_meanflow_loss()`。
- 修复 `MeanFlowPolicy.forward()` 中 `cfm_loss_r` 和 `cfm_loss_eps` 传参顺序错误，避免后续 FPO/DAMP replay 取错张量。
- 已通过语法检查：`python3 -m py_compile pretrain_flow_bc.py src/meanflow_model.py src/meanflow_model_config.py`。
- 已用 state-only dummy batch 验证 `MeanFlowPolicy.get_cfm_loss()` 可反传，`predict_action_chunk()` 输出 action chunk 形状 `(2, 16, 7)`，`mdp_x_t_path` 形状 `(2, 3, 16, 7)`。
- 系统 `python3` 缺少 `imageio`，无法直接启动脚本；使用项目 conda 解释器可显示 CLI help。真实 smoke train 仍需在完整实验环境中执行。

## 阶段 3：跑通 MeanFlow checkpoint 保存与加载

确认 `save_pretrained()` 输出的 `policy/` 目录包含 config、model 权重、normalization stats 和 EMA 状态。随后用 `eval_checkpoint.py` 加载：

- 确保评估脚本 import 了 `MeanFlowPolicy`，避免 LeRobot 自动加载时找不到 policy class。
- `eval_checkpoint.load_policy()` 能根据 `config.json` 中的 `type="meanflow"` 构造 `MeanFlowPolicy`。
- `predict_action_chunk()` 返回 `(B, n_action_steps, action_dim)`，并能填充 action buffer。

验收标准：`eval_checkpoint.py` 能加载 MeanFlow BC checkpoint 并完成至少 5 条 Can 评估轨迹。

实施记录（2026-06-02）：

- `eval_checkpoint.py` 已导入 `MeanFlowConfig` 和 `MeanFlowPolicy`。
- `load_policy()` 已从 `policy/config.json` 的 `type` 字段自动选择 `flowmatching` 或 `meanflow`。
- `eval_checkpoint.py` 的 rollout policy 类型已兼容 `FlowMatchingPolicy | MeanFlowPolicy`。
- W&B eval logging 已取消默认 `far-wandb`；只有显式传入 `--wandb-entity` 时才传 entity。若从 W&B artifact 下载 checkpoint，仍要求显式提供 entity。
- 已用 `/tmp` 生成最小 MeanFlow policy checkpoint，验证 `eval_checkpoint.load_policy()` 可重新加载，并且 `select_action()` 输出 action 形状 `(2, 7)`、path 形状 `(2, 2, 7)`。

## 阶段 4：最小 Can 训练与评估命令

先用低成本配置验证链路：

```bash
python pretrain_flow_bc.py \
  --policy meanflow \
  --dataset <can_dataset_name_or_path> \
  --horizon 16 \
  --n-action-steps 16 \
  --sampling-steps 5 \
  --network-architecture mlp \
  --mlp-dims "[512, 512, 512]" \
  --cfm-loss-mode u \
  --batch-size 64 \
  --steps 500 \
  --wandb-enable False \
  --save-freq 500 \
  --seed 0
```

再评估保存的 checkpoint：

```bash
python eval_checkpoint.py \
  --local-checkpoint-path runs/<meanflow_run>/checkpoints/step_500 \
  --eval-env Can \
  --eval-num-episodes 5 \
  --eval-num-envs 2 \
  --zero-sampling True \
  --load-ema True \
  --wandb-enable False \
  --save-video False
```

如果 CLI 参数名与当前脚本不一致，以 `python pretrain_flow_bc.py --help` 和 `python eval_checkpoint.py --help` 为准，并把最终可运行命令记录回本文档。

## 阶段 5：训练稳定性检查

记录并对齐以下指标：

- `meanflow_loss` 是否稳定下降。
- action 均值、方差、min/max 是否在归一化后合理。
- zero sampling 与 random sampling 的评估差异。
- `sampling_steps=1/5/10` 的成功率和耗时。
- `meanflow_encode_t_minus_r`、`meanflow_separate_time_encoders` 的开关对 loss 和评估的影响。

建议第一轮只改一个变量，默认组合为 `meanflow_separate_time_encoders=False`、`meanflow_encode_t_minus_r=False`，后续再对齐 IsaacLab/MVPO 的时间编码习惯。

实施记录（2026-06-02）：

- 修复 `src/flow_net_mlp.py` 的 `MeanflowMLPModel.encode_time_pair()`，开启 `meanflow_encode_t_minus_r` 时返回 `[emb(t), emb(r), emb(t-r)]`。
- `predict_action_chunk()` 和 `get_meanflow_loss()` 已统一调用 `encode_time_pair()`，避免训练和采样路径手写拼接产生维度不一致。
- 保留 `meanflow_encode_t_minus_r` 参数名以兼容 CLI 和 checkpoint，配置注释已改为“额外加入区间编码”。

## 阶段 6：为 DAMP 预留接口

MeanFlow BC 跑通后，再进入 DAMP 迁移。BC 阶段需要保留以下可复用接口：

- chunk 级动作 latent 或初始噪声采样入口。
- 可 replay 的 MeanFlow loss：`eps/t/r` 外部传入时必须和 rollout 时一致。
- `predict_action_chunk()` 的 `mdp_x_t_path` 形状稳定，供在线 RL 存储使用。
- checkpoint config 中必须保留 MeanFlow 专属参数，避免 RL 加载后默认值漂移。

完成这些后，DAMP 才能在当前项目中以“chunk action”为单位增加 ADN、ADN KL damping 和 MeanFlow proxy ratio。
