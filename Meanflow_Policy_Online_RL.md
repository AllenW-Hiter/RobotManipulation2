# MeanFlow Policy 在线强化微调实施方案

## 目标

在当前 manipulation 项目中接入 MeanFlow Policy 的在线强化微调，优先实现 `loss_mode=fpo` + `rollout_granularity=chunk` 的稳定版本。该版本以 action chunk 作为 RL 的宏动作，沿用已有 FPO++ 的 PPO ratio、critic、chunk reward 和 GAE 框架，但把 FlowMatching 的 CFM replay loss 替换为 MeanFlow 的 replay loss。

首个目标任务为 Can，基础 checkpoint 使用已验证较好的 MeanFlow BC：

```text
runs/meanflow_sampling_steps_ablation_can_seed3_step1000/train_s2/checkpoints/step_999
```

## 当前状态

已有基础：

- `MeanFlowPolicy` 已能 BC 训练、保存、加载和评估。
- `MeanFlowPolicy.forward_fpo()` 已返回 MeanFlow replay 所需的 `loss/t/r/eps`。
- `finetune_online_rl.py` 已有 chunk 级 rollout、chunk reward、chunk GAE、chunk-level PPO ratio 和 critic update。
- Can BC ablation 中，`train_s2_eval_s2` 在 50 episodes 上达到最高成功率 `88%`。

主要缺口：

- `finetune_online_rl.py` 目前硬编码 `FlowMatchingConfig` 和 `FlowMatchingPolicy`。
- 现有 FPO replay 只存 `cfm_loss_t` 和 `cfm_loss_eps`，MeanFlow 还必须存 `cfm_loss_r`。
- `get_cfm_values()` 当前按 FlowMatching 的 3 返回值设计；MeanFlow 返回 4 个值。
- 训练更新阶段 replay 时只传 `t/eps`，MeanFlow 必须同时传 `t/r/eps`，否则 replay loss 语义不一致。
- MeanFlow 当前不支持 DPPO/SDE 路径，在线 RL 第一版必须显式限制为 `loss_mode=fpo`。

## 设计原则

1. 第一版只支持 `--policy meanflow --loss-mode fpo --rollout-granularity chunk`。
2. 不实现 MeanFlow DPPO，不启用 `sde_sampling`、`learn_sde_sigma` 或 `forward_dppo()`。
3. 保持 FlowMatching FPO++ 原路径不变，MeanFlow 分支用类型判断隔离。
4. 采样、回报、advantage、ratio 和 value loss 全部按 chunk 级组织。
5. replay loss 必须复用 rollout 时采样到的 `t/r/eps`，不能更新阶段重新采样。

## 阶段 1：策略类型加载

修改 `finetune_online_rl.py`：

- 导入 `MeanFlowConfig` 和 `MeanFlowPolicy`。
- 从 checkpoint `config.json` 读取 `type` 字段。
- 当 `type == "meanflow"` 或 CLI `--policy meanflow` 时，构建 `MeanFlowConfig/MeanFlowPolicy`。
- 当 checkpoint 类型和 CLI `--policy` 不一致时直接报错。
- 日志中打印 `policy_type`、`sampling_steps`、`meanflow_encode_t_minus_r` 和 image keys。

验收：

```bash
python3 -m py_compile finetune_online_rl.py src/meanflow_model.py
```

并能加载 MeanFlow checkpoint 到创建 actor/critic 之前。

## 阶段 2：MeanFlow FPO Replay Buffer

在 `finetune_online_rl.py` 的 rollout storage 中新增：

```python
cfm_loss_rs_stored = torch.zeros((steps_per_iteration, num_envs_per_process, n_action_samples))
```

chunk rollout 中：

- `get_cfm_values()` 对 FlowMatching 返回 `(loss, t, None, eps)`。
- `get_cfm_values()` 对 MeanFlow 返回 `(loss, t, r, eps)`。
- 存储 `cfm_loss_rs_stored[storage_step] = cfm_loss_r[action_idx].cpu()`。
- step rollout 分支如果暂不支持 MeanFlow，直接报错，避免半支持。

重塑 batch 时新增：

```python
b_cfm_loss_rs = cfm_loss_rs_stored.reshape(...)
```

形状与 `b_cfm_loss_ts` 一致：

```text
(local_batch_size, n_action_steps, n_action_samples)
```

验收：

- FlowMatching 路径不需要 `r`，`b_cfm_loss_rs` 可为 `None` 或全 0。
- MeanFlow 路径中 `t/r/eps` 的 batch 维度都能与 `n_action_samples` 对齐。

## 阶段 3：更新阶段 MeanFlow FPO Ratio

在 actor update 的 `cfg.loss_mode == "fpo"` 分支中增加 MeanFlow replay：

```python
old_t = mb_cfm_loss_ts.permute(0, 2, 1).reshape(-1, n_action_steps, 1)[:, 0:1, :]
old_r = mb_cfm_loss_rs.permute(0, 2, 1).reshape(-1, n_action_steps, 1)[:, 0:1, :]
old_eps = mb_cfm_loss_epsilons.permute(0, 2, 1, 3).reshape(-1, n_action_steps, action_dim)
curr_loss, _, _, _ = actor(
    obs_chunk2,
    n_action_samples=n_action_samples,
    cfm_loss_t=old_t,
    cfm_loss_r=old_r,
    cfm_loss_eps=old_eps,
)
```

FlowMatching 保持原调用：

```python
curr_loss, _, _ = actor(..., cfm_loss_t=old_t, cfm_loss_eps=old_eps)
```

随后沿用现有逻辑：

- mask 掉 chunk 内 done 后无效动作。
- 按 chunk 求和或平均 CFM loss。
- `logratio = old_cfm_loss - curr_cfm_loss`。
- 使用 PPO/SPO/ASPO 计算 policy loss。

验收：

- `old_t` 和 `old_r` 在 chunk 内应保持一致，可加 assert：

```python
assert (old_t_full[:, 0, 0] == old_t_full[:, -1, 0]).all()
assert (old_r_full[:, 0, 0] == old_r_full[:, -1, 0]).all()
```

## 阶段 4：配置限制与错误防护

在参数校验阶段加入：

- `policy == "meanflow"` 时，要求 `loss_mode == "fpo"`。
- `policy == "meanflow"` 时，要求 `rollout_granularity == "chunk"`。
- `policy == "meanflow"` 时，禁止 `learn_sde_sigma=True` 和 `sde_sampling=True`。
- `policy == "meanflow"` 时，`do_chunk_level_ppo=True`。
- 第一版建议在线 RL 使用 `n_action_steps=horizon=16`，让一个 chunk 级宏动作覆盖完整预测窗口。BC checkpoint 的 `n_action_steps=8` 只表示 BC 评估时执行前缀，不应限制 chunk 级 PPO 的动作粒度。

推荐默认配置：

```text
policy=meanflow
loss_mode=fpo
rollout_granularity=chunk
chunk_reward_mode=discounted_sum
do_chunk_level_ppo=True
do_average_cfm_loss_in_chunk=False
sampling_steps=2
n_action_steps=16
n_action_samples=8 或 16
freeze_vision_encoder=True
eval_ema=False
```

## 阶段 5：Smoke Test

先跑极小规模，只验证能完成 rollout、update、保存和评估：

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --policy meanflow \
  --base-policy-local-path runs/meanflow_sampling_steps_ablation_can_seed3_step1000/train_s2/checkpoints/step_999 \
  --load-ema True \
  --experiment finetune-meanflow-fpo-can-chunk-smoke \
  --total-timesteps 20000 \
  --task Can \
  --eval-env Can \
  --eval-num-episodes 10 \
  --rollout-freq 2 \
  --eval-save-video False \
  --wandb-upload-eval-video False \
  --wandb-enable False \
  --rollout-granularity chunk \
  --chunk-reward-mode discounted_sum \
  --data-collection-steps 256 \
  --num-envs 4 \
  --n-action-steps 16 \
  --sampling-steps 2 \
  --discount 0.99 \
  --gae-lambda 0.99 \
  --gradient-accumulation-steps 1 \
  --num-minibatches 4 \
  --update-epochs 2 \
  --log-freq 1 \
  --save-freq 2 \
  --loss-mode fpo \
  --cfm-loss-average-group-size 1 \
  --cfm-loss-use-huber False \
  --clip-coef 0.02 \
  --max-grad-norm 5 \
  --clamp-logratio 5 \
  --clamp-old-cfm-loss 4 \
  --trust-region-mode ppo \
  --n-action-samples 8 \
  --do-chunk-level-ppo True \
  --freeze-vision-encoder True \
  --eval-ema False \
  --seed 0
```

验收标准：

- 完成至少 2 个 iteration。
- W&B 关闭时不报错。
- 日志中有 `cfm/curr_cfm_loss_mean`、`cfm/logratio_mean`、`chunk/reward_mean`。
- 保存 `checkpoints/latest`。
- eval 能加载保存后的 MeanFlow RL checkpoint。

## 阶段 6：短程 Can 微调

Smoke 通过后跑 50 万到 100 万 env steps：

```bash
torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --policy meanflow \
  --base-policy-local-path runs/meanflow_sampling_steps_ablation_can_seed3_step1000/train_s2/checkpoints/step_999 \
  --load-ema True \
  --experiment finetune-meanflow-fpo-can-chunk-s2-1m \
  --total-timesteps 1000000 \
  --task Can \
  --eval-env Can \
  --eval-num-episodes 50 \
  --rollout-freq 5 \
  --eval-save-video False \
  --wandb-upload-eval-video False \
  --wandb-enable True \
  --wandb-project flow-bc-fpo-finetuning \
  --rollout-granularity chunk \
  --chunk-reward-mode discounted_sum \
  --data-collection-steps 1600 \
  --num-envs 16 \
  --n-action-steps 16 \
  --sampling-steps 2 \
  --discount 0.99 \
  --gae-lambda 0.99 \
  --gradient-accumulation-steps 1 \
  --num-minibatches 8 \
  --update-epochs 10 \
  --log-freq 1 \
  --save-freq 5 \
  --loss-mode fpo \
  --cfm-loss-average-group-size 1 \
  --cfm-loss-use-huber False \
  --clip-coef 0.02 \
  --max-grad-norm 5 \
  --clamp-logratio 5 \
  --clamp-old-cfm-loss 4 \
  --trust-region-mode ppo \
  --n-action-samples 8 \
  --do-chunk-level-ppo True \
  --freeze-vision-encoder True \
  --eval-ema False \
  --seed 0
```

如果本地 robosuite async 不稳定，改用服务器运行；本机只建议 smoke：

```text
--env-vectorization sync --num-envs 1
```

## 阶段 7：正式对齐 FPO++ Can

短程稳定后，对齐 FPO++ 主实验的数据量：

- `total_timesteps=5000000`
- `data_collection_steps=1600`
- `num_envs` 按显存与环境稳定性设置，服务器 32G 可用 16 起步。
- `eval_num_episodes=200`
- seeds 使用 `0,1,2`

重点对比：

1. BC starting point：MeanFlow `train_s2_eval_s2` vs FlowMatching `95j3noe4_step_1000`。
2. RL 前后成功率提升。
3. `sampling_steps=2` 在线微调是否比 `sampling_steps=10/5/1` 更稳定。
4. `cfm/logratio` 是否频繁触达 clamp。
5. `chunk/reward_mean`、`chunk/done_rate`、`chunk/valid_length_mean` 是否正常。

## 风险与后续方向

- MeanFlow loss 使用 JVP，在线更新比 FlowMatching 更耗显存和时间；必要时降低 `n_action_samples` 或 `update_epochs`。
- `sampling_steps=1` BC 表现偏弱，不建议作为第一版 RL 起点。
- MeanFlow DPPO/SDE 目前未定义，第一版不要混入 `loss_mode=dppo`。
- 若 FPO ratio 抖动大，可优先调小 `clip_coef`、启用/调小 `clamp_logratio`，或把 `n_action_samples` 从 16 降到 8。
- 若 critic 不稳定，先增加 `n_iterations_train_only_value` 或降低 `learning_rate_actor`。

## 实施顺序清单

1. 接入 `MeanFlowConfig/MeanFlowPolicy` 加载。
2. 新增 `cfm_loss_r` storage、reshape 和 minibatch 读取。
3. 改造 `get_cfm_values()`，同时兼容 FlowMatching 与 MeanFlow。
4. 改造 FPO replay 调用，MeanFlow 传入 `t/r/eps`。
5. 加入 MeanFlow 在线 RL 配置限制。
6. 跑 smoke test。
7. 跑 1M Can 短程。
8. 跑 5M Can 正式对齐。

## 实施记录

### 2026-06-03：阶段 1-3

- `finetune_online_rl.py` 已导入 `MeanFlowConfig` 和 `MeanFlowPolicy`。
- checkpoint 加载已读取 `config.json` 的 `type` 字段，并按 `flowmatching/meanflow` 分别构造 policy；CLI `--policy` 与 checkpoint 类型不一致时直接报错。
- MeanFlow 在线 RL 第一版已限制为 `loss_mode=fpo`、`rollout_granularity=chunk`、`do_chunk_level_ppo=True`，并禁止 `learn_sde_sigma=True`。
- rollout storage 新增 `cfm_loss_rs_stored`，形状为 `(steps_per_iteration, num_envs_per_process, n_action_samples)`。
- `get_cfm_values()` 已统一返回 `(loss, t, r, eps)`；FlowMatching 路径中 `r=None`，MeanFlow 路径中保存真实 `r`。
- chunk rollout、旧 step rollout、batch reshape、minibatch 读取都已贯通 `cfm_loss_r`。
- FPO 更新阶段已按 policy 类型分支：FlowMatching replay 传 `t/eps`，MeanFlow replay 传 `t/r/eps`。
- 已通过 `python3 -m py_compile finetune_online_rl.py src/meanflow_model.py src/flow_model.py`。
- 已用 `train_s2` MeanFlow checkpoint 做 CPU replay shape 测试：`old_loss=(8,2,2)`、`old_t/old_r=(8,2,2)`、`old_eps=(8,2,2,7)`，重放后 `curr_loss=(8,2,2)` 且全为 finite。
- 已完成单进程最小在线 RL smoke：`num_envs=1`、`data_collection_steps=8`、`n_action_steps=8`、`update_epochs=1`，能完成 1 个 chunk rollout、1 次 update，并保存 `/tmp/meanflow_online_rl_smoke/checkpoints/step_8`。
- smoke 中发现 minibatch size 为 1 时 advantage 标准差会因 unbiased std 产生 NaN；已在 `finetune_online_rl.py` 中改为 `std(unbiased=False)`。
- 已将在线 RL 推荐配置改为 `n_action_steps=horizon=16`。BC checkpoint 中的 `n_action_steps=8` 只作为 BC 执行前缀设置，chunk 级 PPO 训练时覆盖为 16，使宏动作与完整预测窗口对齐。
- 已完成 `n_action_steps=16` 的单进程最小在线 RL smoke：`num_envs=1`、`data_collection_steps=16`、`update_epochs=1`，日志确认 `N action steps: 16, prediction horizon: 16`，能完成 1 个 chunk rollout、1 次 update，并保存 `/tmp/meanflow_online_rl_smoke_n16_clean/checkpoints/step_16`。
- 已屏蔽 MeanFlow 路径中的无效 `sde_sigma` 覆盖日志；该参数只保留给非 MeanFlow/SDE 路径使用。
