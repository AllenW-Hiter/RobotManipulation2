# Chunk 级别在线强化微调转换计划

## 1. 背景与问题

当前 FPO++ 在线微调中，策略一次生成 `n_action_steps` 个动作组成的 action chunk，但 rollout、reward、GAE 和 value loss 仍主要按环境的 primitive step 组织。策略更新时又把 FPO/DPPO 的 ratio 聚合到 chunk 级，并使用 chunk 起点的 advantage。这会造成时间尺度不一致：actor 的“动作”是 chunk，而 critic/advantage 的估计来源是单步序列。

新分支目标是把强化学习中的一个时间步定义为一个 action chunk。环境仍按单步 API 执行，但采样、回报、advantage、AC 更新全部按 chunk 级 transition 组织。

建议分支名：

```bash
git checkout -b chunk-level-ac-finetune
```

## 2. 设计原则

- 策略动作：`a_j = [u_{j,0}, ..., u_{j,K-1}]`，其中 `K = n_action_steps`。
- RL 时间步：从 `s_j` 执行完整 chunk 到 `s_{j+1}`，而不是每个 primitive action 都作为一次策略决策。
- 环境交互：仍调用 `env.step(u_{j,i})`，但只在 chunk 完成或 episode 结束后写入一个 macro transition。
- 回报估计、advantage、PPO ratio 和 value target 必须使用同一 chunk 时间尺度。
- chunk 内 done 之后的动作无效，不能进入 reward、logprob、CFM loss、DPPO path 或 value loss 聚合。

## 3. 新数据结构

新增 chunk 级 rollout buffer，替代当前按 step 存储再 reshape 的方式。建议字段如下：

```python
ChunkTransition:
    obs_start              # chunk 起点观测 s_j
    obs_next               # chunk 结束后观测 s_{j+1}
    action_chunk           # (num_envs, K, action_dim)
    valid_action_mask      # (num_envs, K)，done 后为 0
    inner_rewards          # (num_envs, K)，仅用于日志和调试
    chunk_reward           # (num_envs,)
    chunk_discount         # (num_envs,)，通常是 gamma ** valid_len；done 时为 0
    chunk_done             # (num_envs,)
    value_start            # V(s_j)
    value_next             # V(s_{j+1})
    old_cfm_loss           # FPO 使用，保留 chunk 维度
    old_cfm_t              # FPO 使用
    old_cfm_epsilon        # FPO 使用
    mdp_x_t_path           # DPPO 使用，(num_envs, K, sampling_steps, action_dim)
    old_dppo_log_prob      # DPPO 使用，(num_envs, K, sampling_steps)
```

buffer 主维度应是：

```text
(chunks_per_iteration, num_envs, ...)
```

其中：

```text
chunks_per_iteration = data_collection_steps // n_action_steps
```

`data_collection_steps` 仍表示 primitive env steps 总数；训练 batch size 改为 `chunks_per_iteration * num_envs`。

## 4. Chunk 采样流程

每个 chunk 的采样流程：

1. 保存 `obs_start = next_obs`。
2. 调用策略生成完整 action chunk，并同时保存 FPO/DPPO 需要的旧策略信息。
3. 对 `i in range(K)` 逐个执行 `env.step(action_chunk[:, i])`。
4. 累积 chunk 内 reward、done 和 valid mask。
5. 若某个 env 在 chunk 内 done，则该 env 后续 primitive action 标记为无效。
6. chunk 结束后保存 `obs_next`，并计算 `value_start` 与 `value_next`。

推荐将环境单步到 chunk transition 的转换封装成独立函数：

```python
def rollout_one_chunk(actor, critic, env, obs_start, done_start, cfg):
    ...
    return transition, obs_next, done_next
```

这样 `finetune_online_rl.py` 的主循环可以直接按 chunk 推进，避免后续再做复杂 reshape。

## 5. Chunk 回报计算

对每个环境单独计算：

```text
R_j = sum_{i=0}^{L_j-1} gamma^i * r_{j,i}
```

其中 `L_j` 是该 chunk 内有效 primitive step 数。如果没有 done，`L_j = K`；如果第 `m` 个 primitive step 发生 done，则 `L_j = m + 1`。

bootstrap 折扣：

```text
D_j = gamma ^ L_j
```

若 chunk 内 done：

```text
D_j = 0
```

因此 TD residual 为：

```text
delta_j = R_j + D_j * V(s_{j+1}) - V(s_j)
```

这比直接取 `mb_advantages[:, 0:1]` 更一致，因为 `R_j` 明确对应整段 chunk 的执行结果。

对于 sparse success reward，可保留默认求和；如任务中 success reward 会在完成后持续为 1，需要增加配置控制：

```python
chunk_reward_mode: Literal["discounted_sum", "sum", "max", "last"] = "discounted_sum"
```

默认使用 `discounted_sum`，调试时可比较 `max` 是否更适合二值成功奖励。

## 6. Chunk 级 GAE

在 chunk 维度上计算 GAE：

```text
A_j = delta_j + D_j * lambda * A_{j+1}
return_j = A_j + V(s_j)
```

注意这里不再使用固定 `gamma`，而使用每个 chunk 的 `chunk_discount = gamma ** valid_len`。这样可以正确处理 chunk 内提前结束的 episode。

建议新增函数：

```python
def calculate_chunk_advantage(
    values_start,
    values_next_last,
    chunk_rewards,
    chunk_discounts,
    chunk_dones,
    gae_lambda,
):
    ...
```

返回形状：

```text
advantages: (chunks_per_iteration, num_envs)
returns:    (chunks_per_iteration, num_envs)
```

## 7. Actor 更新改造

### FPO

FPO 的 ratio 应继续按 chunk 聚合：

```text
logratio_chunk = old_cfm_loss_chunk - curr_cfm_loss_chunk
ratio_chunk = exp(logratio_chunk)
```

但 advantage 改为新的 `chunk_advantage`：

```python
pg_loss = PPO_or_SPO(ratio_chunk, chunk_advantage)
```

chunk 内无效动作用 `valid_action_mask` 过滤。若使用 loss 求和，长度短的 chunk 会天然贡献更少；若使用平均，需要用有效长度归一化。建议默认使用求和，以匹配 chunk log-prob / loss 的联合动作含义。

### DPPO

DPPO 的 log-prob 也按 chunk 聚合：

```text
logprob_chunk = sum_over_valid_actions_and_denoising_steps(logprob)
ratio_chunk = exp(logprob_chunk_new - logprob_chunk_old)
```

再与 `chunk_advantage` 相乘。`mdp_x_t_path` 继续保留 `(B, K, sampling_steps, action_dim)`，但只在 `valid_action_mask == 1` 的位置参与 log-prob 聚合。

## 8. Critic 更新改造

critic 的主 value loss 改为只训练 chunk 起点：

```text
v_loss = 0.5 * (V(s_j) - return_j)^2
```

这使 critic 与 chunk-level actor 的决策点一致。可以保留一个可选辅助项训练 chunk 内 primitive state value，但默认关闭，避免重新引入 step-level 目标。

建议配置：

```python
train_primitive_value_aux: bool = False
primitive_value_aux_coef: float = 0.1
```

## 9. Done 与 Reset 处理

chunk 内某个 env done 后：

- `valid_action_mask` 从 done 后下一步开始为 0。
- `chunk_done=True`。
- `chunk_discount=0`，不 bootstrap `V(s_{j+1})`。
- actor 的 action buffer、`mdp_x_t_path_buffers` 必须 reset 对应 env。
- 若环境自动 reset，需要确保 `obs_next` 是 reset 后的新 episode 初始观测；若非自动 reset，则在 chunk 边界统一 reset。

不建议在一个 chunk 内把 done 后 reset 出来的新 episode 继续塞进同一个 transition，否则一个 macro action 会跨 episode，回报语义会混乱。

## 10. 配置开关

新增配置，先允许新旧实现共存：

```python
rollout_granularity: Literal["step", "chunk"] = "step"
chunk_reward_mode: Literal["discounted_sum", "sum", "max", "last"] = "discounted_sum"
chunk_value_loss_only_at_start: bool = True
train_primitive_value_aux: bool = False
```

迁移初期默认仍使用 `"step"`，新分支实验显式设置：

```bash
python finetune_online_rl.py --rollout_granularity chunk ...
```

验证稳定后，再考虑将 chunk 级实现设为默认。

## 11. 实施步骤

1. 新增 chunk rollout buffer 和 `rollout_one_chunk`，先只支持 `loss_mode="fpo"`。
2. 实现 `calculate_chunk_advantage`，用小张量单元测试验证 done、truncated、短 chunk 和 bootstrap。
3. 改造 FPO 更新路径，直接消费 `(B, K, ...)` 的 chunk batch，不再从 step buffer reshape。
4. 接入 DPPO 的 `mdp_x_t_path` 和 log-prob 聚合。
5. 调整日志，分别记录 `chunk_reward_mean`、`chunk_return_mean`、`valid_chunk_len_mean`、`inner_reward_sum`。
6. 对比旧实现和新实现：同一 checkpoint、同一 seed、短 rollout，确认采样数量和环境成功率统计一致。
7. 跑 reduced-step 训练，检查 ratio、advantage、value loss 是否数值稳定。

## 12. 验证清单

- `chunk_reward` 等于 chunk 内有效 reward 的折扣和。
- chunk 内 done 后没有动作、CFM loss、DPPO log-prob 继续参与 actor loss。
- `chunk_discount` 在完整 chunk 时为 `gamma ** K`，在 done chunk 时为 0。
- `advantages` 和 `returns` 形状为 `(chunks_per_iteration, num_envs)`。
- FPO/DPPO 的 ratio 形状与 `chunk_advantage` 对齐。
- value loss 只使用 chunk 起点观测，除非显式开启 primitive auxiliary value。
- W&B 日志能同时看到 chunk 级 reward 和原始 inner reward，方便排查奖励聚合是否符合任务定义。

## 13. 预期收益与风险

预期收益是 actor、advantage 和 value target 在同一 chunk 时间尺度上优化，更符合 flow policy 一次输出动作块的决策结构。对于 chunk 内局部动作好坏交替的情况，更新会根据整段执行后的累计效果推动策略，而不是把同一个 chunk 拆成多个不独立的 step-level 决策。

主要风险是 credit assignment 会更粗：chunk 越长，越难判断 chunk 内哪个 primitive action 导致收益变化。因此需要同时记录 `inner_rewards` 和 `valid_action_mask`，并对比不同 `n_action_steps`、`chunk_reward_mode` 和 advantage normalization 设置。
