# FPO 微调实验总结

所有 run 都使用 `far_manipulation-fpo` 中的 `finetune_online_rl.py`，并设置 `--distributed True`。W&B 项目为 `far-wandb/flow-bc-fpo-finetuning`。

## 目录

1. [主 Benchmark](#1-主-benchmark)
2. [Base Policy Checkpoint Ablation](#2-base-policy-checkpoint-ablation)
3. [FPO Ablation Study](#3-fpo-ablation-study)
4. [运行方式](#运行方式)

## Base Policy

| 任务 | Base Policy Run | Checkpoint Step | Project |
|---|---|---|---|
| Can | `95j3noe4` | `step_1000`（主实验）/ `step_6000`（ablation） | `flow-bc` |
| Square | `trc7rbt0` | `step_110000` | `flow-bc` |
| Box Clearance (`TwoArmBoxCleanup`) | `lainyisy` | `step_10000` | `flow-bc` |
| Tray Lifting (`TwoArmLiftTray`) | `ri0w9j39` | `step_20000` | `flow-bc` |
| Threading (`TwoArmThreading`) | `6vqrn614` | `step_10000` | `flow-bc` |

## 1. 主 Benchmark

主 Benchmark 包含 5 个任务、4 个模型、3 个 seed（0、1、2），共 60 个 run。每个任务和模型组合内的 seed 使用相同超参数。

### 共享参数

```text
--distributed True
--load-ema True
--gradient_accumulation_steps 1
--num_minibatches 8
--log_freq 1
--save_freq 2
--rollout_freq 2
--eval_num_episodes 200
--wandb_enable True
--data_collection_steps=1600
--do_chunk_level_ppo=True
--eval_ema=False
--exploration_noise_std=None
--freeze_vision_encoder=True
--gae_lambda=0.99
--n_action_samples=8
--n_action_steps=16
--num_envs=30
--sampling_steps=10
--spo_clip_coef=0.01
--trust_region_mode=ppo
--zero_sampling=True
```

### 任务级共享参数

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `base_policy_wandb_run_id` | `95j3noe4` | `trc7rbt0` | `lainyisy` | `ri0w9j39` | `6vqrn614` |
| `checkpoint_step` | `step_1000` | `step_110000` | `step_10000` | `step_20000` | `step_10000` |
| `total_timesteps` | 5000000 | 8000000 | 5000000 | 8000000 | 8000000 |
| `discount` | 0.99 | 0.995 | 0.995 | 0.999 | 0.999 |

### FPO++（15 个 run）

Run ID：

| 任务 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Can | `wbxzw7z3`（顺序未确认） | `e0h4wy1r` | `i6rmkgrh`（seed=0） |
| Square | `rsmunbo4` | `wzyv707a` | `z2u9ryms` |
| Box Clearance | `ujjdjtov` | `qlux3x9d` | `o6kv0feo` |
| Tray Lifting | `dcwh6cja` | `rdigmvk4` | `oi516dvb` |
| Threading | `bt3sl4ex` | `g2ldgoss` | `fu8wpmbd` |

模型参数：`loss_mode=fpo`（默认）、`sde_sigma=0`、`cfm_loss_average_group_size=1`、`cfm_loss_use_huber=True`。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.02 | 0.01 | 0.03 | 0.03 | 0.01 |
| `max_grad_norm` | 5 | 25 | 5 | 1 | 1 |
| `cfm_loss_huber_delta` | 0.5 | 1 | 0.1 | 1 | 0.1 |
| `clamp_logratio` | 5 | None | 5 | 5 | 5 |
| `clamp_old_cfm_loss` | 4 | None | 4 | 4 | 4 |

### DPPO - Learned Noise（15 个 run）

模型参数：`loss_mode=dppo`、`sde_sigma=0.18`、`learn_sde_sigma=True`（Can/Square/BoxClearance），并启用 noise injection。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| `max_grad_norm` | 25 | 5 | 1 | 25 | 1 |
| `cfm_loss_huber_delta` | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| `noise_injection_min` | 0.3 | 0.3 | 0.2 | 0.3 | 0.3 |
| `noise_injection_max` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `learn_sde_sigma` | True | True | True | 默认 False | 默认 False |

### Vanilla FPO（15 个 run）

模型参数：`loss_mode=fpo`（默认）、`sde_sigma=0`、`cfm_loss_average_group_size=-1`、`cfm_loss_use_huber=False`、`clamp_logratio=None`、`clamp_old_cfm_loss=None`。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| `max_grad_norm` | 25 | 25 | 5 | 25 | 25 |
| `cfm_loss_huber_delta` | 0.5 | 1 | 0.5 | 0.5 | 0.5 |

### DPPO - Fixed Noise（15 个 run）

模型参数：`loss_mode=dppo`、`cfm_loss_average_group_size=1`、`cfm_loss_use_huber=True`，并使用固定 `sde_sigma`。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 |
| `max_grad_norm` | 25 | 25 | 25 | 5 | 5 |
| `cfm_loss_huber_delta` | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| `sde_sigma` | 0.3 | 0.3 | 0.24 | 0.24 | 0.24 |

## 2. Base Policy Checkpoint Ablation

该实验只使用 Can 任务，将 base policy checkpoint 改为 `step_6000`，共 4 个模型 x 3 个 seed = 12 个 run。除 `checkpoint_step=step_6000` 外，任务参数与主 Benchmark 中的 Can 设置一致。

| 模型 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| FPO++ | `txf5crib` | `ylxhw9uu` | `6zp46k32` |
| Vanilla FPO | `gzm22fqh` | `opygcs30` | `89jg66ll` |
| DPPO Learned | `7lz0maky` | `ss13djab` | `nqkx8w0s` |
| DPPO Fixed | `4iz08o2h` | `0wo72noe` | `0yibse8t` |

| 参数 | FPO++ | Vanilla FPO | DPPO Learned | DPPO Fixed |
|---|---|---|---|---|
| `loss_mode` | fpo | fpo | dppo | dppo |
| `cfm_loss_average_group_size` | 1 | -1 | 1 | 1 |
| `cfm_loss_use_huber` | True | False | True | True |
| `clamp_logratio` | 5 | None | 5 | 5 |
| `clamp_old_cfm_loss` | 4 | None | 4 | 4 |
| `clip_coef` | 0.02 | 0.02 | 0.01 | 0.01 |
| `max_grad_norm` | 5 | 5 | 1 | 5 |
| `cfm_loss_huber_delta` | 0.5 | 0.5 | 0.5 | 0.5 |
| `sde_sigma` | 0 | 0 | 0.18 | 0.3 |

## 3. FPO Ablation Study

该实验在 Square 和 Threading 上比较 FPO++、ASPO 和 per-action ratio 变体，共 3 个模型 x 3 个 seed x 2 个任务 = 18 个 run。

Square：

| 模型 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| FPO++ | `z2u9ryms` | `wzyv707a` | `rsmunbo4` |
| ASPO | `whhhg3oc` | `safs53cm` | `069ligh5` |
| Per-action ratio | `eikmrvae` | `kf0ywqzj` | `4x5cxo0w` |

Threading：

| 模型 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| FPO++ | `bt3sl4ex` | `fu8wpmbd` | `g2ldgoss` |
| ASPO | `4pwr3dzu` | `ir471vi8` | `nx0ktwru` |
| Per-action ratio | `l5hdijzn` | `pcu57hf1` | `xmklzaca` |

核心 ablation 参数：

| 参数 | FPO++ | ASPO | Per-action ratio |
|---|---|---|---|
| `trust_region_mode` | ppo | aspo | ppo |
| `cfm_loss_average_group_size` | 1 | 1 | -1 |
| `cfm_loss_use_huber` | True | True | True |
| `clamp_logratio` | None（Square）/ 5（Threading） | 5 | 5 |
| `clamp_old_cfm_loss` | None（Square）/ 4（Threading） | 4 | 4 |

## 运行方式

各脚本都支持 `DRY_RUN=1` 预览命令，支持 `NUM_GPUS=N` 设置 `torchrun` 使用的 GPU 数。

```bash
# 预训练 base policy
DRY_RUN=1 bash scripts/run_pretrain_base_policies.sh
bash scripts/run_pretrain_base_policies.sh

# 主 Benchmark
DRY_RUN=1 bash scripts/run_main_benchmark.sh
bash scripts/run_main_benchmark.sh

# Checkpoint ablation
DRY_RUN=1 bash scripts/run_checkpoint_ablation.sh
bash scripts/run_checkpoint_ablation.sh

# FPO ablation
DRY_RUN=1 bash scripts/run_fpo_ablation.sh
bash scripts/run_fpo_ablation.sh
```

Can 任务的 FPO++ 示例命令：

```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_1000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-can-base-uul2gclip25-dec25-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=5 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```
