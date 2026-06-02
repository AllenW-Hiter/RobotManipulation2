# 复现 FPO++ 实验

本文说明如何复现《Flow Policy Gradients for Robot Control》中的机械臂操作实验。实验使用预训练行为克隆（BC）checkpoint，在 5 个任务上评估 4 种微调方法：FPO++、Vanilla FPO、DPPO Learned Noise 和 DPPO Fixed Noise。参考开发环境为 Ubuntu 24.04、单张 46GB VRAM 的 L40S GPU。

## 目录

1. [概览](#概览)
2. [环境准备](#环境准备)
3. [Base Policy](#base-policy)
4. [实验 1：主 Benchmark](#实验-1主-benchmark)
5. [实验 2：Checkpoint Ablation](#实验-2checkpoint-ablation)
6. [实验 3：FPO Ablation](#实验-3fpo-ablation)
7. [评估 Base Policy](#评估-base-policy)
8. [绘制结果](#绘制结果)
9. [W&B Run ID 对照](#wb-run-id-对照)
10. [硬件要求](#硬件要求)
11. [辅助脚本](#辅助脚本)

## 概览

总计 90 个训练 run 和 10 个 base policy 评估。

| 实验 | 说明 | Run 数 |
|---|---|---|
| 主 Benchmark | 5 个任务 x 4 个模型 x 3 个 seed | 60 |
| Checkpoint Ablation | Can 任务，`step_6000`，4 个模型 x 3 个 seed | 12 |
| FPO Ablation | 2 个任务 x 3 个模型 x 3 个 seed | 18 |

训练脚本为 `finetune_online_rl.py`，所有 run 使用 `--distributed True`。W&B 项目为 `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning`。

| 方法 | `loss_mode` | 关键区别参数 |
|---|---|---|
| FPO++ | `fpo` | `sde_sigma=0`、`cfm_loss_average_group_size=1`、`cfm_loss_use_huber=True`、启用 clamp |
| Vanilla FPO | `fpo` | `sde_sigma=0`、`cfm_loss_average_group_size=-1`、`cfm_loss_use_huber=False`、不启用 clamp |
| DPPO Learned Noise | `dppo` | `sde_sigma=0.18`、`learn_sde_sigma=True`、启用噪声注入 |
| DPPO Fixed Noise | `dppo` | 每个任务使用固定 `sde_sigma`，`learn_sde_sigma=False` |

## 环境准备

首次安装环境：

```bash
bash setup_env.sh
```

该脚本会安装 Python 3.10、conda 环境 `fpo_manipulation`、`thirdparty/` 中的 robosuite、DexMimicGen、LeRobot、ffmpeg 7.1.1，以及 `thirdparty/lerobot_requirements.txt` 中的依赖。

每次运行实验前激活环境：

```bash
source source_env.sh
```

如需从 W&B 下载 checkpoint 或记录结果，请先登录：

```bash
wandb login
```

## Base Policy

所有微调实验都从预训练 BC checkpoint 开始。推荐从 Google Drive 下载：

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t -O downloaded_checkpoints
```

也可以通过 W&B artifact 下载：

```bash
python eval_checkpoint.py \
  --wandb_run_id 95j3noe4 \
  --wandb_project flow-bc \
  --checkpoint_step step_1000 \
  --eval_env Can \
  --eval_num_episodes 0
```

可用 checkpoint：

| 任务 | 环境名 | Base Policy Run ID | 主实验 Step | Ablation Step | Google Drive 目录 |
|---|---|---|---|---|---|
| Can | `Can` | `95j3noe4` | `step_1000` | `step_6000` | `95j3noe4_step_1000`、`95j3noe4_step_6000` |
| Square | `Square` | `trc7rbt0` | `step_110000` | -- | `trc7rbt0_step_110000` |
| Box Clearance | `TwoArmBoxCleanup` | `lainyisy` | `step_10000` | -- | `lainyisy_step_10000` |
| Tray Lifting | `TwoArmLiftTray` | `ri0w9j39` | `step_20000` | -- | `ri0w9j39_step_20000` |
| Threading | `TwoArmThreading` | `6vqrn614` | `step_10000` | -- | `6vqrn614_step_10000` |

从本地 checkpoint 微调时，使用 `--base_policy_local_path` 替代 W&B run ID。checkpoint 目录结构如下：

```text
<run_id>_<step>/
├── optimizer.pt
└── policy/
    ├── config.json
    └── model.safetensors
```

从零预训练 base policy 时使用 `pretrain_flow_bc.py`。所有 base policy 共享 `--policy flowmatching`、`--network_architecture mlp`、`--vision_backbone vit`、`--horizon 16`、`--n_action_steps 8`、`--sampling_steps 10` 和 `--ema_power 0.995` 等配置。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `dataset` | `ankile/robomimic-mh-can-image` | `ankile/robomimic-mh-square-image` | `ankile/dexmg-two-arm-box-cleanup` | `ankile/dexmg-two-arm-lift-tray` | `ankile/dexmg-two-arm-threading` |
| `eval_env` | `Can` | `Square` | `TwoArmBoxCleanup` | `TwoArmLiftTray` | `TwoArmThreading` |
| `steps` | 500000 | 1000000 | 1000000 | 1000000 | 1000000 |

启动 5 个预训练任务：

```bash
DRY_RUN=1 bash scripts/run_pretrain_base_policies.sh
bash scripts/run_pretrain_base_policies.sh
```

## 实验 1：主 Benchmark

主实验包含 5 个任务、4 个模型、3 个 seed（0、1、2），共 60 个 run。每个任务和模型组内的所有 seed 使用相同超参数。

所有 run 共享：

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
--data_collection_steps 1600
--do_chunk_level_ppo True
--eval_ema False
--exploration_noise_std None
--freeze_vision_encoder True
--gae_lambda 0.99
--n_action_samples 8
--n_action_steps 16
--num_envs 30
--sampling_steps 10
--spo_clip_coef 0.01
--zero_sampling True
```

任务级共享参数：

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `task` / `eval_env` | `Can` | `Square` | `TwoArmBoxCleanup` | `TwoArmLiftTray` | `TwoArmThreading` |
| `base_policy_wandb_run_id` | `95j3noe4` | `trc7rbt0` | `lainyisy` | `ri0w9j39` | `6vqrn614` |
| `checkpoint_step` | `step_1000` | `step_110000` | `step_10000` | `step_20000` | `step_10000` |
| `total_timesteps` | 5000000 | 8000000 | 5000000 | 8000000 | 8000000 |
| `discount` | 0.99 | 0.995 | 0.995 | 0.999 | 0.999 |

### FPO++

模型参数：`loss_mode=fpo`、`sde_sigma=0`、`cfm_loss_average_group_size=1`、`cfm_loss_use_huber=True`。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.02 | 0.01 | 0.03 | 0.03 | 0.01 |
| `max_grad_norm` | 5 | 25 | 5 | 1 | 1 |
| `cfm_loss_huber_delta` | 0.5 | 1 | 0.1 | 1 | 0.1 |
| `clamp_logratio` | 5 | None | 5 | 5 | 5 |
| `clamp_old_cfm_loss` | 4 | None | 4 | 4 | 4 |

Can 任务示例：

```bash
torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_1000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-can \
  --total_timesteps 5000000 \
  --task Can \
  --eval_env Can \
  --eval_num_episodes 200 \
  --wandb_enable True \
  --discount 0.99 \
  --sde_sigma 0 \
  --cfm_loss_average_group_size 1 \
  --cfm_loss_use_huber True \
  --cfm_loss_huber_delta 0.5 \
  --clip_coef 0.02 \
  --max_grad_norm 5 \
  --clamp_logratio 5 \
  --clamp_old_cfm_loss 4 \
  --trust_region_mode ppo \
  --seed 0
```

### DPPO Learned Noise

模型参数：`loss_mode=dppo`、`sde_sigma=0.18`、`cfm_loss_use_huber=True`、`cfm_loss_average_group_size=1`、`clamp_logratio=5`、`clamp_old_cfm_loss=4`、`trust_region_mode=ppo`。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| `max_grad_norm` | 25 | 5 | 1 | 25 | 1 |
| `cfm_loss_huber_delta` | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| `noise_injection_min` | 0.3 | 0.3 | 0.2 | 0.3 | 0.3 |
| `noise_injection_max` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `learn_sde_sigma` | True | True | True | True | True |

### Vanilla FPO

模型参数：`loss_mode=fpo`、`sde_sigma=0`、`cfm_loss_average_group_size=-1`、`cfm_loss_use_huber=False`、`clamp_logratio=None`、`clamp_old_cfm_loss=None`、`trust_region_mode=ppo`。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| `max_grad_norm` | 25 | 25 | 5 | 25 | 25 |
| `cfm_loss_huber_delta` | 0.5 | 1 | 0.5 | 0.5 | 0.5 |

### DPPO Fixed Noise

模型参数：`loss_mode=dppo`、`cfm_loss_average_group_size=1`、`cfm_loss_use_huber=True`、`clamp_logratio=5`、`clamp_old_cfm_loss=4`、`trust_region_mode=ppo`。

| 参数 | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 |
| `max_grad_norm` | 25 | 25 | 25 | 5 | 5 |
| `cfm_loss_huber_delta` | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| `sde_sigma` | 0.3 | 0.3 | 0.24 | 0.24 | 0.24 |

启动全部 60 个 run：

```bash
DRY_RUN=1 bash scripts/run_main_benchmark.sh
bash scripts/run_main_benchmark.sh
```

## 实验 2：Checkpoint Ablation

该实验只使用 Can 任务，将 base policy checkpoint 从 `step_1000` 替换为 `step_6000`，共 4 个模型 x 3 个 seed = 12 个 run。

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
| `learn_sde_sigma` | -- | -- | True | False |
| `noise_injection_min` / `max` | -- | -- | 0.2 / 0.5 | -- |

启动命令：

```bash
DRY_RUN=1 bash scripts/run_checkpoint_ablation.sh
bash scripts/run_checkpoint_ablation.sh
```

## 实验 3：FPO Ablation

该实验在 Square 和 Threading 两个任务上比较 FPO++、ASPO 和 per-action ratio 变体，共 18 个 run。

| 参数 | FPO++ | ASPO | Per-action ratio |
|---|---|---|---|
| `trust_region_mode` | ppo | aspo | ppo |
| `cfm_loss_average_group_size` | 1 | 1 | -1 |
| `cfm_loss_use_huber` | True | True | True |
| `clamp_logratio` | None (Square) / 5 (Threading) | 5 | 5 |
| `clamp_old_cfm_loss` | None (Square) / 4 (Threading) | 4 | 4 |

其他任务参数与主 Benchmark 中对应任务的 FPO++ 设置一致。

```bash
DRY_RUN=1 bash scripts/run_fpo_ablation.sh
bash scripts/run_fpo_ablation.sh
```

## 评估 Base Policy

评估全部 5 个预训练 BC base policy：

```bash
bash scripts/eval_base_policies.sh
```

脚本会同时评估 zero sampling（确定性）和 random sampling（`--zero-sampling False`，随机推理）。每次评估使用 30 个并行环境运行 200 个 episode。

示例：

```bash
python eval_checkpoint.py \
  --wandb_run_id 95j3noe4 \
  --wandb_project flow-bc \
  --checkpoint_step step_1000 \
  --eval_env Can \
  --eval_num_episodes 200 \
  --eval-num-envs 30 \
  --load-ema True
```

## 绘制结果

使用 `plot_results.py` 从 W&B run 数据生成训练曲线。脚本从 `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning` 获取指标，并输出 PDF。

| Mode | 说明 | 默认输出 |
|---|---|---|
| `main_benchmark` | 5 个任务、4 种方法，zero/random sampling 并排展示 | `main_benchmark_plot.pdf` |
| `fpoplusplus_ablation` | Square 与 Threading 上的 FPO++ ablation | `fpoplusplus_ablation_plot.pdf` |
| `base_policy_ablation` | Can 任务的多视角 checkpoint ablation | `base_policy_ablation_plot.pdf` |

```bash
python plot_results.py --mode main_benchmark
python plot_results.py --mode fpoplusplus_ablation
python plot_results.py --mode base_policy_ablation
python plot_results.py --mode main_benchmark --output my_custom_plot.pdf
```

## W&B Run ID 对照

### 实验 1：主 Benchmark

#### FPO++

| 任务 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Can | `wbxzw7z3` | `e0h4wy1r` | `i6rmkgrh` |
| Square | `rsmunbo4` | `wzyv707a` | `z2u9ryms` |
| Box Clearance | `ujjdjtov` | `qlux3x9d` | `o6kv0feo` |
| Tray Lifting | `dcwh6cja` | `rdigmvk4` | `oi516dvb` |
| Threading | `bt3sl4ex` | `g2ldgoss` | `fu8wpmbd` |

DPPO Learned Noise、Vanilla FPO 和 DPPO Fixed Noise 的 run ID 可在 `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning` 中查看。

### 实验 2：Checkpoint Ablation（Can，`step_6000`）

| 模型 | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| FPO++ | `txf5crib` | `ylxhw9uu` | `6zp46k32` |
| Vanilla FPO | `gzm22fqh` | `opygcs30` | `89jg66ll` |
| DPPO Learned | `7lz0maky` | `ss13djab` | `nqkx8w0s` |
| DPPO Fixed | `4iz08o2h` | `0wo72noe` | `0yibse8t` |

### 实验 3：FPO Ablation

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

## 硬件要求

- GPU：脚本通过 `torchrun` 支持 DDP，默认 `NUM_GPUS=1`，可按需调整。
- 内存：每个 run 默认启动 30 个并行仿真环境，并运行 200 个评估 episode，会占用较多 CPU 和内存。
- 存储：每 2 次迭代保存 checkpoint，并记录到 W&B；90 个 run 会产生大量 checkpoint。
- 时间：每个 run 的 `total_timesteps` 为 5M 到 8M，单 GPU 上通常需要数小时到数天。

## 辅助脚本

| 脚本 | 说明 |
|---|---|
| `scripts/run_pretrain_base_policies.sh` | 预训练 5 个 base policy |
| `scripts/run_main_benchmark.sh` | 启动 60 个主 Benchmark run |
| `scripts/run_checkpoint_ablation.sh` | 启动 12 个 checkpoint ablation run |
| `scripts/run_fpo_ablation.sh` | 启动 18 个 FPO ablation run |
| `scripts/eval_base_policies.sh` | 评估 5 个 base policy |

所有脚本支持：

- `DRY_RUN=1`：只打印命令，不实际执行。
- `NUM_GPUS=N`：设置 `torchrun` 使用的 GPU 数，默认 1。
