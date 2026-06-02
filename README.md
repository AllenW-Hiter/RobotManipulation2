# 用于机械臂操作的 FPO++

本仓库包含用于机械臂操作场景的 FPO++ 代码。主要目标是复现 [FPO++ paper](https://arxiv.org/pdf/2602.02481) 的结果，并为后续研究提供稳健 baseline。

本仓库实现：

- 预训练：在五个操作任务上通过行为克隆训练任务专用 base policy。
- 微调：将四类微调算法（FPO++、Vanilla FPO 和 DPPO 变体）适配到 flow matching，使它们可以共享同一个 base policy。

本 README 概述实验和实现。完整复现实验请参考 `docs/reproduce.md`。

## 目录结构

```text
.
├── docs/
│   └── reproduce.md                # 复现指南和绘图说明
├── downloaded_checkpoints/         # 预训练 base policy checkpoint（通过 gdown 下载）
├── scripts/
│   ├── eval_base_policies.sh       # 评估预训练 base policy
│   ├── run_checkpoint_ablation.sh  # checkpoint ablation 实验
│   ├── run_fpo_ablation.sh         # FPO ablation 实验
│   ├── run_main_benchmark.sh       # 主 benchmark 实验
│   ├── run_pretrain_base_policies.sh  # 预训练 base policy
│   ├── skypilot/                   # SkyPilot 云端启动脚本
│   └── sweeps/                     # W&B sweep 配置
├── src/
│   ├── dexmg_env.py                # DexMimicGen 环境封装
│   ├── flow_model.py               # Flow matching policy
│   ├── flow_model_config.py        # Flow model 配置
│   ├── flow_net_mlp.py             # MLP flow 网络
│   ├── flow_net_residual_mlp.py    # Residual MLP flow 网络
│   ├── flow_net_unet.py            # UNet flow 网络
│   ├── noise_injection_network.py  # DPPO 使用的噪声注入网络
│   ├── utils.py                    # 工具函数
│   └── vit.py                      # Vision Transformer backbone
├── thirdparty/                     # robosuite、DexMimicGen、LeRobot 等依赖
├── pretrain_flow_bc.py             # 预训练入口
├── finetune_online_rl.py           # 在线 RL 微调入口
├── eval_checkpoint.py              # checkpoint 评估脚本
├── plot_results.py                 # 结果绘图脚本
├── setup_env.sh                    # 首次环境安装
├── source_env.sh                   # 每次会话的环境激活脚本
└── pyproject.toml                  # 项目配置
```

## 环境配置

运行任何命令前，先配置并激活 conda 环境：

```bash
bash setup_env.sh          # 首次安装
source source_env.sh       # 每次会话激活环境
```

## 预训练 Checkpoint

所有 5 个任务的预训练 base policy checkpoint 可从 [Google Drive](https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t?usp=sharing) 下载：

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t -O downloaded_checkpoints
```

从本地 checkpoint 微调时使用 `--base_policy_local_path`：

```bash
torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base_policy_local_path downloaded_checkpoints/95j3noe4_step_1000 \
  --load-ema True \
  --task Can --eval_env Can \
  ...
```

各 checkpoint 的完整说明和微调命令见 `docs/reproduce.md`。

## 训练

### 通过行为克隆预训练 Flow Matching Base Policy

可用的 `(dataset, task)` 组合：

- `ankile/robomimic-ph-can-image` / `PickPlaceCan`
- `ankile/robomimic-ph-square-image` / `NutAssemblySquare`
- `ankile/dexmg-two-arm-box-cleanup` / `TwoArmBoxCleanup`
- `ankile/dexmg-two-arm-lift-tray` / `TwoArmLiftTray`
- `ankile/dexmg-two-arm-threading` / `TwoArmThreading`

示例：

```bash
python pretrain_flow_bc.py --dataset ankile/dexmg-two-arm-threading --policy flowmatching --network_architecture mlp --horizon 8 --n_action_steps 8 --sampling_steps 10 --image_observation_keys "agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image" --eval_env TwoArmThreading --eval_num_envs 1 --eval_num_episodes 5 --log_freq 10 --save_freq 200 --rollout_freq 200 --steps 6000 --wandb_enable True --wandb_project flow-bc
```

常用可调参数包括 `network_architecture`、`image_observation_keys`、`horizon`、`n_action_steps`、`sampling_steps`、`ema_power`、`grad_clip_norm`、`batch_size`、`num_workers`、`learning_rate`、`lr_backbone`、`weight_decay`、`flow_network_output_param`、`cfm_loss_mode` 和 `enable_geometric_augmentations`。

### 通过 Online RL 微调 Base Policy

可用微调算法：

- FPO++
- Vanilla FPO
- DPPO with learned noise injection
- DPPO with fixed noise injection

FPO++ 与 Vanilla FPO 的主要差异：

- Ratio 计算：FPO++ 使用 per-sample PPO ratio；Vanilla FPO 使用 per-action ratio。
- Trust region：两者都使用 PPO trust region；在微调设置中使用 ASPO 会降低性能。

示例：

```bash
python finetune_online_rl.py --base-policy-wandb-project flow-bc --base_policy_wandb_run_id wd9xdji9 --wandb_enable True --wandb_project flow-bc-fpo-finetuning --experiment finetune-fpo-can-image-v1 --log_freq 10 --save_freq 10 --rollout_freq 10 --eval_env Can --eval_num_envs 5 --eval_num_episodes 10 --num_envs 4 --n_action_steps 4 --task Can --load-ema True --data-collection-steps 300
```

如需调整特定任务的 horizon，可修改 `src/dexmg_env.py` 中的 `self.horizon` 映射。

## 评估 Checkpoint 与绘图

使用 `eval_checkpoint.py` 评估任意预训练或微调 checkpoint：

```bash
python eval_checkpoint.py \
  --wandb_run_id 95j3noe4 \
  --wandb_project flow-bc \
  --checkpoint_step step_1000 \
  --eval_env Can \
  --eval_num_episodes 200 \
  --eval-num-envs 10 \
  --load-ema True
```

关键参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--wandb_run_id` | 从 W&B 下载 checkpoint 的 run ID | -- |
| `--wandb_project` | W&B 项目名；预训练用 `flow-bc`，微调用 `flow-bc-fpo-finetuning` | -- |
| `--checkpoint_step` | 要评估的 checkpoint，可为 `latest`、`best` 或具体 step | `latest` |
| `--local_checkpoint_path` | 本地 checkpoint 目录，可替代 W&B | -- |
| `--load-ema` | 是否加载 EMA 权重；预训练 base policy 通常设为 True | `False` |
| `--eval_env` | 评估环境名 | `Lift` |
| `--eval_num_episodes` | 评估 episode 数 | 50 |
| `--eval-num-envs` | 并行环境数 | 2 |
| `--zero-sampling` | 使用确定性 zero sampling；设为 `False` 启用随机采样 | `True` |

视频渲染选项：

| 参数 | 用途 | 默认值 |
|---|---|---|
| `eval_camera_size` | policy 输入图像尺寸 | 84 |
| `render_size` | 保存 rollout 视频帧的分辨率 | (240, 320) |

```bash
# 以 480x640 分辨率保存视频
python eval_checkpoint.py --wandb_run_id wbxzw7z3 --wandb_project flow-bc-fpo-finetuning --eval_env Can --render_size 480 640

# 保存不带文字标注的视频
python eval_checkpoint.py --wandb_run_id wbxzw7z3 --wandb_project flow-bc-fpo-finetuning --eval_env Can --annotate_video False
```

绘图说明见 `docs/reproduce.md`，使用 `plot_results.py` 生成结果图。
