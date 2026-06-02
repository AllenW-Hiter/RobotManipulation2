# Pretrained Base Policy Checkpoints

These are pretrained Flow Matching behavior cloning (BC) checkpoints used as starting points for
finetuning in the FPO++ experiments. All checkpoints were trained in the `far-wandb/flow-bc`
W&B project.

## Checkpoints

| Checkpoint | Task | Env Name | W&B Run ID | Step | Size |
|------------|------|----------|------------|------|------|
| `95j3noe4_step_1000` | Can | `Can` | `95j3noe4` | 1,000 | 205M |
| `95j3noe4_step_6000` | Can (ablation) | `Can` | `95j3noe4` | 6,000 | 205M |
| `trc7rbt0_step_110000` | Square | `Square` | `trc7rbt0` | 110,000 | 205M |
| `lainyisy_step_10000` | Box Clearance | `TwoArmBoxCleanup` | `lainyisy` | 10,000 | 538M |
| `ri0w9j39_step_20000` | Tray Lifting | `TwoArmLiftTray` | `ri0w9j39` | 20,000 | 538M |
| `6vqrn614_step_10000` | Threading | `TwoArmThreading` | `6vqrn614` | 10,000 | 532M |

## Directory Structure

Each checkpoint directory contains:

```
<run_id>_<step>/
├── optimizer.pt          # Optimizer state (not needed for finetuning)
└── policy/
    ├── config.json       # Model architecture and training config
    └── model.safetensors # Model weights (includes EMA weights)
```

## Usage

### Finetuning from a local checkpoint

Use `--base_policy_local_path` instead of `--base_policy_wandb_run_id`:

```bash
torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base_policy_local_path downloaded_checkpoints/95j3noe4_step_1000 \
  --load-ema True \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo++-can \
  --total_timesteps 5000000 \
  --task Can --eval_env Can \
  --discount 0.99 \
  --cfm_loss_average_group_size 1 --cfm_loss_use_huber True \
  --cfm_loss_huber_delta 0.5 --clip_coef 0.02 \
  --max_grad_norm 5 --sde_sigma 0 \
  --clamp_logratio 5 --clamp_old_cfm_loss 4 \
  --trust_region_mode ppo \
  --seed 0
```

### Evaluating a checkpoint

```bash
python eval_checkpoint.py \
  --local_checkpoint_path downloaded_checkpoints/95j3noe4_step_1000 \
  --eval_env Can \
  --eval_num_episodes 200 \
  --eval-num-envs 10 \
  --load-ema True
```

## Training Details

All base policies were trained with identical hyperparameters:

- **Architecture:** MLP `[1024, 1024, 1024]` with ViT vision backbone
- **Flow matching:** velocity prediction (`flow_network_output_param=u`, `cfm_loss_mode=u`)
- **Loss:** L2 (no Huber)
- **Optimizer:** AdamW, lr=1e-4, backbone lr=1e-5, weight_decay=1e-6
- **Training:** batch_size=512, grad_clip_norm=25, EMA power=0.995
- **Policy:** horizon=16, n_action_steps=8, sampling_steps=10
- **Augmentation:** geometric augmentations enabled
- **Seed:** 3

### Task-specific differences

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|-----------|-----|--------|---------------|--------------|-----------|
| Dataset | ankile/robomimic-mh-can-image | ankile/robomimic-mh-square-image | ankile/dexmg-two-arm-box-cleanup | ankile/dexmg-two-arm-lift-tray | ankile/dexmg-two-arm-threading |
| Image keys | robot0_eye_in_hand | agentview | agentview, robot0_eye_in_hand, robot1_eye_in_hand | agentview, robot0_eye_in_hand, robot1_eye_in_hand | agentview, robot0_eye_in_hand, robot1_eye_in_hand |
| Training steps | 500K | 1M | 1M | 1M | 1M |

The single-arm tasks (Can, Square) use one camera and are ~205M, while the two-arm tasks
(Box Clearance, Tray Lifting, Threading) use three cameras and are ~530M.
