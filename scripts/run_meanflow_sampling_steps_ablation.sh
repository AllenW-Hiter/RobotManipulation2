#!/usr/bin/env bash
#
# Train MeanFlow Can BC policies with different sampling_steps and evaluate each
# checkpoint under its native sampling_steps plus low-step inference settings.
#
# Usage:
#   bash scripts/run_meanflow_sampling_steps_ablation.sh
#   DRY_RUN=1 bash scripts/run_meanflow_sampling_steps_ablation.sh
#   EVAL_EPISODES=200 EVAL_NUM_ENVS=10 bash scripts/run_meanflow_sampling_steps_ablation.sh
#   DEBUG=True EVAL_NUM_ENVS=1 bash scripts/run_meanflow_sampling_steps_ablation.sh
#
# The script assumes the fpo_manipulation environment is already active. To use a
# specific interpreter, set PYTHON_BIN=/path/to/python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

DATASET="${DATASET:-/home/allen/03_Data/03_Project_Muti_System/RL03_FPOPP/fpo-control/manipulation_experiments/BC_dataset/ankile_robomimic-ph-can-image}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
TRAIN_SAMPLING_STEPS="${TRAIN_SAMPLING_STEPS:-10 5 2 1}"
EXTRA_EVAL_SAMPLING_STEPS="${EXTRA_EVAL_SAMPLING_STEPS:-2 1}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SEED="${SEED:-3}"

EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-10}"
DEBUG="${DEBUG:-False}"
ENV_VECTORIZATION="${ENV_VECTORIZATION:-async}"
ASYNC_ENV_CONTEXT="${ASYNC_ENV_CONTEXT:-spawn}"
ASYNC_ENV_SHARED_MEMORY="${ASYNC_ENV_SHARED_MEMORY:-True}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
WANDB_ENABLE="${WANDB_ENABLE:-False}"
SKIP_EXISTING="${SKIP_EXISTING:-True}"

RUN_ROOT="${RUN_ROOT:-runs/meanflow_sampling_steps_ablation_can_seed${SEED}_step${TRAIN_STEPS}}"

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@"
        echo
    else
        echo "=== Running: $* ==="
        "$@"
    fi
}

contains_step() {
    local target="$1"
    shift
    local step
    for step in "$@"; do
        [[ "$step" == "$target" ]] && return 0
    done
    return 1
}

echo "Repo: $REPO_DIR"
echo "Dataset: $DATASET"
echo "Train sampling steps: $TRAIN_SAMPLING_STEPS"
echo "Extra eval sampling steps: $EXTRA_EVAL_SAMPLING_STEPS"
echo "Eval episodes/envs: $EVAL_EPISODES / $EVAL_NUM_ENVS"
echo "Debug sync env: $DEBUG"
echo "Eval vectorization: $ENV_VECTORIZATION (context=$ASYNC_ENV_CONTEXT, shared_memory=$ASYNC_ENV_SHARED_MEMORY)"
echo "Skip existing checkpoints: $SKIP_EXISTING"
echo "Run root: $RUN_ROOT"

for train_sampling_steps in $TRAIN_SAMPLING_STEPS; do
    train_run_dir="${RUN_ROOT}/train_s${train_sampling_steps}"
    checkpoint_dir="${train_run_dir}/checkpoints/step_$((TRAIN_STEPS - 1))"

    echo "====================================================================="
    echo "Training MeanFlow Can BC | train sampling_steps=${train_sampling_steps}"
    echo "====================================================================="

    if [[ "$SKIP_EXISTING" == "True" && -d "$checkpoint_dir" && "$DRY_RUN" != "1" ]]; then
        echo "Checkpoint already exists, skipping training: $checkpoint_dir"
    else
        run_cmd "$PYTHON_BIN" pretrain_flow_bc.py \
            --policy meanflow \
            --dataset "$DATASET" \
            --image-observation-keys "robot0_eye_in_hand_image" \
            --horizon 16 \
            --n-action-steps 8 \
            --sampling-steps "$train_sampling_steps" \
            --network-architecture mlp \
            --mlp-dims "[1024, 1024, 1024]" \
            --vision-backbone vit \
            --flow-network-output-param u \
            --cfm-loss-mode u \
            --cfm-loss-use-huber False \
            --cfm-loss-huber-delta 0.5 \
            --batch-size "$BATCH_SIZE" \
            --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
            --learning-rate 1e-4 \
            --lr-backbone 1e-5 \
            --weight-decay 1e-6 \
            --grad-clip-norm 25 \
            --ema-power 0.995 \
            --enable-geometric-augmentations True \
            --meanflow-encode-t-minus-r True \
            --meanflow-flow-ratio 0.5 \
            --meanflow-time-sampling logit_normal_pair \
            --num-workers "$NUM_WORKERS" \
            --steps "$TRAIN_STEPS" \
            --save-freq "$TRAIN_STEPS" \
            --log-freq 5 \
            --wandb-enable "$WANDB_ENABLE" \
            --output-dir "$train_run_dir" \
            --experiment "meanflow_bc_can_s${train_sampling_steps}_seed${SEED}" \
            --seed "$SEED"
    fi

    eval_steps=("$train_sampling_steps")
    for extra_step in $EXTRA_EVAL_SAMPLING_STEPS; do
        if ! contains_step "$extra_step" "${eval_steps[@]}"; then
            eval_steps+=("$extra_step")
        fi
    done

    for eval_sampling_steps in "${eval_steps[@]}"; do
        eval_run_dir="${RUN_ROOT}/eval_train_s${train_sampling_steps}_eval_s${eval_sampling_steps}"

        echo "---------------------------------------------------------------------"
        echo "Evaluating train_s${train_sampling_steps} checkpoint with sampling_steps=${eval_sampling_steps}"
        echo "---------------------------------------------------------------------"

        run_cmd "$PYTHON_BIN" eval_checkpoint.py \
            --local-checkpoint-path "$checkpoint_dir" \
            --eval-env Can \
            --eval-num-episodes "$EVAL_EPISODES" \
            --eval-num-envs "$EVAL_NUM_ENVS" \
            --load-ema True \
            --zero-sampling True \
            --override-sampling-steps "$eval_sampling_steps" \
            --save-video "$SAVE_VIDEO" \
            --wandb-enable "$WANDB_ENABLE" \
            --debug "$DEBUG" \
            --env-vectorization "$ENV_VECTORIZATION" \
            --async-env-context "$ASYNC_ENV_CONTEXT" \
            --async-env-shared-memory "$ASYNC_ENV_SHARED_MEMORY" \
            --output-dir "$eval_run_dir" \
            --experiment "eval_meanflow_can_train_s${train_sampling_steps}_eval_s${eval_sampling_steps}" \
            --seed "$SEED"
    done
done

echo "====================================================================="
echo "MeanFlow sampling_steps ablation finished"
echo "Results are under: $RUN_ROOT"
echo "====================================================================="
