#!/usr/bin/env bash
#
# Run three long MeanFlow online RL tests sequentially:
#   1. Continue from the previous 5M staged-BC run for another 5M env steps.
#   2. Train from the S2 BC baseline with anchor loss only for 10M env steps.
#   3. Train from the S2 BC baseline with shared-rollout joint anchor + BC loss for 10M env steps.
#
# Usage:
#   bash scripts/run_meanflow_long_three_tests.sh
#   DRY_RUN=1 bash scripts/run_meanflow_long_three_tests.sh
#   RUN_TESTS="anchoronly10m jointbc10m" bash scripts/run_meanflow_long_three_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DRY_RUN="${DRY_RUN:-0}"
RUN_TESTS="${RUN_TESTS:-continue5m anchoronly10m jointbc10m}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/root/autodl-tmp/RobotManipulation/thirdparty/miniconda3/envs/fpo_manipulation/bin/torchrun}"
if [[ ! -x "$TORCHRUN_BIN" && "$DRY_RUN" != "1" ]]; then
    echo "TORCHRUN_BIN is not executable: $TORCHRUN_BIN" >&2
    echo "Set TORCHRUN_BIN=/path/to/torchrun or activate the correct environment." >&2
    exit 1
fi

WANDB_ENTITY="${WANDB_ENTITY:-wxyhitphd-hit}"
WANDB_PROJECT="${WANDB_PROJECT:-meanflow-rl-finetuning}"
WANDB_ENABLE="${WANDB_ENABLE:-True}"

RESUME_WANDB_RUN_ID="${RESUME_WANDB_RUN_ID:-oqajwx9d}"
RESUME_BASE_POLICY_LOCAL_PATH="${RESUME_BASE_POLICY_LOCAL_PATH:-}"
S2_BASE_POLICY_LOCAL_PATH="${S2_BASE_POLICY_LOCAL_PATH:-runs/meanflow_sampling_steps_ablation_can_seed3_step1000/train_s2/checkpoints/step_999}"

NUM_ENVS="${NUM_ENVS:-16}"
DATA_COLLECTION_STEPS="${DATA_COLLECTION_STEPS:-3008}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
ROLLOUT_FREQ="${ROLLOUT_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
SEED="${SEED:-0}"

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@"
        echo
    else
        echo "====================================================================="
        echo "Running: $*"
        echo "====================================================================="
        "$@"
    fi
}

contains_test() {
    local target="$1"
    local item
    for item in $RUN_TESTS; do
        [[ "$item" == "$target" ]] && return 0
    done
    return 1
}

common_args=(
    --distributed True
    --env-vectorization async
    --async-env-context forkserver
    --async-env-shared-memory True
    --policy meanflow
    --load-ema True
    --task Can
    --eval-env Can
    --eval-num-episodes "$EVAL_NUM_EPISODES"
    --rollout-freq "$ROLLOUT_FREQ"
    --eval-save-video False
    --wandb-upload-eval-video False
    --wandb-enable "$WANDB_ENABLE"
    --wandb-project "$WANDB_PROJECT"
    --wandb-entity "$WANDB_ENTITY"
    --rollout-granularity chunk
    --chunk-reward-mode discounted_sum
    --data-collection-steps "$DATA_COLLECTION_STEPS"
    --num-envs "$NUM_ENVS"
    --n-action-steps 16
    --sampling-steps 2
    --discount 0.99
    --gae-lambda 0.99
    --learning-rate-actor 1e-5
    --learning-rate-critic 1e-4
    --gradient-accumulation-steps 1
    --num-minibatches 8
    --update-epochs 10
    --n-iterations-train-only-value 0
    --log-freq 1
    --save-freq "$SAVE_FREQ"
    --loss-mode fpo
    --meanflow-fpo-loss-source anchor
    --meanflow-anchor-sampling-mode schedule
    --meanflow-anchor-sampling-steps 2
    --meanflow-anchor-logratio-coef 1.0
    --meanflow-value-update-mode joint
    --cfm-loss-average-group-size 1
    --cfm-loss-use-huber True
    --cfm-loss-huber-delta 0.5
    --clip-coef 0.1
    --max-grad-norm 5
    --clamp-logratio 5
    --clamp-old-cfm-loss 4
    --clamp-old-anchor-loss 4
    --trust-region-mode ppo
    --n-action-samples 16
    --do-chunk-level-ppo True
    --do-average-cfm-loss-in-chunk False
    --freeze-vision-encoder True
    --eval-ema False
    --seed "$SEED"
)

run_finetune() {
    run_cmd "$TORCHRUN_BIN" --nproc_per_node=1 finetune_online_rl.py "$@"
}

echo "Repo: $REPO_DIR"
echo "Run tests: $RUN_TESTS"
echo "W&B: entity=$WANDB_ENTITY project=$WANDB_PROJECT enable=$WANDB_ENABLE"
echo "Torchrun: $TORCHRUN_BIN"
echo "S2 baseline: $S2_BASE_POLICY_LOCAL_PATH"
echo "Resume W&B run id: $RESUME_WANDB_RUN_ID"

if contains_test "continue5m"; then
    resume_base_args=()
    if [[ -n "$RESUME_BASE_POLICY_LOCAL_PATH" ]]; then
        resume_base_args=(--base-policy-local-path "$RESUME_BASE_POLICY_LOCAL_PATH")
    else
        resume_base_args=(
            --base-policy-wandb-run-id "$RESUME_WANDB_RUN_ID"
            --base-policy-wandb-project "$WANDB_PROJECT"
            --checkpoint-step latest
        )
    fi

    run_finetune \
        "${common_args[@]}" \
        "${resume_base_args[@]}" \
        --experiment finetune-meanflow-can-s2-anchor2-clip01-successbc-resume5m-from-oqajwx9d \
        --total-timesteps 5000000 \
        --meanflow-bc-update-mode staged \
        --meanflow-bc-stage-epochs 5 \
        --meanflow-bc-stage-selection success_or_top \
        --meanflow-bc-stage-top-fraction 0.2 \
        --meanflow-bc-stage-adv-weight binary \
        --meanflow-bc-stage-loss-coef 0.5 \
        --meanflow-bc-stage-interval 10 \
        --meanflow-bc-stage-rollout-repeats 8 \
        --meanflow-bc-stage-cache-mode memory \
        --meanflow-stage-sampling-mode separate \
        --meanflow-stage-update-order anchor_first
fi

if contains_test "anchoronly10m"; then
    run_finetune \
        "${common_args[@]}" \
        --base-policy-local-path "$S2_BASE_POLICY_LOCAL_PATH" \
        --experiment finetune-meanflow-can-s2-anchor2-clip01-anchoronly-10m \
        --total-timesteps 10000000 \
        --meanflow-bc-update-mode staged \
        --meanflow-bc-stage-epochs 0 \
        --meanflow-bc-stage-loss-coef 0.0 \
        --meanflow-stage-sampling-mode shared
fi

if contains_test "jointbc10m"; then
    run_finetune \
        "${common_args[@]}" \
        --base-policy-local-path "$S2_BASE_POLICY_LOCAL_PATH" \
        --experiment finetune-meanflow-can-s2-anchor2-clip01-jointbc-c02-10m \
        --total-timesteps 10000000 \
        --meanflow-bc-update-mode joint \
        --meanflow-bc-stage-epochs 0 \
        --meanflow-bc-stage-selection success_or_top \
        --meanflow-bc-stage-top-fraction 0.2 \
        --meanflow-bc-stage-adv-weight binary \
        --meanflow-bc-stage-loss-coef 0.2 \
        --meanflow-bc-stage-interval 1 \
        --meanflow-stage-sampling-mode shared
fi

