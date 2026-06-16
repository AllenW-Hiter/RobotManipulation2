#!/usr/bin/env bash
#
# Evaluate the Threading MeanFlow BC checkpoint with 10/5/2/1 inference steps
# and compare it against the 6vqrn614_step_10000 FlowMatching BC checkpoint.
#
# Usage:
#   bash scripts/eval_threading_meanflow_vs_flow_bc.sh
#   DRY_RUN=1 bash scripts/eval_threading_meanflow_vs_flow_bc.sh
#   RUN_TARGETS=meanflow_s10 bash scripts/eval_threading_meanflow_vs_flow_bc.sh
#
# Useful overrides:
#   EVAL_NUM_EPISODES=200 EVAL_NUM_ENVS=30 bash scripts/eval_threading_meanflow_vs_flow_bc.sh
#   WANDB_ENABLE=True SAVE_VIDEO=True bash scripts/eval_threading_meanflow_vs_flow_bc.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MEANFLOW_CKPT="${MEANFLOW_CKPT:-/home/allen/03_Data/03_Project_Muti_System/RL03_FPOPP/fpo-control/manipulation_experiments/runs/meanflow_bc_threading_2026-05-26_21-21-51/checkpoints/step_10000}"
FLOW_BC_CKPT="${FLOW_BC_CKPT:-downloaded_checkpoints/6vqrn614_step_10000}"

EVAL_ENV="${EVAL_ENV:-TwoArmThreading}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-16}"
SEED="${SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
WANDB_ENABLE="${WANDB_ENABLE:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-threading-checkpoint-evaluation}"
WANDB_ENTITY="${WANDB_ENTITY:-wxyhitphd-hit}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/eval_threading_meanflow_vs_flow_bc}"
RUN_TARGETS="${RUN_TARGETS:-meanflow_s10 meanflow_s5 meanflow_s2 meanflow_s1 flowmatching_s10}"

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

common_args=(
  --eval-env "$EVAL_ENV"
  --eval-num-episodes "$EVAL_NUM_EPISODES"
  --eval-num-envs "$EVAL_NUM_ENVS"
  --env-vectorization async
  --async-env-context forkserver
  --async-env-shared-memory True
  --load-ema True
  --zero-sampling True
  --save-video "$SAVE_VIDEO"
  --wandb-enable "$WANDB_ENABLE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
  --seed "$SEED"
)

contains_target() {
  local target="$1"
  local item
  for item in $RUN_TARGETS; do
    [[ "$item" == "$target" ]] && return 0
  done
  return 1
}

for sampling_steps in 10 5 2 1; do
  target="meanflow_s${sampling_steps}"
  contains_target "$target" || continue

  run_cmd "$PYTHON_BIN" eval_checkpoint.py \
    --local-checkpoint-path "$MEANFLOW_CKPT" \
    --checkpoint-step "meanflow_step_10000_s${sampling_steps}" \
    --override-sampling-steps "$sampling_steps" \
    --experiment "eval-meanflow-threading-bc-s${sampling_steps}" \
    --output-dir "$OUTPUT_ROOT/meanflow_s${sampling_steps}" \
    "${common_args[@]}"
done

if contains_target "flowmatching_s10"; then
  run_cmd "$PYTHON_BIN" eval_checkpoint.py \
    --local-checkpoint-path "$FLOW_BC_CKPT" \
    --checkpoint-step flowmatching_6vqrn614_step_10000_s10 \
    --override-sampling-steps 10 \
    --experiment eval-flowmatching-threading-bc-6vqrn614-s10 \
    --output-dir "$OUTPUT_ROOT/flowmatching_6vqrn614_s10" \
    "${common_args[@]}"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  echo "====================================================================="
  echo "Summary files:"
  find "$OUTPUT_ROOT" -name 'eval_summary_*.txt' -print | sort
  echo "====================================================================="
  echo "Success rates:"
  find "$OUTPUT_ROOT" -name 'eval_summary_*.txt' -print0 \
    | sort -z \
    | xargs -0 grep -H 'Success Rate'
fi
