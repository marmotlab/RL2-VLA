#!/bin/bash
# NOTE: Activate the environment first (from repo root):
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# ==========================================================================
# Eval config (TODO: Change here)
# ==========================================================================
GPU=0
SEEDS=(42 0 7)
NUM_TRIALS_PER_TASK=100

# Action sampling for all states
LANG_REPHRASE_NUM_PREFAIL=8
ACTION_SAMPLES_PREFAIL=5
COMPOSED_SAMPLES_PREFAIL=0

# Log Directory
LOCAL_LOG_DIR="./safe_rollouts"
# LOCAL_LOG_DIR="/mnt/hdd/SAFE_ds/training_latents/rollouts"

# ==========================================================================
# Other config
# ==========================================================================

# Set the base directory to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
# Add repo root so bridge_verifier can be imported (go up 3 levels to cover-vla root)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Add CoVer_VLA root so robot_utils can be imported
INFERENCE_ROOT="$REPO_ROOT/CoVer_VLA"
export PYTHONPATH="$REPO_ROOT:$INFERENCE_ROOT:$PYTHONPATH"
export PRISMATIC_DATA_ROOT=.

# HF pretrained checkpoint for INTACT Pi0 finetuned on Bridge-V2
PRETRAINED_CHECKPOINT="juexzz/INTACT-pi0-finetune-bridge"

# Only collect rollouts for IID tasks
TASK_SUITES=(
    simpler_put_eggplant_in_basket
    simpler_spoon_on_towel
    simpler_stack_cube
    simpler_carrot_on_plate
)

# ==========================================================================
# Collect Rollouts for Safe Training
# ==========================================================================
for seed in "${SEEDS[@]}"; do
    for task_suite in "${TASK_SUITES[@]}"; do
        CUDA_VISIBLE_DEVICES=$GPU python ../run_simpler_eval_with_openpi.py \
            --task_suite_name "$task_suite" \
            --lang_transform_type rephrase \
            --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
            --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
            --use_failure_prediction False \
            --lang_rephrase_num_prefail "$LANG_REPHRASE_NUM_PREFAIL" \
            --action_samples_prefail "$ACTION_SAMPLES_PREFAIL" \
            --composed_samples_prefail "$COMPOSED_SAMPLES_PREFAIL" \
            --use_verifier True \
            --critic cover \
            --seed "$seed" \
            --local_log_dir "$LOCAL_LOG_DIR" \
            --wandb_project Rephrase-Safe-Collect   \
            --log_safe_training_data True
    done
done
