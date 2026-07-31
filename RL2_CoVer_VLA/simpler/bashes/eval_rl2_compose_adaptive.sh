#!/bin/bash
# NOTE: Activate the environment first (from repo root):
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# ==========================================================================
# Eval config (TODO: Change here)
# ==========================================================================
GPU=0
SEEDS=(42 0 7)
EVALUATE_TOP=3   # 1 = only evaluate TOP 1 alpha from huristics
NUM_TRIALS_PER_TASK=50

# Action sampling for non-failure states
LANG_REPHRASE_NUM_PREFAIL=8
ACTION_SAMPLES_PREFAIL=5
COMPOSED_SAMPLES_PREFAIL=0

# Action sampling for failure states
LANG_REPHRASE_NUM=8
ACTION_SAMPLES=1
COMPOSED_SAMPLES=5

# Log Directory
LOCAL_LOG_DIR="./experiments"
# LOCAL_LOG_DIR="/mnt/hdd/SAFE_ds/training_latents/rollouts"

# Set to "IID" or "OOD" to select which task-suite type to evaluate.
TASK_SUITE_TYPE="IID"

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

# IID ckpt has Taskwise CP bands
FAILURE_CHECKPOINT_DIR_IID="$REPO_ROOT/third_party/SAFE/scripts/batch_training/logs/SAVED/rl2_pi0_bridge_safe_ckpt_per_task_cp"
# OOD ckpt has Combined CP bands
FAILURE_CHECKPOINT_DIR_OOD="$REPO_ROOT/third_party/SAFE/scripts/batch_training/logs/SAVED/rl2_pi0_bridge_safe_ckpt_combined_cp"

# QAM checkpoint trained on Bridge-V2
QAM_CKPT="$REPO_ROOT/third_party/qam/exp/SAVED/rl2-vla-qam-bridge/rl2_vla_qam_bridge_500k.pkl"

# HF pretrained checkpoint for INTACT Pi0 finetuned on Bridge-V2
PRETRAINED_CHECKPOINT="juexzz/INTACT-pi0-finetune-bridge"

if [[ "$TASK_SUITE_TYPE" == "IID" ]]; then
    TASK_SUITES=(
        simpler_put_eggplant_in_basket
        simpler_spoon_on_towel
        simpler_stack_cube
        simpler_carrot_on_plate
    )
    USE_TASKWISE_CP_BAND=True
    FAILURE_CHECKPOINT_DIR="$FAILURE_CHECKPOINT_DIR_IID"
else
    TASK_SUITES=(
        simpler_orange_juice_on_plate
        simpler_spoon_on_towel_google
        simpler_tape_measure_in_basket
        simpler_toy_dinosaur_on_towel
    )
    USE_TASKWISE_CP_BAND=False
    FAILURE_CHECKPOINT_DIR="$FAILURE_CHECKPOINT_DIR_OOD"
fi

# Per-(task_suite, seed) tuned CP alpha sweep values (no shared pattern, so kept in JSON
# and looked up individually rather than encoded as a bash literal).
CP_ALPHAS_JSON_IID="$SCRIPT_DIR/rl2_cp_alphas_per_task.json"
CP_ALPHAS_JSON_OOD="$SCRIPT_DIR/rl2_cp_alphas_combined.json"

get_cp_alphas() {
    local task_suite=$1 seed=$2
    if [[ "$TASK_SUITE_TYPE" == "IID" ]]; then
        python3 -c "
import json
alphas = json.load(open('$CP_ALPHAS_JSON_IID'))['alpha']['$task_suite']['$seed']
print(' '.join(str(a) for a in alphas))
"
    else
        python3 -c "
import json
alphas = json.load(open('$CP_ALPHAS_JSON_OOD'))['alpha']['combined']['$seed']
print(' '.join(str(a) for a in alphas))
"
    fi
}

# ==========================================================================
# RL2 (Compose - Adaptive)
# ==========================================================================
for task_suite in "${TASK_SUITES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        read -ra cp_alphas <<< "$(get_cp_alphas "$task_suite" "$seed")"
        if [ "$EVALUATE_TOP" = "1" ]; then
            cp_alphas=("${cp_alphas[0]}")
        fi
        for cp_alpha in "${cp_alphas[@]}"; do
            CUDA_VISIBLE_DEVICES=$GPU python ../run_simpler_eval_with_openpi.py \
                --task_suite_name "$task_suite" \
                --lang_transform_type rephrase \
                --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
                --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
                --use_failure_prediction True \
                --use_taskwise_cp_band "$USE_TASKWISE_CP_BAND" \
                --failure_checkpoint_dir "$FAILURE_CHECKPOINT_DIR" \
                --lang_rephrase_num_prefail "$LANG_REPHRASE_NUM_PREFAIL" \
                --action_samples_prefail "$ACTION_SAMPLES_PREFAIL" \
                --composed_samples_prefail "$COMPOSED_SAMPLES_PREFAIL" \
                --lang_rephrase_num "$LANG_REPHRASE_NUM" \
                --action_samples "$ACTION_SAMPLES" \
                --composed_samples "$COMPOSED_SAMPLES" \
                --use_verifier True \
                --qam_ckpt "$QAM_CKPT" \
                --critic cover \
                --failure_cp_alpha "$cp_alpha" \
                --seed "$seed" \
                --local_log_dir "$LOCAL_LOG_DIR" \
                --wandb_project RL2
        done
    done
done