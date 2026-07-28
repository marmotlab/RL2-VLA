"""
SIMPLER Environment Evaluation with RL2 and CoVer Verifier

This script evaluates vision-language-action policies on SIMPLER benchmark tasks using:
- PI0 policy with batch inference
- Language instruction rephrasing
- Failure detection using SAFE model
- RL Latent (RL2) compositional steering when failing 
- Action verification and selection using CoVer verifier model

For paper: "RL2-VLA: Adaptive RL Latent Compositional Steering with Test-Time Scaling for Vision-Language-Action Models"
"""

import itertools
import os
import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
import wandb
import warnings
warnings.filterwarnings("ignore", message="os.fork().*JAX", category=RuntimeWarning)


# SIMPLER environment imports
from simpler_benchmark import get_benchmark
from eval_utils import (
    convert_maniskill_with_bridge_adapter,
    create_bridge_adapter_wrapper,
    get_simpler_dummy_action,
    get_simpler_env,
    load_rephrases,
    process_inputs,
    process_raw_image_to_jpg,
    save_episode_data_openpi,
    save_rollout_video_openpi,
    set_seed_everywhere,
)
from rl2_utils import (
    QAMInference,
    SAFE_TASK_MAP_DICT,
    EpisodeVizTracker,
    load_failure_detection_model,
    run_policy_warmup,
    compute_composed_actions,
    check_failure_prediction_lstm,
    save_logs_pkl,
    save_episode_meta_json,
    _write_cp_plots_mp4,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# PI0 policy imports (installed as module via env_simpler_pi.sh)
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

# Ensemble verifier imports (installed as module via env_simpler_pi.sh)
from bridge_verifier.ensemble_eval import EfficientEnsembleMerged

# Constants
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# =========================================================================================
# Configuration
# =========================================================================================

@dataclass
class GenerateConfig:
    """Configuration for SIMPLER evaluation with RL2 and CoVer verifier."""
    
    # Model parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "juexzz/INTACT-pi0-finetune-bridge"
    obs_history: int = 1

    # Environment parameters
    task_suite_name: str = "simpler_widowx"
    num_steps_wait: int = 0
    num_trials_per_task: int = 50

    # Logging parameters
    local_log_dir: str = "./experiments"
    use_wandb: bool = False
    wandb_project: str = "RL2"
    wandb_entity: Optional[str] = None
    seed: int = 42
    
    # CoVer verifier parameters
    use_verifier: bool = True
    use_verifier_always: bool = True
    critic: str = "cover"                           

    # Action chunking parameters
    n_action_steps: int = 4
    action_ensemble_temp: float = -0.8
    
    # Language transformation parameters
    lang_transform_type: str = "rephrase"       # "rephrase", "no_transform" (in-domain prompt)
    lang_rephrase_num_prefail: int = 1
    lang_rephrase_num: int = 1

    log_misc_episode_data_openpi: bool = False     # Set to False if collecting SAFE data
    log_safe_training_data: bool = False
    save_3d_html_viz: bool = False

    # Samples before/after failure detection
    action_samples_prefail: int = 1
    augmented_samples_prefail: int = 0
    action_samples: int = 1
    augmented_samples: int = 0
    warmup: bool = True                 # Pass in a MAX of pre/post-fail batch size at step 0 to avoid OOM Crash

    # QAM Params
    merge_rel_weight: float = -1        # -1 means gaussian sampling of weights
    qam_ckpt: str = "third_party/qam/exp/SAVED/rl2-vla-qam-bridge/rl2_vla_qam_bridge_500k.pkl"

    # SAFE Params
    use_failure_prediction: bool = False
    use_rephrased_latents_for_qam: bool = True
    failure_checkpoint_dir: str = "third_party/SAFE/scripts/batch_training/logs/SAVED/rl2_pi0_ckpt_per_task_cp"
    failure_cp_alpha: float = 0.20
    use_taskwise_cp_band: bool = True

@draccus.wrap()
def eval_simpler(cfg: GenerateConfig) -> None:
    """Main evaluation function for SIMPLER benchmark with RL2 and CoVer.
    
    Args:
        cfg: Configuration object containing all evaluation parameters
    """
    # =========================================================================================
    # Setup: Config Validation & Logging
    # =========================================================================================

    # Validate configuration
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Set action un-normalization key
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    else:
        cfg.unnorm_key = "bridge_orig"

    # Initialize logging
    if cfg.use_failure_prediction:
        run_id = f"[{cfg.wandb_project}]-SEED-{cfg.seed}_{cfg.task_suite_name}-batch-{cfg.lang_rephrase_num_prefail}-{cfg.action_samples_prefail}-{cfg.augmented_samples_prefail}_{cfg.lang_rephrase_num}-{cfg.action_samples}-{cfg.augmented_samples}-alpha-[{cfg.failure_cp_alpha}]-{DATE_TIME}"
    else:
        run_id = f"[{cfg.wandb_project}]-SEED-{cfg.seed}_{cfg.task_suite_name}-batch-{cfg.lang_rephrase_num_prefail}-{cfg.action_samples_prefail}-{cfg.augmented_samples_prefail}-{DATE_TIME}"
    
    logs_dir = os.path.join(cfg.local_log_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    local_log_filepath = os.path.join(logs_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Write config to top of log file
    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    log_file.write("=" * 80 + "\n")
    log_file.write("EVALUATION CONFIGURATION\n")
    log_file.write("=" * 80 + "\n")

    for field in fields(cfg):
        value = getattr(cfg, field.name)
        print(f"{field.name}: {value}")
        log_file.write(f"{field.name}: {value}\n")
    print("=" * 80)
    print()
    log_file.write("=" * 80 + "\n")

    # Initialize Weights & Biases logging
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # =========================================================================================
    # Initialize Policy, Verifier, and Task Suite
    # =========================================================================================

    # Initialize PI0 policy
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    pi0_policy = PI0Policy.from_pretrained(cfg.pretrained_checkpoint)
    
    if torch.cuda.is_available():
        pi0_policy.to("cuda")
        pi0_policy.config.device = "cuda"
    
    pi0_policy.config.n_action_steps = int(cfg.n_action_steps)
    print(f"PI0Policy device: {pi0_policy.config.device}")
    
    # Initialize verifier model
    if cfg.use_verifier or cfg.augmented_samples_prefail > 0 or cfg.augmented_samples > 0:
        print("Loading ensemble model for similarity scoring...")
        # Use dynamic path relative to the VLA-CLIP root
        vla_clip_root = Path(__file__).resolve().parents[2]
        ensemble_checkpoint_path = vla_clip_root / "bridge_verifier" / "cover_verifier_bridge.pt"
        ensemble_model = EfficientEnsembleMerged(str(ensemble_checkpoint_path))
        print("Ensemble model loaded successfully!")
    else:
        ensemble_model = None
    
    # Initialize SIMPLER task suite
    task_suite = get_benchmark(cfg.task_suite_name)()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    
    # Load pre-generated rephrases
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)
    
    # Create adapter for preprocessing
    action_queue = None
    if not hasattr(pi0_policy, '_preprocess_adapter'):
        pi0_policy._preprocess_adapter = create_bridge_adapter_wrapper(cfg.action_ensemble_temp)
    preprocess_adapter = pi0_policy._preprocess_adapter
    
    # Action noise for batch inference
    action_noise_std = 1.0

    # =========================================================================================
    # Main Evaluation Loop (tasks -> episodes -> steps)
    # =========================================================================================
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        seeds = itertools.count(1000)

        # Initialize environment and task description
        env = get_simpler_env(task, cfg.model_family)
        original_task_description = env.get_language_instruction()

        curr_task = SAFE_TASK_MAP_DICT.get(original_task_description, cfg.task_suite_name)
        print("original_task_description", original_task_description)
        print("curr_task", curr_task)
        
        # Load QAM model if enabled
        if cfg.augmented_samples > 0 or cfg.augmented_samples_prefail > 0:
            print(f"Loading QAM checkpoint from {cfg.qam_ckpt}")
            qam = QAMInference(checkpoint_path = cfg.qam_ckpt)
            print("QAM checkpoint loaded successfully")

        # Load failure prediction model if enabled
        failure_model = None
        cp_band = None
        if cfg.use_failure_prediction:
            failure_model, cp_band = load_failure_detection_model(cfg, curr_task)

        # Load rephrased instructions if using language transformation
        if cfg.lang_transform_type == "no_transform":
            assert cfg.lang_rephrase_num == 1, "Language rephrase number must be 1 for no transformation"
            task_description = original_task_description
            rephrased_list = None
            matching_task_id = None
        else:
            # Find matching task in preloaded rephrases
            matching_task_id = None
            for task_key, task_data in preloaded_rephrases.items():
                if task_key == original_task_description:
                    matching_task_id = task_key
                    break
            
            if matching_task_id is not None:
                rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"][:max(cfg.lang_rephrase_num, cfg.lang_rephrase_num_prefail)] 
            else:
                raise ValueError(f"No preloaded rephrases found for task: {original_task_description}")

        # Run episodes for this task
        task_episodes, task_successes = 0, 0
        for eps_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if matching_task_id is not None and cfg.lang_transform_type == "rephrase":
                task_description = ([preloaded_rephrases[matching_task_id]["original"]] + preloaded_rephrases[matching_task_id]["ert_rephrases"])[0]
                print(f"\nUsing rephrased instruction (idx={0}): {task_description}")
            elif cfg.lang_transform_type == "no_transform":
                task_description = original_task_description
            
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            
            if eps_idx % 50 == 0:
                seeds = itertools.count(1000)
            
            obs, reset_info = env.reset(seed=next(seeds))

            # Initialize episode
            t = 0
            replay_images = []
            action_history = []
            all_features_episode = []     # For LSTM sequence history
            
            episode_data = {
                'verifier_scores': [],
                'selected_instructions': [],
                'execute_actions': [],
                'step_timestamps': [],
                'original_task_description': original_task_description,
                'used_task_description': task_description,
                'success': False,
                'episode_length': 0
            }
            # Per-episode logs and 3D-viz accumulator (reset every episode)
            logs = []
            viz = EpisodeVizTracker()
            cp_raw_data = []
            any_failure_detected = False
            
            # Set max steps based on task suite
            if cfg.task_suite_name.startswith("simpler"):
                max_steps = 150
            else:
                raise NotImplementedError

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            # Start episode
            pbar = tqdm.tqdm(total=max_steps + cfg.num_steps_wait, desc=f"Episode steps")
            done = False
            while t < max_steps + cfg.num_steps_wait:
                # Wait for objects to stabilize in simulator
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(get_simpler_dummy_action(cfg.model_family))
                    t += 1
                    pbar.update(1)
                    continue

                # PREFAIL init (i.e. before failure detection)
                action_samples = cfg.action_samples_prefail
                augmented_samples = cfg.augmented_samples_prefail
                policy_batch_inference_size = (action_samples) if augmented_samples <= 0 else augmented_samples
                lang_rephrase_num = cfg.lang_rephrase_num_prefail

                # Get raw image from environment
                raw_img = get_image_from_maniskill2_obs_dict(env, obs)
                replay_images.append(raw_img)
                
                # Prepare observations for adapter
                obs_for_adapter = {
                    'observation.images.top': raw_img,
                    'observation.state': obs,
                    'task': task_description
                }
                processed_obs = preprocess_adapter.preprocess(obs_for_adapter)
                
                # Move to policy device
                policy_device = torch.device(pi0_policy.config.device)
                processed_obs = {
                    k: (v.to(device=policy_device) if isinstance(v, torch.Tensor) else v)
                    for k, v in processed_obs.items()
                }
                
                # Get image feature key from policy config
                image_feature_keys = list(pi0_policy.config.image_features.keys())
                image_key = image_feature_keys[0]
                
                # Create batch of language instructions
                batch_size = policy_batch_inference_size * lang_rephrase_num
                
                # Build unique instruction list
                # Best task description from previous step is at the top of the list
                if rephrased_list is not None and lang_rephrase_num > 1:
                    unique_prompts = [task_description] + rephrased_list[:lang_rephrase_num - 1]
                else:
                    unique_prompts = [task_description]

                # Run a warmup with max num_actions in the config
                if eps_idx == 0 and t == 0 and cfg.warmup:
                    run_policy_warmup(cfg, task_description, rephrased_list, processed_obs, image_key, pi0_policy, action_noise_std)
                # Repeat each instruction for batch inference
                task_list = []
                for p in unique_prompts:
                    task_list.extend([p] * policy_batch_inference_size)
                    
                assert len(task_list) == batch_size, "Batch size mismatch"
                    
                # Create batch observation dict
                batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
                batch_state = processed_obs['observation.state'].repeat(batch_size, 1)
                
                observation = {
                    image_key: batch_image,
                    "observation.state": batch_state,
                    "task": task_list,
                }
                
                # Call select_action every n_action_steps
                if t % cfg.n_action_steps == 0:
                    with torch.no_grad():
                        pi0_policy.reset()
                        output_action_queue, action_embeds = pi0_policy.select_action(observation, noise_std=action_noise_std, return_action_embeds=True)
                        action_queue = output_action_queue.copy()
                        output_action_queue.clear()

                    # =====================================================================================
                    # Failure Detection (SAFE)
                    # =====================================================================================
                    # Process hidden states into the latent used by the failure classifier
                    sampled_action_embeds = torch.from_numpy(action_embeds)
                    action_embeds_h = sampled_action_embeds[:, :, 1:, :]  # NOTE: INTACT-pi0 has redundant first dim

                    if cfg.use_failure_prediction:
                        # TODO: Change Latent Config: horizon_idx_rel=mean, diff_idx_rel=0
                        assert action_embeds_h[0].shape == (10, 4, 1024)
                        action_embeds_h = action_embeds_h[0, :, :, :].mean(dim=1)  # (T, E) = (10, 1024)
                        assert action_embeds_h.shape == (10, 1024)
                        hidden_states_last_token = action_embeds_h[0, :]  # (1024,)
                        assert hidden_states_last_token.shape == (1024,)

                    if cfg.use_failure_prediction:
                        if not hidden_states_last_token.is_cuda:
                            hidden_states_last_token = hidden_states_last_token.cuda()

                        cumulative_prob, is_failure = check_failure_prediction_lstm(
                            hidden_states_last_token=hidden_states_last_token,
                            failure_model=failure_model,
                            cp_band=cp_band,
                            timestep=t - cfg.num_steps_wait,
                            all_features=all_features_episode,
                        )

                        cp_threshold = cp_band[min(t - cfg.num_steps_wait, len(cp_band) - 1)]
                        if is_failure:
                            print(f"\033[91m⚠️  FAILURE PREDICTED at t={t}: "
                                    f"cumulative_prob={cumulative_prob:.4f} >= "
                                    f"CP_band[{t - cfg.num_steps_wait}]={cp_threshold:.4f}\033[0m")
                        else:
                            print(f"\033[92m✅  NO FAILURE PREDICTED at t={t}: "
                                    f"cumulative_prob={cumulative_prob:.4f} < "
                                    f"CP_band[{t - cfg.num_steps_wait}]={cp_threshold:.4f}\033[0m")

                        # If not failure, stick to prefail config
                        if is_failure:
                            action_samples = cfg.action_samples
                            augmented_samples = cfg.augmented_samples
                            policy_batch_inference_size = (action_samples) if augmented_samples <= 0 else augmented_samples
                            lang_rephrase_num = cfg.lang_rephrase_num

                            # Create batch of language instructions
                            batch_size = policy_batch_inference_size * lang_rephrase_num

                            # Truncate extra actions from queue
                            if cfg.action_samples_prefail > cfg.action_samples and cfg.lang_rephrase_num < cfg.lang_rephrase_num_prefail and augmented_samples <= 0:
                                action_queue = list(action_queue)
                                action_queue = [action_queue[i*cfg.action_samples_prefail + j] for i in range(lang_rephrase_num) for j in range(action_samples)]
                                action_queue = deque(action_queue)
                            
                            # Resample if necessary
                            elif augmented_samples <= 0:
                                # Build unique instruction list
                                if rephrased_list is not None and lang_rephrase_num > 1:
                                    unique_prompts = [task_description] + rephrased_list[:lang_rephrase_num - 1]
                                else:
                                    unique_prompts = [task_description]

                                # Repeat each instruction for batch inference
                                task_list = []
                                for p in unique_prompts:
                                    task_list.extend([p] * policy_batch_inference_size)
                                    
                                assert len(task_list) == batch_size, "Batch size mismatch"

                                # Create batch observation dict
                                batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
                                batch_state = processed_obs['observation.state'].repeat(batch_size, 1)
                                
                                observation = {
                                    image_key: batch_image,
                                    "observation.state": batch_state,
                                    "task": task_list,
                                }
                                # Free PyTorch cached allocations from prefail batch before
                                # allocating a larger post-fail batch (prevents OOM on batch jump)
                                torch.cuda.empty_cache()

                                # Clear the queue to force re-generation in select_action
                                # (select_action only generates new actions when the queue is empty)
                                # Call select_action every n_action_steps
                                with torch.no_grad():
                                    pi0_policy.reset()
                                    output_action_queue, action_embeds = pi0_policy.select_action(observation, noise_std=action_noise_std, return_action_embeds=True)
                                    action_queue = output_action_queue.copy()
                                    output_action_queue.clear()

                    # =====================================================================================
                    # Compositional Action Steering
                    # =====================================================================================
                    augmented_actions = None
                    augmented_actions_queue = None
                    w = None

                    if augmented_samples > 0:
                        augmented_actions, augmented_actions_queue, w = compute_composed_actions(
                            cfg=cfg,
                            task_description=task_description,
                            rephrased_list=rephrased_list,
                            lang_rephrase_num=lang_rephrase_num,
                            augmented_samples=augmented_samples,
                            processed_obs=processed_obs,
                            image_key=image_key,
                            pi0_policy=pi0_policy,
                            qam=qam,
                            action_noise_std=action_noise_std,
                        )

                # =========================================================================================
                # Verifier-Based Action Selection
                # =========================================================================================
                if (cfg.use_verifier or augmented_actions_queue is not None) and t % cfg.n_action_steps == 0:
                    if augmented_actions_queue is not None:
                        assert len(augmented_actions_queue) == cfg.n_action_steps, \
                            f"Augmented action queue length should be {cfg.n_action_steps}, but got {len(augmented_actions_queue)}"
                        action_queue = augmented_actions_queue
                    assert len(action_queue) == cfg.n_action_steps, \
                        f"Action queue length should be {cfg.n_action_steps}, but got {len(action_queue)}"
                    
                    num_past = min(len(action_history), 6)
                    predefined_action_queue = list(action_queue)
                    action_queue.popleft()
                    
                    # Process actions for verifier
                    action_histories_list = process_inputs(
                        batch_size, predefined_action_queue, 
                        verifier_action=True, action_history=action_history.copy(), cfg=cfg
                    )
                    images_list = [process_raw_image_to_jpg(raw_img)] * batch_size

                    # Note: RL2 does not adaptively adjust number of samples 
                    if not cfg.use_verifier_always:
                        with torch.no_grad():
                            # First try with original instruction only (high confidence)
                            max_score, _, max_action_history, global_action_idx = \
                                ensemble_model.compute_max_similarity_scores_batch(
                                    images=images_list[0:1],
                                    instructions=[task_description],
                                    all_action_histories=action_histories_list[0:1],
                                    cfg_repeat_language_instructions=1
                                )
                    else:
                        max_score = 0
                    
                    # If score is too low, try with all rephrased instructions (low confidence)
                    if max_score < 0.1:
                        with torch.no_grad():
                            max_score, _ , max_action_history, global_action_idx = \
                                ensemble_model.compute_max_similarity_scores_batch(
                                    images=images_list,
                                    instructions=[task_description] * batch_size,
                                    all_action_histories=action_histories_list,
                                    cfg_repeat_language_instructions=policy_batch_inference_size
                                )
                        # Map global_action_idx back to the corresponding rephrase instruction
                        max_instruction = task_list[global_action_idx]

                    print("Total samples got: ", len(action_histories_list), f"| policy_batch_inference_size * lang_rephrase_num = {policy_batch_inference_size} * {int(len(action_histories_list) / policy_batch_inference_size)}")
                    
                    # Get execution-format actions (not verification-format)
                    execution_action_histories_list = process_inputs(
                        batch_size, predefined_action_queue, 
                        verifier_action=False, action_history=action_history.copy(), cfg=cfg
                    )
                    execute_action = execution_action_histories_list[global_action_idx][num_past].copy()
                    
                    # Perform gripper voting
                    group_start = (global_action_idx // policy_batch_inference_size) * policy_batch_inference_size
                    group_end = group_start + policy_batch_inference_size
                    stacked_histories = np.stack(execution_action_histories_list[group_start:group_end])
                    grippers = stacked_histories[:, num_past, -1]
                    
                    # Count votes: >= 0 is closed, < 0 is open
                    close_votes = int((grippers >= 0).sum())
                    open_votes = int((grippers < 0).sum())

                    if close_votes > open_votes:
                        execute_action[-1] = 1.0
                    elif open_votes > close_votes:
                        execute_action[-1] = -1.0
                    else:
                        # Tie: use selected action's sign
                        execute_action[-1] = 1.0 if execute_action[-1] >= 0 else -1.0

                    execute_action[-1] = float(np.sign(execute_action[-1]))
                    
                    # Extract remaining actions from selected batch item
                    selected_action_chunk = deque()
                    for timestep_idx in range(1, cfg.n_action_steps):
                        timestep_actions = predefined_action_queue[timestep_idx]
                        selected_action = timestep_actions[global_action_idx:global_action_idx+1]
                        selected_action_chunk.append(selected_action)
                    
                    action_queue = selected_action_chunk
                    
                    # Store episode data
                    episode_data['verifier_scores'].append(max_score)
                    episode_data['selected_instructions'].append(max_instruction)
                    episode_data['execute_actions'].append(execute_action.copy())
                    episode_data['step_timestamps'].append(t)
                    
                    task_description = max_instruction  # Note: Updates task_description with best prompt
                else:
                    # At inference steps without verifier: set up log variables before popleft
                    if t % cfg.n_action_steps == 0:
                        global_action_idx = 0   # single sample, always pick index 0
                        num_past = min(len(action_history), 6)
                        predefined_action_queue = list(action_queue)   
                        execution_action_histories_list = process_inputs(
                            1, predefined_action_queue, verifier_action=False,
                            action_history=action_history.copy(), cfg=cfg
                        )
                    # Use actions from queue
                    single_action = action_queue.popleft().cpu().numpy()
                    action_for_env = single_action[0:1]
                    execute_action = convert_maniskill_with_bridge_adapter(
                        action_for_env, verifier_action=False, action_ensemble_temp=cfg.action_ensemble_temp
                    )
                    
                    # Store episode data
                    episode_data['verifier_scores'].append(None)
                    episode_data['selected_instructions'].append(task_description)
                    episode_data['execute_actions'].append(execute_action.copy())
                    episode_data['step_timestamps'].append(t)
                    
                # Update action history for verifier
                if cfg.use_verifier:
                    if t % cfg.n_action_steps == 0:
                        processed_action_for_history = max_action_history[num_past].copy()
                    else:
                        processed_action_for_history = convert_maniskill_with_bridge_adapter(
                            single_action[0:1], verifier_action=True, action_ensemble_temp=cfg.action_ensemble_temp
                        )
                    action_history.append(processed_action_for_history)

                # Collect 3D viz data BEFORE env.step so EEF shows position before current action
                if cfg.save_3d_html_viz:
                    viz.update(
                        t=t,
                        n_action_steps=cfg.n_action_steps,
                        num_steps_wait=cfg.num_steps_wait,
                        predefined_action_queue=predefined_action_queue,
                        global_action_idx=global_action_idx,
                        augmented_actions_queue=augmented_actions_queue,
                        w=w,
                        task_description=task_description,
                        raw_img=raw_img,
                        use_failure_prediction=cfg.use_failure_prediction,
                        cumulative_prob=cumulative_prob if cfg.use_failure_prediction else None,
                        is_failure=is_failure if cfg.use_failure_prediction else None,
                        cp_band=cp_band,
                        max_steps=max_steps,
                    )

                # Collect raw failure prediction data (matplotlib rendering deferred to post-episode)
                if cfg.use_failure_prediction:
                    any_failure_detected = any_failure_detected or is_failure
                    cp_raw_data.append((cumulative_prob, is_failure, t - cfg.num_steps_wait))

                # Execute action in environment
                obs, reward, done, trunc, info = env.step(execute_action)

                # Log at inference-step boundaries, matching open-pi-zero format
                if t % cfg.n_action_steps == 0 and cfg.log_safe_training_data:
                    sampled_actions_np        = torch.stack(predefined_action_queue, dim=1).cpu().float().numpy()
                    sampled_actions_denorm_np = np.stack(
                        [execution_action_histories_list[n][num_past:num_past + cfg.n_action_steps]
                         for n in range(len(execution_action_histories_list))], axis=0)
                    selected_action_np        = torch.stack(
                        [predefined_action_queue[h][global_action_idx]
                         for h in range(len(predefined_action_queue))], dim=0).cpu().float().numpy()
                    selected_action_denorm_np = execution_action_histories_list[global_action_idx][
                        num_past:num_past + cfg.n_action_steps].copy()
                    logs.append({
                        "timestep":               t,
                        "instruction":            task_description,
                        "reward":                 float(reward),
                        "success":                bool(done),
                        "truncated":              bool(trunc),
                        "info":                   info,
                        "predicted_terminated":   False,
                        "sampled_actions":        sampled_actions_np,
                        "sampled_actions_denorm": sampled_actions_denorm_np,
                        "selected_action":        selected_action_np,
                        "selected_action_denorm": selected_action_denorm_np,
                        "sampled_action_embeds":  sampled_action_embeds[0:1].clone().cpu(),
                    })  # idx0 = top lang instruction from prev step used for SAFE

                if done and not cfg.log_safe_training_data:
                    task_successes += 1
                    total_successes += 1
                    break
                
                t += 1
                
                # Update progress bar
                if cfg.use_verifier:
                    pbar.set_description(f"Episode steps (score: {max_score:.3f})")
                pbar.update(1)

            if done and cfg.log_safe_training_data:
                task_successes += 1
                total_successes += 1

            pbar.close()
            task_episodes += 1
            total_episodes += 1

            # =========================================================================================
            # Episode Finalization & Logging
            # =========================================================================================
            # Finalize episode data
            episode_data['success'] = done
            episode_data['episode_length'] = t
            action_queue.clear()
            
            # Save rollout video
            video_save_path = save_rollout_video_openpi(
                replay_images, total_episodes, success=done,
                task_description=original_task_description,
                transformation_type=cfg.lang_transform_type,
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=policy_batch_inference_size,
                local_log_dir=cfg.local_log_dir,
                task_name=curr_task,
                log_file=log_file,
                run_id=run_id
            )

            # Render and write failure prediction plot MP4
            if cfg.use_failure_prediction and len(cp_raw_data) > 0:
                failure_tag = "FAILURE_DETECTED" if any_failure_detected else "NO_FAILURE_DETECTED"
                cp_mp4_path = video_save_path.replace(".mp4", f"_failure_plots_{failure_tag}.mp4")
                print(f"Saving failure prediction plots MP4 at path {cp_mp4_path}")
                log_file.write(f"Saving failure prediction plots MP4 at path {cp_mp4_path}\n")
                _write_cp_plots_mp4(list(cp_raw_data), cp_mp4_path, cp_band, max_steps)

            # Save episode data
            if cfg.log_misc_episode_data_openpi:
                save_episode_data_openpi(
                    episode_data, total_episodes, success=done,
                    task_description=original_task_description,
                    transformation_type=cfg.lang_transform_type,
                    lang_rephrase_num=cfg.lang_rephrase_num,
                    policy_batch_inference_size=policy_batch_inference_size,
                    local_log_dir=cfg.local_log_dir,
                    task_name=curr_task,
                    log_file=log_file,
                    run_id=run_id
                )

            # Save open-pi-zero–compatible logs pkl and meta.json (same dir as video)
            if cfg.log_safe_training_data:
                rollout_dir = os.path.dirname(video_save_path)
                logs_pkl_path = save_logs_pkl(
                    logs, total_episodes, done, rollout_dir, log_file
                )
                save_episode_meta_json(
                    curr_task, total_episodes, done, t, video_save_path, logs_pkl_path,
                    cfg, rollout_dir, log_file
                )

            # Save 3D action visualization
            if cfg.save_3d_html_viz and len(viz.states) > 0:
                viz_3d_path = Path(os.path.dirname(video_save_path)) / f"episode_{total_episodes}_success_{done}_3d_viz"
                viz.create_3d_action_visualization(viz_3d_path, use_failure_prediction=cfg.use_failure_prediction)

            # Log videos to W&B (limit 5 successes and 5 failures per task)
            if cfg.use_wandb and ((done and task_successes < 5) or 
                                 (not done and task_episodes - task_successes < 5)):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                video_array = np.array(replay_images).transpose(0, 3, 1, 2)
                wandb.log({f"{task_description}/{group}/{idx}": wandb.Video(video_array)})

            # Log episode results
            success_rate = total_successes / total_episodes * 100
            print(f"Success: {done}")
            print(f"Episodes: {total_episodes} | Successes: {total_successes} ({success_rate:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Episodes: {total_episodes} | Successes: {total_successes} ({success_rate:.1f}%)\n")
            log_file.flush()

        # Log task results
        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        print(f"\nTask success rate: {task_success_rate:.3f}")
        print(f"Total success rate: {total_success_rate:.3f}")
        log_file.write(f"\nTask success rate: {task_success_rate:.3f}\n")
        log_file.write(f"Total success rate: {total_success_rate:.3f}\n")
        log_file.flush()
        
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            })

    # =========================================================================================
    # Finalize Logging
    # =========================================================================================
    log_file.close()

    # Log final metrics to W&B
    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        })
        wandb.save(local_log_filepath)


# =========================================================================================
# Entry Point
# =========================================================================================

if __name__ == "__main__":
    eval_simpler()
