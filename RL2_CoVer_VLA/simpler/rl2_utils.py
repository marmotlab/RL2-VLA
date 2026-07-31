# This script is a utility module for the SIMPLER benchmark evaluation.
# For paper: "RL2-VLA: Adaptive RL Latent Compositional Steering with Test-Time Scaling for Vision-Language-Action Models"

import os
import sys
import json
import pickle
from collections import deque
from pathlib import Path

import imageio
import numpy as np
import torch

import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

# QAM imports
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'   
import jax
import jax.numpy as jnp

# Failure prediction imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root_path = os.path.abspath(os.path.join(script_dir, "../.."))
safe_path = os.path.join(repo_root_path, "third_party/SAFE")
sys.path.insert(0, safe_path)
from failure_prob.model.base import BaseModel
from failure_prob.model import get_model as get_model_safe
from omegaconf import OmegaConf

qam_path = os.path.join(repo_root_path, "third_party/qam")
# QAM already cached sys.modules for 'agents.*' and 'utils.*' with its own modules.
# Temporarily evict all QAM-cached modules so QAM imports resolve against qam_path.
_QAM_PREFIXES = ('agents', 'utils')
_saved_qam_modules = {k: v for k, v in sys.modules.items()
                     if any(k == p or k.startswith(p + '.') for p in _QAM_PREFIXES)}
for _k in _saved_qam_modules:
    del sys.modules[_k]
sys.path.insert(0, qam_path)
from agents import agents as qam_agents
from utils.flax_utils import restore_agent_with_file as qam_restore_agent_with_file
sys.path.remove(qam_path)
# Restore QAM's modules so downstream QAM code continues to work
sys.modules.update(_saved_qam_modules)


SAFE_TASK_MAP_DICT: dict = {
    "put eggplant into yellow basket": "widowx_put_eggplant_in_basket",
    "put carrot on plate": "widowx_carrot_on_plate",
    "put the spoon on the towel": "widowx_spoon_on_towel",
    "stack the green block on the yellow block": "widowx_stack_cube",
}

class QAMInference:
    def __init__(self, checkpoint_path: str):
        """Load QAM model from checkpoint."""
        # Load config from flags.json saved alongside the checkpoint
        flags_path = os.path.join(os.path.dirname(checkpoint_path), "flags.json")
        flags = json.load(open(flags_path)) if os.path.exists(flags_path) else {}
        agent_config = flags.get('agent', {})

        print("QAM Seed: ---> ", flags["seed"])
        # Create dummy example batch — only shapes matter for agent initialization.
        # ob_dims and action_dim are stored in flags.json by main.py after create().
        obs_dim = agent_config['ob_dims']   # list, e.g. [1028]
        action_dim = agent_config['action_dim']  # int, e.g. 7
        example_batch = {
            'observations': np.zeros(obs_dim, dtype=np.float32),
            'actions': np.zeros(action_dim, dtype=np.float32),
        }

        agent_class = qam_agents[agent_config['agent_name']]
        agent = agent_class.create(
            flags["seed"],
            example_batch['observations'],
            example_batch['actions'],
            agent_config,
        )

        params = agent.network.params
        params = {k: v for k, v in params.items() if "target" not in k}
        print(params.keys())
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print("param count:", param_count)

        # Restore agent weights from checkpoint file
        self.agent = qam_restore_agent_with_file(agent, checkpoint_path)
        
        if self.agent.config["inv_temp"] == 0.:
            self.model="slow"
        else:
            self.model="slow,fast" if self.agent.config["residual"] else "fast"
        self.networks = tuple(self.agent.network.select(f'actor_{m}') for m in self.model.split(","))

    def get_denoising_vector(self, observation, noisy_action, i):
        obs = jnp.asarray(observation)
        if obs.ndim == 1:
            obs = jnp.tile(obs[None, :], (noisy_action.shape[0], 1)) 
        return self.agent.get_denoising_vector(obs, noisy_action, self.model, self.networks, i)

    # def sample_actions(self, observations, rng):
    #     actions = self.agent.sample_actions(
    #         observations=observations,
    #         rng=rng,
    #     )
    #     return actions
        
def load_failure_detection_model(cfg, curr_task):
    print(f"\n{'='*80}")
    print("Loading Failure Prediction Model")
    print(f"{'='*80}")

    config_path = os.path.join(cfg.failure_checkpoint_dir, "config.yaml")
    failure_cfg = OmegaConf.load(config_path)

    input_dim = 1024   
    failure_model = get_model_safe(failure_cfg, input_dim)

    # Check model type
    if failure_cfg.model.name == "lstm":
        pass
    else:
        raise ValueError(f"Unsupported failure model name: {failure_cfg.model.name}")
    failure_model.to("cuda")
    failure_model.eval()

    checkpoint_path = os.path.join(cfg.failure_checkpoint_dir, "model_final.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        failure_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        failure_model.load_state_dict(checkpoint)
    print(f"Loaded failure prediction model from {checkpoint_path}")

    if cfg.use_taskwise_cp_band:
        cp_band_path = os.path.join(cfg.failure_checkpoint_dir, "cp_bands_by_task.npy")
    else:
        cp_band_path = os.path.join(cfg.failure_checkpoint_dir, "cp_band_by_alpha.npy")

    if os.path.exists(cp_band_path):
        cp_band_data = np.load(cp_band_path, allow_pickle=True).item()

        if cfg.use_taskwise_cp_band:
            # Task-wise format: {task_name: {alpha: cp_band}}
            if curr_task not in cp_band_data:
                available_tasks = list(cp_band_data.keys())
                raise ValueError(f"Task '{curr_task}' not found in CP band file. Available: {available_tasks}")

            task_cp_data = cp_band_data[curr_task]
            if cfg.failure_cp_alpha not in task_cp_data:
                available_alphas = list(task_cp_data.keys())
                raise ValueError(f"Alpha={cfg.failure_cp_alpha} not found for task '{curr_task}'. Available: {available_alphas}")

            cp_band = task_cp_data[cfg.failure_cp_alpha]
            print(f"Loaded task-specific CP band for '{curr_task}' with alpha={cfg.failure_cp_alpha}")
        else:
            # Pooled format: {alpha: cp_band}
            if cfg.failure_cp_alpha not in cp_band_data:
                available_alphas = list(cp_band_data.keys())
                raise ValueError(f"CP band for alpha={cfg.failure_cp_alpha} not found. Available: {available_alphas}")

            cp_band = cp_band_data[cfg.failure_cp_alpha]
            print(f"Loaded pooled CP band with alpha={cfg.failure_cp_alpha}")

        # Common processing (same for both formats)
        if len(cp_band.shape) == 2 and cp_band.shape[0] == 1:
            cp_band = cp_band[0]
        print(f"CP band shape={cp_band.shape}, range=[{cp_band.min():.4f}, {cp_band.max():.4f}]")

        # Stretch CP band to account for cfg.n_action_steps
        # CP band was trained at frequency of cfg.n_action_steps, so interpolate to get all timesteps
        original_length = len(cp_band)
        x_original = np.arange(original_length) * cfg.n_action_steps
        x_new = np.arange(original_length * cfg.n_action_steps)
        cp_band = np.interp(x_new, x_original, cp_band)
        print(f"Stretched CP band from {original_length} to {len(cp_band)} steps (act_steps={cfg.n_action_steps})")
    else:
        raise FileNotFoundError(f"CP band not found at {cp_band_path}")
    print(f"{'='*80}\n")
    return failure_model, cp_band

def run_policy_warmup(cfg, task_description, rephrased_list, processed_obs, image_key, pi0_policy, action_noise_std):
    dummy_policy_batch_inference_size = max(cfg.action_samples_prefail, cfg.composed_samples_prefail, cfg.action_samples, cfg.composed_samples)
    dummy_batch_size = dummy_policy_batch_inference_size * max(cfg.lang_rephrase_num, cfg.lang_rephrase_num_prefail)

    # Build unique prompts for warmup (postfail uses rephrased prompts regardless of use_failure_prediction)
    if rephrased_list is not None and max(cfg.lang_rephrase_num, cfg.lang_rephrase_num_prefail) > 1:
        dummy_unique_prompts = [task_description] + rephrased_list[:max(cfg.lang_rephrase_num, cfg.lang_rephrase_num_prefail) - 1]
    else:
        dummy_unique_prompts = [task_description] * max(cfg.lang_rephrase_num, cfg.lang_rephrase_num_prefail)

    # Repeat each instruction for batch inference
    dummy_task_list = []
    for p in dummy_unique_prompts:
        dummy_task_list.extend([p] * dummy_policy_batch_inference_size)

    assert len(dummy_task_list) == dummy_batch_size, "Batch size mismatch"

    # Create batch observation dict
    dummy_batch_image = processed_obs['observation.images.top'].repeat(dummy_batch_size, 1, 1, 1)
    dummy_batch_state = processed_obs['observation.state'].repeat(dummy_batch_size, 1)

    dummy_observation = {
        image_key: dummy_batch_image,
        "observation.state": dummy_batch_state,
        "task": dummy_task_list,
    }
    with torch.no_grad():
        pi0_policy.reset()
        pi0_policy.select_action(dummy_observation, noise_std=action_noise_std, return_action_embeds=True)

def get_composed_actions(pi0_policy, qam, processed_obs, image_key, task_list, batch_size, hidden_states_np, w, action_noise_std):
    # Create batch observation dict
    _batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
    _batch_state = processed_obs['observation.state'].repeat(batch_size, 1)

    _observation = {
        image_key: _batch_image,
        "observation.state": _batch_state,
        "task": task_list,
    }
    with torch.no_grad():
        _past_key_values, _state, _prefix_pad_masks = pi0_policy.forward_pass_vlm(_observation, noise_std=action_noise_std)
        # ########################################################################
        # Sample noise
        # ########################################################################
        bsz, horizon, action_dim = _state.shape[0], pi0_policy.model.config.chunk_size, pi0_policy.model.config.max_action_dim
        actions_shape = (bsz, horizon, action_dim)

        noisy_action_torch = pi0_policy.model.sample_noise(actions_shape, _state.device, noise_std=action_noise_std)
        noisy_action_np = noisy_action_torch.detach().cpu().float().numpy()
        noisy_action_jax = jnp.asarray(noisy_action_np[:,:,:pi0_policy.config.action_feature.shape[0]]).reshape(bsz, -1)

        # ########################################################################
        # Denoise by Composing
        # ########################################################################
        dt = -1.0 / pi0_policy.model.config.num_steps
        time = torch.tensor(1.0, dtype=torch.float32, device=_state.device)
        qam_t = 0
        while time >= -dt / 2:
            expanded_time = time.expand(bsz)
            vel_vla, _ = pi0_policy.model.denoise_step(
                _state,
                _prefix_pad_masks,
                _past_key_values,
                noisy_action_torch,
                expanded_time,
            )
            vel_vla = vel_vla.detach().cpu().float().numpy()
            vel_qam = np.asarray(qam.get_denoising_vector(
                observation=hidden_states_np,
                noisy_action=noisy_action_jax,
                i=qam_t
            )).reshape(bsz, horizon, -1)

            # # Compositional steering (pad vel_qam with vel_vla)
            # vel_vla -- (bsz, horizon, action_dim_padded) -- B, H, 32
            # vel_qam -- (bsz, horizon, action_dim) -- B, H, 7
            vel_qam = np.concatenate([vel_qam, vel_vla[..., vel_qam.shape[-1]:]], axis=-1)
            noisy_action_np += (w * vel_vla - (1 - w) * vel_qam) * dt

            noisy_action_torch = torch.from_numpy(noisy_action_np).to(_state.device, _state.dtype)
            noisy_action_jax = jnp.asarray(noisy_action_np[:,:,:pi0_policy.config.action_feature.shape[0]]).reshape(bsz, -1)

            time += dt
            qam_t += 1

    composed_actions = noisy_action_torch[:, :pi0_policy.config.n_action_steps]

    # Unpad composed_actions
    original_action_dim = pi0_policy.config.action_feature.shape[0]
    composed_actions = composed_actions[:, :, :original_action_dim]
    composed_actions = pi0_policy.unnormalize_outputs({"action": composed_actions})["action"]

    return composed_actions

def compute_composed_actions(cfg, task_description, rephrased_list, lang_rephrase_num, composed_samples,
                              processed_obs, image_key, pi0_policy, qam, action_noise_std):
    """Run compositional (QAM-weighted) action denoising for one action-selection step.

    Returns:
        composed_actions: (B, n_action_steps, action_dim) denoised actions
        composed_actions_queue: deque of per-timestep actions, maxlen=cfg.n_action_steps
        w: resolved compositional weight array, shape (B, 1, 1)
    """
    w = cfg.merge_rel_weight

    # ########################################################################
    # VLA sampling
    # ########################################################################
    policy_batch_inference_size = composed_samples
    batch_size = policy_batch_inference_size * lang_rephrase_num
    composed_actions_queue = deque([], maxlen=cfg.n_action_steps)

    # Set weights for compositional denoising
    # NOTE: If w == -1.0, sample from normal distribution
    if w == -1.0:
        w = np.random.normal(loc=0.5, scale=0.25, size=batch_size)
        w = np.clip(w, 0, 1)             # shape = B,
        w = w[:, None, None]             # shape = B, 1, 1
    # If  0 <= w <= 1, use fixed weights
    else:
        w = np.ones(batch_size) * w
        w = w[:, None, None]             # shape = B, 1, 1

    # Build unique instruction list
    if rephrased_list is not None and lang_rephrase_num > 1:
        unique_prompts = [task_description] + rephrased_list[:lang_rephrase_num - 1]
    else:
        unique_prompts = [task_description]

    # ########################################################################
    # IF "use_rephrased_latents_for_qam" is True, use rephrased latents for QAM
    # So, "hidden_states_np" becomes (B, 1024) instead of (1024,)
    # ########################################################################
    if cfg.use_rephrased_latents_for_qam:
        # Create batch observation dict
        _batch_image = processed_obs['observation.images.top'].repeat(len(unique_prompts), 1, 1, 1)
        _batch_state = processed_obs['observation.state'].repeat(len(unique_prompts), 1)
        _observation = {
            image_key: _batch_image,
            "observation.state": _batch_state,
            "task": unique_prompts,
        }

        with torch.no_grad():
            pi0_policy.reset()
            _, action_embeds = pi0_policy.select_action(_observation, noise_std=action_noise_std, return_action_embeds=True)

            #########################################
            # Process Hidden States
            #########################################
            sampled_action_embeds = torch.from_numpy(action_embeds)
            action_embeds_h = sampled_action_embeds[:, :, 1:, :]  # NOTE: INTACT-pi0 has redundant first dim

            # TODO: Adjust Latent config: horizon_idx_rel=mean, diff_idx_rel=0
            assert action_embeds_h[0].shape == (10, 4, 1024)
            action_embeds_h = action_embeds_h[:, :, :, :].mean(dim=2)  # (B, T, E) = (B, 10, 1024)
            assert action_embeds_h[0].shape == (10, 1024)
            hidden_states_last_token = action_embeds_h[:,0]  # (B, 1024)
            assert hidden_states_last_token[0].shape == (1024,)

            hidden_states_np = hidden_states_last_token.detach().cpu().to(torch.float32).numpy()

            # B = len(unique_prompts) but we need B = len(unique_prompts) * composed_samples
            rephrased_hidden_states_np = []
            for i in range(len(unique_prompts)):
                rephrased_hidden_states_np.extend([hidden_states_np[i]] * policy_batch_inference_size)

            hidden_states_np = np.array(rephrased_hidden_states_np)   # (B, 1024)

    # Repeat each instruction for batch inference
    task_list = []
    for p in unique_prompts:
        task_list.extend([p] * policy_batch_inference_size)

    assert len(task_list) == batch_size, "Batch size mismatch"

    composed_actions = get_composed_actions(pi0_policy, qam, processed_obs, image_key, task_list, batch_size, hidden_states_np, w, action_noise_std)

    composed_actions_queue.extend(composed_actions.transpose(0, 1))

    return composed_actions, composed_actions_queue, w

# Failure Prediction Fns
def check_failure_prediction(
    hidden_states_last_token: torch.Tensor,
    failure_model: BaseModel,
    cp_band: np.ndarray,
    timestep: int,
    raw_predictions: list,
    ) -> tuple[float, bool]:
    """
    Check if failure is predicted based on cumulative failure probability.
    Adapted from save_predictions_v4.py for online/streaming inference.
    """
    latent = hidden_states_last_token.unsqueeze(0).unsqueeze(0)  # (1, 1, F)

    with torch.no_grad():
        # Ensure dtype consistency between latent and projector weights
        projector_dtype = next(failure_model.projector.parameters()).dtype
        latent = latent.to(dtype=projector_dtype)
        raw_output = failure_model.projector(latent)  # (1, 1, 1)
        raw_pred = raw_output.squeeze().item()
        raw_predictions.append(raw_pred)

        if failure_model.cfg.model.cumsum or failure_model.cfg.model.rmean:
            cumulative_prob = sum(raw_predictions)
            if failure_model.cfg.model.rmean:
                cumulative_prob = cumulative_prob / len(raw_predictions)
        else:
            cumulative_prob = raw_pred

    cp_threshold = cp_band[min(timestep, len(cp_band) - 1)]
    is_failure = cumulative_prob >= cp_threshold

    return cumulative_prob, is_failure

def check_failure_prediction_lstm(
    hidden_states_last_token: torch.Tensor,
    failure_model: BaseModel,
    cp_band: np.ndarray,
    timestep: int,
    all_features: list,  # Accumulated feature history for LSTM
    ) -> tuple[float, bool]:
    """
    Check if failure is detected using LSTM model.
    LSTM processes the full sequence history each timestep.
    """
    # Add current feature to history
    all_features.append(hidden_states_last_token)
    print(f"[LSTM] Timestep {timestep}: Added feature, history length = {len(all_features)}")

    with torch.no_grad():
        # LSTM: Stack all features into full sequence (1, T, F)
        sequence = torch.stack(all_features, dim=0).unsqueeze(0)

        # Ensure dtype consistency - convert to model's dtype (float32)
        model_dtype = next(failure_model.parameters()).dtype
        sequence = sequence.to(dtype=model_dtype)

        batch = {"features": sequence}

        # Forward through LSTM - processes entire sequence
        output = failure_model(batch)  # (1, T, 1)

        # LSTM outputs per-timestep probabilities (cumsum=false in config)
        # Take the last timestep's prediction
        failure_prob = output[0, -1, 0].item()

    # Get conformal prediction threshold for this timestep
    cp_threshold = cp_band[min(timestep, len(cp_band) - 1)]
    is_failure = failure_prob >= cp_threshold

    return failure_prob, is_failure


# =========================================================================================
# Open-pi-zero–Compatible Logging Helpers
# =========================================================================================

def save_logs_pkl(logs, idx, success, rollout_dir, log_file=None):
    """Save open-pi-zero–format list-of-step-dicts as episode_{idx}_success_{success}.pkl."""

    os.makedirs(rollout_dir, exist_ok=True)
    pkl_path = os.path.join(rollout_dir, f"episode_{idx}_success_{success}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(logs, f)
    print(f"Saved open-pi-zero logs pkl at {pkl_path}")
    if log_file:
        log_file.write(f"Saved open-pi-zero logs pkl at {pkl_path}\n")
    return pkl_path


def save_episode_meta_json(curr_task, idx, success, final_timestep, video_path, log_path,
                            cfg, rollout_dir, log_file=None):
    """Save open-pi-zero–format meta.json alongside the logs pkl."""

    os.makedirs(rollout_dir, exist_ok=True)
    meta = {
        "episode_id":     idx,
        "success":        success,
        "final_timestep": final_timestep,
        "video_path":     video_path,
        "log_path":       log_path,
        "task":           curr_task,
        **vars(cfg),    # all cfg fields flattened, matching open-pi-zero args.__dict__ pattern
    }
    json_path = os.path.join(rollout_dir, f"episode_{idx}_success_{success}_meta.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, default=str)   # default=str handles Path / non-JSON-serialisable values
    print(f"Saved meta.json at {json_path}")
    if log_file:
        log_file.write(f"Saved meta.json at {json_path}\n")
    return json_path


# =========================================================================================
# Episode 3D-Viz State Tracker
# =========================================================================================

class EpisodeVizTracker:
    """Accumulates per-step 3D-viz data for one episode.

    Action samples/selection/instruction only change once per n_action_steps chunk
    (at substep 0); `update()` freezes that chunk's state internally and repeats it
    for every sub-step so per-frame lists stay aligned with the episode timeline.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.arrow_origins = []
        self.vla_actions = []
        self.qam_actions = []
        self.sim_images = []
        self.selected_idx = []
        self.composed_flag = []
        self.w_values = []
        self.instructions = []
        self.cp_plot_frames = []
        self.cumulative_probs = []

        self._eef = None
        self._inf_eef = None
        self._vla_prev = None
        self._sel_prev = 0
        self._vla_frozen = None
        self._qam_frozen = None
        self._arrow_origin = None
        self._w_frozen = None
        self._sel_frozen = 0
        self._is_composed_frozen = False
        self._instruction_frozen = ""

    def update(self, *, t, n_action_steps, num_steps_wait, predefined_action_queue,
               global_action_idx, composed_actions_queue, w, task_description, raw_img,
               use_failure_prediction, cumulative_prob=None, is_failure=None,
               cp_band=None, max_steps=None):
        substep = t % n_action_steps
        if substep == 0:
            # --- New inference step ---
            # 1. Advance accumulated EEF to end of the previous chunk
            if self._inf_eef is not None and self._vla_prev is not None:
                self._eef = self._inf_eef + self._vla_prev[self._sel_prev, :, :3].sum(axis=0)
            elif self._eef is None:
                self._eef = np.zeros(3)
            self._inf_eef = self._eef.copy()
            # 2. Capture the FINAL action samples used for this chunk
            #    (predefined_action_queue is set after all resampling/compose/verifier)
            vla_now = torch.stack(predefined_action_queue, dim=1).cpu().float().numpy()  # (N, H, 7)
            # 3. Freeze arrows and origin for the whole chunk duration
            self._vla_frozen = vla_now
            self._sel_frozen = global_action_idx
            self._qam_frozen = None
            self._arrow_origin = self._inf_eef.copy()
            # 4. Freeze composed flag: True when compose mode produced the samples
            self._is_composed_frozen = (composed_actions_queue is not None)
            # 5a. Freeze per-sample w values (only valid in composed mode)
            self._w_frozen = w[:, 0, 0].copy() if (self._is_composed_frozen and w is not None) else None
            # 5b. Freeze the current instruction text
            self._instruction_frozen = task_description
            # 5. Save for EEF advance at the next inference step
            self._vla_prev = vla_now
            self._sel_prev = global_action_idx

        # EEF position for this frame: inference EEF + cumulative selected deltas so far
        if self._inf_eef is None:
            eef_pos = np.zeros(3)
        elif substep == 0 or self._vla_prev is None:
            eef_pos = self._inf_eef.copy()
        else:
            eef_pos = self._inf_eef + self._vla_prev[self._sel_prev, :substep, :3].sum(axis=0)

        # Append per-step data; action arrows are FROZEN for all sub-steps
        self.states.append(eef_pos)
        self.arrow_origins.append(self._arrow_origin.copy() if self._arrow_origin is not None else np.zeros(3))
        self.vla_actions.append(self._vla_frozen)     # same frozen chunk for all sub-steps
        self.qam_actions.append(self._qam_frozen)
        self.selected_idx.append(self._sel_frozen)
        self.composed_flag.append(self._is_composed_frozen)
        self.w_values.append(self._w_frozen)
        self.instructions.append(self._instruction_frozen)
        self.sim_images.append(raw_img)

        # CP failure prediction plot (duplicate inference-step values across sub-steps)
        if use_failure_prediction:
            self.cumulative_probs.append(cumulative_prob)
            cp_img = create_failure_prediction_plot(
                cumulative_prob=cumulative_prob,
                cumulative_probs_episode=self.cumulative_probs,
                cp_band=cp_band,
                is_failure=is_failure,
                t=t - num_steps_wait,
                max_steps=max_steps,
            )
            self.cp_plot_frames.append(cp_img)

    def create_3d_action_visualization(self, output_path, critic_values_episode=None, use_failure_prediction=False):
        """
        Create an interactive 3D Plotly visualization showing:
        - Black dots for end-effector positions (XYZ from state_vector)
        - Blue dots for VLA action deltas (first 3 values), chained across horizon H
        - Yellow dots for QAM actor action deltas (first 3 values)
        - Black lines connecting chained VLA actions (EEF -> action[0] -> action[1] -> ... -> action[H-1])
        - Simulation images (side-by-side with 3D plot)
        - SAFE failure prediction plots (below simulation images, if provided)

        Creates an interactive HTML file with a slider to step through frames.

        Uses the episode data accumulated on self (states, vla_actions, qam_actions,
        sim_images, selected_idx, arrow_origins, composed_flag, w_values, instructions,
        cp_plot_frames).

        Args:
            output_path: Path to save the HTML file (without extension)
            critic_values_episode: List of critic values for each timestep (can be None)
            use_failure_prediction: Whether to include the SAFE failure prediction plots
        """
        import plotly.graph_objects as go
        import io
        import base64
        from PIL import Image as PILImage

        states_episode = self.states
        vla_actions_episode = self.vla_actions
        qam_actions_episode = self.qam_actions
        sim_images = self.sim_images
        selected_indices = self.selected_idx
        arrow_origins = self.arrow_origins
        composed_flags = self.composed_flag
        w_values = self.w_values
        instructions = self.instructions
        cp_plot_images = self.cp_plot_frames if (use_failure_prediction and len(self.cp_plot_frames) > 0) else None

        if len(states_episode) == 0:
            print("No states to visualize")
            return

        # Extract XYZ positions from states
        positions = np.array([state[:3] for state in states_episode])

        # Prepare VLA action deltas (first 3 values: XYZ)
        vla_deltas = []
        if vla_actions_episode:
            for actions in vla_actions_episode:
                if actions is not None and len(actions) > 0:
                    if actions.ndim == 1:
                        vla_deltas.append(np.array([[actions[:3]]]))
                    elif actions.ndim == 2:
                        vla_deltas.append(actions[:, np.newaxis, :3])
                    else:
                        vla_deltas.append(actions[:, :, :3])
                else:
                    vla_deltas.append(None)

        # Prepare QAM action deltas (first 3 values: XYZ)
        qam_deltas = []
        if qam_actions_episode:
            for actions in qam_actions_episode:
                if actions is not None and len(actions) > 0:
                    if actions.ndim == 1:
                        qam_deltas.append(np.array([[actions[:3]]]))
                    elif actions.ndim == 2:
                        qam_deltas.append(actions[:, np.newaxis, :3])
                    else:
                        qam_deltas.append(actions[:, :, :3])
                else:
                    qam_deltas.append(None)

        # Resolve arrow origins (frozen inference EEF per frame, or fall back to positions)
        arrow_origins_arr = (np.array(arrow_origins) if arrow_origins is not None and len(arrow_origins) == len(positions)
                             else positions)

        # Determine axis limits
        all_points = [positions]
        for i, deltas in enumerate(vla_deltas):
            if deltas is not None:
                N, H, _ = deltas.shape
                cumulative_deltas = np.cumsum(deltas, axis=1)
                for h in range(H):
                    all_points.append(arrow_origins_arr[i:i+1] + cumulative_deltas[:, h, :])
        for i, deltas in enumerate(qam_deltas):
            if deltas is not None:
                N, H, _ = deltas.shape
                cumulative_deltas = np.cumsum(deltas, axis=1)
                for h in range(H):
                    all_points.append(arrow_origins_arr[i:i+1] + cumulative_deltas[:, h, :])

        all_points = np.vstack(all_points)
        margin = 0.05
        x_range = [all_points[:, 0].min() - margin, all_points[:, 0].max() + margin]
        y_range = [all_points[:, 1].min() - margin, all_points[:, 1].max() + margin]
        z_range = [all_points[:, 2].min() - margin, all_points[:, 2].max() + margin]

        fig = go.Figure()
        has_images = sim_images is not None

        if has_images and len(sim_images) != len(positions):
            print(f"Warning: sim_images length ({len(sim_images)}) != positions length ({len(positions)})")
            if len(sim_images) > len(positions):
                sim_images = sim_images[:len(positions)]
            else:
                while len(sim_images) < len(positions):
                    sim_images.append(sim_images[-1])

        max_vla_actions = 0
        max_vla_horizon = 0
        if vla_deltas:
            for deltas in vla_deltas:
                if deltas is not None:
                    N, H, _ = deltas.shape
                    max_vla_actions = max(max_vla_actions, N)
                    max_vla_horizon = max(max_vla_horizon, H)

        max_qam_actions = 0
        max_qam_horizon = 0
        if qam_deltas:
            for deltas in qam_deltas:
                if deltas is not None:
                    N, H, _ = deltas.shape
                    max_qam_actions = max(max_qam_actions, N)
                    max_qam_horizon = max(max_qam_horizon, H)

        frames = []
        for frame_idx in range(len(positions)):
            frame_data = []

            # Trajectory up to current frame
            if frame_idx > 0:
                trace = go.Scatter3d(
                    x=positions[:frame_idx+1, 0],
                    y=positions[:frame_idx+1, 1],
                    z=positions[:frame_idx+1, 2],
                    mode='lines',
                    line=dict(color='black', width=2),
                    opacity=0.3,
                    name='Trajectory',
                    showlegend=(frame_idx == 0)
                )
                frame_data.append(trace)

            # Current end-effector position
            trace = go.Scatter3d(
                x=[positions[frame_idx, 0]],
                y=[positions[frame_idx, 1]],
                z=[positions[frame_idx, 2]],
                mode='markers',
                marker=dict(size=3, color='black'),
                name='EE Position',
                showlegend=(frame_idx == 0)
            )
            frame_data.append(trace)

            # Arrow origin: frozen at inference EEF for the full chunk duration
            arrow_origin = arrow_origins_arr[frame_idx]

            # VLA action deltas (blue dots with lines)
            deltas = vla_deltas[frame_idx] if (frame_idx < len(vla_deltas) and vla_deltas[frame_idx] is not None) else None
            num_vla = deltas.shape[0] if deltas is not None else 0
            horizon = deltas.shape[1] if deltas is not None else 0

            for delta_idx in range(max_vla_actions):
                if delta_idx < num_vla:
                    current_pos = arrow_origin.copy()
                    # Check if this action chain is the selected one
                    if selected_indices is not None and frame_idx < len(selected_indices):
                        action_is_selected = (delta_idx == selected_indices[frame_idx])
                    elif critic_values_episode is not None and frame_idx < len(critic_values_episode) and critic_values_episode[frame_idx] is not None:
                        critic_vals = critic_values_episode[frame_idx]
                        action_is_selected = (delta_idx < len(critic_vals['vla_values']) and
                                              critic_vals['selected_idx'] == delta_idx and
                                              critic_vals['selected_source'] == 'VLA')
                    else:
                        action_is_selected = False
                    critic_values_for_chunk = None
                    if critic_values_episode is not None and frame_idx < len(critic_values_episode) and critic_values_episode[frame_idx] is not None:
                        critic_vals = critic_values_episode[frame_idx]
                        if delta_idx < len(critic_vals['vla_values']):
                            critic_values_for_chunk = critic_vals['vla_values'][delta_idx]

                    # Base color: W-gradient (yellow=0 to blue=1) for composed, blue for normal VLA
                    is_composed_frame = (composed_flags is not None and frame_idx < len(composed_flags) and composed_flags[frame_idx])
                    source_label = 'Composed' if is_composed_frame else 'VLA'
                    if is_composed_frame and w_values is not None and frame_idx < len(w_values) and w_values[frame_idx] is not None:
                        w_arr = w_values[frame_idx]
                        w_val = float(np.clip(w_arr[delta_idx], 0.0, 1.0)) if delta_idx < len(w_arr) else 0.5
                        # yellow (255,255,0) at w=0, blue (0,0,255) at w=1
                        r = int(255 * (1.0 - w_val))
                        g = int(255 * (1.0 - w_val))
                        b = int(255 * w_val)
                        base_color = f'rgb({r},{g},{b})'
                    else:
                        base_color = 'blue' if not is_composed_frame else 'green'

                    for h_idx in range(max_vla_horizon):
                        if h_idx < horizon:
                            delta = deltas[delta_idx, h_idx]
                            delta_scaled = delta * 1.0
                            target = current_pos + delta_scaled
                            trace = go.Scatter3d(
                                x=[current_pos[0], target[0]], y=[current_pos[1], target[1]], z=[current_pos[2], target[2]],
                                mode='lines', line=dict(color='black', width=1), opacity=0.5, showlegend=False, hoverinfo='skip'
                            )
                            frame_data.append(trace)
                            # Build w-value label for composed samples
                            w_label = ''
                            if is_composed_frame and w_values is not None and frame_idx < len(w_values) and w_values[frame_idx] is not None:
                                w_arr = w_values[frame_idx]
                                if delta_idx < len(w_arr):
                                    w_label = f'<br>W: {float(w_arr[delta_idx]):.4f}'
                            if critic_values_for_chunk is not None and h_idx < len(critic_values_for_chunk):
                                selected_marker = ' [SELECTED]' if action_is_selected else ''
                                hover_template = f'Source: {source_label}{selected_marker}<br>Index: {delta_idx}<br>Horizon: {h_idx}{w_label}<br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<br>z: %{{z:.4f}}<br>Q-value: {critic_values_for_chunk[h_idx]:.4f}<extra></extra>'
                            else:
                                hover_template = f'Source: {source_label}<br>Index: {delta_idx}<br>Horizon: {h_idx}{w_label}<br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<br>z: %{{z:.4f}}<extra></extra>'
                            marker_color = 'red' if action_is_selected else base_color
                            marker_size = 6 if action_is_selected else 4
                            trace = go.Scatter3d(
                                x=[target[0]], y=[target[1]], z=[target[2]],
                                mode='markers', marker=dict(size=marker_size, color=marker_color),
                                opacity=0.9 if action_is_selected else 0.7,
                                name=f'{source_label} Actions' if (frame_idx == 0 and delta_idx == 0 and h_idx == 0) else None,
                                showlegend=(frame_idx == 0 and delta_idx == 0 and h_idx == 0),
                                hovertemplate=hover_template
                            )
                            frame_data.append(trace)
                            current_pos = target
                        else:
                            frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='black', width=1), opacity=0, showlegend=False, hoverinfo='skip'))
                            frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='blue'), opacity=0, showlegend=False, hoverinfo='skip'))
                else:
                    for h_idx in range(max_vla_horizon):
                        frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='black', width=1), opacity=0, showlegend=False, hoverinfo='skip'))
                        frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='blue'), opacity=0, showlegend=False, hoverinfo='skip'))

            # QAM actor action deltas (yellow dots with lines)
            deltas_qam = qam_deltas[frame_idx] if (frame_idx < len(qam_deltas) and qam_deltas[frame_idx] is not None) else None
            num_qam = deltas_qam.shape[0] if deltas_qam is not None else 0
            qam_horizon = deltas_qam.shape[1] if deltas_qam is not None else 0

            for delta_idx in range(max_qam_actions):
                if delta_idx < num_qam:
                    current_pos = arrow_origin.copy()
                    action_is_selected = False
                    critic_values_for_chunk = None
                    if critic_values_episode is not None and frame_idx < len(critic_values_episode) and critic_values_episode[frame_idx] is not None:
                        critic_vals = critic_values_episode[frame_idx]
                        if delta_idx < len(critic_vals['qam_values']):
                            critic_values_for_chunk = critic_vals['qam_values'][delta_idx]
                            n_vla_actions = len(critic_vals['vla_values'])
                            action_is_selected = (critic_vals['selected_idx'] == n_vla_actions + delta_idx and critic_vals['selected_source'] == 'Actor')

                    for h_idx in range(max_qam_horizon):
                        if h_idx < qam_horizon:
                            delta = deltas_qam[delta_idx, h_idx]
                            delta_scaled = delta * 1.0
                            target = current_pos + delta_scaled
                            trace = go.Scatter3d(
                                x=[current_pos[0], target[0]], y=[current_pos[1], target[1]], z=[current_pos[2], target[2]],
                                mode='lines', line=dict(color='black', width=1), opacity=0.5, showlegend=False, hoverinfo='skip'
                            )
                            frame_data.append(trace)
                            if critic_values_for_chunk is not None:
                                selected_marker = ' [SELECTED]' if action_is_selected else ''
                                hover_template = f'Source: QAM Actor{selected_marker}<br>Index: {delta_idx}<br>Horizon: {h_idx}<br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<br>z: %{{z:.4f}}<br>Q-value: {critic_values_for_chunk:.4f}<extra></extra>'
                            else:
                                hover_template = f'Source: QAM Actor<br>Index: {delta_idx}<br>Horizon: {h_idx}<br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<br>z: %{{z:.4f}}<extra></extra>'
                            marker_color = 'red' if action_is_selected else 'yellow'
                            marker_size = 6 if action_is_selected else 4
                            trace = go.Scatter3d(
                                x=[target[0]], y=[target[1]], z=[target[2]],
                                mode='markers', marker=dict(size=marker_size, color=marker_color),
                                opacity=0.9 if action_is_selected else 0.7,
                                name='QAM Actions' if (frame_idx == 0 and delta_idx == 0 and h_idx == 0) else None,
                                showlegend=(frame_idx == 0 and delta_idx == 0 and h_idx == 0),
                                hovertemplate=hover_template
                            )
                            frame_data.append(trace)
                            current_pos = target
                        else:
                            frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='black', width=1), opacity=0, showlegend=False, hoverinfo='skip'))
                            frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='yellow'), opacity=0, showlegend=False, hoverinfo='skip'))
                else:
                    for h_idx in range(max_qam_horizon):
                        frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='black', width=1), opacity=0, showlegend=False, hoverinfo='skip'))
                        frame_data.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=4, color='yellow'), opacity=0, showlegend=False, hoverinfo='skip'))

            frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

        for trace in frames[0].data:
            fig.add_trace(trace)

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers', marker=dict(size=6, color='red'),
            name='Selected Action', showlegend=True, hoverinfo='skip'
        ))

        fig.frames = frames

        if has_images:
            image_html_parts = []
            for idx, img in enumerate(sim_images):
                pil_img = PILImage.fromarray(img.astype('uint8'))
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_html_parts.append(f'<img id="sim_img_{idx}" src="data:image/png;base64,{img_str}" style="display:none; width:100%; height:auto;" />')
            image_container_html = '\n'.join(image_html_parts)

        has_cp_plots = cp_plot_images is not None and len(cp_plot_images) > 0
        if has_cp_plots:
            if len(cp_plot_images) != len(positions):
                if len(cp_plot_images) > len(positions):
                    cp_plot_images = cp_plot_images[:len(positions)]
                else:
                    while len(cp_plot_images) < len(positions):
                        cp_plot_images.append(cp_plot_images[-1])
            cp_plot_html_parts = []
            for idx, img in enumerate(cp_plot_images):
                pil_img = PILImage.fromarray(img.astype('uint8'))
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                cp_plot_html_parts.append(f'<img id="cp_img_{idx}" src="data:image/png;base64,{img_str}" style="display:none; width:100%; height:auto;" />')
            cp_plot_container_html = '\n'.join(cp_plot_html_parts)

        layout_dict = dict(
            title='3D Action Visualization - Interactive',
            height=700,
            updatemenus=[dict(
                type='buttons', showactive=False,
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]),
                    dict(label='Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
                ],
                direction='left', pad=dict(r=10, t=87), x=0.1, xanchor='right', y=0, yanchor='top'
            )],
            sliders=[dict(
                active=0, yanchor='top', y=0.02, xanchor='left', x=0.1, len=0.9,
                currentvalue=dict(prefix='Step: ', visible=True, xanchor='right'),
                steps=[dict(
                    args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))],
                    label=str(k), method='animate'
                ) for k, f in enumerate(fig.frames)]
            )]
        )
        layout_dict['scene'] = dict(
            xaxis=dict(range=x_range, title='X'),
            yaxis=dict(range=y_range, title='Y'),
            zaxis=dict(range=z_range, title='Z'),
            aspectmode='cube'
        )
        fig.update_layout(layout_dict)

        output_path = Path(output_path)
        html_path = output_path.parent / (output_path.stem + '.html')
        try:
            if has_images:
                if has_cp_plots:
                    cp_plot_section = '''
                <h3>Failure Prediction</h3>
                <div id="cp-plot-display">
                    <img id="current-cp-plot" src="" />
                </div>'''
                    cp_plot_hidden_container = f'''
        <div style="display: none;">
            {cp_plot_container_html}
        </div>'''
                    cp_plot_update_function = '''
            function updateCPPlot(frameNum) {{
                var img = document.getElementById('cp_img_' + frameNum);
                var display = document.getElementById('current-cp-plot');
                if (img && display) {{ display.src = img.src; }}
            }}'''
                    cp_plot_init = "updateCPPlot(0);"
                else:
                    cp_plot_section = ""
                    cp_plot_hidden_container = ""
                    cp_plot_update_function = ""
                    cp_plot_init = ""

                # Build JS array of per-frame instructions
                instructions_list = instructions if instructions is not None else []
                # Pad/trim to match positions length
                while len(instructions_list) < len(positions):
                    instructions_list.append('')
                instructions_list = instructions_list[:len(positions)]
                instructions_js_array = json.dumps(instructions_list)

                fig_dict = fig.to_dict()
                combined_html = f'''<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>3D Action Visualization with Simulation View</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
            #instruction-display {{ margin-bottom: 10px; padding: 8px 12px; background: #eef4ff; border-left: 4px solid #4a7eff; font-size: 14px; border-radius: 4px; }}
            #instruction-display strong {{ margin-right: 6px; color: #333; }}
            #instruction-text {{ color: #1a3a7a; font-style: italic; }}
            .container {{ display: flex; gap: 20px; height: 700px; }}
            .plot-container {{ flex: 0 0 73%; }}
            #plotly-div {{ width: 100%; height: 100%; }}
            .image-container {{ flex: 0 0 25%; display: flex; flex-direction: column; gap: 10px; }}
            .image-container h3 {{ margin: 0 0 5px 0; font-size: 14px; }}
            #sim-image-display {{ flex: 0 0 {('45%' if has_cp_plots else '100%')}; display: flex; align-items: center; justify-content: center; border: none; background: #f5f5f5; padding: 0; }}
            #sim-image-display img {{ width: 100%; height: 100%; object-fit: contain; margin: 0; padding: 0; }}
            #cp-plot-display {{ flex: 0 0 45%; display: flex; align-items: center; justify-content: center; border: none; background: #f5f5f5; padding: 0; }}
            #cp-plot-display img {{ width: 100%; height: 100%; object-fit: contain; margin: 0; padding: 0; }}
        </style>
    </head>
    <body>
        <h1>3D Action Visualization - Interactive</h1>
        <div id="instruction-display"><strong>Instruction:</strong><span id="instruction-text"></span></div>
        <div class="container">
            <div class="plot-container">
                <div id="plotly-div"></div>
            </div>
            <div class="image-container">
                <h3>Simulation View</h3>
                <div id="sim-image-display">
                    <img id="current-sim-image" src="" />
                </div>{cp_plot_section}
            </div>
        </div>
        <div style="display: none;">
            {image_container_html}
        </div>{cp_plot_hidden_container}
        <script>
            var figData = {json.dumps(fig_dict['data'])};
            var figLayout = {json.dumps(fig_dict['layout'])};
            var figFrames = {json.dumps(fig_dict.get('frames', []))};
            var frameInstructions = {instructions_js_array};
            Plotly.newPlot('plotly-div', figData, figLayout, {{"responsive": true}}).then(function(gd) {{
                if (figFrames.length > 0) {{ Plotly.addFrames('plotly-div', figFrames); }}
            }});
            function updateSimImage(frameNum) {{
                var img = document.getElementById('sim_img_' + frameNum);
                var display = document.getElementById('current-sim-image');
                if (img && display) {{ display.src = img.src; }}
            }}
            function updateInstruction(frameNum) {{
                var el = document.getElementById('instruction-text');
                if (el && frameInstructions.length > frameNum) {{ el.textContent = frameInstructions[frameNum]; }}
            }}
            {cp_plot_update_function}
            updateSimImage(0);
            updateInstruction(0);
            {cp_plot_init}
            var plotDiv = document.getElementById('plotly-div');
            plotDiv.on('plotly_animating', function() {{
                var activeStep = plotDiv.layout.sliders[0].active;
                updateSimImage(activeStep);
                updateInstruction(activeStep);
                {('updateCPPlot(activeStep);' if has_cp_plots else '')}
            }});
            plotDiv.on('plotly_sliderchange', function(data) {{
                updateSimImage(data.step.index);
                updateInstruction(data.step.index);
                {('updateCPPlot(data.step.index);' if has_cp_plots else '')}
            }});
            plotDiv.on('plotly_animated', function() {{
                var activeStep = plotDiv.layout.sliders[0].active;
                updateSimImage(activeStep);
                updateInstruction(activeStep);
                {('updateCPPlot(activeStep);' if has_cp_plots else '')}
            }});
        </script>
    </body>
    </html>'''
                with open(html_path, 'w') as f:
                    f.write(combined_html)
            else:
                fig.write_html(str(html_path))
            print(f"Saved interactive 3D visualization HTML at {html_path}")
        except Exception as e:
            print(f"Error saving 3D visualization: {e}")


# =========================================================================================
# Failure Detection Plot
# =========================================================================================

def create_failure_prediction_plot(
    cumulative_prob: float,
    cumulative_probs_episode: list,
    cp_band: np.ndarray,
    is_failure: bool,
    t: int,
    max_steps: int
) -> np.ndarray:
    """Create a plot frame for CP band and cumulative failure probabilities.

    Returns:
        plot_img: RGB image array of the plot
    """

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    if is_failure:
        fig.patch.set_edgecolor('red')
        fig.patch.set_linewidth(10)
        ax.set_facecolor('#ffebee')  # Light red background

    cp_band_steps = np.arange(len(cp_band))
    ax.plot(cp_band_steps, cp_band, color='green', linewidth=2, label='CP Band')
    ax.fill_between(cp_band_steps, 0, cp_band, color='green', alpha=0.2)

    actual_timesteps = np.arange(len(cumulative_probs_episode))
    ax.plot(actual_timesteps, cumulative_probs_episode, color='blue', linewidth=2,
            label='Failure Probs', marker='o', markersize=2)

    ax.set_xlabel('Time step $t$', fontsize=12)
    ax.set_ylabel('Score Threshold $s_t$', fontsize=12)
    title_prefix = '⚠️ FAILURE DETECTED - ' if is_failure else ''
    ax.set_title(f'{title_prefix}Failure Prediction (t={t})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max_steps)
    y_max = max(1.0, np.max(cp_band) * 1.1 if len(cp_band) > 0 else 1.0)
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plot_img = plot_img[:, :, :3]  # RGBA → RGB
    plt.close(fig)

    return plot_img




# =========================================================================================
# Main Evaluation Function
# =========================================================================================

# =========================================================================================
# Video Writers
# =========================================================================================

def _write_cp_plots_mp4(raw_data, path, cp_band, max_steps):
    """Render all failure-prediction plot frames from raw (prob, is_fail, t) tuples and write MP4.
    Reuses a single matplotlib figure across all frames to avoid per-frame figure overhead.
    """
    fig = matplotlib.figure.Figure(figsize=(10, 6), dpi=160)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    writer = imageio.get_writer(path, fps=30)
    probs_so_far = []
    cp_band_steps = np.arange(len(cp_band))
    y_max = max(1.0, float(np.max(cp_band)) * 1.1 if len(cp_band) > 0 else 1.0)
    for cumprob, is_fail, t_val in raw_data:
        probs_so_far.append(cumprob)
        ax.clear()
        if is_fail:
            fig.patch.set_edgecolor('red')
            fig.patch.set_linewidth(10)
            ax.set_facecolor('#ffebee')
        else:
            fig.patch.set_edgecolor('none')
            ax.set_facecolor('white')
        ax.plot(cp_band_steps, cp_band, color='green', linewidth=2, label='CP Band')
        ax.fill_between(cp_band_steps, 0, cp_band, color='green', alpha=0.2)
        ax.plot(np.arange(len(probs_so_far)), probs_so_far, color='blue', linewidth=2,
                label='Failure Probs', marker='o', markersize=2)
        ax.set_xlabel('Time step $t$', fontsize=12)
        ax.set_ylabel('Score Threshold $s_t$', fontsize=12)
        title_prefix = '⚠️ FAILURE DETECTED - ' if is_fail else ''
        ax.set_title(f'{title_prefix}Failure Prediction (t={t_val})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, y_max)
        fig.tight_layout()
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        writer.append_data(frame)
    writer.close()

