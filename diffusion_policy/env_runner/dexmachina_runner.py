import os
import gym
import tqdm
import dill
import math
import torch
import wandb
import pathlib
import logging
import numpy as np
import collections
from gym import spaces
from copy import deepcopy
import multiprocessing as mp
import wandb.sdk.data_types.video as wv
from moviepy import ImageSequenceClip

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.dexmachina.dexmachina_wrapper import DexmachinaMultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder



module_logger = logging.getLogger(__name__)

import yaml
# from dexmachina.envs import GraspEnv
from dexmachina.distill import DistillEnvWrapper
from omegaconf import OmegaConf
import genesis as gs

class DexMachinaEnvRunner(BaseLowdimRunner):
    """
    Supports both
    """
    def __init__(self,
            output_dir,
            dataset_dir,
            n_latency_steps=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=280,
            n_obs_steps=2,
            n_action_steps=8,
            fps=12.5,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            abs_action=False,
            robot_noise_ratio=0.1,
            skip_env=False,
            vis_camera="front_960",
            vis_render_hw=(240,360),
            state_keys=[],
            camera_keys=[],
            env_kwargs_path="data_env.yaml",
            repeat_depth=True,
            renderer="batched", # or rasterizer
            device="cpu",
        ):
        super().__init__(output_dir)
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps
        self.video_fname = os.path.join(output_dir, "eval_video.mp4")
        self.state_keys = state_keys
        self.camera_keys = camera_keys
        if renderer == 'rasterizer':
            print("Using rasterizer renderer only supports num_envs=1")
            n_test = 1
        self.device = device
        def load_sim_env():
            # try loading env
            backend = gs.cuda if 'cuda' in device else gs.cpu
            gs.init(backend=backend)
            assert os.path.exists(env_kwargs_path), f"Env kwargs path {env_kwargs_path} does not exist."
            with open(env_kwargs_path, "r") as f:
                kwargs_clean = yaml.safe_load(f)
            rl_env_kwargs = {k: OmegaConf.create(v) for k, v in kwargs_clean['env_kwargs'].items()}

            wrapper_cfg = OmegaConf.create(kwargs_clean['wrapper_cfg'])
            wrapper_cfg.num_envs = n_test
            wrapper_cfg.state_keys = state_keys
            wrapper_cfg.camera_keys = camera_keys # overwrite this for img policy
            wrapper_cfg.renderer = renderer
            # ==== handle cameras ===
            # NOTE we assume each camera_key is a subset of the loaded config files from data_env.yaml!
            # optionally add another vis_camera for low-dim policy
            loaded_cam_kwargs = deepcopy(wrapper_cfg.camera_kwargs) # these are all the cameras used for data collection! assume the policy uses a subset of these

            use_cam_kwargs = dict()
            for key in camera_keys: # e.g. imgs/front_960/depth
                cam_name = key.split('/')[-2]
                assert cam_name in loaded_cam_kwargs, f"Camera {cam_name} not found in loaded camera kwargs: {list(loaded_cam_kwargs.keys())}"
                if not cam_name in use_cam_kwargs:
                    use_cam_kwargs[cam_name] = loaded_cam_kwargs[cam_name]

            if vis_camera is not None:
                kwargs = loaded_cam_kwargs.get(vis_camera, {})
                kwargs['res'] = vis_render_hw
                use_cam_kwargs[vis_camera] = kwargs

            wrapper_cfg.camera_kwargs = use_cam_kwargs
            wrapper_cfg.repeat_depth = repeat_depth
            # set device to cpu
            rl_env_kwargs['device'] = torch.device(device)
            uenv = DistillEnvWrapper(wrapper_cfg, rl_env_kwargs)

            env = DexmachinaMultiStepWrapper(
                uenv,
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps,
            )
            return uenv, env
        self.skip_env = skip_env
        self.uenv, self.env = None, None
        if not skip_env:
            self.uenv, self.env = load_sim_env()
        self.n_env = n_test
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.n_latency_steps = n_latency_steps
        self.vis_camera_key = None
        if vis_camera is not None: # optionally use rgb rendering for visualizing low-dim policy
            self.vis_camera_key = f"imgs/{vis_camera}/rgb"
        elif len(camera_keys) > 0: # use policy input camera for vis if not specified other cameras
            self.vis_camera_key = camera_keys[0]


    def run(self, policy: BaseLowdimPolicy, epoch: int):
        if self.skip_env:
            return {'test_mean_score': 0.0, 'reward_std': 0.0,}, None
        env = self.env
        # allocate data
        # just do one batch step
        obs = env.reset()
        past_action = None
        policy.reset()
        done = False
        video_frames = []
        step = 0
        while not done:
            if len(self.camera_keys) > 0:
                obs_dict = {}
                for key, vals in obs.items():
                    if self.vis_camera_key == key:
                        video_frames.append(vals[:,0].cpu().numpy())  # collect video frames in channel last!
                    if 'imgs' in key and key not in self.camera_keys: # ignore vis_camera
                        continue
                    vals = vals[:, :self.n_obs_steps].to(policy.device, dtype=policy.dtype)
                    if 'imgs' in key:
                        # move channel dim to first # (B, n_obs_steps, H, W, C) -> (B, n_obs_steps, C, H, W)
                        vals = vals.permute(0, 1, 4, 2, 3)
                    if 'rgb' in key:
                        vals = vals / 255.0  # normalize to [0, 1]
                    obs_dict[key] = vals  # (B, n_obs_steps, ...)
                del obs  # remove from obs to save memory
            else: # LowDim policy!
                obs_dict = {'obs': obs['state'][:, :self.n_obs_steps].to(policy.device, dtype=policy.dtype)}
                # additional rendering
                for k, v in obs.items():
                    if 'rgb' in k:
                        # video export assumes imgs are (B, n_obs_steps, H, W, C)
                        video_frames.append(v[:,0].cpu().numpy())  # collect video frames
            breakpoint()
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            # handle latency_steps, we discard the first n_latency_steps actions
            # to simulate latency
            action = action_dict['action'][:,self.n_latency_steps:] # (B, n_action_steps, action_dim)
            # send action to env runner's
            action = action.to(self.device)
            obs, reward, done, info = env.step(action)
            # print(f"step {step+1}/{env.max_episode_steps}, reward: {reward} done: {done}")
            past_action = action
            step += 1
        log_data = {
            'test_mean_score': torch.mean(reward).item(),
            'reward_std': torch.std(reward).item(),
            'reward_min': torch.min(reward).item(),
            'reward_max': torch.max(reward).item(),
        }
        fname = None
        if len(video_frames) > 0:
            frames = np.stack(video_frames, axis=0) # (T, B, H, W, C)
            concat_frames = []
            for t in range(frames.shape[0]):
                # Stack B environments side by side: (H, B*W, C)
                side_by_side = np.concatenate([frames[t, b] for b in range(frames.shape[1])], axis=1)
                concat_frames.append(side_by_side)

            # normalize depth
            is_depth = 'depth' in self.vis_camera_key
            if is_depth:
                depth_cap = 5
                frames_capped = [np.clip(frame, 0, depth_cap) for frame in concat_frames]
                depth_min, depth_max = np.min(frames_capped), np.max(frames_capped)
                concat_frames = [
                    (frame - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(frame)
                    for frame in frames_capped
                ]
                concat_frames = [
                    (frame * 255).astype(np.uint8) for frame in concat_frames
                ]
            clip = ImageSequenceClip(concat_frames, fps=30)
            fname = self.video_fname.replace(".mp4", f"_epoch{epoch}.mp4")
            clip.write_videofile(fname, logger=None)
            log_data['video_path'] = fname

        del obs
        del obs_dict
        del video_frames
        return log_data
