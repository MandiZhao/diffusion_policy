import copy
import torch
import numpy as np
from typing import Dict, List, Optional
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
            SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer   

class DexmachinaLowDimDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_path: str,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            state_keys: List = ['robot/dof_pos', 'robot/kpt_pos', 'object/part_pos', 'object/part_quat', 'task/kpt_dists'],
            action_key: str = 'actions', 
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None
            ):
        super().__init__()
        
        # Load replay buffer with required keys
        required_keys = state_keys + [action_key] 
            
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=required_keys)

        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )
        
        self.state_keys = state_keys
        self.action_key = action_key 
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer[self.action_key],
        }
        states = []
        for key in self.state_keys:
            vals = self.replay_buffer[key]
            if len(vals.shape) > 2: # flatten last dim, e.g. (T, n_kpts, 3) -> (T, n_kpts*3)
                vals = vals.reshape(vals.shape[0], -1)
            states.append(vals)
        obs = np.concatenate(states, axis=-1)  # Concatenate all observations along last dim
        data['obs'] = obs  # T, obs_dim (joint positions as observation)

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample): 
        action = sample[self.action_key].astype(np.float32)  # T, action_dim
        obs = []
        for key in self.state_keys:
            val = sample[key].astype(np.float32)
            val = val.reshape(val.shape[0], -1)  # Flatten last dim if needed
            obs.append(val)  # Append flattened observation 
        obs = np.concatenate(obs, axis=-1)  # Concatenate all observations along last dim
        data = {
            'obs': obs,  # T, obs_dim (joint positions as observation)
            'action': action,  # T, action_dim
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data 

class DexmachinaImgDataset(BaseImageDataset):
    def __init__(self,
        zarr_path: str, 
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        camera_names: List[str] = ['front_960'],
        state_keys: List[str] = ['robot/dof_pos', 'robot/kpt_pos', 'object/part_pos', 'object/part_quat', 'task/kpt_dists'],
        action_key: str = 'actions',
        use_rgb: bool = True,
        use_depth: bool = True,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None
        ):
        
        super().__init__()
        
        img_keys = []
        for cam_name in camera_names:
            if use_rgb:
                img_keys.append(f'imgs/{cam_name}/rgb')
            if use_depth:
                img_keys.append(f'imgs/{cam_name}/depth')
        
        self.camera_names = camera_names
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.state_keys = state_keys
        self.action_key = action_key
        
        # Build required keys list
        required_keys = state_keys + [action_key] + img_keys 
        
        # Load replay buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=required_keys)
        
        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer[self.action_key],
        }
        
        # Add observation keys to normalizer
        for key in self.state_keys:
            obs_data = self.replay_buffer[key]
            if len(obs_data.shape) > 2:  # Flatten if multi-dimensional
                obs_data = obs_data.reshape(obs_data.shape[0], -1)
            data[key] = obs_data
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Add image normalizer for all cameras
        for cam_name in self.camera_names:
            if self.use_rgb:
                normalizer[f'image_{cam_name}_rgb'] = get_image_range_normalizer()
            if self.use_depth:
                normalizer[f'image_{cam_name}_depth'] = get_image_range_normalizer()
                
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Extract observations
        obs_dict = {}
        
        # Add low-dimensional observations
        for key in self.state_keys:
            obs_data = sample[key].astype(np.float32)
            if len(obs_data.shape) > 2:  # Flatten if needed
                obs_data = obs_data.reshape(obs_data.shape[0], -1)
            obs_dict[key] = obs_data
        
        # Add image observations for each camera
        for cam_name in self.camera_names:
            if self.use_rgb:
                # Convert from (T, H, W, C) to (T, C, H, W) and normalize to [0, 1]
                rgb_key = f'imgs/{cam_name}/rgb'
                rgb_image = np.moveaxis(sample[rgb_key], -1, 1) / 255.0
                obs_dict[rgb_key] = rgb_image.astype(np.float32)
            
            if self.use_depth:
                # Handle depth images - assuming they come as (T, H, W, 1)
                depth_key = f'imgs/{cam_name}/depth'
                depth_image = np.moveaxis(sample[depth_key], -1, 1)
                # Normalize depth if needed (this depends on your depth range)
                obs_dict[depth_key] = depth_image.astype(np.float32)

        data = {
            'obs': obs_dict,
            'action': sample[self.action_key].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
