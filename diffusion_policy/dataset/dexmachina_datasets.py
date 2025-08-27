import copy
import torch
import numpy as np
from typing import Dict, List, Optional
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
            SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.noise_adder import StateNoiseAdder, DepthImageAugmentor

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
            val = sample[key]
            if val.dtype != np.float32:
                val = np.asarray(val, dtype=np.float32) # (can avoid copy)
            val = val.reshape(val.shape[0], -1)  # Flatten last dim if needed
            obs.append(val)  # Append flattened observation
            del sample[key]  # Remove from sample to avoid duplication
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
        camera_keys: List[str] = ['imgs/front_960/depth'],
        state_keys: List[str] = ['robot/dof_pos', 'robot/kpt_pos', 'object/part_pos', 'object/part_quat', 'task/kpt_dists'],
        action_key: str = 'actions',
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        n_obs_steps: int = 1,
        n_action_steps: int = 1,
        ):

        super().__init__()

        self.camera_keys = camera_keys
        self.state_keys = state_keys
        self.action_key = action_key

        # Build required keys list
        required_keys = state_keys + [action_key] + camera_keys
        print(f"Loading replay buffer with keys: {required_keys} | {zarr_path}")
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
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.get_noise_adder()

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
        for key in self.camera_keys:
            if 'rgb' in key:
                normalizer[key] = get_image_range_normalizer()
            elif 'depth' in key:
                depth_normalizer = SingleFieldLinearNormalizer.create_fit(
                    data=self.replay_buffer[key], mode='limits',
                    output_min=-1, output_max=1
                )
                normalizer[key] = depth_normalizer

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        obs contains both low-dim state and image observations.
        """
        obs_dict = {}
        state_dict = {}
        T_slice = slice(self.n_obs_steps)
        # Add low-dim observations
        for key in self.state_keys:
            obs_data = sample[key][T_slice]
            if obs_data.dtype != np.float32:
                obs_data = np.asarray(obs_data, dtype=np.float32)
            if len(obs_data.shape) > 2:  # Flatten if needed
                obs_data = obs_data.reshape(obs_data.shape[0], -1)
            state_dict[key] = obs_data
            del sample[key]
        obs_dict.update(state_dict)  # Add all state keys to obs_dict

        action = sample[self.action_key]
        if action.dtype != np.float32:
            action = np.asarray(action, dtype=np.float32)
        # Add image observations for each camera
        for cam_key in self.camera_keys:
            image = sample[cam_key][T_slice]
            # NEW(0818): handled img cap & /255 during data collection, saved data as channel first
            if 'depth' in cam_key:
                image = np.repeat(image, repeats=3, axis=1) # repeat (T, 1, H, W) to (T, 3, H, W)
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            obs_dict[cam_key] = image
            del sample[cam_key]  # Remove from sample to avoid duplication

        data = {
            'obs': obs_dict,
            'action': action,
        }

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        for key in self.camera_keys:
            if 'depth' in key:
                torch_data['obs'][key] = self.depth_augmentor(torch_data['obs'][key])

        for key in self.state_keys:
            torch_data['obs'][key] = self.state_noise_adder(torch_data['obs'][key])

        return torch_data

    def get_noise_adder(self):
        self.state_noise_adder = StateNoiseAdder(std=0.1) # was 0.02
        self.depth_augmentor = DepthImageAugmentor(
            noise_std=0.01, dropout_prob=0.7, patch_size=(10,20), num_patches=10
            )
