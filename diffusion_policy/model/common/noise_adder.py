
from typing import Union, Dict, Tuple

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin

class StateNoiseAdder(DictOfTensorMixin):
    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std

    def __call__(self, x: Union[Dict, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            return {k: self._add_noise(v) for k, v in x.items()}
        return self._add_noise(x)

    def _add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise

class DepthImageAugmentor(DictOfTensorMixin):
    def __init__(self, noise_std: float = 0.01, dropout_prob: float = 0.2, patch_size: Tuple[int, int] = (20, 20), num_patches: int = 5):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, x: Union[Dict, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            return {k: self._augment(v) for k, v in x.items()}
        return self._augment(x)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std
        inp_img = img.clone()
        if self.dropout_prob > 0 and torch.rand(1).item() < self.dropout_prob:
            img = self._apply_random_patches(img)
        return img

    def _apply_random_patches(self, img: torch.Tensor) -> torch.Tensor:
        """Apply zero patches to the image (img: [T, C, H, W] or [C, H, W])"""
        img_aug = img.clone()
        if img.ndim == 4:
            # Assume shape is [T, C, H, W]
            for t in range(img.shape[0]):
                img_aug[t] = self._drop_patches(img[t])
        elif img.ndim == 3:
            # [C, H, W]
            img_aug = self._drop_patches(img)
        else:
            raise ValueError(f"Unexpected depth image shape: {img.shape}")
        return img_aug

    def _drop_patches(self, single_img: torch.Tensor) -> torch.Tensor:
        C, H, W = single_img.shape
        ph, pw = self.patch_size

        for _ in range(self.num_patches):
            top = torch.randint(0, H - ph + 1, (1,)).item()
            left = torch.randint(0, W - pw + 1, (1,)).item()
            single_img[:, top:top+ph, left:left+pw] = 0.0

        return single_img
