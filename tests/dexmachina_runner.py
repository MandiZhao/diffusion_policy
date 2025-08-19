import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str('../diffusion_policy/config')
)
def main(cfg):

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir="tests/")
    action = torch.zeros((1, 8, 23))
    obs, reward, done, info = env_runner.env.step(action)
    model = hydra.utils.instantiate(cfg.policy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"Dataset created with {len(dataset)} samples")
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    model.to(device) # set everything to device, including normalizer
    runner_log, video_path = env_runner.run(model, epoch=0)
    breakpoint()


if __name__ == "__main__":
    import os
    import sys
    import hydra
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    print(ROOT_DIR  ) # this should be ${HOME}/diffusion_policy
    from diffusion_policy.common.pytorch_util import dict_apply
    main()
