"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import random
import os
import pathlib
import click
import hydra
import torch
import dill
import shutil
# import wandb
import json
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.murp_env_runner import MurpImageRunner
import numpy as np

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-r', '--robot_operation_mode', required=True, default="JOINT_IMPEDANCE")
@click.option('-e', '--is_real_robot', required=True, type=lambda x: x.lower() == "true", default=False)
@click.option('-ov', '--override_action_deploy', required=True, type=lambda x: x.lower() == "true", default=False)
@click.option('-d', '--device', default='cuda:0')
@click.option('-s', '--max_steps', default=500)
@click.option('-ad', '--action_deploy', default=8)
@click.option('-o', '--output_dir', default="data/outputs_eval")
def main(checkpoint, robot_operation_mode, is_real_robot, override_action_deploy, device, max_steps, action_deploy, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # cfg.policy.noise_scheduler.num_train_timesteps = 100
    cfg.n_action_steps = action_deploy if override_action_deploy else cfg.n_action_steps
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.n_action_steps = cfg.n_action_steps
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    print('policy loaded') 
    # run eval
    murp_config_path = os.environ.get("MURP_CONFIG_PATH")
    path_to_transform = os.environ.get("BASE_T_RIGHT_BASE", "base_T_right_base.npy")
    
    if not is_real_robot:
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        # configure validation dataset
        # with open("videos_sorted.json") as f:
        #     episode_filters = json.load(f)
        #     filter_episodes = np.concatenate([episode_filters["right"][:2]], axis=0)
        # val_dataset = dataset.get_validation_dataset(filter_episodes)
        val_set = val_dataset.train_mask.copy()
        val_indices = np.where(val_set)[0].tolist()
        if not len(val_indices):
            val_indices = [0]
        del val_dataset
        del dataset
        demo_path = cfg.task.dataset.dataset_path
    else:
        val_indices = [0]
        demo_path = ""
    try:
        shape_meta = cfg.task.shape_meta
    except:
        shape_meta = cfg.shape_meta
    abs_action = False 
    fitness = []
    for demo_index in val_indices:
        env_runner = MurpImageRunner(
            os.path.dirname(os.path.dirname(checkpoint)),
            device=device,
            robot_cfg=murp_config_path,
            path_to_transform=path_to_transform,
            robot_operation_mode=cfg.task.env_runner.get('robot_operation_mode', robot_operation_mode),
            n_obs_steps=cfg.task.dataset.n_obs_steps,
            n_action_steps=cfg.n_action_steps,
            shape_meta=shape_meta,
            fps=30,
            abs_action=abs_action,
            max_steps=max_steps,
            is_real_robot=is_real_robot,
            eval_threshold=cfg.training.get('eval_threshold', 0.05),
            folder_name='figures_eval',
            demo_path=demo_path,
            demo_index=demo_index
            )
        
        runner_log = env_runner.run(policy, epoch=-1)
        fitness.append(runner_log['rollout_fitness']) if 'rollout_fitness' in runner_log else None
    
    print(f"Average fitness {np.mean(fitness)}, {fitness=}")
    

if __name__ == '__main__':
    main()
