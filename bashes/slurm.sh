#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --account=siro
#SBATCH --qos=h100_siro_high
#SBATCH --cpus-per-task=24
#SBATCH --job-name=devmachina
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

# Ensure slurms directory exists
mkdir -p slurms

# Enable debugging
set -x
set -e

# Set up environment
cd /checkpoint/siro/mandizhao/diffusion_policy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robodiff310

# Set environment variables
export WANDB_CACHE_DIR=/checkpoint/siro/mandizhao/.cache
export WANDB_CONFIG_DIR=/checkpoint/siro/mandizhao/.config
export WANDB_DATA_DIR=/checkpoint/siro/mandizhao/.cache/wandb-data
export WANDB_ARTIFACT_DIR=/checkpoint/siro/mandizhao/.artifacts

export HYDRA_FULL_ERROR=1

# EXP=cam2-ho16-b64-obs-dof
# python train.py --config-name=train_dexmachina_grasp_depth_2cam  \
#     data_name=ep500 \
#     task.env_runner.skip_env=0 \
#     training.rollout_every=100 training.checkpoint_every=100 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
#     horizon=16 n_action_steps=4 n_obs_steps=2 training.num_epochs=600  logging.mode=offline

# EXP=cam2-ho16-b128-obs-dof
# python train.py --config-name=train_dexmachina_grasp_depth_2cam  \
#     data_name=ep500 \
#     task.env_runner.skip_env=0 \
#     training.rollout_every=100 training.checkpoint_every=100 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=128  dataloader.num_workers=8 exp_name=$EXP  \
#     horizon=16 n_action_steps=4 n_obs_steps=2 training.num_epochs=600  logging.mode=offline


# EXP=head-torso-ho32-b64-crop
# python train.py --config-name=train_dexmachina_head_torso  \
#     data_name=ep500 \
#     task.env_runner.skip_env=0 \
#     training.rollout_every=50 training.checkpoint_every=100 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
#     horizon=32 n_action_steps=4 n_obs_steps=2 training.num_epochs=500  logging.mode=offline


# EXP=head-torso-ho32-b64-nocrop
# python train.py --config-name=train_dexmachina_head_torso  \
#     data_name=ep500 \
#     task.env_runner.skip_env=0 \
#     training.rollout_every=50 training.checkpoint_every=100 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
#     horizon=32 n_action_steps=4 n_obs_steps=2 training.num_epochs=500  \
#     logging.mode=offline policy.obs_encoder.random_crop=False policy.obs_encoder.crop_shape=[200,200]



# EXP=head-ho16-b64-nocrop
# python train.py --config-name=train_dexmachina_grasp_depth  \
#     data_name=ep500 \
#     task.env_runner.skip_env=0 \
#     training.rollout_every=50 training.checkpoint_every=100 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
#     horizon=16 n_action_steps=4 n_obs_steps=2 training.num_epochs=500 \
#     logging.mode=offline

EXP=head-ho16-b64-crop
python train.py --config-name=train_dexmachina_grasp_depth  \
    data_name=ep500 \
    task.env_runner.skip_env=0 \
    training.rollout_every=50 training.checkpoint_every=100 \
    task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
    dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
    horizon=16 n_action_steps=4 n_obs_steps=2 training.num_epochs=500 \
    logging.mode=offline \
    policy.obs_encoder.random_crop=True policy.obs_encoder.crop_shape=[200,200]
