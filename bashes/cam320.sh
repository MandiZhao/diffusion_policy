# conda activate robodiff310
export HYDRA_FULL_ERROR=1
wandb online
# EXP=cam320-nocrop
# python train.py --config-name=train_dexmachina_grasp_depth  \
#     data_name=cam320 \
#     task.env_runner.skip_env=0 \
#     training.rollout_every=50 training.checkpoint_every=100 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
#     horizon=36 n_action_steps=4 n_obs_steps=2 \
#     policy.obs_encoder.random_crop=False \
#     policy.obs_encoder.crop_shape=[240,320] \
#     training.num_epochs=800

EXP=img-noise-front-nocrop
python train.py --config-name=train_dexmachina_grasp_depth  \
    data_name=cam320 \
    task.env_runner.skip_env=0 \
    training.rollout_every=50 training.checkpoint_every=100 \
    task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
    dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP \
    horizon=36 n_action_steps=4 n_obs_steps=2 \
    policy.obs_encoder.random_crop=False \
    policy.obs_encoder.crop_shape=[240,320] \
    training.num_epochs=800
