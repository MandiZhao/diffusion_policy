# NOTE need to manually change task.shape_meta to switch cameras!!
# conda activate robodiff310
export HYDRA_FULL_ERROR=1
wandb online
EXP=b64-dep-ddpm-crop-down256-time512 #front-kpts
DATA=ep500
python train.py --config-name=train_dexmachina_grasp_depth  \
    data_name=$DATA \
    task.env_runner.skip_env=0 \
    training.rollout_every=100 training.checkpoint_every=100 \
    task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
    dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
    horizon=36 n_action_steps=4 \
    state_keys=['robot/dof_pos','robot/kpt_pos','object/part_pos','task/kpt_dists'] \
    policy.down_dims=[256,512,1024] policy.obs_encoder.random_crop=True \
    policy.obs_encoder.crop_shape=[200,200] policy.diffusion_step_embed_dim=512

    # policy.obs_encoder.debug_zero_emb=True

    # policy.obs_encoder.random_crop=True policy.obs_encoder.crop_shape=[210,210]

    # dataloader.num_workers=0 val_dataloader.num_workers=0 # training.debug=1
    # dataloader.pin_memory=False dataloader.persistent_workers=False
    # state_keys=['robot/dof_pos','robot/kpt_pos','object/part_pos','task/kpt_dists']

    # state_keys=['robot/dof_pos','object/part_pos','task/kpt_dists']

# policy.obs_encoder.random_crop=True policy.obs_encoder.crop_shape=[200,200]

# - 'robot/dof_pos'
# - 'robot/kpt_pos'
# - 'object/part_pos'
# - 'object/part_quat'
# - 'task/kpt_dists'

# wandb offline
# EXP=2cam
# DATA=debug
# python train.py --config-name=train_dexmachina_grasp_depth_2cam  \
#     data_name=$DATA rl_run=no-vel-bbox-wrist7_box_ctrljoint_B6072_ho16 \
#     cam1_name=front_256 cam2_name=torso_256 \
#     task.env_runner.skip_env=0 \
#     training.rollout_every=500 training.checkpoint_every=500 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=64  exp_name=$EXP dataloader.num_workers=12 val_dataloader.num_workers=1


# wandb offline
# EXP=data0818
# python train.py --config-name=train_dexmachina_grasp_depth  \
#     data_name=debug rl_run=no-vel-bbox-wrist7_box_ctrljoint_B6072_ho16 \
#     camera_keys=[imgs/front_256/depth] task.env_runner.skip_env=0 \
#     training.rollout_every=2 training.checkpoint_every=2 \
#     task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#     dataloader.batch_size=32 exp_name=$EXP dataloader.num_workers=12 \
#     val_dataloader.num_workers=1 training.debug=1


# EXP=lowdim
# python train.py --config-name=train_dexmachina_grasp_lowdim  \
#    data_name=ep500 camera_keys=[] task.env_runner.skip_env=0 \
#    task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
#    dataloader.batch_size=128 exp_name=$EXP
