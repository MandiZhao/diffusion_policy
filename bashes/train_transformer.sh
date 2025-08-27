export HYDRA_FULL_ERROR=1
wandb online
DATA=ep500
EXP=transformer-rgb-ho36-obs4 #front-kpts
python train.py --config-name=train_dexmachina_grasp_transformer \
	data_name=$DATA \
	task.env_runner.skip_env=0 \
	training.rollout_every=100 training.checkpoint_every=100 \
	task.env_runner.renderer=rasterizer task.env_runner.n_test=1 \
	dataloader.batch_size=64  dataloader.num_workers=8 exp_name=$EXP  \
	horizon=36 n_action_steps=4 n_obs_steps=4  \
	state_keys=['robot/dof_pos','robot/kpt_pos','object/part_pos','task/kpt_dists']
