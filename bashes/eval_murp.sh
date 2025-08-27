#!/bin/bash
# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

# Conda activate your robot-skills conda env
conda activate murp_policy_diffusion

export HYDRA_FULL_ERROR=1

export MURP_CONFIG_PATH="/home/kavit/fair/murp/core/murp/murp/config/robot/static_tmr_right.yaml"
export BASE_T_RIGHT_BASE="base_T_right_base.npy"

IS_REAL_ROBOT=False
OPERATION_MODE="OSC_POSE" # deployment or training mode
CHECKPOINT_PATH="data/outputs/2025.08.01/horizon_20_control_mode_OSC_POSE_n_obs_2_right_object_transformer_binarized/checkpoints/latest.ckpt"
OVERRIDE_ACTION_DEPLOY=False
ACTION_DEPLOY=32

python eval_murp.py \
--checkpoint=$CHECKPOINT_PATH \
--robot_operation_mode=$OPERATION_MODE \
--is_real_robot=$IS_REAL_ROBOT \
--device='cuda:0' \
--max_steps=500 \
--override_action_deploy=$OVERRIDE_ACTION_DEPLOY \
--action_deploy=$ACTION_DEPLOY

#this is for robomimic
# IS_REAL_ROBOT=False
# OPERATION_MODE="OSC_POSE" # deployment or training mode
# CHECKPOINT_PATH="data/outputs/2025.06.10/18.43.54_train_diffusion_unet_image_can_image/checkpoints/epoch=0400-test_mean_score=0.960.ckpt"
# OVERRIDE_ACTION_DEPLOY=True
# ACTION_DEPLOY=8

# python eval_murp.py \
# --checkpoint=$CHECKPOINT_PATH \
# --robot_operation_mode=$OPERATION_MODE \
# --is_real_robot=$IS_REAL_ROBOT \
# --device='cuda:0' \
# --max_steps=500 \
# --override_action_deploy=$OVERRIDE_ACTION_DEPLOY \
# --action_deploy=$ACTION_DEPLOY
