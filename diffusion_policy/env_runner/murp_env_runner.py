import numpy as np
import torch
import time
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.murp.murp_image_wrapper import MurpImageWrapper
from tqdm import tqdm
import os
import wandb
from diffusion_policy.common.plot import plot_image

class MurpImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self,
            output_dir,
            device="cuda:0",
            robot_cfg="/home/kavit/fair/murp/core/murp/murp/config/robot/static_tmr_right.yaml",
            path_to_transform:str="base_T_right_base.npy",
            robot_operation_mode:str="JOINT_IMPEDANCE", # JOINT_IMPEDANCE | OSC_POSE
            n_obs_steps=2,
            n_action_steps=8,
            shape_meta={},
            fps=15,
            abs_action=True,
            max_steps=500,
            is_real_robot=True,
            eval_threshold=0.05,
            folder_name="figures_eval",
            demo_path="",
            demo_index=0
        ):
        super().__init__(output_dir)
        self.is_real_robot = is_real_robot
        self.env = MurpImageWrapper(
                                    n_obs_steps,
                                    n_action_steps,
                                    device,
                                    robot_cfg=robot_cfg, path_to_transform=path_to_transform,
                                    robot_operation_mode=robot_operation_mode,
                                    fps=fps,
                                    shape_meta=shape_meta,
                                    is_real_robot=is_real_robot,
                                    demo_path=demo_path,
                                    demo_index=demo_index
                                )
        self.abs_action = abs_action
        self.robot_operation_mode = robot_operation_mode
        self.max_steps = max_steps if is_real_robot else self.env.get_max_steps()
        self.abs_action = abs_action #if self.is_real_robot else False
        self.eval_threshold = eval_threshold
        self.demo_index = demo_index
        self.folder_name = folder_name
        self.figure_plot_path = os.path.join(self.output_dir, self.folder_name)
        self.gripper_dims = 16 if shape_meta['action']['shape'][0] > 16 else 1
        if self.abs_action:
            from_rep = "matrix" if self.is_real_robot else "axis_angle"
            self.rotation_transformer = RotationTransformer(from_rep, 'rotation_6d')

    def run(self, policy: BaseImagePolicy, epoch=0):
        env = self.env
        env.init_robot()
        env.init_camera()
        env.reset_robot()
        env.reset()
        predicted_actions = list()
        correct_actions = list()
        for _ in tqdm(range(self.max_steps), total=self.max_steps):

            obs_dict = env.process_obs()

            # run policy
            with torch.no_grad():
                start_time = time.time()
                action_dict = policy.predict_action(obs_dict)
                # print(f"Time consuming {time.time() - start_time} secs")

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().squeeze(0).to('cpu').numpy())

            action = np_action_dict['action']
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")

            # step env
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)
            # breakpoint()
            action_start_time = time.time()
            obs_accumulate_at = [3*len(env_action)//4, len(env_action)-1]
            for a_idx, action in enumerate(env_action):
                predicted_actions.append(action)
                correct_action =  env.step(action)
                correct_actions.append(correct_action) if correct_action is not None else None
                env.get_observation() # if a_idx in obs_accumulate_at else None
            action_end_time = time.time()
            # print(f"Time taken for action execution {action_end_time - action_start_time}")

        if len(correct_actions):
            gts = np.array(correct_actions)
            preds = np.array(predicted_actions)
            # gts = self.transform_actions(gts) if self.robot_operation_mode == "OSC_POSE" else gts
            loss_pred = np.abs(preds - gts) # T X D
            loss_pred = loss_pred.reshape(-1) # T*D
            mask = loss_pred < self.eval_threshold
            count_below = mask.sum(axis=-1).astype(np.float32)
            percent_below = count_below / loss_pred.shape[-1]  # shape: [B]
            average_percent_below = np.mean(percent_below)
            print(f"Fit {average_percent_below}")
            step_log = {'rollout_fitness':0.0}
            step_log = plot_image(gts, name=f"demo_{self.demo_index}", pred=preds, step_log=step_log, wandb=wandb, folder_name=self.figure_plot_path)
            result_file_path = os.path.join(self.figure_plot_path, f"demo_{self.demo_index}_rollout_fitness.txt")
            with open(result_file_path, "a") as f:
                f.write(f"average_fit: {np.mean(average_percent_below):.2f}\n")
            step_log.update({'rollout_fitness': np.mean(average_percent_below)})
            return step_log
        return {}

    def undo_transform_action(self, action):
        d_rot = action.shape[-1] - self.gripper_dims
        uaction = None
        if self.robot_operation_mode == "OSC_POSE":
            pos = action[...,:3]
            rot = action[...,3:d_rot]
            grippers = action[...,d_rot:]
            rots = self.rotation_transformer.inverse(rot)
            if self.is_real_robot:
                rotmats = np.ones((rot.shape[0], 4, 4), dtype=np.float64)
                rotmats[:, :3, :3] = rots
                rotmats[:, :3, -1] = pos
                uaction = [(rotmat, gripper) for rotmat, gripper in zip(rotmats.astype(np.float32), grippers.astype(np.float32))]
            else:
                return np.concatenate([pos, rots, grippers], axis=-1)
        elif self.is_real_robot and self.robot_operation_mode == "JOINT_IMPEDANCE":
            arm_joints = action[..., :d_rot]
            grippers = action[...,d_rot:]
            uaction = [(arm_joint, gripper) for arm_joint, gripper in zip(arm_joints.astype(np.float64), grippers.astype(np.float32))]
            return uaction

        return action
