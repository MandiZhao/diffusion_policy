import numpy as np
import time
import torch
from collections import deque
import cv2, os
import h5py as h5
from diffusion_policy.env.murp.transform_utils import mat2pose, pose2mat
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import hydra
from torch.utils.data import DataLoader

Robot = SensorEnum = None


class MurpImageWrapper:
    def __init__(self,
        n_obs_steps=2,
        n_action_steps=8,
        device="cuda:0",
        robot_cfg:str="/home/kavit/fair/murp/core/murp/murp/config/robot/static_tmr_right.yaml",
        path_to_transform:str="base_T_right_base.npy",
        robot_operation_mode:str="JOINT_IMPEDANCE", # JOINT | POSE,
        fps = 15,
        shape_meta={},
        is_real_robot=True,
        # cfg=None,
        demo_path="data/teleop/datasets/avocado/mh/image_abs.hdf5",
        demo_index=0
        ):

        # self.reset_arm_pos = np.array([0.14936262, -0.65780519, -0.26952777, -2.65130757, 0.6578265, 2.40055512, 0.56525831])#np.array([0.14936262, -0.65780519, -0.26952777, -2.65130757, 0.6578265, 2.40055512, 0.56525831])
        self.is_real_robot = is_real_robot
        self.demo_path = demo_path
        self.demo_index = demo_index
        self.n_action_steps = n_action_steps
        # previous reset_arm from avocado set
        # self.reset_arm_pos = np.array([[-0.02356606, -1.00290606, -0.11727795, -2.47662257,  0.39520455, 1.79568671,  1.01451966]])
        self.reset_arm_pos = np.array([-0.02356606, -1.00290606, -0.11727795, -2.47662257,  0.39520455, 1.79568671,  1.01451966])
        # self.reset_hand_pos = np.zeros(16)
        self.shape_meta = shape_meta
        # self.reset_hand_pos = np.array([
        #     -1.77366529e-04, -1.22294221e-01,  1.79051511e-01,  2.39001397e-01,
        #     -1.41893223e-03, -1.95103181e-01,  1.83574357e-01,  2.50352855e-01,
        #     -1.59629876e-03, -2.13017201e-01,  1.31694647e-01,  2.02197843e-01,
        #     8.99957766e-01,  1.08681340e+00,  1.05178351e-01,  2.79352283e-01
        # ])
        self.reset_hand_pos = np.array([
            -1.77366529e-04, -1.22294221e-01,  1.79051511e-01,  2.39001397e-01,
            -1.41893223e-03, -1.95103181e-01,  1.83574357e-01,  2.50352855e-01,
            -1.59629876e-03, -2.13017201e-01,  1.31694647e-01,  2.02197843e-01,
            8.99957766e-01,  1.08681340e+00,  1.05178351e-01,  2.79352283e-01
        ])
        self.close_hand_joints = np.array([
            -0.00860228,  0.77766354,  0.74520547,  0.79451336, -0.00940043,
            0.56970129,  0.92026623,  1.23934862, -0.00416811,  0.57590912,
            0.83734738,  1.3024911 ,  1.504955  ,  1.18596129,  0.11537693,
        0.29895128])
        self.robot_cfg = robot_cfg
        self.max_steps = 500
        self.sleep_timer = 1/fps
        self.n_obs_steps = n_obs_steps
        self.device = device
        self.base_T_right_base = np.load(path_to_transform) if os.path.exists(path_to_transform) else None
        self.right_base_T_base = np.linalg.inv(self.base_T_right_base) if self.base_T_right_base is not None else None
        self.robot_operation_mode = robot_operation_mode
        self._init_dequeus(maxlen=self.n_obs_steps)
        self.demo_data = None
        self.apply_frame_transform = False
        for obs_key in self.shape_meta['obs']:
            if 'eef_pos' in obs_key:
                self.apply_frame_transform = True
                if 'right_base' in obs_key:
                    self.apply_frame_transform = False
        self.quat_obs_key_name = [obs_key for obs_key in self.shape_meta['obs'] if "quat" in obs_key]
        # self.cfg = cfg

    def _init_dequeus(self, maxlen=2):
        self.obs_queue_dict = {}
        for obs_key in self.shape_meta['obs']:
            self.obs_queue_dict[obs_key] = deque(maxlen=maxlen)

    def get_max_steps(self):
        if self.is_real_robot:
            return None
        else:
            self.init_robot()
            return self.episode_len // self.n_action_steps

    def init_robot(self):
        if self.is_real_robot:
            global Robot
            from murp.robot.robot import Robot
            st = time.perf_counter_ns()
            self.robot = Robot.construct_from_config(config_file=self.robot_cfg)
            et = time.perf_counter_ns()
            print(f"Robot init time: {((et - st) / 1_000_000):.3f} ms")
            time.sleep(2)
        elif self.demo_data is None:
            self.file = h5.File(self.demo_path, "r")
            self.demo_data = self.file['data'][f'demo_{self.demo_index}']
            self.action_key = self.file.attrs.get('action_key', 'actions')
            self.episode_len = len(self.demo_data['actions'])
        self.i = 0

    def reset_robot(self):
        if self.is_real_robot:
            print('resetting robot')
            # reset_arm_pos = np.array([0.14936262, -0.65780519, -0.26952777, -2.65130757, 0.6578265, 2.40055512, 0.56525831])
            reset_arm_pos = np.array(self.reset_arm_pos)
            reset_hand_pos = np.array(self.reset_hand_pos)
            st = time.perf_counter_ns()
            self.robot.right_arm.set_joint_positions(reset_arm_pos, interpolate=True)
            self.robot.right_arm.hand.set_joint_positions(reset_hand_pos)

            # TODO: set the left arm pose here too for resting!
            et = time.perf_counter_ns()
            print(f"set joint arm and hand t_d: {((et - st) / 1_000_000):.3f} ms")
            time.sleep(5)

    def init_camera(self):
        if self.is_real_robot:
            global SensorEnum
            from murp.common.constants import RobotEnum, SensorEnum, SensorType
            time.sleep(2)
            print('warming up camera')
            for i in range(100):
                rgbd_data = self.robot.perception.get_sensor_data(SensorEnum.View.HEAD)
                print(i, rgbd_data.rgb.image.shape)

    def reset(self):
        for _ in range(self.n_obs_steps):
            self.get_observation()  # warm up the camera


    def _resize_image(self, orig_image, slice_index:int=160, cv2_resize_shape=(320, 240)):
        return cv2.resize(orig_image[:, slice_index:, :], cv2_resize_shape).transpose((2, 0, 1))

    def process_depth_image(self, depth_image):
        depth_image = depth_image.astype(np.float32)
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        # convert from 0 to 1 to 0 to 255
        depth_image = (depth_image - np.min(depth_image)) / (
            np.max(depth_image) - np.min(depth_image)
        )
        depth_image = (depth_image * 255).astype(np.uint8)
        # breakpoint()
        return depth_image

    def get_observation(self):

        for obs_key in self.shape_meta['obs']:
            if self.is_real_robot:
                slice_index = 160
                is_rgb = 'image' in  obs_key
                is_depth = 'depth' in obs_key

                if 'front' in obs_key: 
                    print("WARNING: don't have the front camera on real yet")
                    is_depth = True
                    depth = self.robot.perception.get_sensor_data(SensorEnum.View.HEAD).depth.image
                    depth = np.stack([depth] * 3, axis=-1)

                if 'head_image' in obs_key or ('head' in obs_key and 'rgb' in obs_key): # for dexmachina sime env
                    rgb = self.robot.perception.get_sensor_data(SensorEnum.View.HEAD).rgb.image
                if 'torso_image' in obs_key or ('torso' in obs_key and 'rgb' in obs_key):
                    rgb = self.robot.perception.get_sensor_data(SensorEnum.View.TORSO).rgb.image
                if 'wrist_right_image' in obs_key:
                    rgb = self.robot.perception.get_sensor_data(SensorEnum.View.RIGHT_WRIST).rgb.image
                if 'head_depth' in obs_key or ('head' in obs_key and 'depth' in obs_key):
                    depth = self.robot.perception.get_sensor_data(SensorEnum.View.HEAD).depth.image
                    depth = np.stack([depth] * 3, axis=-1)
                if 'torso_depth' in obs_key or ('torso' in obs_key and 'depth' in obs_key):
                    depth = self.robot.perception.get_sensor_data(SensorEnum.View.TORSO).depth.image
                    depth = np.stack([depth] * 3, axis=-1)
                if 'wrist_right_depth' in obs_key:
                    depth = self.robot.perception.get_sensor_data(SensorEnum.View.RIGHT_WRIST).depth.image
                    depth = np.stack([depth] * 3, axis=-1)

                if 'eef_pos' in obs_key:
                    eef_pos_quat = self.robot.right_arm.get_end_effector_pose().copy()
                    eef_pos = eef_pos_quat[:3]
                    eef_quat = eef_pos_quat[3:] #xyzw
                    if self.apply_frame_transform:
                        #convert right base to base
                        eef_pos, eef_quat = mat2pose(self.base_T_right_base @ pose2mat((eef_pos, eef_quat)))
                    self.obs_queue_dict[obs_key].append(eef_pos)
                    self.obs_queue_dict[self.quat_obs_key_name[0]].append(eef_quat)
                if 'gripper_qpos' in obs_key:
                    hand_joints = self.robot.right_arm.hand.c().copy()
                    self.obs_queue_dict[obs_key].append(hand_joints)
                if 'arm_qpos' in obs_key:
                    arm_joints = self.robot.right_arm.get_joint_positions()
                    self.obs_queue_dict[obs_key].append(arm_joints)
                
                if 'robot/dof_pos' == obs_key:
                    # 46dim! left & right stacked -> TODO check joint order 
                    robot_dof = np.concatenate([
                        ent.get_joint_positions() for ent in \
                            [self.robot.left_arm, self.robot.left_arm.hand, self.robot.right_arm, self.robot.right_hand]
                    ])
                    self.obs_queue_dict[obs_key].append(robot_dof)

                if is_rgb:
                    self.obs_queue_dict[obs_key].append(self._resize_image(rgb, slice_index=slice_index))
                if is_depth: 
                    self.obs_queue_dict[obs_key].append(self.process_depth_image(self._resize_image(depth, slice_index=slice_index)))
            elif self.i < self.episode_len:
                obs = self.demo_data['obs'][obs_key][self.i]
                if "image" in obs_key:
                    obs = self._resize_image(obs, slice_index=0)
                self.obs_queue_dict[obs_key].append(obs)
            else:
                return

    def process_obs(self):
        np_obs_dict = {key:np.array(list(value)) for key, value in self.obs_queue_dict.items()}
        # device transfer
        obs_dict = dict_apply(np_obs_dict,
            lambda x: torch.from_numpy(x).unsqueeze_(0).to(
                device=self.device))
        return obs_dict

    def step(self, action):
        # x,y,z, 6d rotation, 16 gripper joints
        # convert from rotvec to quaternion wxyz
        start_time = time.time()
        correct_action = None
        if self.is_real_robot:
            print('got action shaped', action.shape)
            # rotmat_pose_in_base, gripper = action
            # if self.robot_operation_mode == "OSC_POSE":
            #     if self.apply_frame_transform:
            #         # convert the action predicted in base to right base
            #         ee_pos, ee_rot_xyzw = mat2pose(self.right_base_T_base @ rotmat_pose_in_base)
            #     else:
            #         ee_pos, ee_rot_xyzw = mat2pose(rotmat_pose_in_base)
            #     arm_action = np.concatenate((ee_pos, ee_rot_xyzw))
            #     self.robot.right_arm.set_end_effector_pose(target_pose=arm_action, interpolation_steps=None) # use default interpolation
            # else:
            #     arm_joint_states = rotmat_pose_in_base
            #     self.robot.right_arm.set_joint_positions(np.array(arm_joint_states).flatten(), interpolate=False)
            # if len(gripper) == 16:
            #     self.robot.right_arm.hand.set_joint_positions(target_positions=gripper)
            # else:
            #     #TODO : If gripper action is binary substitute with recorded closing gripper angles
            #     if gripper[0] > 0.5:
            #         self.robot.right_arm.hand.set_joint_positions(target_positions=self.close_hand_joints)
            #     else:
            #         self.robot.right_arm.hand.set_joint_positions(target_positions=self.reset_hand_pos)
            end_time = time.time()
            time.sleep(max(self.sleep_timer - (end_time - start_time), 0))
        elif self.i < self.episode_len:
            correct_action = self.demo_data[getattr(self, 'action_key', 'actions')][self.i]
        self.i += 1
        return correct_action


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
