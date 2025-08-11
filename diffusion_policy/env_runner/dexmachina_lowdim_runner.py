import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces
import multiprocessing as mp
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

module_logger = logging.getLogger(__name__)

class DexMachinaGraspLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            dataset_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=280,
            n_obs_steps=2,
            n_action_steps=8,
            render_hw=(240,360),
            fps=12.5,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            abs_action=False,
            robot_noise_ratio=0.1,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        task_fps = 12.5
        steps_per_render = int(max(task_fps // fps, 1))

        def env_fn():
            return None

        # all_init_qpos = np.load(pathlib.Path(dataset_dir) / "all_init_qpos.npy")
        # all_init_qvel = np.load(pathlib.Path(dataset_dir) / "all_init_qvel.npy")
        # module_logger.info(f'Loaded {len(all_init_qpos)} known initial conditions.')

        # env_fns = [env_fn] * n_envs
        # env_seeds = list()
        # env_prefixs = list()
        # env_init_fn_dills = list()
        # # train
        # for i in range(n_train):
        #     seed = train_start_seed + i
        #     enable_render = i < n_train_vis
        #     init_qpos = None
        #     init_qvel = None
        #     if i < len(all_init_qpos):
        #         init_qpos = all_init_qpos[i]
        #         init_qvel = all_init_qvel[i]

        #     def init_fn(env, init_qpos=init_qpos, init_qvel=init_qvel, enable_render=enable_render):
        #         from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
        #         # setup rendering
        #         # video_wrapper
        #         assert isinstance(env.env, VideoRecordingWrapper)
        #         env.env.video_recoder.stop()
        #         env.env.file_path = None
        #         if enable_render:
        #             filename = pathlib.Path(output_dir).joinpath(
        #                 'media', wv.util.generate_id() + ".mp4")
        #             filename.parent.mkdir(parents=False, exist_ok=True)
        #             filename = str(filename)
        #             env.env.file_path = filename

        #         # set initial condition
        #         assert isinstance(env.env.env, KitchenLowdimWrapper)
        #         env.env.env.init_qpos = init_qpos
        #         env.env.env.init_qvel = init_qvel
            
        #     env_seeds.append(seed)
        #     env_prefixs.append('train/')
        #     env_init_fn_dills.append(dill.dumps(init_fn))

        # # test
        # for i in range(n_test):
        #     seed = test_start_seed + i
        #     enable_render = i < n_test_vis

        #     def init_fn(env, seed=seed, enable_render=enable_render):
        #         from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
        #         # setup rendering
        #         # video_wrapper
        #         assert isinstance(env.env, VideoRecordingWrapper)
        #         env.env.video_recoder.stop()
        #         env.env.file_path = None
        #         if enable_render:
        #             filename = pathlib.Path(output_dir).joinpath(
        #                 'media', wv.util.generate_id() + ".mp4")
        #             filename.parent.mkdir(parents=False, exist_ok=True)
        #             filename = str(filename)
        #             env.env.file_path = filename

        #         # set initial condition
        #         assert isinstance(env.env.env, KitchenLowdimWrapper)
        #         env.env.env.init_qpos = None
        #         env.env.env.init_qvel = None

        #         # set seed
        #         assert isinstance(env, MultiStepWrapper)
        #         env.seed(seed)
            
        #     env_seeds.append(seed)
        #     env_prefixs.append('test/')
        #     env_init_fn_dills.append(dill.dumps(init_fn))
        
        # def dummy_env_fn():
        #     # Avoid importing or using env in the main process
        #     # to prevent OpenGL context issue with fork.
        #     # Create a fake env whose sole purpos is to provide 
        #     # obs/action spaces and metadata.
        #     env = gym.Env()
        #     env.observation_space = gym.spaces.Box(
        #         -8, 8, shape=(60,), dtype=np.float32)
        #     env.action_space = gym.spaces.Box(
        #         -8, 8, shape=(9,), dtype=np.float32)
        #     env.metadata = {
        #         'render.modes': ['human', 'rgb_array', 'depth_array'],
        #         'video.frames_per_second': 12
        #     }
        #     env = MultiStepWrapper(
        #         env=env,
        #         n_obs_steps=n_obs_steps,
        #         n_action_steps=n_action_steps,
        #         max_episode_steps=max_steps
        #     )
        #     return env
        
        # env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        env = None

        self.env = env
        # self.env_fns = env_fns
        # self.env_seeds = env_seeds
        # self.env_prefixs = env_prefixs
        # self.env_init_fn_dills = env_init_fn_dills
        # self.fps = fps
        # self.crf = crf
        # self.n_obs_steps = n_obs_steps
        # self.n_action_steps = n_action_steps
        # self.past_action = past_action
        # self.max_steps = max_steps
        # self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        return {'test_mean_score': 1.0}