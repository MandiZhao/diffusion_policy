import gym
import dill
import torch
import numpy as np
from gym import spaces
from collections import defaultdict, deque
from diffusion_policy.gym_util.multistep_wrapper import (
    repeated_space, 
    # stack_last_n_obs, 
    aggregate, 
    # dict_take_last_n
)
from dexmachina.distill import DistillEnvWrapper
"""
Multi-step wrapper for dexmachina RL environments.
"""
# def torch_aggregate(data, method='max'):
#     if method == 'max':
#         # equivalent to any
#         return torch.max(data)
#     elif method == 'min':
#         # equivalent to all
#         return torch.min(data)
#     elif method == 'mean':
#         return torch.mean(data)
#     elif method == 'sum': 
#         return torch.sum(data)
#     else:
#         raise NotImplementedError()

def torch_stack_last_n_obs(all_obs, n_steps):
    """
    (B, num_obs_steps, obs_shape)
    """
    assert(len(all_obs) > 0)
    all_obs = list(all_obs) 
    result = torch.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs)) 
    result[start_idx:] = torch.stack(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    # result = result.permute(1,0,2)  # (B, num_obs_steps, obs_shape)
    result = result.permute(1, 0, *range(2, len(result.shape)))
    return result

def torch_dict_take_last_n(x, n):
    def take_last_n_torch(x, n):
        x = list(x)
        n = min(len(x), n)
        return np.array(x[-n:].cpu().numpy())
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n_torch(value, n)
    return result

class DexmachinaMultiStepWrapper:
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None, 
        ): 
        """
        Handled stepping multiple times in the environment.
        """
        self.env = env
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps 
        self.n_obs_steps = n_obs_steps 
        self.obs = deque(maxlen=n_obs_steps+1)
        self.num_envs = env.num_envs
        self.rewards = torch.zeros((self.num_envs, ), dtype=torch.float32) # cumulative rewards
        self.dones = torch.zeros((self.num_envs, ), dtype=torch.bool)
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs.""" 
        obs = self.env.reset()  
        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.rewards[:] = 0.0
        self.dones[:] = False
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (B, n_action_steps,) + action_shape
        """
        act_idx = action.shape[1]
        for i in range(act_idx):
            act = action[:, i, :] 
            obs, reward, done, info = self.env.step(act)
            self.obs.append(obs)  
            self.dones[:] = self.dones | done # some might be done earlier than other threads
            # accumulate rewards only for the envs that are not done
            self.rewards[~self.dones] += reward[~self.dones]
            # print(f'action_idx: {i+1}/{act_idx}, reward: {reward} done: {done} | cum reward: {self.rewards} | cum dones: {self.dones}')
            self._add_info({})

        observation = self._get_obs(self.n_obs_steps)
        # reward = aggregate(self.reward, self.reward_agg_method)
        # done = aggregate(self.done, 'max')
        reward = self.rewards 
        # return done only if all are done 
        done = self.dones.all().item()
        info = torch_dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """ 
        assert(len(self.obs) > 0)
        if isinstance(self.env.observation_space, spaces.Box):
            return torch_stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.env.observation_space, spaces.Dict):
            result = dict() 
            for key in self.env.observation_space.keys(): 
                result[key] = torch_stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
