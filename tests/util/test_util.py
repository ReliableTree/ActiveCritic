import unittest

import gym
import numpy as np
import torch as th
from ActiveCritic.metaworld.metaworld.envs import \
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from ActiveCritic.utils.gym_utils import make_policy_dict, new_epoch_reach
from ActiveCritic.utils.pytorch_utils import make_partially_observed_seq
from gym.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from ActiveCritic.utils.gym_utils import DummyExtractor


class TestUtils(unittest.TestCase):

    def test_make_partially_observed_seq(self):
        seq_len = 3
        d_out = 2
        d_in = 3
        acts_array_low = [0]*d_out
        acts_array_high = [1]*d_out
        action_space = gym.spaces.box.Box(
            np.array(acts_array_low), np.array(acts_array_high), (d_out,), float)
        obs = th.ones([2, 1, d_in], dtype=th.float, device='cuda')
        pos = make_partially_observed_seq(
            obs=obs, acts=None, seq_len=seq_len, act_space=action_space)
        expected_shape = list(obs.shape)
        expected_shape[1] = seq_len
        expected_shape[2] += action_space.shape[0]

        assert list(
            pos.shape) == expected_shape, f'Output shape is not as expected. Expected: {expected_shape}, but got {pos.shape}'
        assert th.all(pos[:, :, obs.shape[-1]:] ==
                      0), 'No action input, but some action fields are not zero.'
        assert th.equal(pos[:, :obs.shape[1], :obs.shape[2]],
                        obs), 'Observation not correctly inserted into sequence.'

        current_obs_len = 2
        obs = th.ones([2, current_obs_len, d_in],
                      dtype=th.float, device='cuda')
        acts = 2*th.ones([2, current_obs_len-1, d_out],
                         dtype=th.float, device='cuda')
        pos = make_partially_observed_seq(
            obs=obs, acts=acts, seq_len=seq_len, act_space=action_space)
        expected_shape = list(obs.shape)
        expected_shape[1] = seq_len
        expected_shape[2] += action_space.shape[0]
        assert list(
            pos.shape) == expected_shape, f'Output shape is not as expected. Expected: {expected_shape}, but got {pos.shape}'
        assert th.all(pos[:, acts.shape[1], obs.shape[-1]:] ==
                      0), 'Action that werent input are displayed.'
        assert th.equal(pos[:, :obs.shape[1], :obs.shape[2]],
                        obs), 'Observation not correctly inserted into sequence.'
        assert th.equal(pos[:, :acts.shape[1], obs.shape[2]:],
                        acts), 'Not all actions have been inserted correctly.'

    def test_new_epoch(self):
        policy_dict = make_policy_dict()
        env_tag = 'reach'
        max_episode_steps = 5
        re = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
        re._freeze_rand_vec = False
        timelimit = TimeLimit(env=re, max_episode_steps=max_episode_steps)
        dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(timelimit)])
        de = DummyExtractor()
        obs1 = de.forward(dv1.reset())
        self.assertFalse(new_epoch_reach(obs1, obs1), 'new_epoch cant diff the same input.')

        obs2_np, _,_,_ = dv1.step(np.array([[0,1,0,0]]))
        obs2 = de.forward(obs2_np)
        
        self.assertFalse(th.equal(obs2, obs1), 'new observation after step is same as old one.') 
        self.assertFalse(new_epoch_reach(obs1, obs2), 'Same epoch, different observation. Should not be a new epoch.') 
        new_obs = de.forward(dv1.reset())
        self.assertTrue(new_epoch_reach(obs1, new_obs), 'New epoch was not recognized.')