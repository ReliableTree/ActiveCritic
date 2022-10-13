import unittest
import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_dummy_vec_env, make_policy_dict, parse_sampled_transitions, sample_expert_transitions \
    ,DummyExtractor, new_epoch_reach, sample_new_episode, make_policy_dict, TimeLimit, RolloutInfoWrapper, ImitationLearningWrapper, \
        make_vec_env
from active_critic.utils.pytorch_utils import make_part_obs_data, count_parameters
from active_critic.utils.dataset import DatasetAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from active_critic.utils.dataset import DatasetAC
from active_critic.model_src.whole_sequence_model import (
    WholeSequenceModelSetup, WholeSequenceModel)
from active_critic.model_src.transformer import (
    ModelSetup, generate_square_subsequent_mask)
from active_critic.policy.active_critic_policy import ActiveCriticPolicySetup, ActiveCriticPolicy
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np
from gym import Env

def make_wsm_setup(seq_len, d_output, device='cuda'):
    wsm = WholeSequenceModelSetup()
    wsm.model_setup = ModelSetup()
    seq_len = seq_len
    d_output = d_output
    wsm.model_setup.d_output = d_output
    wsm.model_setup.nhead = 1
    wsm.model_setup.d_hid = 512
    wsm.model_setup.d_model = 512
    wsm.model_setup.nlayers = 4
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout = 0
    wsm.lr = 1e-4
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.Adam
    wsm.optimizer_kwargs = {}
    return wsm


def make_acps(seq_len, extractor, new_epoch, batch_size=32):
    acps = ActiveCriticPolicySetup()
    acps.device = 'cuda'
    acps.epoch_len = seq_len
    acps.extractor = extractor
    acps.new_epoch = new_epoch
    acps.opt_steps = 100
    acps.optimisation_threshold = 0.95
    acps.inference_opt_lr = 5e-2
    acps.optimize = True
    acps.batch_size = 32
    return acps


def setup_ac_reach(seq_len, num_cpu):
    seq_len = seq_len
    env, expert = make_vec_env('reach', num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = WholeSequenceModel(wsm_critic_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env, expert


def make_acl():
    device = 'cuda'
    acla = ActiveCriticLearnerArgs()
    acla.data_path = '/home/hendrik/Documents/master_project/LokalData/TransformerImitationLearning/'
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = False
    acla.logname = 'reach_org_test'
    acla.tboard = True
    acla.batch_size = 32
    acla.val_every = 10
    acla.add_data_every = 1
    acla.validation_episodes = 10
    acla.training_epsiodes = 4
    acla.actor_threshold = 5e-2
    acla.critic_threshold = 5e-2
    acla.num_cpu = 5
    seq_len = 10
    ac, acps, env, expert = setup_ac_reach(seq_len=seq_len, num_cpu=acla.num_cpu)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, network_args_obj=acla)
    return acl, env, expert, seq_len, device

class TestVec(unittest.TestCase):
    def test_vec_env(self):
        acl, env, expert, seq_len, device = make_acl()
        self.assertTrue(len(acl.train_data) == 0)
        acl.add_training_data()
        self.assertTrue(len(acl.train_data) == acl.network_args.training_epsiodes * seq_len)
        opt_actions, gen_actions, observations, rewards, expected_rewards_before, expected_rewards_after = \
            sample_new_episode(acl.policy, env, episodes=acl.network_args.validation_episodes, return_gen_trj=True)
        self.assertTrue(list(opt_actions.shape) == [acl.network_args.validation_episodes, seq_len, env.action_space.shape[0]])
        self.assertTrue(list(gen_actions.shape) == [acl.network_args.validation_episodes, seq_len, env.action_space.shape[0]])
        self.assertTrue(list(observations.shape) == [acl.network_args.validation_episodes, seq_len, env.observation_space.shape[0]])
        self.assertTrue(list(rewards.shape) == [acl.network_args.validation_episodes, seq_len, 1])
        self.assertTrue(list(expected_rewards_before.shape) == [acl.network_args.validation_episodes, seq_len, acl.policy.critic.model.model_setup.d_output])
        self.assertTrue(list(expected_rewards_after.shape) == [acl.network_args.validation_episodes, seq_len, acl.policy.critic.model.model_setup.d_output])
        self.assertTrue(th.all(expected_rewards_after.mean(dim=[1,2]) > expected_rewards_before.mean(dim=[1,2])))

if __name__ == '__main__':
    unittest.main()