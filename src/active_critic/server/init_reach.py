import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_dummy_vec_env, make_vec_env, parse_sampled_transitions, DummyExtractor, new_epoch_reach, sample_new_episode
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
import argparse
from datetime import datetime

from gym import Env
th.manual_seed(0)


def make_wsm_setup(seq_len, d_output, device='cuda'):
    wsm = WholeSequenceModelSetup()
    wsm.model_setup = ModelSetup()
    seq_len = seq_len
    d_output = d_output
    wsm.model_setup.d_output = d_output
    wsm.model_setup.nhead = 32
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


def make_acps(seq_len, extractor, new_epoch, device, batch_size=32):
    acps = ActiveCriticPolicySetup()
    acps.device = device
    acps.epoch_len = seq_len
    acps.extractor = extractor
    acps.new_epoch = new_epoch
    acps.opt_steps = 0
    acps.optimisation_threshold = 0.95
    acps.inference_opt_lr = 5e-2
    acps.optimize = True
    acps.batch_size = 32
    acps.opt_end = True
    acps.stop_opt = True
    acps.clip = False
    return acps


def setup_ac_reach(seq_len, num_cpu, device):
    seq_len = seq_len
    env, expert = make_vec_env('reach', num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = WholeSequenceModel(wsm_critic_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env, expert


def make_acl(device, id):
    device = device
    acla = ActiveCriticLearnerArgs()
    date = datetime.today().strftime('%Y-%m-%d')
    acla.data_path = f'/data/bing/hendrik/reach_0_imitation_vanilla/{date}_id:{id}/'
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = False
    acla.logname = 'reach 0 imitation'
    acla.tboard = True
    acla.batch_size = 32
    acla.val_every = 1
    acla.add_data_every = 1
    acla.validation_episodes = 20
    acla.training_epsiodes = 1
    acla.actor_threshold = 5e-1
    acla.critic_threshold = 5e-1
    acla.num_cpu = 20
    acla.num_training_samples = 300

    seq_len = 100
    epsiodes = 30
    ac, acps, env, expert = setup_ac_reach(seq_len=seq_len, num_cpu=min(acla.training_epsiodes, acla.num_cpu), device=device)
    eval_env, expert = make_vec_env('reach', num_cpu=acla.num_cpu, seq_len=seq_len)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=eval_env, network_args_obj=acla)
    return acl, env, expert, seq_len, epsiodes, device


def run_experiment_init_reach(device):
    for id in [i for i in range(5)]:
        acl, env, expert, seq_len, epsiodes, device = make_acl(device=device, id=id)
        acl.train(epochs=10000)
