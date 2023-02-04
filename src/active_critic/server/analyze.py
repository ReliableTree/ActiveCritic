import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_dummy_vec_env, make_vec_env, parse_sampled_transitions, sample_expert_transitions, DummyExtractor, ReductiveExtractor, new_epoch_reach, sample_new_episode
from active_critic.utils.pytorch_utils import make_part_obs_data, count_parameters
from active_critic.utils.dataset import DatasetAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from active_critic.utils.dataset import DatasetAC
from active_critic.model_src.whole_sequence_model import (
    WholeSequenceModelSetup, WholeSequenceModel, CriticSequenceModel)
from active_critic.model_src.transformer import (
    ModelSetup, generate_square_subsequent_mask)
from active_critic.policy.active_critic_policy import ActiveCriticPolicySetup, ActiveCriticPolicy
import argparse
from prettytable import PrettyTable
from sb3_contrib import TQC

from gym import Env
th.manual_seed(0)




def make_wsm_setup(seq_len, d_output, device='cuda'):
    wsm = WholeSequenceModelSetup()
    wsm.model_setup = ModelSetup()
    seq_len = seq_len
    d_output = d_output
    wsm.model_setup.d_output = d_output
    wsm.model_setup.nhead = 8
    wsm.model_setup.d_hid = 200
    wsm.model_setup.d_model = 200
    wsm.model_setup.nlayers = 5
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout = 0.2
    wsm.lr = 1e-4
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.AdamW
    wsm.optimizer_kwargs = {}
    return wsm



def make_acps(seq_len, extractor, new_epoch, device, batch_size=32):
    acps = ActiveCriticPolicySetup()
    acps.device = device
    acps.epoch_len = seq_len
    acps.extractor = extractor
    acps.new_epoch = new_epoch
    acps.opt_steps = 100
    acps.optimisation_threshold = 1
    acps.inference_opt_lr = 5e-3
    
    acps.optimize = True
    acps.batch_size = 32
    acps.stop_opt = True
    acps.clip = False
    return acps


def setup_ac(seq_len, num_cpu, device, tag):
    seq_len = seq_len
    env, expert = make_vec_env(tag, num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device)
    acps = make_acps(
        seq_len=seq_len, extractor=ReductiveExtractor(), new_epoch=new_epoch_reach, device=device)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = CriticSequenceModel(wsm_critic_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env, expert


def make_acl(device):
    device = device
    acla = ActiveCriticLearnerArgs()
    acla.data_path = '/data/bing/hendrik/'
    acla.device = device
    acla.extractor = ReductiveExtractor()
    acla.imitation_phase = True
    tag = 'push'
    acla.logname = tag + ' 20 imitation'
    acla.tboard = True
    acla.batch_size = 16
    number = 10
    acla.val_every = 200
    acla.add_data_every = 200
    acla.validation_episodes = 200
    acla.training_epsiodes = 10
    acla.actor_threshold = 1e-3
    acla.critic_threshold = 1e-4
    acla.num_cpu = 50
    acla.compute_steps = 20000

    seq_len = 200
    epsiodes = 30
    ac, acps, env, expert = setup_ac(seq_len=seq_len, num_cpu=min(acla.num_cpu, acla.training_epsiodes), device=device, tag=tag)
    eval_env, expert = make_vec_env(tag, num_cpu=acla.num_cpu, seq_len=seq_len)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=eval_env, network_args_obj=acla)
    return acl, env, expert, seq_len, epsiodes, device


def run_experiment_analyze(device):
    acl, env, expert, seq_len, epsiodes, device = make_acl(device)

    acl.add_training_data(policy=expert.predict, episodes=20, add_to_actor=True)
    #acl.run_validation()
    acl.train(epochs=100000)

if __name__ == '__main__':
    run_experiment_analyze(device='cuda')
