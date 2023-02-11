import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_dummy_vec_env, make_vec_env, make_vec_env_pomdp, parse_sampled_transitions, sample_expert_transitions_rollouts, make_pomdp_rollouts, ReductiveExtractor, DummyExtractor, new_epoch_reach, sample_new_episode
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
from imitation.data import rollout

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
    acps.opt_steps = 5
    acps.optimisation_threshold = 1
    acps.inference_opt_lr = 5e-3
    acps.inference_opt_lr = 1e-2
    
    acps.optimize = True
    acps.batch_size = 32
    acps.stop_opt = False
    acps.clip = False
    return acps


def setup_ac(seq_len, num_cpu, device, tag, extractor):
    seq_len = seq_len
    env, expert = make_vec_env(tag, num_cpu, seq_len=seq_len)
    env_pomdp, _ = make_vec_env_pomdp(env_id=tag, num_cpu=num_cpu, seq_len=seq_len, lookup_freq=1000)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device)
    acps = make_acps(
        seq_len=seq_len, extractor=extractor, new_epoch=new_epoch_reach, device=device)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = CriticSequenceModel(wsm_critic_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env, expert, env_pomdp


def make_acl(device):
    device = device
    acla = ActiveCriticLearnerArgs()
    acla.data_path = '/data/bing/hendrik/'
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = False
    tag = 'pickplace'
    acla.logname = tag + ' reinit sparse POMDP'
    acla.tboard = True
    acla.batch_size = 16
    number = 10
    acla.val_every = 200
    acla.add_data_every = 200
    acla.validation_episodes = 200
    acla.training_epsiodes = 10
    acla.actor_threshold = 1e-2
    acla.critic_threshold = 1e-2
    acla.num_cpu = 10
    acla.patients = 200000
    acla.num_sampled_episodes = 400

    seq_len = 200
    epsiodes = 1000000
    ac, acps, env, expert, env_pomdp = setup_ac(seq_len=seq_len, num_cpu=min(acla.num_cpu, acla.training_epsiodes), device=device, tag=tag, extractor=acla.extractor)
    eval_env_pomdp, _ = make_vec_env_pomdp(env_id=tag, num_cpu=acla.num_cpu, seq_len=seq_len, lookup_freq=1000)

    acl = ActiveCriticLearner(ac_policy=ac, env=env_pomdp, eval_env=eval_env_pomdp, network_args_obj=acla)
    return acl, env, expert, seq_len, epsiodes, device


def run_experiment_analyze(device):
    number_expert_demonstrations = 10

    lookup_freq = 200
    acl, env, expert, seq_len, epsiodes, device = make_acl(device)
    acl.network_args.num_imitation = number_expert_demonstrations

    transitions, rollouts = sample_expert_transitions_rollouts(expert.predict, env, number_expert_demonstrations)
    pomdp_rollouts = make_pomdp_rollouts(rollouts, lookup_frq=lookup_freq, count_dim=10)
    pomdp_transitions = rollout.flatten_trajectories(pomdp_rollouts)


    datas = parse_sampled_transitions(
    transitions=pomdp_transitions, seq_len=seq_len, extractor=acl.network_args.extractor, device=device)
    
    device_data = []
    for data in datas:
        device_data.append(data.to(device))
    actions, observations, rewards = device_data
    print(f'avr reward: {rewards.mean()}')
    acl.add_data(
            actions=actions,
            observations=observations,
            rewards=rewards
        )
    #acl.add_training_data(policy=expert, episodes=10, seq_len=seq_len)
    #acl.run_validation()
    acl.train(epochs=100000)

if __name__ == '__main__':
    run_experiment_analyze(device='cuda')
