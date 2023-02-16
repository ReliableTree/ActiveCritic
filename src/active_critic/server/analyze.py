import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_dummy_vec_env, make_vec_env, parse_sampled_transitions, sample_expert_transitions, DummyExtractor, new_epoch_reach, sample_new_episode
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
import os
import pickle
import numpy as np
from gym import Env
import random



def make_wsm_setup(seq_len, d_output, weight_decay, device='cuda'):
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
    wsm.optimizer_kwargs = {'weight_decay':weight_decay}
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


def setup_ac(seq_len, num_cpu, device, tag, weight_decay):
    env, expert = make_vec_env(tag, num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device, weight_decay=weight_decay)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device, weight_decay=weight_decay)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = CriticSequenceModel(wsm_critic_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env, expert


def make_acl(device, logname,  seq_len , imitation_phase, total_training_epsiodes, training_episodes, min_critic_threshold, data_path, weight_decay):
    device = device
    acla = ActiveCriticLearnerArgs()
    acla.data_path = data_path
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = imitation_phase
    tag = 'pickplace'
    acla.logname = tag + logname
    acla.tboard = True
    acla.batch_size = 16
    number = 10
    acla.val_every = 10000
    acla.add_data_every = 10000
    acla.validation_episodes = 25 #(*8)
    acla.validation_rep = 8
    acla.training_epsiodes = training_episodes
    acla.actor_threshold = 1e-2
    acla.critic_threshold = 1e-2
    acla.min_critic_threshold = min_critic_threshold
    acla.num_cpu = 25
    acla.patients = 40000
    acla.total_training_epsiodes = total_training_epsiodes
    acla.start_critic = False

    epsiodes = 30
    ac, acps, env, expert = setup_ac(seq_len=seq_len, num_cpu=min(acla.num_cpu, acla.training_epsiodes), device=device, tag=tag, weight_decay=weight_decay)
    eval_env, expert = make_vec_env(tag, num_cpu=acla.num_cpu, seq_len=seq_len)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=eval_env, network_args_obj=acla)
    return acl, env, expert, seq_len, epsiodes, device

def make_acl_fast(device, logname,  seq_len , imitation_phase, total_training_epsiodes, training_episodes, min_critic_threshold, data_path):
    device = device
    acla = ActiveCriticLearnerArgs()
    acla.data_path = data_path
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = imitation_phase
    tag = 'pickplace'
    acla.logname = tag + logname
    acla.tboard = True
    acla.batch_size = 16
    number = 10
    acla.val_every = 10000
    acla.add_data_every = 10000
    acla.validation_episodes = 10 #(*8)
    acla.validation_rep = 2
    acla.training_epsiodes = training_episodes
    acla.actor_threshold = 10
    acla.critic_threshold = 10
    acla.min_critic_threshold = min_critic_threshold
    acla.num_cpu = 10
    acla.patients = 40000
    acla.total_training_epsiodes = total_training_epsiodes
    acla.start_critic = False

    epsiodes = 30
    ac, acps, env, expert = setup_ac(seq_len=seq_len, num_cpu=min(acla.num_cpu, acla.training_epsiodes), device=device, tag=tag)
    eval_env, expert = make_vec_env(tag, num_cpu=acla.num_cpu, seq_len=seq_len)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=eval_env, network_args_obj=acla)
    return acl, env, expert, seq_len, epsiodes, device


def run_experiment(device, data_path, logname, weight_decay=1e-2, demos=14, imitation_phase=False, total_training_epsiodes=20, training_episodes=10, min_critic_threshold=1e-4, fast=False):
    seq_len = 100
    if fast:
        acl, env, expert, seq_len, epsiodes, device = make_acl_fast(
            device,
            data_path=data_path, 
            seq_len=seq_len, 
            logname=logname, 
            imitation_phase=imitation_phase, 
            total_training_epsiodes=total_training_epsiodes,
            training_episodes=training_episodes,
            min_critic_threshold=min_critic_threshold,
            weight_decay=weight_decay)    
    else:
        acl, env, expert, seq_len, epsiodes, device = make_acl(
                                device,
                                data_path=data_path, 
                                seq_len=seq_len, 
                                logname=logname, 
                                imitation_phase=imitation_phase, 
                                total_training_epsiodes=total_training_epsiodes,
                                training_episodes=training_episodes,
                                min_critic_threshold=min_critic_threshold,
                                weight_decay=weight_decay)    
    acl.network_args.num_expert_demos = demos

    path_to_expert_trjs = acl.network_args.data_path + '/demonstrations'

    if not os.path.exists(path_to_expert_trjs):
        if not os.path.exists(acl.network_args.data_path):
            os.makedirs(acl.network_args.data_path)
        
        actions, observations, rewards, _, expected_rewards = sample_new_episode(
            policy=expert,
            env=acl.env,
            extractor=acl.network_args.extractor,
            device=acl.network_args.device,
            episodes=20,
            seq_len=seq_len)
        
        print(f'expert success rate: {rewards.mean()}')
        
        actions = actions.detach().to('cpu')
        observations = observations.detach().to('cpu')
        rewards = rewards.detach().to('cpu')

        exp_dict = {
            'actions': actions,
            'observations': observations,
            'rewards': rewards
        }

        with open(path_to_expert_trjs, 'wb') as handle:
            pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(path_to_expert_trjs, 'rb') as handle:
            exp_dict = pickle.load(handle)
        print('loaded exp trjs')
    acl.add_data(actions=exp_dict['actions'][:demos], observations=exp_dict['observations'][:demos], rewards=exp_dict['rewards'][:demos])

    acl.train(epochs=100000)

def run_eval(device):
    #'/data/bing/hendrik/EvalAC_Fast_Incline'
    imitation_phases = [True, False]
    seq_lens = [100]
    demonstrations = [20, 30, 40]
    training_episodes_list = [1, 10]
    min_critic_thresholds = [1e-4, 5e-5]
    for seq_len in seq_lens:
        for demos in demonstrations:
            for min_critic_threshold in min_critic_thresholds:
                for training_episodes in training_episodes_list:
                    for imitation_phase in imitation_phases:
                        if imitation_phase:
                            total_training_epsiodes = 400
                            logname = f'seq_len: {seq_len}, demonstrations: {demos}, training_episodes: {training_episodes}, min critic: {min_critic_threshold} BC'
                        else:
                            total_training_epsiodes = 200
                            logname = f'seq_len: {seq_len}, demonstrations: {demos}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}'

                        acl, env, expert, seq_len, epsiodes, device = make_acl(
                            device, seq_len=seq_len, 
                            logname=logname, 
                            imitation_phase=imitation_phase, 
                            total_training_epsiodes=total_training_epsiodes,
                            training_episodes=training_episodes,
                            min_critic_threshold=min_critic_threshold)
                        acl.network_args.num_expert_demos = demos
                        acl.add_training_data(policy=expert, episodes=demos, seq_len=seq_len)
                        acl.train(epochs=100000)

def run_eval_stats(device):
    imitation_phases = [True, False]
    seq_lens = [100]
    demonstrations = [20, 30, 40]
    training_episodes_list = [1, 10]
    min_critic_thresholds = [1e-4, 5e-5]
    for seq_len in seq_lens:
        for demos in demonstrations:
            for min_critic_threshold in min_critic_thresholds:
                for training_episodes in training_episodes_list:
                    for imitation_phase in imitation_phases:
                        if imitation_phase:
                            total_training_epsiodes = 400
                            logname = f'seq_len: {seq_len}, demonstrations: {demos}, training_episodes: {training_episodes}, min critic: {min_critic_threshold} BC'
                        else:
                            total_training_epsiodes = 200
                            logname = f'seq_len: {seq_len}, demonstrations: {demos}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}'

                        acl, env, expert, seq_len, epsiodes, device = make_acl(
                            device, seq_len=seq_len, 
                            logname=logname, 
                            imitation_phase=imitation_phase, 
                            total_training_epsiodes=total_training_epsiodes,
                            training_episodes=training_episodes,
                            min_critic_threshold=min_critic_threshold)
                        acl.network_args.num_expert_demos = demos
                        acl.add_training_data(policy=expert, episodes=demos, seq_len=seq_len)
                        acl.train(epochs=100000)

def run_eval_stats(device, demos, weight_decay):
    imitation_phase = False
    demonstrations = demos
    run_ids = [0,1,2,3,4,6]
    training_episodes = 10
    total_training_epsiodes = 200
    min_critic_threshold = 5e-5
    data_path = '/data/bing/hendrik/AC_var_test_200'
    for run_id in run_ids:
        logname = f'demonstrations: {demonstrations}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, wd: {weight_decay}, run id: {run_id}'
        run_experiment(device=device,
                       logname=logname,
                       data_path=data_path,
                       demos=demonstrations,
                       imitation_phase=imitation_phase,
                       total_training_epsiodes=total_training_epsiodes,
                       training_episodes=training_episodes,
                       min_critic_threshold=min_critic_threshold,
                       weight_decay = weight_decay,
                       fast=False)




if __name__ == '__main__':
    run_eval(device='cuda')
