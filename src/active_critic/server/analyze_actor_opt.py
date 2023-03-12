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

from datetime import datetime


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
    wsm.model_setup.dropout = 0.5
    wsm.lr = 1e-4
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.AdamW
    wsm.optimizer_kwargs = {'weight_decay':weight_decay}
    return wsm

def make_wsm_setup_small(seq_len, d_output, weight_decay, device='cuda'):
    wsm = WholeSequenceModelSetup()
    wsm.model_setup = ModelSetup()
    seq_len = seq_len
    d_output = d_output
    wsm.model_setup.d_output = d_output
    wsm.model_setup.nhead = 1
    wsm.model_setup.d_hid = 64
    wsm.model_setup.d_model = 64
    wsm.model_setup.nlayers = 3
    wsm.model_setup.dropout = 0.3
    wsm.lr = 1e-4
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.AdamW
    wsm.optimizer_kwargs = {'weight_decay':weight_decay}
    return wsm



def make_acps(seq_len, extractor, new_epoch, device, n_bins, batch_size=32):
    acps = ActiveCriticPolicySetup()
    acps.device = device
    acps.epoch_len = seq_len
    acps.extractor = extractor
    acps.new_epoch = new_epoch
    acps.opt_steps = 20
    acps.optimisation_threshold = 1
    acps.inference_opt_lr = 1e-6
    
    acps.optimize = True
    acps.batch_size = 32
    acps.stop_opt = False
    acps.clip = True
    acps.optimizer_mode = 'actor'
    acps.n_bins = n_bins
    return acps


def setup_ac(seq_len, num_cpu, device, tag, weight_decay, n_bins):
    env, expert = make_vec_env(tag, num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device, weight_decay=weight_decay)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device, weight_decay=weight_decay)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device, n_bins=n_bins)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = CriticSequenceModel(wsm_critic_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env, expert


def make_acl(device, env_tag, data_path, logname,  seq_len, val_every, imitation_phase, n_bins, total_training_epsiodes, training_episodes, min_critic_threshold, weight_decay, fast=False):
    device = device
    acla = ActiveCriticLearnerArgs()
    acla.data_path = data_path
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = imitation_phase
    tag = env_tag
    acla.logname = tag + logname
    acla.tboard = True
    acla.batch_size = 16
    number = 10

    if fast:
        acla.val_every = val_every
        acla.add_data_every = val_every
        acla.validation_episodes = 2 #(*8)
        acla.validation_rep = 1
        acla.training_epsiodes = training_episodes
        acla.actor_threshold = 10
        acla.critic_threshold = 10
        acla.min_critic_threshold = min_critic_threshold
        acla.num_cpu = 2
    else:
        acla.val_every = val_every
        acla.add_data_every = val_every

        acla.validation_episodes = 15 #(*8)
        acla.validation_rep = 3
        acla.training_epsiodes = training_episodes
        acla.actor_threshold = 1e-2
        acla.critic_threshold = 1e-2
        acla.min_critic_threshold = min_critic_threshold
        acla.num_cpu = 15

    acla.patients = 40000
    acla.total_training_epsiodes = total_training_epsiodes
    acla.start_critic = False

    epsiodes = 30
    ac, acps, env, expert = setup_ac(seq_len=seq_len, num_cpu=min(acla.num_cpu, acla.training_epsiodes), device=device, tag=tag, weight_decay=weight_decay, n_bins=n_bins)
    eval_env, expert = make_vec_env(tag, num_cpu=acla.num_cpu, seq_len=seq_len)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=eval_env, network_args_obj=acla)
    return acl, env, expert, seq_len, epsiodes, device

def run_experiment(device, data_path, env_tag, logname, fast, val_every, n_bins, weight_decay=1e-2, demos=14, imitation_phase=False, total_training_epsiodes=20, training_episodes=10, min_critic_threshold=1e-4):
    seq_len = 100

    acl, env, expert, seq_len, epsiodes, device = make_acl(
                            device,
                            env_tag=env_tag,
                            data_path=data_path, 
                            seq_len=seq_len, 
                            logname=logname, 
                            imitation_phase=imitation_phase, 
                            total_training_epsiodes=total_training_epsiodes,
                            training_episodes=training_episodes,
                            min_critic_threshold=min_critic_threshold,
                            weight_decay=weight_decay,
                            val_every=val_every,
                            n_bins=n_bins,
                            fast=fast)    
    acl.network_args.num_expert_demos = demos

        
    actions, observations, rewards, _, expected_rewards = sample_new_episode(
        policy=expert,
        env=acl.env,
        extractor=acl.network_args.extractor,
        device=acl.network_args.device,
        episodes=20,
        seq_len=seq_len)

    acl.add_data(actions=actions[:demos], observations=observations[:demos], rewards=rewards[:demos])

    acl.train(epochs=100000)

def run_eval_env_actor_opt(device, weight_decay):
    imitation_phases = [False]
    demonstrations_list = [14]
    run_ids = [i for i in range(1)]
    s = datetime.today().strftime('%Y-%m-%d')
    training_episodes = 10
    total_training_epsiodes = 200
    min_critic_threshold = 5e-5
    data_path = '/data/bing/hendrik/AC_var_' + s
    env_tags = ['pickplace']
    val_everys = [20000]
    for demonstrations in demonstrations_list:
        for env_tag in env_tags:
            for im_ph in imitation_phases:
                for val_every in val_everys:
                    for run_id in run_ids:
                        logname = f'quantized actions: {demonstrations}, im_ph:{im_ph}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, wd: {weight_decay}, val_every: {val_every} run id: {run_id}'
                        run_experiment(device=device,
                                    env_tag=env_tag,
                                    logname=logname,
                                    data_path=data_path,
                                    demos=demonstrations,
                                    imitation_phase=im_ph,
                                    total_training_epsiodes=total_training_epsiodes,
                                    training_episodes=training_episodes,
                                    min_critic_threshold=min_critic_threshold,
                                    weight_decay = weight_decay,
                                    val_every=val_every,
                                    fast=False)

def run_eval_stats_env_actor_opt(device, weight_decay):
    imitation_phases = [False]
    demonstrations_list = [14]
    run_ids = [i for i in range(5)]
    s = datetime.today().strftime('%Y-%m-%d')
    training_episodes = 10
    total_training_epsiodes = 1000
    min_critic_threshold = 5e-5
    data_path = '/data/bing/hendrik/AC_var_' + s
    env_tags = ['pickplace']
    val_everys = [20000]
    n_bins = [50, 200, 1000]
    for demonstrations in demonstrations_list:
        for env_tag in env_tags:
            for im_ph in imitation_phases:
                for val_every in val_everys:
                    for bin in n_bins:
                        for run_id in run_ids:
                            logname = f' quantized actions n_bins: {n_bins}, {demonstrations}, im_ph:{im_ph}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, wd: {weight_decay}, val_every: {val_every} run id: {run_id}'
                            run_experiment(device=device,
                                        env_tag=env_tag,
                                        logname=logname,
                                        data_path=data_path,
                                        demos=demonstrations,
                                        imitation_phase=im_ph,
                                        total_training_epsiodes=total_training_epsiodes,
                                        training_episodes=training_episodes,
                                        min_critic_threshold=min_critic_threshold,
                                        weight_decay = weight_decay,
                                        val_every=val_every,
                                        n_bins=bin,
                                        fast=False)