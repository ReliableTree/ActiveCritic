import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_dummy_vec_env, make_vec_env, parse_sampled_transitions, sample_expert_transitions, DummyExtractor, new_epoch_reach, sample_new_episode
from active_critic.utils.pytorch_utils import make_part_obs_data, count_parameters, get_steps_from_actions
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
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout = 0
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
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout = 0
    wsm.lr = 1e-4
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.AdamW
    wsm.optimizer_kwargs = {'weight_decay':weight_decay}
    return wsm

def make_wsm_setup_tiny(seq_len, d_output, weight_decay, device='cuda'):
    wsm = WholeSequenceModelSetup()
    wsm.model_setup = ModelSetup()
    seq_len = seq_len
    d_output = d_output
    wsm.model_setup.d_output = d_output
    wsm.model_setup.nhead = 1
    wsm.model_setup.d_hid = 1
    wsm.model_setup.d_model = 1
    wsm.model_setup.nlayers = 1
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout =0
    wsm.lr = 1e-4
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.AdamW
    wsm.optimizer_kwargs = {'weight_decay':weight_decay}
    return wsm



def make_acps(seq_len, extractor, new_epoch, device, opt_mode, opt_steps):
    acps = ActiveCriticPolicySetup()
    acps.device = device
    acps.epoch_len = seq_len
    acps.extractor = extractor
    acps.new_epoch = new_epoch
    acps.optimisation_threshold = 1
    if opt_mode == 'actor':
        acps.inference_opt_lr = 1e-6
        acps.opt_steps = 100
    elif opt_mode == 'plan':
        acps.inference_opt_lr = 1e-3
        acps.opt_steps = 100
    elif opt_mode == 'actions':
        acps.inference_opt_lr = 1e-3
        acps.opt_steps = 5
    elif opt_mode == 'goal':
        acps.inference_opt_lr = 1e-3
        acps.opt_steps = 5
    elif opt_mode == 'actor+plan':
        acps.inference_opt_lr = 1e-5
        acps.opt_steps = opt_steps
    else:
        1/0

    acps.optimize = True
    acps.stop_opt = True
    acps.clip = True

    acps.optimizer_mode = opt_mode
    return acps


def setup_ac(seq_len, num_cpu, device, tag, weight_decay, opt_mode, training_episodes, opt_steps, sparse):
    env, expert = make_vec_env(tag, num_cpu, seq_len=seq_len, sparse=sparse)
    d_output = env.action_space.shape[0]
    d_plan = 1
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device, weight_decay=weight_decay)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device, weight_decay=weight_decay)
    
    wsm_critic_setup.sparse = True

    wsm_planner_setup = make_wsm_setup_tiny(
        seq_len=seq_len, d_output=d_plan, weight_decay=weight_decay, device=device)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device, opt_mode=opt_mode, opt_steps=opt_steps)
    acps.buffer_size = 2*training_episodes
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = CriticSequenceModel(wsm_critic_setup)
    planner = WholeSequenceModel(wsm_planner_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, planner=planner, acps=acps)
    return ac, acps, env, expert


def make_acl(
        device, 
        env_tag, 
        data_path, 
        logname,  
        seq_len, 
        val_every, 
        add_data_every,
        imitation_phase, 
        total_training_epsiodes, 
        opt_mode, training_episodes, 
        min_critic_threshold, 
        weight_decay, 
        make_graphs,
        opt_steps,
        sparse,
        max_epoch_steps,
        explore_until,
        fast=False):
    device = device
    acla = ActiveCriticLearnerArgs()

    if fast:
        acla.val_every = val_every
        acla.add_data_every = add_data_every
        acla.validation_episodes = 2 #(*8)
        acla.validation_rep = 1
        acla.training_epsiodes = training_episodes
        acla.actor_threshold = 10
        acla.critic_threshold = 10
        acla.min_critic_threshold = min_critic_threshold
        acla.num_cpu = 2
    else:
        acla.val_every = val_every
        acla.add_data_every = add_data_every

        acla.validation_episodes = 20 
        acla.validation_rep = 1
        acla.training_epsiodes = training_episodes
        acla.actor_threshold = 1e-2
        acla.critic_threshold = 1e-2
        acla.min_critic_threshold = min_critic_threshold
        acla.num_cpu = 20

    acla.data_path = data_path
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = imitation_phase
    tag = env_tag
    acla.logname = tag + logname
    acla.tboard = True
    acla.batch_size = 16
    acla.make_graphs = make_graphs
    acla.explore_until = explore_until
    acla.total_training_epsiodes = total_training_epsiodes
    acla.start_critic = True
    acla.dense = True
    acla.max_epoch_steps = max_epoch_steps

    acla.use_pred_loss = True

    epsiodes = 30
    ac, acps, env, expert = setup_ac(
        seq_len=seq_len, 
        num_cpu=min(acla.num_cpu, acla.training_epsiodes), 
        device=device, 
        opt_mode=opt_mode, 
        tag=tag, 
        weight_decay=weight_decay, 
        training_episodes=acla.training_epsiodes,
        opt_steps=opt_steps,
        sparse=sparse
        )
    
    eval_env, expert = make_vec_env(tag, num_cpu=acla.num_cpu, seq_len=seq_len, sparse=sparse)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=eval_env, network_args_obj=acla)
    ac.write_tboard_scalar = acl.write_tboard_scalar
    return acl, env, expert, seq_len, epsiodes, device


def run_experiment(
        device, 
        data_path, 
        env_tag, 
        logname, 
        fast, 
        val_every, 
        add_data_every,
        opt_mode, 
        opt_steps,
        sparse,
        seq_len,
        max_epoch_steps,
        explore_until,
        weight_decay=1e-2, 
        demos=14, 
        make_graphs=False,
        imitation_phase=False, 
        total_training_epsiodes=20, 
        training_episodes=10, 
        min_critic_threshold=1e-4):

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
                            add_data_every = add_data_every,
                            opt_mode=opt_mode,
                            make_graphs=make_graphs,
                            opt_steps=opt_steps,
                            fast=fast,
                            sparse=sparse,
                            explore_until=explore_until,
                            max_epoch_steps=max_epoch_steps)    
    acl.network_args.num_expert_demos = demos
    if demos > 0:
        
        actions, observations, rewards, _, expected_rewards, _ = sample_new_episode(
            policy=expert,
            env=acl.env,
            dense=True,
            extractor=acl.network_args.extractor,
            device=acl.network_args.device,
            episodes=demos,
            seq_len=seq_len)
    
    
        exp_trjs = th.ones([actions.shape[0]], device=acl.network_args.device, dtype=th.bool)
        actions_history = actions.unsqueeze(1).repeat([1, actions.shape[1], 1, 1])
        print(rewards)
        acl.add_data(actions=actions[:demos], observations=observations[:demos], rewards=rewards[:demos], expert_trjs=exp_trjs[:demos], action_history=actions_history[:demos])
    acl.train(epochs=100000)


def run_eval_stats_env(device, weight_decay):
    imitation_phases = [False]
    demonstrations_list = [0]
    run_ids = [i for i in range(3)]
    s = datetime.today().strftime('%Y-%m-%d')
    training_episodes = 10
    total_training_epsiodes = 5000
    min_critic_threshold = 1e-5
    data_path = '/data/bing/hendrik/AC_var_' + s
    env_tags = ['windowopen']
    val_everys = [1000]
    add_data_everys = [1000]
    opt_modes = ['actor+plan']
    opt_steps_list = [3]
    sparse = True
    seq_len = 100
    max_epoch_steps = 15000
    manual_seed = 1
    explore_until = 100
    th.manual_seed(manual_seed)
    for demonstrations in demonstrations_list:
        for env_tag in env_tags:
            for im_ph in imitation_phases:
                for val_step, val_every in enumerate(val_everys):
                    for run_id in run_ids:
                        for opt_mode in opt_modes:
                            for opt_steps in opt_steps_list:
                                logname = f' high inf lr ms {manual_seed} trainin eps: {total_training_epsiodes} opt mode: {opt_mode} demonstrations: {demonstrations}, im_ph:{im_ph}, {training_episodes}, run id: {run_id}'
                                print(f'____________________________________logname: {logname}')
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
                                            add_data_every = add_data_everys[val_step],
                                            opt_mode=opt_mode,
                                            make_graphs = True,
                                            fast=False,
                                            opt_steps=opt_steps,
                                            sparse=sparse,
                                            seq_len=seq_len,
                                            max_epoch_steps=max_epoch_steps,
                                            explore_until=explore_until)

