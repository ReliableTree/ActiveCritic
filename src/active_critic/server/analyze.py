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
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout = 0.1
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
    wsm.model_setup.dropout = 0.1
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
    wsm.model_setup.dropout =0.1
    wsm.lr = 1e-4
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.AdamW
    wsm.optimizer_kwargs = {'weight_decay':weight_decay}
    return wsm



def make_acps(seq_len, extractor, new_epoch, device, opt_mode, batch_size=32):
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
        acps.inference_opt_lr = 1e-4
        acps.opt_steps = 100
    elif opt_mode == 'goal':
        acps.inference_opt_lr = 1e-3
        acps.opt_steps = 5
    elif opt_mode == 'actor+plan':
        acps.inference_opt_lr = 1e-6
        acps.opt_steps = 100
    else:
        1/0

    acps.optimize = True
    acps.batch_size = 32
    acps.stop_opt = True
    acps.clip = True

    acps.optimizer_mode = opt_mode
    return acps


def setup_ac(seq_len, num_cpu, device, tag, weight_decay, opt_mode):
    env, expert = make_vec_env(tag, num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    d_plan = 1
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device, weight_decay=weight_decay)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device, weight_decay=weight_decay)
    wsm_planner_setup = make_wsm_setup_small(
        seq_len=seq_len, d_output=d_plan, weight_decay=weight_decay, device=device)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device, opt_mode=opt_mode)
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
        fast=False):
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
    acla.make_graphs = make_graphs
    number = 10

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

        acla.validation_episodes = 15 
        acla.validation_rep = 3
        acla.training_epsiodes = training_episodes
        acla.actor_threshold = 1e-2
        acla.critic_threshold = 1e-2
        acla.min_critic_threshold = min_critic_threshold
        acla.num_cpu = 15

    acla.plan_decay = 0.1
    acla.patients = 40000
    acla.total_training_epsiodes = total_training_epsiodes
    acla.start_critic = True
    acla.train_inference = False

    epsiodes = 30
    ac, acps, env, expert = setup_ac(seq_len=seq_len, num_cpu=min(acla.num_cpu, acla.training_epsiodes), device=device, opt_mode=opt_mode, tag=tag, weight_decay=weight_decay)
    eval_env, expert = make_vec_env(tag, num_cpu=acla.num_cpu, seq_len=seq_len)
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
        weight_decay=1e-2, 
        demos=14, 
        make_graphs=False,
        imitation_phase=False, 
        total_training_epsiodes=20, 
        training_episodes=10, 
        min_critic_threshold=1e-4):
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
                            add_data_every = add_data_every,
                            opt_mode=opt_mode,
                            make_graphs=make_graphs,
                            fast=fast)    
    acl.network_args.num_expert_demos = demos

    if demos > 0:
        actions, observations, rewards, _, expected_rewards = sample_new_episode(
            policy=expert,
            env=acl.env,
            extractor=acl.network_args.extractor,
            device=acl.network_args.device,
            episodes=demos,
            seq_len=seq_len)
    
        exp_trjs = th.ones([actions.shape[0]], device=acl.network_args.device, dtype=th.bool)

        acl.add_data(actions=actions[:demos], observations=observations[:demos], rewards=rewards[:demos], expert_trjs=exp_trjs[:demos])

    acl.train_data.last_success_is_role = False

    acl.train(epochs=100000)

def run_eval(device):
    imitation_phases = [False]
    seq_lens = [100]
    demonstrations = [14,25]
    training_episodes_list = [10, 20]
    min_critic_thresholds = [5e-5]
    ids = [0,1,2,3,4]
    for seq_len in seq_lens:
        for demos in demonstrations:
            for min_critic_threshold in min_critic_thresholds:
                for training_episodes in training_episodes_list:
                    for imitation_phase in imitation_phases:
                        for id in ids:
                            if imitation_phase:
                                total_training_epsiodes = 50
                                logname = f'demonstrations: {demos}, training_episodes: {training_episodes}, min critic: {min_critic_threshold} BC'
                            else:
                                total_training_epsiodes = 200
                                logname = f'demonstrations: {demos}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, id: {id}'

                            acl, env, expert, seq_len, epsiodes, device = make_acl(
                                device, seq_len=seq_len, 
                                logname=logname, 
                                imitation_phase=imitation_phase, 
                                total_training_epsiodes=total_training_epsiodes,
                                training_episodes=training_episodes,
                                min_critic_threshold=min_critic_threshold,
                                fast=False)
                            acl.network_args.num_expert_demos = demos
                            acl.add_training_data(policy=expert, episodes=demos, seq_len=seq_len)
                            acl.train(epochs=100000)

def run_eval_stats(device, demos, weight_decay):
    imitation_phases = [True, False]
    demonstrations = demos
    run_ids = [0,1,2,3,4]
    training_episodes = 10
    total_training_epsiodes = 200
    min_critic_threshold = 5e-5
    data_path = '/data/bing/hendrik/AC_var_test_19'
    env_tags = ['push, reach, windowopen, pickplace']
    for env_tag in env_tags:
        for im_ph in imitation_phases:
            for run_id in run_ids:
                logname = f' demonstrations: {demonstrations}, im_ph:{im_ph}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, wd: {weight_decay}, run id: {run_id}'
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
                            fast=False)

def run_eval_stats_demos(device, weight_decay):
    imitation_phases = [True, False]
    demonstrations_list = [14, 14, 6, 6]
    run_ids = [i for i in range(5)]
    s = datetime.today().strftime('%Y-%m-%d')
    training_episodes = 10
    total_training_epsiodes = 200
    min_critic_threshold = 5e-5
    data_path = '/data/bing/hendrik/AC_var_' + s
    env_tags = ['pickplace', 'push', 'reach', 'windowopen']
    val_everys = [2000, 20000]
    for i in range(len(env_tags)):
        demonstrations = demonstrations_list[i]
        env_tag = env_tags[i]
        for im_ph in imitation_phases:
            for val_every in val_everys:
                for run_id in run_ids:
                    logname = f' demonstrations: {demonstrations}, im_ph:{im_ph}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, wd: {weight_decay}, val_every: {val_every} run id: {run_id}'
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


def run_eval_stats_pp(device, weight_decay):
    imitation_phases = [True]
    demonstrations_list = [8, 12, 16]
    run_ids = [i for i in range(5)]
    s = datetime.today().strftime('%Y-%m-%d')
    training_episodes = 10
    total_training_epsiodes = 200
    min_critic_threshold = 5e-5
    data_path = '/data/bing/hendrik/AC_var_' + s
    env_tags = ['pickplace']
    val_everys = [2000]
    for demonstrations in demonstrations_list:
        for env_tag in env_tags:
            for im_ph in imitation_phases:
                for val_every in val_everys:
                    for run_id in run_ids:
                        logname = f' demonstrations: {demonstrations}, im_ph:{im_ph}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, wd: {weight_decay}, val_every: {val_every} run id: {run_id}'
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

def run_eval_stats_env(device, weight_decay):
    imitation_phases = [False]
    demonstrations_list = [1]
    th.manual_seed(0)
    run_ids = [i for i in range(4)]
    s = datetime.today().strftime('%Y-%m-%d')
    training_episodes = 10
    total_training_epsiodes = 1000
    min_critic_threshold = 5e-5
    data_path = '/data/bing/hendrik/AC_var_' + s
    env_tags = ['windowopen']
    val_everys = [6000]
    add_data_everys = [6000]
    opt_modes = ['actor+plan']
    for run_id in run_ids:
        for demonstrations in demonstrations_list:
            for env_tag in env_tags:
                for im_ph in imitation_phases:
                    for val_step, val_every in enumerate(val_everys):
                        for opt_mode in opt_modes:
                            logname = f' history eps: {total_training_epsiodes} opt mode: {opt_mode} demonstrations: {demonstrations}, im_ph:{im_ph}, training_episodes: {training_episodes}, min critic: {min_critic_threshold}, wd: {weight_decay}, val_every: {val_every} run id: {run_id}'
                            print(f'____________________________________logname: {env_tag}  {logname}')
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
                                        make_graphs = True ,
                                        fast=False)

if __name__ == '__main__':
    run_eval(device='cuda')
