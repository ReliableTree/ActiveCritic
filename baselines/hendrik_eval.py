from active_critic.utils.gym_utils import (
    make_vec_env, 
    make_dummy_vec_env, 
    sample_expert_transitions_rollouts, 
    make_pomdp_rollouts, 
    make_dummy_vec_env_pomdp,
    get_avr_succ_rew_det
)
import gym
from stable_baselines3 import PPO
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance,  get_schedule_fn

from torch.utils.data import DataLoader
from imitation.algorithms.adversarial import gail 
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.ppo import MlpPolicy
from active_critic.utils.tboard_graphs import TBoardGraphs
from active_critic.model_src.transformer import PositionalEncoding
import os.path
from imitation.algorithms.adversarial.gail import GAIL


import copy

from active_critic.TQC.tqc import TQC
from active_critic.TQC.tqc_policy import TQCPolicyEval

def evaluate_learner(env_tag, logname, save_path, seq_len, n_demonstrations, bc_epochs, n_samples, device, bc_logname, learner:TQC=None):
    lookup_freq = 1000
    env, vec_expert = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    val_env, _ = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    transitions, rollouts = sample_expert_transitions_rollouts(vec_expert.predict, val_env, n_demonstrations)

    pomdp_rollouts = make_pomdp_rollouts(rollouts, lookup_frq=lookup_freq, count_dim=10)
    pomdp_transitions = rollout.flatten_trajectories(pomdp_rollouts)

    if learner is None:
        bc_learner = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=pomdp_transitions,
            device=device)
    else:
        bc_learner = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=pomdp_transitions,
            device=device,
            policy=learner.policy)

    pomdp_env_val, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
    if (not os.path.isfile(save_path + bc_logname + ' BC best')):
        tboard = TBoardGraphs(logname=logname + ' BC' , data_path='/data/bing/hendrik/gboard/')
        best_succes_rate = -1
        best_model = None
        runs_per_epoch = 20
        for i in range(bc_epochs):
            bc_learner.train(n_epochs=runs_per_epoch)
            success, rews = get_avr_succ_rew_det(env=pomdp_env_val, learner=bc_learner.policy, epsiodes=200)
            success_rate = success.mean()
            tboard.addValidationScalar('Reward', value=th.tensor(rews.mean()), stepid=i)
            tboard.addValidationScalar('Success Rate', value=th.tensor(success_rate), stepid=i)
            if success_rate > best_succes_rate:
                best_succes_rate = success_rate
                th.save(bc_learner.policy.state_dict(), save_path + bc_logname + ' BC best')
                print(save_path + logname + ' BC best')
    else:
        print('skipping BC')

    if learner is not None:

        tboard = TBoardGraphs(logname=logname + str(' Reinforcement') , data_path='/data/bing/hendrik/gboard/')
        learner.policy.load_state_dict(th.load(save_path + bc_logname + ' BC best'))


        success, rews = get_avr_succ_rew_det(env=pomdp_env_val, learner=learner.policy, epsiodes=200)
        tboard.addValidationScalar('Reloaded Success Rate', value=th.tensor(success.mean()), stepid=0)
        tboard.addValidationScalar('Reloaded Reward', value=th.tensor(rews.mean()), stepid=0)

        tboard.addValidationScalar('Reward', value=th.tensor(rews.mean()), stepid=learner.env.envs[0].reset_count)
        tboard.addValidationScalar('Success Rate', value=th.tensor(success.mean()), stepid=learner.env.envs[0].reset_count)

        while learner.env.envs[0].reset_count <= n_samples:
            print('before learn')
            learner.learn(2000)
            print('after learn')
            print(learner.env.envs[0].reset_count)
            success, rews = get_avr_succ_rew_det(env=pomdp_env_val, learner=learner.policy, epsiodes=200)
            success_rate = success.mean()
            tboard.addValidationScalar('Reward', value=th.tensor(rews.mean()), stepid=learner.env.envs[0].reset_count)
            tboard.addValidationScalar('Success Rate', value=th.tensor(success_rate), stepid=learner.env.envs[0].reset_count)

def run_eval_TQC(device, lr, demonstrations, seq_len):
    env_tag = 'pickplace'
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=1000)
    tqc_learner = TQC(policy='MlpPolicy', env=pomdp_env, device=device, learning_rate=lr)
    logname = 'TQC ' + f'lr: {lr} demonstrations: {demonstrations} seq_len: {seq_len}'
    bc_logname = 'TQC ' + f'demonstrations: {demonstrations} seq_len: {seq_len}'
    evaluate_learner(env_tag, logname=logname, save_path='/data/bing/hendrik/Evaluate Baseline/', seq_len=seq_len, n_demonstrations=demonstrations, bc_epochs=400, n_samples=400, device=device, learner=tqc_learner, bc_logname=bc_logname)
    

def run_eval_PPO(device, lr, demonstrations, seq_len):
    env_tag = 'pickplace'
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=1000)
    PPO_learner = PPO("MlpPolicy", pomdp_env, verbose=0, device=device, learning_rate=lr)
    logname = 'PPO ' + f'lr: {lr} demonstrations: {demonstrations} seq_len: {seq_len}'
    bc_logname = 'PPO ' + f'demonstrations: {demonstrations} seq_len: {seq_len}'
    evaluate_learner(env_tag, logname=logname, save_path='/data/bing/hendrik/Evaluate Baseline/', seq_len=seq_len, n_demonstrations=demonstrations, bc_epochs=400, n_samples=400, device=device, learner=PPO_learner, bc_logname=bc_logname)
    
def run_eval_BC(device):
    env_tag = 'pickplace'
    seq_len = 200
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=1000)
    evaluate_learner(env_tag, 'BC 10', save_path='/data/bing/hendrik/Evaluate Baseline/', seq_len=seq_len, n_demonstrations=10, bc_epochs=400, n_samples=400, device=device)

def run_tune_TQC(device):
    lr = 1e-3
    seq_lens = [50, 100, 200]
    for i in range(5):
        for seq_len in seq_lens:
            demonstrations = 6
            for j in range(4):
                demonstrations += 2
                run_eval_TQC(device=device, lr=lr, demonstrations=demonstrations, seq_len=seq_len)
        lr = lr * 0.6

def run_tune_PPO(device):
    lr = 1e-3
    seq_lens = [50, 100, 200]
    for i in range(5):
        for seq_len in seq_lens:
            demonstrations = 6
            for j in range(4):
                demonstrations += 2
                run_eval_PPO(device=device, lr=lr, demonstrations=demonstrations, seq_len=seq_len)
        lr = lr * 0.6


def evaluate_GAIL(env_tag, logname, seq_len, n_demonstrations, n_samples, learner, pomdp_env, save_path, bc_epochs, bc_logname, device):
    lookup_freq = 1000
    env, vec_expert = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    val_env, _ = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    transitions, rollouts = sample_expert_transitions_rollouts(vec_expert.predict, val_env, n_demonstrations)

    pomdp_rollouts = make_pomdp_rollouts(rollouts, lookup_frq=lookup_freq, count_dim=10)
    pomdp_transitions = rollout.flatten_trajectories(pomdp_rollouts)

    pomdp_env_val, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
    if (not os.path.isfile(save_path + bc_logname + ' BC best')):
        print('BC')
        bc_learner = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=pomdp_transitions,
            device=device,
            policy=learner.policy)
        
        tboard = TBoardGraphs(logname=logname + ' BC' , data_path=save_path + '/' + logname)
        best_succes_rate = -1
        best_model = None
        fac = 40
        runs_per_epoch = 20*fac
        for i in range(int(bc_epochs/fac)):
            bc_learner.train(n_epochs=runs_per_epoch)
            success, rews = get_avr_succ_rew_det(env=pomdp_env_val, learner=bc_learner.policy, epsiodes=50)
            success_rate = success.mean()
            tboard.addValidationScalar('Reward', value=th.tensor(rews.mean()), stepid=i)
            tboard.addValidationScalar('Success Rate', value=th.tensor(success_rate), stepid=i)
            if success_rate > best_succes_rate:
                best_succes_rate = success_rate
                th.save(bc_learner.policy.state_dict(), save_path + bc_logname + ' BC best')
                print(save_path + logname + ' BC best')
    else:
        print('skipping BC')

    reward_net = BasicRewardNet(
        pomdp_env.observation_space, pomdp_env.action_space, normalize_input_layer=RunningNorm
    )

    learner.policy.load_state_dict(th.load(save_path + bc_logname + ' BC best'))

    gail_trainer = GAIL(
        demonstrations=pomdp_transitions,
        demo_batch_size=min(1024, len(transitions)),
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=pomdp_env,
        gen_algo=learner,
        reward_net=reward_net,
    ) 
    print(f'logname: {logname}')
    print(f'save_path: {save_path}')
    tboard = TBoardGraphs(logname=logname , data_path=save_path + '/' + logname)
    success, rews = get_avr_succ_rew_det(env=pomdp_env_val, learner=learner.policy, epsiodes=200)
    success_rate = success.mean()
    tboard.addValidationScalar('Reward', value=th.tensor(rews.mean()), stepid=min(learner.env.envs[0].reset_count, n_samples))
    tboard.addValidationScalar('Success Rate', value=th.tensor(success_rate), stepid=min(learner.env.envs[0].reset_count, n_samples))

    while learner.env.envs[0].reset_count <= n_samples:
        print(f'nsamples: {n_samples}')
        print(f'learner.env.envs[0].reset_count')
        print('before learn')
        gail_trainer.train(2048)
        print('after learn')
        print(learner.env.envs[0].reset_count)
        success, rews = get_avr_succ_rew_det(env=pomdp_env_val, learner=learner.policy, epsiodes=200)
        success_rate = success.mean()
        tboard.addValidationScalar('Reward', value=th.tensor(rews.mean()), stepid=min(learner.env.envs[0].reset_count, n_samples))
        tboard.addValidationScalar('Success Rate', value=th.tensor(success_rate), stepid=min(learner.env.envs[0].reset_count, n_samples))

def evaluate_GAIL_PPO(device):
    env_tag = 'pickplace'
    lookup_freq = 1000
    lr = 1e-3
    seq_lens = [50, 100, 200]
    for i in range(5):
        for seq_len in seq_lens:
            demonstrations = 6
            for j in range(4):
                demonstrations += 2
                pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
                learner = PPO(
                        env=pomdp_env,
                        policy=MlpPolicy,
                        batch_size=64,
                        ent_coef=0.0,
                        learning_rate=lr,
                        n_epochs=10,
                        device=device
                    )
                logname = f'GAIL + PPO lr: {lr}, Demonstrations: {demonstrations}, seq_len: {seq_len}'
                bc_logname = f'GAIL + PPO Demonstrations: {demonstrations}, seq_len: {seq_len}'
                evaluate_GAIL(
                    env_tag=env_tag, 
                    logname=logname, 
                    seq_len=seq_len, 
                    n_demonstrations=demonstrations, 
                    n_samples = 400, 
                    learner = learner, 
                    pomdp_env = pomdp_env, 
                    save_path='/data/bing/hendrik/Evaluate Baseline/',
                    bc_epochs = 400,
                    bc_logname = bc_logname,
                    device=device)
        lr = lr * 0.6

def evaluate_GAIL_PPO_Fast(device):
    env_tag = 'pickplace'
    lookup_freq = 1000
    lr = 1e-4
    seq_len = 100
    for i in range(5):
        demonstrations = 10
        pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
        learner = PPO(
                env=pomdp_env,
                policy=MlpPolicy,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=lr,
                n_epochs=10,
                device=device
            )
        logname = f'GAIL + PPO lr: {lr}, Demonstrations: {demonstrations}, seq_len: {seq_len}'
        bc_logname = f'GAIL + PPO Demonstrations: {demonstrations}, seq_len: {seq_len}'
        evaluate_GAIL(
            env_tag=env_tag, 
            logname=logname, 
            seq_len=seq_len, 
            n_demonstrations=demonstrations, 
            n_samples = 400, 
            learner = learner, 
            pomdp_env = pomdp_env, 
            save_path='/data/bing/hendrik/Evaluate Baseline/',
            bc_epochs = 400,
            bc_logname = bc_logname,
            device=device)
        lr = lr * 0.6

def evaluate_GAIL_TQC(device):
    env_tag = 'pickplace'
    lookup_freq = 1000
    lr = 1e-3
    seq_lens = [50, 100, 200]
    for i in range(5):
        for seq_len in seq_lens:
            demonstrations = 6
            for j in range(4):
                demonstrations += 2
                pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
                learner = TQC(policy='MlpPolicy', env=pomdp_env, device=device, learning_rate=lr)
                logname = f'GAIL + TQC lr: {lr}, Demonstrations: {demonstrations}, seq_len: {seq_len}'
                bc_logname = f'GAIL + TQC Demonstrations: {demonstrations}, seq_len: {seq_len}'
                evaluate_GAIL(
                    env_tag=env_tag, 
                    logname=logname, 
                    seq_len=seq_len, 
                    n_demonstrations=demonstrations, 
                    n_samples = 400, 
                    learner = learner, 
                    pomdp_env = pomdp_env, 
                    save_path='/data/bing/hendrik/Evaluate Baseline/',
                    bc_epochs = 400,
                    bc_logname = bc_logname,
                    device=device)
        lr = lr * 0.6

def evaluate_GAIL_TQC_Fast(device):
    env_tag = 'pickplace'
    lookup_freq = 1000
    lr = 1e-3
    seq_len = 100
    demonstrations = 10
    for i in range(5):
        pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
        learner = TQC(policy='MlpPolicy', env=pomdp_env, device=device, learning_rate=lr)
        logname = f'GAIL + TQC lr: {lr}, Demonstrations: {demonstrations}, seq_len: {seq_len}'
        bc_logname = f'GAIL + TQC Demonstrations: {demonstrations}, seq_len: {seq_len}'
        evaluate_GAIL(
            env_tag=env_tag, 
            logname=logname, 
            seq_len=seq_len, 
            n_demonstrations=demonstrations, 
            n_samples = 400, 
            learner = learner, 
            pomdp_env = pomdp_env, 
            save_path='/data/bing/hendrik/Evaluate Baseline/',
            bc_epochs = 400,
            bc_logname = bc_logname,
            device=device)
        lr = lr * 0.6

def evaluate_GAIL_PPO_custom(device, lr, seq_len, demonstrations):
    env_tag = 'pickplace'
    lookup_freq = 1000
    if lr is None:
        lr = 1e-4
    if seq_len is None:
        seq_len = 100
    if demonstrations is None:
        demonstrations = 10

    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
    learner = PPO(
            env=pomdp_env,
            policy=MlpPolicy,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=lr,
            n_epochs=10,
            device=device
        )
    logname = f'GAIL + PPO lr: {lr}, Demonstrations: {demonstrations}, seq_len: {seq_len}'
    bc_logname = f'GAIL + PPO Demonstrations: {demonstrations}, seq_len: {seq_len}'
    evaluate_GAIL(
        env_tag=env_tag, 
        logname=logname, 
        seq_len=seq_len, 
        n_demonstrations=demonstrations, 
        n_samples = 400, 
        learner = learner, 
        pomdp_env = pomdp_env, 
        save_path='/data/bing/hendrik/Evaluate Baseline/',
        bc_epochs = 400,
        bc_logname = bc_logname,
        device=device)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str,
                    help='Choose free GPU')
    parser.add_argument('-learner', type=str,
                help='pick Learner')
    parser.add_argument('-lr', type=float,
                help='pick lr')
    parser.add_argument('-seq', type=int,
                help='pick seq_len')
    parser.add_argument('-exp', type=int,
                help='pick num expert demos')
    args = parser.parse_args()
    if args.learner == 'TQC':
        print('running TQC')
        run_eval_TQC(device=args.device)
    elif args.learner == 'PPO':
        print('running PPO')
        run_eval_PPO(device=args.device)
    elif args.learner == 'BC':
        print('running BC')
        run_eval_BC(device=args.device)
    elif args.learner == 'PPO_f':
        print('running BC')
        run_tune_PPO(device=args.device)
    elif args.learner == 'TQC_f':
        print('running BC')
        run_tune_TQC(device=args.device)
    elif args.learner == 'GAIL_TQC':
        evaluate_GAIL_TQC(device=args.device)
    elif args.learner == 'GAIL_PPO':
        evaluate_GAIL_PPO(device=args.device)
    elif args.learner == 'GAIL_PPO_fast':
        evaluate_GAIL_PPO_Fast(device=args.device)
    elif args.learner == 'GAIL_TQC_fast':
        evaluate_GAIL_TQC_Fast(device=args.device)
    elif args.learner == 'GAIL_PPO_custom':
        lr = None
        seq_len = None
        expert_samples = None
        if args.lr:
            lr = args.lr
        if args.seq:
            seq_len = args.seq
        if args.exp:
            expert_samples = args.exp
        evaluate_GAIL_PPO_custom(device=args.device, lr=lr, seq_len=seq_len, demonstrations=expert_samples)

    else:
        print('choose other algo')
