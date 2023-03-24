from active_critic.utils.gym_utils import (
    make_vec_env,
    make_dummy_vec_env,
    sample_expert_transitions_rollouts,
    make_pomdp_rollouts,
    make_dummy_vec_env_pomdp,
    get_avr_succ_rew_det,
    get_avr_succ_rew_det_rec,
    make_ppo_rec_data_loader,
    make_dummy_vec_env_rec_pomdp
)
import gym
from stable_baselines3 import PPO, SAC
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
import pickle

import copy

from active_critic.TQC.tqc import TQC
from active_critic.TQC.tqc_policy import TQCPolicyEval
from datetime import datetime
from sb3_contrib import RecurrentPPO
from active_critic.PPO_Recurrent.ppo_recurrent_bc import Rec_PPO_BC


def evaluate_learner(env_tag, logname_save_path, seq_len, n_demonstrations, bc_epochs, n_samples, device, logname, eval_every,  learner: TQC = None):
    history = None
    lookup_freq = 1000
    if not os.path.exists(logname_save_path):
        os.makedirs(logname_save_path)
    env, vec_expert = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    val_env, _ = make_dummy_vec_env(name=env_tag, seq_len=seq_len)

    pomdp_env_val, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
    if bc_epochs > 0:
        transitions, rollouts = sample_expert_transitions_rollouts(
            vec_expert.predict, val_env, n_demonstrations)

        pomdp_rollouts = make_pomdp_rollouts(
            rollouts, lookup_frq=lookup_freq, count_dim=10)
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

        bc_file_path = logname_save_path+'bc_best'
        bc_stats_path = logname_save_path + 'bc_stats'
        if (not os.path.isfile(bc_file_path)):
            print('BC Phase')
            tboard = TBoardGraphs(logname=logname + '_BC',
                                data_path=logname_save_path)

            best_succes_rate = -1
            fac = 40
            runs_per_epoch = 20 * fac
            for i in range(int(bc_epochs/fac)):
                bc_learner.train(n_epochs=runs_per_epoch)
                success, rews, history = get_avr_succ_rew_det(
                    env=pomdp_env_val, 
                    learner=bc_learner.policy, 
                    epsiodes=50,
                    path=bc_stats_path,
                    history=history,
                    step=i)
                success_rate = success.mean()
                tboard.addValidationScalar(
                    'Reward', value=th.tensor(rews.mean()), stepid=i)
                tboard.addValidationScalar(
                    'Success Rate', value=th.tensor(success_rate), stepid=i)
                if success_rate > best_succes_rate:
                    best_succes_rate = success_rate
                    th.save(bc_learner.policy.state_dict(),
                            bc_file_path)
        else:
            print('skipping BC')

    if learner is not None:
        learner_stats_path = logname_save_path + 'stats_learner'
        tboard = TBoardGraphs(
            logname=logname + str(' Reinforcement'), data_path=logname_save_path)
        if bc_epochs > 0:
            learner.policy.load_state_dict(
                th.load(bc_file_path))


        history = None

        success, rews, history = get_avr_succ_rew_det(
            env=pomdp_env_val, 
            learner=learner.policy, 
            epsiodes=100,
            path=learner_stats_path,
            history=history,
            step=0)
        
        tboard.addValidationScalar(
            'Reloaded Success Rate', value=th.tensor(success.mean()), stepid=0)
        tboard.addValidationScalar(
            'Reloaded Reward', value=th.tensor(rews.mean()), stepid=0)

        tboard.addValidationScalar('Reward', value=th.tensor(
            rews.mean()), stepid=learner.env.envs[0].reset_count)
        tboard.addValidationScalar('Success Rate', value=th.tensor(
            success.mean()), stepid=learner.env.envs[0].reset_count)

        while learner.env.envs[0].reset_count <= n_samples:
            print('before learn')
            learner.learn(eval_every)
            print('after learn')
            print(learner.env.envs[0].reset_count)
            success, rews, history = get_avr_succ_rew_det(
                env=pomdp_env_val, 
                learner=learner.policy, 
                epsiodes=200,
                path=learner_stats_path,
                history=history,
                step=learner.env.envs[0].reset_count)
            success_rate = success.mean()
            tboard.addValidationScalar('Reward', value=th.tensor(
                rews.mean()), stepid=learner.env.envs[0].reset_count)
            tboard.addValidationScalar('Success Rate', value=th.tensor(
                success_rate), stepid=learner.env.envs[0].reset_count)


def run_eval_TQC(device, lr, demonstrations, save_path, n_samples, id, env_tag):
    seq_len=100
    env_tag = env_tag
    logname = f'TQC_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_n_samples_{n_samples}_id_{id}'
    print(logname)
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=1000)
    tqc_learner = TQC(policy='MlpPolicy', env=pomdp_env,
                      device=device, learning_rate=lr)
    evaluate_learner(env_tag, logname_save_path=logname_save_path, logname=logname, seq_len=seq_len, n_demonstrations=demonstrations,
                     bc_epochs=0, n_samples=n_samples, device=device, eval_every=1000, learner=tqc_learner)


def run_eval_PPO(device, lr, demonstrations, save_path, n_samples, id, env_tag):
    seq_len=100
    env_tag = env_tag
    logname = f'PPO_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_n_samples_{n_samples}_id_{id}'
    print(logname)
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=1000)
    PPO_learner = PPO("MlpPolicy", pomdp_env, verbose=0,
                      device=device, learning_rate=lr)

    evaluate_learner(env_tag, logname_save_path=logname_save_path, logname=logname, seq_len=seq_len, n_demonstrations=demonstrations,
                     bc_epochs=0, n_samples=n_samples, device=device, eval_every=1000, learner=PPO_learner)


def run_tune_TQC(device):
    lr = 1e-4
    seq_lens = [100, 200]
    for i in range(3):
        for seq_len in seq_lens:
            demonstrations = 14
            for j in range(3):
                run_eval_TQC(device=device, lr=lr,
                             demonstrations=demonstrations, seq_len=seq_len)
                demonstrations += 2
        lr = lr * 0.4

def stats_PPO(device, path, demonstration, lr, env_tag):
    ids = [i for i in range(5)]
    for id in ids:
        run_eval_PPO(device=device, lr=lr, demonstrations=demonstration, save_path=path, n_samples=20000, id=id, env_tag=env_tag)

def stats_TQC(device, path, demonstration, lr, env_tag):
    ids = [i for i in range(5)]
    for id in ids:
        run_eval_TQC(device=device, lr=lr, demonstrations=demonstration, save_path=path, n_samples=20000, id=id, env_tag=env_tag)

def run_tune_PPO(device):
    lr = 1e-4
    seq_lens = [100, 200]
    for i in range(3):
        for seq_len in seq_lens:
            demonstrations = 14
            for j in range(3):
                run_eval_PPO(device=device, lr=lr,
                             demonstrations=demonstrations, seq_len=seq_len)
                demonstrations += 2
        lr = lr * 0.4

def evaluate_GAIL(env_tag, logname_save_path, seq_len, n_demonstrations, bc_epochs, n_samples, device, logname, learner, pomdp_env, eval_every):
    history = None
    lookup_freq = 1000
    if not os.path.exists(logname_save_path):
        os.makedirs(logname_save_path)
    env, vec_expert = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    val_env, _ = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    if bc_epochs > 0:
        transitions, rollouts = sample_expert_transitions_rollouts(
            vec_expert.predict, val_env, n_demonstrations)

        pomdp_rollouts = make_pomdp_rollouts(
            rollouts, lookup_frq=lookup_freq, count_dim=10)
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

        pomdp_env_val, pomdp_vec_expert = make_dummy_vec_env_pomdp(
            name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq)
        bc_file_path = logname_save_path+'bc_best'
        bc_stats_path = logname_save_path + 'bc_stats_gail'
        #if (not os.path.isfile(bc_file_path)):
        print('BC Phase')
        tboard = TBoardGraphs(logname=logname + '_BC',
                                data_path=logname_save_path)

        best_succes_rate = -1
        fac = 40
        runs_per_epoch = 20 * fac
        for i in range(int(bc_epochs/fac)):
            bc_learner.train(n_epochs=runs_per_epoch)
            success, rews, history = get_avr_succ_rew_det(
                env=pomdp_env_val, 
                learner=bc_learner.policy, 
                epsiodes=50,
                path=bc_stats_path,
                history=history,
                step=i)
            success_rate = success.mean()
            tboard.addValidationScalar(
                'Reward', value=th.tensor(rews.mean()), stepid=i*fac)
            tboard.addValidationScalar(
                'Success Rate', value=th.tensor(success_rate), stepid=i*fac)
            if success_rate > best_succes_rate:
                best_succes_rate = success_rate
                th.save(bc_learner.policy.state_dict(),
                        bc_file_path)

    if learner is not None:
        learner_stats_path = logname_save_path + 'learner_stats_gail'
        tboard = TBoardGraphs(
            logname=logname + str(' Reinforcement'), data_path=logname_save_path)
        
        reward_net = BasicRewardNet(
            learner.env.observation_space, learner.env.action_space, normalize_input_layer=RunningNorm
        )
        if bc_epochs > 0:
            learner.policy.load_state_dict(
                th.load(bc_file_path))

        gail_trainer = GAIL(
            demonstrations=pomdp_transitions,
            demo_batch_size=64,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=pomdp_env,
            gen_algo=learner,
            reward_net=reward_net,
        )


        history = None

        success, rews, history = get_avr_succ_rew_det(
            env=pomdp_env_val, 
            learner=learner.policy, 
            epsiodes=50,
            path=learner_stats_path,
            history=history,
            step=0)
        
        tboard.addValidationScalar(
            'Reloaded Success Rate', value=th.tensor(success.mean()), stepid=0)
        tboard.addValidationScalar(
            'Reloaded Reward', value=th.tensor(rews.mean()), stepid=0)

        tboard.addValidationScalar('Reward', value=th.tensor(
            rews.mean()), stepid=learner.env.envs[0].reset_count)
        tboard.addValidationScalar('Success Rate', value=th.tensor(
            success.mean()), stepid=learner.env.envs[0].reset_count)

        while learner.env.envs[0].reset_count <= n_samples:
            print('before learn')
            gail_trainer.train(eval_every)
            print('after learn')
            print(learner.env.envs[0].reset_count)
            success, rews, history = get_avr_succ_rew_det(
                env=pomdp_env_val, 
                learner=learner.policy, 
                epsiodes=50,
                path=learner_stats_path,
                history=history,
                step=learner.env.envs[0].reset_count)
            success_rate = success.mean()
            tboard.addValidationScalar('Reward', value=th.tensor(
                rews.mean()), stepid=learner.env.envs[0].reset_count)
            tboard.addValidationScalar('Success Rate', value=th.tensor(
                success_rate), stepid=learner.env.envs[0].reset_count)
            
def evaluate_Rec_PPO(env_tag, logname_save_path, seq_len, n_demonstrations, bc_epochs, n_samples, device, logname, eval_every, lr, bc_mult):
    history = None
    if not os.path.exists(logname_save_path):
        os.makedirs(logname_save_path)

    env, vec_expert = make_dummy_vec_env(name=env_tag, seq_len=seq_len)
    dataloader = make_ppo_rec_data_loader(env=env, vec_expert=vec_expert, n_demonstrations=n_demonstrations, seq_len=seq_len, device=device)

    ppo_env, _ = make_dummy_vec_env_rec_pomdp(name=env_tag, seq_len=seq_len)
    learner = RecurrentPPO("MlpLstmPolicy", env=ppo_env, verbose=0, learning_rate=lr, device=device)
    learner.learning_rate

    bc_learner = Rec_PPO_BC(model=learner, dataloader=dataloader, device=device)

    pomdp_env_val, _ = make_dummy_vec_env_rec_pomdp(
        name=env_tag, seq_len=seq_len)
    bc_file_path = logname_save_path+'bc_best'
    bc_stats_path = logname_save_path + 'bc_stats_rec_PPO'
    #if (not os.path.isfile(bc_file_path)):
    print('BC Phase')
    tboard = TBoardGraphs(logname=logname + '_BC',
                            data_path=logname_save_path)

    best_succes_rate = -1
    fac = 40
    runs_per_epoch = 300 * fac
    for i in range(int(bc_epochs/fac)):
        print(f'BC: {i} from {int(bc_epochs/fac)}')
        bc_learner.train(n_epochs=runs_per_epoch, verbose=True, bc_mult=bc_mult)
        success, rews, history = get_avr_succ_rew_det_rec(
            env=pomdp_env_val, 
            learner=bc_learner.policy,
            epsiodes=50,
            path=bc_stats_path,
            history=history,
            step=i)
        print(f'success: {success.mean()}')
        success_rate = success.mean()
        tboard.addValidationScalar(
            'Reward', value=th.tensor(rews.mean()), stepid=i*fac)
        tboard.addValidationScalar(
            'Success Rate', value=th.tensor(success_rate), stepid=i*fac)
        if success_rate > best_succes_rate:
            best_succes_rate = success_rate
    th.save(bc_learner.policy.state_dict(),
            bc_file_path)

    learner_stats_path = logname_save_path + 'learner_stats_rec_PPO'
    tboard = TBoardGraphs(
        logname=logname + str(' Reinforcement'), data_path=logname_save_path)

    learner.policy.load_state_dict(
        th.load(bc_file_path))

    history = None

    success, rews, history = get_avr_succ_rew_det_rec(
        env=pomdp_env_val, 
        learner=learner.policy, 
        epsiodes=50,
        path=learner_stats_path,
        history=history,
        step=0)
    
    tboard.addValidationScalar(
        'Reloaded Success Rate', value=th.tensor(success.mean()), stepid=0)
    tboard.addValidationScalar(
        'Reloaded Reward', value=th.tensor(rews.mean()), stepid=0)

    tboard.addValidationScalar('Reward', value=th.tensor(
        rews.mean()), stepid=learner.env.envs[0].reset_count)
    tboard.addValidationScalar('Success Rate', value=th.tensor(
        success.mean()), stepid=learner.env.envs[0].reset_count)

    while learner.env.envs[0].reset_count <= n_samples:
        learner.learn(eval_every)
        print(learner.env.envs[0].reset_count)
        success, rews, history = get_avr_succ_rew_det_rec(
            env=pomdp_env_val, 
            learner=learner.policy, 
            epsiodes=50,
            path=learner_stats_path,
            history=history,
            step=learner.env.envs[0].reset_count)
        success_rate = success.mean()
        tboard.addValidationScalar('Reward', value=th.tensor(
            rews.mean()), stepid=learner.env.envs[0].reset_count)
        tboard.addValidationScalar('Success Rate', value=th.tensor(
            success_rate), stepid=learner.env.envs[0].reset_count)
            
def run_eval_RPPO(device, lr, demonstrations, save_path, n_samples, id, env_tag, bc_mult):
    seq_len=100
    logname = f'RPPO_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_id_{id}'
    logname_save_path = os.path.join(save_path, logname + '/')
    print(f'learner: RPPO, env: {env_tag}, demos: {demonstrations}')
    evaluate_Rec_PPO(
        env_tag=env_tag,
        logname_save_path=logname_save_path,
        seq_len=seq_len,
        n_demonstrations=demonstrations,
        bc_epochs=2000,
        n_samples=n_samples,
        device=device,
        logname=logname,
        eval_every=2000,
        lr=lr,
        bc_mult=bc_mult
    )

def stats_RPPO(device, lr, demonstrations, save_path, n_samples, env_tag, bc_mult, ids):
    for id in ids:
        run_eval_RPPO(
            device=device,
            lr=lr,
            demonstrations=demonstrations,
            save_path=save_path,
            n_samples=n_samples,
            id=id,
            env_tag=env_tag,
            bc_mult=bc_mult
        )
            
def run_eval_PPO_GAIL(device, lr, demonstrations, save_path, n_samples, id, env_tag):
    seq_len=100
    logname = f'PPO_GAIL_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_id_{id}'
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=2048)
    PPO_learner = PPO("MlpPolicy", pomdp_env, verbose=0,
                      device=device, learning_rate=lr)

    evaluate_GAIL(env_tag, logname_save_path=logname_save_path, logname=logname, seq_len=seq_len, n_demonstrations=demonstrations,
                     bc_epochs=500, n_samples=n_samples, device=device, learner=PPO_learner, pomdp_env=pomdp_env, eval_every=2048)
    
def run_eval_TQC_GAIL(device, lr, demonstrations, save_path, n_samples, id, env_tag):
    seq_len=100
    logname = f'TQC_GAIL_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_id_{id}'
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=50000)
    TQC_learner = TQC(policy='MlpPolicy', env=pomdp_env,
        device=device, learning_rate=lr)

    evaluate_GAIL(env_tag, logname_save_path=logname_save_path, logname=logname, seq_len=seq_len, n_demonstrations=demonstrations,
                     bc_epochs=500, n_samples=n_samples, device=device, learner=TQC_learner, pomdp_env=pomdp_env, eval_every=2000)

def stats_GAIL_PPO(device, lr, demonstrations, save_path, n_samples, env_tag, ids):
    for id in ids:
        run_eval_PPO_GAIL(device=device, lr=lr, demonstrations=demonstrations, save_path=save_path, n_samples=n_samples, id=id, env_tag=env_tag)

def stats_GAIL_TQC(device, lr, demonstrations, save_path, n_samples, env_tag, ids):
    for id in ids:
        run_eval_TQC_GAIL(device=device, lr=lr, demonstrations=demonstrations, save_path=save_path, n_samples=n_samples, id=id, env_tag=env_tag)

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
    s = datetime.today().strftime('%Y-%m-%d')

    list_demonstrations = [4]
    list_env_tags = ['windowopen', 'push', 'pickplace']
    n_samples = 400
    ids = [i for i in range(3)]
    
    path = '/data/bing/hendrik/Baselines_Stats_GAIL_' + s + '/'

    print(f'___________stats: list_demonstrations {list_demonstrations} list_env_tags {list_env_tags}')

    if args.learner == 'TQC':
        print('running TQC')
        run_eval_TQC(device=args.device)
    elif args.learner == 'PPO':
        print('running PPO')
        run_eval_PPO(device=args.device, lr=1e-3, demonstrations=10, save_path=path, n_samples=10, id=0)
    elif args.learner == 'PPO_GAIL':
        print('running PPO_GAIL')
        run_eval_PPO_GAIL(device=args.device, lr=1e-3, demonstrations=10, save_path=path, n_samples=10, id=0)
    elif args.learner == 'TQC_GAIL':
        print('running TQC_GAIL')
        run_eval_TQC_GAIL(device=args.device, lr=1e-3, demonstrations=10, save_path=path, n_samples=10, id=0)
    elif args.learner == 'PPO_f':
        print('running BC')
        run_tune_PPO(device=args.device)
    elif args.learner == 'stats_GAIL_TQC':
        print('running GAIL + TQC')
        
        lrs = [5e-7]
        for env_tag in list_env_tags:
            for demonstrations in list_demonstrations:
                for lr in lrs:
                    stats_GAIL_TQC(device=args.device, lr=lr, demonstrations=demonstrations, save_path=path, n_samples=n_samples, env_tag=env_tag, ids=ids)

    elif args.learner == 'stats_GAIL_PPO':
        print('running GAIL + PPO')

        lrs = [1e-5]
        for env_tag in list_env_tags:
            for demonstrations in list_demonstrations:
                for lr in lrs:
                    stats_GAIL_PPO(device=args.device, lr=lr, demonstrations=demonstrations, save_path=path, n_samples=n_samples, env_tag=env_tag, ids=ids)

    elif args.learner == 'RPPO':
        print('running RPPO')
        run_eval_RPPO(
            device=args.device,
            lr=3e-4,
            demonstrations=10,
            save_path=path,
            n_samples=200,
            id=0,
            env_tag='push'
        )
    elif args.learner == 'stats_RPPO':
        print('running RPPO')
        for lr in [1e-6]:
            for env_tag in list_env_tags:
                for demos in list_demonstrations:
                    stats_RPPO(
                        device=args.device,
                        lr=lr,
                        demonstrations=demos,
                        save_path=path,
                        n_samples=n_samples,
                        env_tag=env_tag,
                        bc_mult = 10,
                        ids = ids
                    )

    elif args.learner == 'stats_PPO':
        print('running stats PPO')
        for lr in [1e-5]:
            for env_tag in list_env_tags:
                for demos in list_demonstrations:
                    stats_PPO(
                        device=args.device,
                        path=path,
                        demonstration=demos,
                        lr=lr,
                        env_tag=env_tag
                    )
    elif args.learner == 'stats_TQC':
        print('running stats TQC')
        for lr in [1e-6, 1e-7]:
            for env_tag in list_env_tags:
                for demos in list_demonstrations:
                    stats_TQC(
                        device=args.device,
                        path=path,
                        demonstration=demos,
                        lr=lr,
                        env_tag=env_tag
                    )
    else:
        print('choose others algo')
