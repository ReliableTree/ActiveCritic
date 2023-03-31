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


def evaluate_learner(env_tag, logname_save_path, seq_len, n_demonstrations, bc_epochs, n_samples, device, logname, eval_every, dense, sparse, learner: TQC = None):
    history = None
    lookup_freq = 1000
    if not os.path.exists(logname_save_path):
        os.makedirs(logname_save_path)
    env, vec_expert = make_dummy_vec_env(name=env_tag, seq_len=seq_len, sparse=sparse)
    val_env, _ = make_dummy_vec_env(name=env_tag, seq_len=seq_len, sparse=sparse)

    pomdp_env_val, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq, dense=dense, sparse=sparse)
    if bc_epochs > 0:
        transitions, rollouts = sample_expert_transitions_rollouts(
            vec_expert.predict, val_env, n_demonstrations)

        pomdp_rollouts = make_pomdp_rollouts(
            rollouts, lookup_frq=lookup_freq, count_dim=10, dense=dense)
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
        print('BC Phase')
        tboard = TBoardGraphs(logname=logname + '_BC',
                            data_path=logname_save_path)

        best_succes_rate = -1
        fac = 40
        runs_per_epoch = 100 * fac
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
            print(f'success rate: {success_rate}')
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
            learner.learn(eval_every, log_interval=1, progress_bar=True, reset_num_timesteps=False)
            print(learner.env.envs[0].reset_count)
            success, rews, history = get_avr_succ_rew_det(
                env=pomdp_env_val, 
                learner=learner.policy, 
                epsiodes=50,
                path=learner_stats_path,
                history=history,
                step=learner.env.envs[0].reset_count)
            success_rate = success.mean()
            print(f'success rate: {success_rate}')
            tboard.addValidationScalar('Reward', value=th.tensor(
                rews.mean()), stepid=learner.env.envs[0].reset_count)
            tboard.addValidationScalar('Success Rate', value=th.tensor(
                success_rate), stepid=learner.env.envs[0].reset_count)


def run_eval_TQC(device, lr, demonstrations, save_path, n_samples, id, env_tag, bc_epochs, dense, sparse, seq_len, learning_starts):
    env_tag = env_tag
    logname = f'TQC_{env_tag}_dense_{dense}_lr_{lr}_demonstrations_{demonstrations}_n_samples_{n_samples}_seqlen_{seq_len}_learning_starts_{learning_starts}_id_{id}'
    print(logname)
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=1000, dense=dense, sparse=sparse)
    
    #reach
    buffer_size = 1000000
    batch_size= 256
    ent_coef= 'auto'
    gamma= 0.999
    tau= 0.02
    train_freq= 8
    gradient_steps= 8
    use_sde = False
    policy_kwargs= dict(log_std_init=-3, net_arch=[400, 300])

    tqc_learner = TQC(policy='MlpPolicy', env=pomdp_env,
                      device=device, learning_rate=lr,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      ent_coef=ent_coef,
                      gamma=gamma,
                      tau=tau,
                      train_freq=train_freq,
                      gradient_steps=gradient_steps,
                      use_sde=use_sde,
                      policy_kwargs=policy_kwargs,
                      learning_starts=learning_starts)
    

    evaluate_learner(
        env_tag, 
        logname_save_path=logname_save_path, 
        logname=logname, 
        seq_len=seq_len, 
        n_demonstrations=demonstrations,
        bc_epochs=bc_epochs, 
        n_samples=n_samples, 
        device=device, 
        eval_every=1000, 
        learner=tqc_learner, 
        dense=dense,
        sparse=sparse)


def run_eval_PPO(device, lr, demonstrations, save_path, n_samples, id, env_tag, bc_epochs, dense, sparse, seq_len):
    env_tag = env_tag
    logname = f'PPO_{env_tag}_dense_{dense}_lr_{lr}_demonstrations_{demonstrations}_n_samples_{n_samples}_seqlen_{seq_len}_id_{id}'
    print(logname)
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=1000, dense=dense, sparse=sparse)
    
    #reach
    n_steps= 512
    batch_size = 32
    gamma= 0.9
    ent_coef= 7.52585e-08
    n_epochs= 5
    gae_lambda= 1.0
    max_grad_norm= 0.9
    vf_coef= 0.950368


    PPO_learner = PPO("MlpPolicy", 
                      pomdp_env, 
                      verbose=0,
                      device=device, 
                      learning_rate=lr,
                      batch_size=batch_size,
                      gamma=gamma,
                      ent_coef=ent_coef,
                      gae_lambda=gae_lambda,
                      max_grad_norm=max_grad_norm,
                      vf_coef=vf_coef,
                      n_steps=n_steps,
                      n_epochs=n_epochs,
                      )

    evaluate_learner(env_tag, logname_save_path=logname_save_path, logname=logname, seq_len=seq_len, n_demonstrations=demonstrations,
                     bc_epochs=bc_epochs, n_samples=n_samples, device=device, eval_every=1000, learner=PPO_learner, dense=dense, sparse=sparse)


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

def stats_PPO(device, path, demonstration, lr, env_tag, n_samples, ids, bc_epochs, dense, sparse, seq_len):
    for id in ids:
        run_eval_PPO(device=device, lr=lr, demonstrations=demonstration, save_path=path, n_samples=n_samples, id=id, env_tag=env_tag, bc_epochs=bc_epochs, dense=dense, sparse=sparse, seq_len=seq_len)

def stats_TQC(device, path, demonstration, lr, env_tag, n_samples, bc_epochs, ids, dense, sparse, seq_len, learning_starts):
    for id in ids:
        run_eval_TQC(device=device, 
                     lr=lr, 
                     demonstrations=demonstration, 
                     save_path=path, 
                     n_samples=n_samples, 
                     id=id, 
                     env_tag=env_tag, 
                     bc_epochs=bc_epochs, 
                     dense=dense, 
                     sparse=sparse, 
                     seq_len=seq_len,
                     learning_starts=learning_starts)

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

def evaluate_GAIL(env_tag, logname_save_path, seq_len, n_demonstrations, bc_epochs, n_samples, device, logname, learner, pomdp_env, eval_every, dense):
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
            rollouts, lookup_frq=lookup_freq, count_dim=10, dense=dense)
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
            name=env_tag, seq_len=seq_len, lookup_freq=lookup_freq, dense=dense)
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
            
def evaluate_Rec_PPO(env_tag, logname_save_path, seq_len, n_demonstrations, bc_epochs, n_samples, device, logname, eval_every, lr, bc_mult, dense):
    history = None
    if not os.path.exists(logname_save_path):
        os.makedirs(logname_save_path)

    env, vec_expert = make_dummy_vec_env(name=env_tag, seq_len=seq_len)

    ppo_env, _ = make_dummy_vec_env_rec_pomdp(name=env_tag, seq_len=seq_len, dense=dense)
    learner = RecurrentPPO("MlpLstmPolicy", env=ppo_env, verbose=0, learning_rate=lr, device=device)
    pomdp_env_val, _ = make_dummy_vec_env_rec_pomdp(
    name=env_tag, seq_len=seq_len, dense=dense)
    learner_stats_path = logname_save_path + 'learner_stats_rec_PPO'

    if n_demonstrations > 0:
        dataloader = make_ppo_rec_data_loader(env=env, vec_expert=vec_expert, n_demonstrations=n_demonstrations, seq_len=seq_len, device=device, dense=dense)
        bc_learner = Rec_PPO_BC(model=learner, dataloader=dataloader, device=device)


        bc_file_path = logname_save_path+'bc_best'
        bc_stats_path = logname_save_path + 'bc_stats_rec_PPO'
        #if (not os.path.isfile(bc_file_path)):
        print('BC Phase')
        tboard = TBoardGraphs(logname=logname + '_BC',
                                data_path=logname_save_path)

        best_succes_rate = -1
        fac = 40
        runs_per_epoch = 500 * fac
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

        learner.policy.load_state_dict(
            th.load(bc_file_path))
        
    tboard = TBoardGraphs(
        logname=logname + str(' Reinforcement'), data_path=logname_save_path)
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
            
def run_eval_RPPO(device, lr, demonstrations, save_path, n_samples, id, env_tag, bc_mult, bc_epochs, dense):
    seq_len=100
    logname = f'RPPO_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_id_{id}'
    logname_save_path = os.path.join(save_path, logname + '/')
    print(f'learner: RPPO, env: {env_tag}, demos: {demonstrations}')
    evaluate_Rec_PPO(
        env_tag=env_tag,
        logname_save_path=logname_save_path,
        seq_len=seq_len,
        n_demonstrations=demonstrations,
        bc_epochs=bc_epochs,
        n_samples=n_samples,
        device=device,
        logname=logname,
        eval_every=2000,
        lr=lr,
        bc_mult=bc_mult,
        dense=dense
    )

def stats_RPPO(device, lr, demonstrations, save_path, n_samples, env_tag, bc_mult, ids, bc_epochs, dense):
    for id in ids:
        run_eval_RPPO(
            device=device,
            lr=lr,
            demonstrations=demonstrations,
            save_path=save_path,
            n_samples=n_samples,
            id=id,
            env_tag=env_tag,
            bc_mult=bc_mult,
            bc_epochs=bc_epochs,
            dense=dense
        )
            
def run_eval_PPO_GAIL(device, lr, demonstrations, save_path, n_samples, id, env_tag, dense):
    seq_len=100
    logname = f'PPO_GAIL_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_id_{id}'
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=2048, dense=dense)
    PPO_learner = PPO("MlpPolicy", pomdp_env, verbose=0,
                      device=device, learning_rate=lr)

    evaluate_GAIL(env_tag, logname_save_path=logname_save_path, logname=logname, seq_len=seq_len, n_demonstrations=demonstrations,
                     bc_epochs=500, n_samples=n_samples, device=device, learner=PPO_learner, pomdp_env=pomdp_env, eval_every=2048)
    
def run_eval_TQC_GAIL(device, lr, demonstrations, save_path, n_samples, id, env_tag, dense):
    seq_len=100
    logname = f'TQC_GAIL_{env_tag}_lr_{lr}_demonstrations_{demonstrations}_id_{id}'
    logname_save_path = os.path.join(save_path, logname + '/')
    pomdp_env, pomdp_vec_expert = make_dummy_vec_env_pomdp(
        name=env_tag, seq_len=seq_len, lookup_freq=50000, dense=dense)
    TQC_learner = TQC(policy='MlpPolicy', env=pomdp_env,
        device=device, learning_rate=lr)

    evaluate_GAIL(env_tag, logname_save_path=logname_save_path, logname=logname, seq_len=seq_len, n_demonstrations=demonstrations,
                     bc_epochs=500, n_samples=n_samples, device=device, learner=TQC_learner, pomdp_env=pomdp_env, eval_every=2000, dense=dense)

def stats_GAIL_PPO(device, lr, demonstrations, save_path, n_samples, env_tag, ids):
    for id in ids:
        run_eval_PPO_GAIL(device=device, lr=lr, demonstrations=demonstrations, save_path=save_path, n_samples=n_samples, id=id, env_tag=env_tag)

def stats_GAIL_TQC(device, lr, demonstrations, save_path, n_samples, env_tag, ids, dense):
    for id in ids:
        run_eval_TQC_GAIL(device=device, lr=lr, demonstrations=demonstrations, save_path=save_path, n_samples=n_samples, id=id, env_tag=env_tag, dense=dense)

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

    list_demonstrations = [0]
    list_env_tags = ['drawerclose']
    n_samples = 1000
    bc_epochs = 0
    ids = [i for i in range(3)]
    dense_list = [True, False]
    sparse = True
    th.manual_seed(0)
    seq_len = 100
    learning_starts = 3000

    path = '/data/bing/hendrik/Baselines_Stats_GAIL_' + s + '/'

    print(f'___________stats: list_demonstrations {list_demonstrations} list_env_tags {list_env_tags}')
    print(f'list dense: {dense_list}')

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
        lrs = [1e-7]
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
                        ids = ids,
                        bc_epochs=bc_epochs,
                        dense=dense
                    )

    elif args.learner == 'stats_PPO':
        print('running stats PPO')
        lrs = [1e-4]
        for env_i, env_tag in enumerate(list_env_tags):
            lr = lrs[env_i]
            for dense in dense_list:
                for demos in list_demonstrations:
                    stats_PPO(
                        device=args.device,
                        path=path,
                        demonstration=demos,
                        lr=lr,
                        env_tag=env_tag,
                        n_samples=n_samples,
                        bc_epochs=bc_epochs,
                        ids=ids,
                        dense=dense,
                        sparse=sparse,
                        seq_len=seq_len
                    )
    elif args.learner == 'stats_TQC':
        print('running stats TQC')
        lrs = [5e-7]
        for env_i, env_tag in enumerate(list_env_tags):
            lr = lrs[env_i]
            for dense in dense_list:
                for demos in list_demonstrations:
                    stats_TQC(
                        device=args.device,
                        path=path,
                        demonstration=demos,
                        lr=lr,
                        env_tag=env_tag,
                        n_samples=n_samples,
                        bc_epochs=bc_epochs,
                        ids=ids,
                        dense=dense,
                        sparse=sparse,
                        seq_len=seq_len,
                        learning_starts=learning_starts
                    )

    elif args.learner == 'stats_TPR':
        print('running RPPO')
        for lr in [5e-7]:
            for env_tag in list_env_tags:
                for demos in list_demonstrations:
                    stats_RPPO(
                        device=args.device,
                        lr=lr,
                        demonstrations=demos,
                        save_path=path,
                        n_samples=n_samples,
                        env_tag=env_tag,
                        bc_mult = 1,
                        ids = ids,
                        bc_epochs=bc_epochs
                    )

        print('running stats PPO')
        for lr in [1e-4]:
            for env_tag in list_env_tags:
                for demos in list_demonstrations:
                    stats_PPO(
                        device=args.device,
                        path=path,
                        demonstration=demos,
                        lr=lr,
                        env_tag=env_tag,
                        n_samples=n_samples,
                        bc_epochs=bc_epochs,
                        ids=ids
                    )

        print('running stats TQC')
        for lr in [1e-7]:
            for env_tag in list_env_tags:
                for demos in list_demonstrations:
                    stats_TQC(
                        device=args.device,
                        path=path,
                        demonstration=demos,
                        lr=lr,
                        env_tag=env_tag,
                        n_samples=n_samples,
                        bc_epochs=bc_epochs,
                        ids=ids
                    )
    else:
        print('choose others algo')
