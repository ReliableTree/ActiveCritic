from active_critic.utils.gym_utils import make_vec_env, make_dummy_vec_env
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
import argparse
from sb3_contrib import TQC

def sample_expert_transitions(expert, env, num):

    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=num),
        exclude_infos=False,
    )
    return rollout.flatten_trajectories(rollouts), rollouts

def asd(env, learner):
    success = []
    rews = []
    for i in range(100):
        obs = env.reset()
        done = False
        while not done:
            action, _ = learner.predict(obs)
            obs, rew, done, info = env.step(action)
            rews.append(rew)
            if info[0]['success'] > 0:
                success.append(info[0]['success'])
                break
            if done:
                success.append(0)
    return np.array(success), np.array(rews)

def run_experiment(device):

    env, vec_expert = make_dummy_vec_env(name='pickplace', seq_len=200)
    val_env, _ = make_dummy_vec_env(name='pickplace', seq_len=200)

    transitions, _ = sample_expert_transitions(vec_expert.predict, env, 100)
    env.envs[0].reset_count = 0

    reward_net = BasicRewardNet(
        env.observation_space, env.action_space, normalize_input_layer=RunningNorm)

    policy_kwargs = dict(n_critics=2, net_arch=[512,512,512])
    learner = TQC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, buffer_size=1000000, batch_size=2048, gamma=0.95, learning_rate=1e-3, tau=0.05, device=device)
    
    gail_trainer = gail.GAIL(
        demonstrations=transitions,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net

    )
    tboard = TBoardGraphs(logname='GAIL + TQC pickplace 100 pure', data_path='/data/bing/hendrik/gboard/')
    current_step = 0
    for i in range(10000):
        gail_trainer.train(1000)
        current_step += 5
        succ, rew = asd(env=val_env, learner=learner)
        print('____________________________________________')
        print(f'succ: {succ.mean()}')
        print(f'rew: {rew.mean()}')
        tboard.addTrainScalar('Reward', value=th.tensor(rew.mean()), stepid=current_step)
        tboard.addTrainScalar('Success Rate', value=th.tensor(succ.mean()), stepid=current_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=str,
                    help='Choose free GPU')
    args = parser.parse_args()
    run_experiment(device=args.device)