from active_critic.policy.active_critic_policy import ActiveCriticPolicy
import numpy as np
import torch as th
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import *
from gym.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from active_critic.utils.rollout import rollout, make_sample_until, flatten_trajectories
from stable_baselines3.common.type_aliases import GymEnv
from active_critic.utils.pytorch_utils import tokenize
from gym import Env
import gym

class DummyExtractor:
    def __init__(self):
        pass

    def forward(self, features):
        if type(features) is np.ndarray:
            features = th.tensor(features)
        return features


def make_policy_dict():
    policy_dict = {}
    for key in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        string_arr = key.split('-')
        v2_ind = string_arr.index('v2')
        string_arr = string_arr[:v2_ind]
        policy_name = ''
        for i, string in enumerate(string_arr):
            policy_name += string
            string_arr[i] = string.capitalize()
        entry = 'policy_dict["' + str(policy_name) + '"] = [Sawyer'
        for string in string_arr:
            entry += string
        entry += 'V2Policy(), "' + key + '"]'
        try:
            exec(entry)
        except (NameError):
            pass
    return policy_dict


def new_epoch_pap(current_obs, check_obsvs):
    result = not th.equal(current_obs.reshape(-1)
                          [-3:], check_obsvs.reshape(-1)[-3:])
    return result


def new_epoch_reach(current_obs, check_obsvs):
    return new_epoch_pap(current_obs, check_obsvs)

class reset_counter(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.reset_count = 0

    def reset(self):
        self.reset_count+=1
        return super().reset()

class ImitationLearningWrapper:
    def __init__(self, policy, env: GymEnv):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.policy = policy

    def predict(self, obsv, deterministic=None):
        actions = []
        for obs in obsv:
            actions.append(self.policy.get_action(obs))
        return actions

class ResetCounterWrapper(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.reset_count = 0

    def reset(self):
        self.reset_count+=1
        return super().reset()

    def step(self, action):
        
        obsv, rew, done, info = super().step(action)
        if info['success']:
            done = True
        else:
            rew = rew - 5
        return obsv, rew, done, info

def make_env(env_id, seq_len):
    policy_dict = make_policy_dict()
    def _init():
        max_episode_steps = seq_len
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_id][1]]()
        env._freeze_rand_vec = False
        #rce = ResetCounterWrapper(env)
        timelimit = TimeLimit(env=env, max_episode_steps=max_episode_steps)
        riw = RolloutInfoWrapper(timelimit)
        return riw
    return _init

def make_dummy_vec_env(name, seq_len):
    policy_dict = make_policy_dict()
    dv1 = DummyVecEnv([lambda: make_env(env_id=name, seq_len=seq_len)()])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[name][0], env=dv1)
    return dv1, vec_expert


def make_vec_env(env_id, num_cpu, seq_len):
    policy_dict = make_policy_dict()    
    env = SubprocVecEnv([make_env(env_id, seq_len) for i in range(num_cpu)])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[env_id][0], env=env)
    return env, vec_expert

def parse_sampled_transitions_legacy(transitions, new_epoch, extractor, seq_len, device='cuda'):
    observations = []
    actions = []
    rewards = []
    epch_actions = []
    epch_observations = []
    epch_rewards = []
    check_obsvs = extractor.forward(transitions[0]['obs'])
    for i in range(len(transitions)):
        current_obs = extractor.forward(features=transitions[i]['obs'])

        if new_epoch(current_obs, check_obsvs):
            check_obsvs = current_obs
            rewards.append(np.array(epch_rewards))
            observations.append(np.array(epch_observations))
            actions.append(np.array(epch_actions))

            epch_actions = []
            epch_observations = []
            epch_rewards = []

        epch_observations.append(current_obs.numpy())
        epch_actions.append(transitions[i]['acts'])
        epch_rewards.append([transitions[i]['infos']['in_place_reward']])

    observations.append(np.array(epch_observations))
    actions.append(np.array(epch_actions))
    rewards.append(np.array(epch_rewards))

    
    actions = th.tensor(fill_arrays(actions, seq_len = seq_len), dtype=th.float, device=device)
    observations = th.tensor(fill_arrays(observations, seq_len = seq_len), dtype=th.float, device=device)
    rewards = th.tensor(fill_arrays(rewards, seq_len = seq_len), dtype=th.float, device=device)
    return actions, observations, rewards

def parse_sampled_transitions(transitions, extractor, seq_len, device='cuda'):
    observations = []
    actions = []
    rewards = []
    epch_actions = []
    epch_observations = []
    epch_rewards = []
    check_obsvs = extractor.forward(transitions[0]['obs'])
    for i in range(len(transitions)):
        current_obs = extractor.forward(features=transitions[i]['obs'])

        epch_observations.append(current_obs.numpy())
        epch_actions.append(transitions[i]['acts'])
        epch_rewards.append([transitions[i]['infos']['unscaled_reward']/10])

        if transitions[i]['dones']:
            rewards.append(np.array(epch_rewards))
            observations.append(np.array(epch_observations))
            actions.append(np.array(epch_actions))

            epch_actions = []
            epch_observations = []
            epch_rewards = []


    actions = th.tensor(fill_arrays(actions, seq_len = seq_len), dtype=th.float, device=device)
    observations = th.tensor(fill_arrays(observations, seq_len = seq_len), dtype=th.float, device=device)
    rewards = th.tensor(fill_arrays(rewards, seq_len = seq_len), dtype=th.float, device=device)
    return actions, observations, rewards

def fill_arrays(inpt, seq_len):
    d = []
    for epoch in inpt:
        if epoch.shape[0] < seq_len:
            fill = -np.ones([seq_len-epoch.shape[0], epoch.shape[-1]])
            nv = np.append(epoch, fill, axis=0)
            d.append(nv)
        else:
            d.append(epoch)
    return np.array(d)

def sample_expert_transitions(policy, env, episodes):

    expert = policy
    print(f"Sampling transitions. {episodes}")
    rollouts = rollout(
        expert,
        env,
        make_sample_until(min_timesteps=None, min_episodes=episodes),
        unwrap=True,
        exclude_infos=False
    )
    return flatten_trajectories(rollouts)


def sample_new_episode(policy:ActiveCriticPolicy, env:Env, device:str, do_tokenize:bool,do_tokenize_reward:bool, ntokens:int, ntokens_reward:int, episodes:int):
        policy.eval()
        policy.reset()
        seq_len = policy.args_obj.epoch_len
        transitions = sample_expert_transitions(
            policy.predict, env, episodes)

        datas = parse_sampled_transitions(
            transitions=transitions, seq_len=seq_len, extractor=policy.args_obj.extractor, device=device)
        device_data = []
        for data in datas:
            device_data.append(data[:episodes].to(policy.args_obj.device))
        actions, observations, rewards = device_data
        if do_tokenize:
            actions = tokenize(inpt=actions, minimum=-1, maximum=1, ntokens=ntokens).type(th.float)
            observations = tokenize(inpt=observations, minimum=-1, maximum=1, ntokens=ntokens).type(th.float)
        if do_tokenize_reward:
            rewards = tokenize(inpt=rewards, minimum=0, maximum=1, ntokens=ntokens_reward).type(th.float)
        return actions, observations, rewards