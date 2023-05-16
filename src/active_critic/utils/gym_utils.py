from active_critic.policy.active_critic_policy import ActiveCriticPolicy
import numpy as np
import torch as th
from metaworld.envs import \
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import *
from gym.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from active_critic.utils.rollout import rollout, make_sample_until, flatten_trajectories
from stable_baselines3.common.type_aliases import GymEnv
from gym import Env
import gym
import copy
import torch.nn as nn
import math
import pickle
import os
from active_critic.utils.dataset import DatasetAC
from torch.utils.data.dataloader import DataLoader

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class DummyExtractor:
    def __init__(self):
        pass

    def forward(self, features):
        if type(features) is np.ndarray:
            features = th.tensor(features)
        return features

class ReductiveExtractor:
    def forward(self, features):
        if type(features) is np.ndarray:
            features = th.tensor(features)
        result = th.cat((features[...,:3], features[...,3:6], features[..., 9:13], features[...,-3:]), dim=-1)

        return result

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

def make_dummy_vec_env(name, seq_len, sparse):
    policy_dict = make_policy_dict()

    env_tag = name
    max_episode_steps = seq_len
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
    env._freeze_rand_vec = False
    timelimit = TimeLimit(env=env, max_episode_steps=max_episode_steps)
    strict_time = StrictSeqLenWrapper(timelimit, seq_len=seq_len + 1, sparse=sparse)
    reset_env = ResetCounterWrapper(env=strict_time)

    dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(reset_env)])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[env_tag][0], env=dv1)
    return dv1, vec_expert


class ResetCounterWrapper(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.reset_count = 0

    def reset(self):
        return super().reset()

    def step(self, action):
        
        obsv, rew, done, info = super().step(action)
        if done:
            self.reset_count += 1
        return obsv, rew, done, info

class StrictSeqLenWrapper(gym.Wrapper):
    def __init__(self, env: Env, seq_len, sparse) -> None:
        super().__init__(env)
        self.seq_len = seq_len
        self.current_step = 0
        self.sparse = sparse

    def reset(self):
        self.current_step = 1
        self.success = float(0)
        obs = super().reset()
        return obs

    def step(self, action):
        self.current_step += 1
        obsv, rew, done, info = super().step(action)
        
        done = self.current_step == self.seq_len
        if info['success'] == 1:
            self.success = float(10)
        if self.sparse and not done:
            info['unscaled_reward'] = float(0)
            return obsv, float(0), done, info
        elif done:
            info['unscaled_reward'] = self.success

            return obsv, self.success, done, info
        else:
            return obsv, rew, done, info
        
def make_vec_env(env_id, num_cpu, seq_len, sparse):
    policy_dict = make_policy_dict()

    def make_env(env_id, rank, seed=0):
        def _init():
            max_episode_steps = seq_len
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_id][1]]()
            env._freeze_rand_vec = False
            #rce = ResetCounterWrapper(env)
            timelimit = TimeLimit(env=env, max_episode_steps=max_episode_steps)
            strict_time = StrictSeqLenWrapper(timelimit, seq_len=seq_len + 1, sparse=sparse)
            riw = RolloutInfoWrapper(strict_time)
            return riw
        return _init
        
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[env_id][0], env=env)
    return env, vec_expert

def make_vec_env_pomdp(env_id, num_cpu, seq_len, lookup_freq, sparse):
    policy_dict = make_policy_dict()

    def make_env(env_id, rank, seed=0):
        def _init():
            max_episode_steps = seq_len
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_id][1]]()
            env._freeze_rand_vec = False
            reset_env = ResetCounterWrapper(env=env)
            timelimit = TimeLimit(env=reset_env, max_episode_steps=max_episode_steps)
            strict_time = StrictSeqLenWrapper(timelimit, seq_len=seq_len + 1, sparse=sparse)
            riw = RolloutInfoWrapper(strict_time)
            pomdp = POMDP_Wrapper(env=riw, lookup_freq=lookup_freq, pe_dim=10, seq_len=seq_len+1)
            return pomdp
        return _init
        
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
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

def parse_sampled_transitions(transitions, extractor, seq_len, device='cuda', dense = False):
    observations = []
    actions = []
    rewards = []
    epch_actions = []
    epch_observations = []
    epch_rewards = []
    check_obsvs = extractor.forward(transitions[0]['obs'])
    current_success = False
    for i in range(len(transitions)):
        current_obs = extractor.forward(features=transitions[i]['obs'])

        epch_observations.append(current_obs.numpy())
        epch_actions.append(transitions[i]['acts'])
        epch_rewards.append([transitions[i]['infos']['unscaled_reward']/10])
        if transitions[i]['infos']['success'] == 1:
            current_success = True

        if transitions[i]['dones']:
            if dense:
                rewards.append(epch_rewards)
            else:
                if current_success:
                    rewards.append(np.ones_like(epch_rewards))
                else:
                    rewards.append(np.zeros_like(epch_rewards))

            observations.append(np.array(epch_observations))
            actions.append(np.array(epch_actions))

            epch_actions = []
            epch_observations = []
            epch_rewards = []
            current_success = False


    actions = th.tensor(fill_arrays(actions, seq_len = seq_len), dtype=th.float, device=device)
    observations = th.tensor(fill_arrays(observations, seq_len = seq_len), dtype=th.float, device=device)
    rewards = th.tensor(fill_arrays(rewards, seq_len = seq_len), dtype=th.float, device=device)
    return actions, observations, rewards


def fill_arrays(inpt, seq_len):
    return inpt
    d = []
    for epoch in inpt:
        if epoch.shape[0] < seq_len:
            fill = np.ones([seq_len-epoch.shape[0], epoch.shape[-1]])
            nv = np.append(epoch, fill, axis=0)
            d.append(nv)
        else:
            d.append(epoch)
    return np.array(d)

def sample_expert_transitions(policy, env, episodes, set_deterministic):

    expert = policy
    print(f"Sampling transitions. {episodes}")

    rollouts = rollout(
        expert,
        env,
        make_sample_until(min_timesteps=None, min_episodes=episodes),
        unwrap=True,
        exclude_infos=False,
        deterministic_policy =True,
        set_deterministic=set_deterministic
    )
    return flatten_trajectories(rollouts)


def sample_new_episode(policy:ActiveCriticPolicy, 
                       env:Env, extractor, 
                       device:str, 
                       dense, 
                       episodes:int=1, 
                       return_gen_trj = False, 
                       seq_len = None,
                       start_training = None):

        if type(policy) is ActiveCriticPolicy:
            policy.eval()
            policy.reset()
            seq_len = policy.args_obj.epoch_len
            policy.start_training = start_training
            print(f'policy train mode: {policy.start_training}')
        else:
            seq_len = seq_len
            
        transitions = sample_expert_transitions(
            policy.predict, env, episodes, set_deterministic=False)

        datas = parse_sampled_transitions(
            transitions=transitions, seq_len=seq_len, extractor=extractor, device=device, dense=dense)
        device_data = []
        for data in datas:
            device_data.append(data[:episodes].to(device))
        actions, observations, rewards = device_data

        if isinstance(policy, ActiveCriticPolicy):
            expected_rewards_before = policy.history.gen_scores[0]
            expected_rewards_after = policy.history.opt_scores[0]
        else:
            expected_rewards_before = th.clone(rewards)
            expected_rewards_after = th.clone(rewards)

        if type(policy) is ActiveCriticPolicy:
            action_history = policy.action_history
        else:
            action_history = None


        if return_gen_trj:
            return actions, policy.history.gen_trj[0][:episodes], observations, rewards, expected_rewards_before[:episodes], expected_rewards_after[:episodes], action_history
        else:
            return actions[:episodes], observations[:episodes], rewards[:episodes], expected_rewards_before[:episodes], expected_rewards_after[:episodes], action_history
        
class POMDP_Wrapper(gym.Wrapper):
    def __init__(self, env, lookup_freq, pe_dim, seq_len) -> None:
        super().__init__(env)
        inpt = th.zeros([1, seq_len, pe_dim])
        positional_encoding = PositionalEncoding(d_model=10, dropout=0)
        self.pe = positional_encoding.forward(inpt).numpy()
        self.current_step = 0
        self.lookup_freq = lookup_freq

    def reset(self):
        obsv =  super().reset()
        obsv[20:30] = self.pe[0, 0]
        self.current_step = 0
        self.current_obv = np.copy(obsv)
        return obsv

    def step(self, action):
        self.current_step += 1
        obsv, rew, done, info = super().step(action)
        if self.current_step % self.lookup_freq == 0:
            self.current_obv = np.copy(obsv)
        
        obsv = np.copy(self.current_obv)
        obsv[20:30] = self.pe[0, self.current_step]
        if done and info['success'] > 0:
            rew = 10.
        else:
            rew = 0.

        return obsv, rew, done, info
    
class REC_POMDP_Wrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.current_step = 0

    def reset(self):
        obsv =  super().reset()
        self.current_step = 0
        self.current_obv = np.copy(obsv)
        return obsv

    def step(self, action):
        self.current_step += 1
        obsv, rew, done, info = super().step(action)
        obsv = np.copy(self.current_obv)
        if done and info['success'] > 0:
            rew = 10.
        else:
            rew = 0.

        return obsv, rew, done, info
    
def make_dummy_vec_env_rec_pomdp(name, seq_len, dense, sparse):
    if dense:
        return make_dummy_vec_env(name=name, seq_len=seq_len)
    policy_dict = make_policy_dict()

    env_tag = name
    max_episode_steps = seq_len
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
    env._freeze_rand_vec = False
    timelimit = TimeLimit(env=env, max_episode_steps=max_episode_steps)
    strict_time = StrictSeqLenWrapper(timelimit, seq_len=seq_len + 1, sparse=sparse)
    reset_env = ResetCounterWrapper(env=strict_time)

    rec_pomdp = REC_POMDP_Wrapper(env=reset_env)

    dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(rec_pomdp)])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[env_tag][0], env=dv1)
    return dv1, vec_expert

def make_dummy_vec_env_pomdp(name, seq_len, lookup_freq, dense, sparse):
    if dense:
        return make_dummy_vec_env(name=name, seq_len=seq_len, sparse=sparse)
    policy_dict = make_policy_dict()

    env_tag = name
    max_episode_steps = seq_len
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
    env._freeze_rand_vec = False
    timelimit = TimeLimit(env=env, max_episode_steps=max_episode_steps)
    strict_time = StrictSeqLenWrapper(timelimit, seq_len=seq_len + 1, sparse=sparse)
    reset_env = ResetCounterWrapper(env=strict_time)

    pomdp = POMDP_Wrapper(env=reset_env, lookup_freq=lookup_freq, pe_dim=10, seq_len=seq_len+1)

    dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(pomdp)])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[env_tag][0], env=dv1)
    return dv1, vec_expert

def get_avr_succ_rew(env, learner, epsiodes):
    success = []
    rews = []
    for i in range(epsiodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = learner.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(action)
            rews.append(rew)
            if info[0]['success'] > 0:
                success.append(info[0]['success'])
                break
            if done:
                success.append(0)
    return np.array(success), np.array(rews)

def sample_expert_transitions_rollouts(expert, env, num):

    rollouts = rollout(
        expert,
        env,
        make_sample_until(min_timesteps=None, min_episodes=num),
        exclude_infos=False,
    )
    return flatten_trajectories(rollouts), rollouts

def make_pomdp_rollouts(rollouts, lookup_frq, count_dim, dense):
    if dense:
        return rollouts
    inpt = th.zeros([1, rollouts[0].obs.shape[0], count_dim])
    positional_encoding = PositionalEncoding(d_model=10, dropout=0)
    pe = positional_encoding.forward(inpt).numpy()
    for ro in rollouts:
        for i in range(ro.obs.shape[0]):
            if i % lookup_frq == 0:
                obsv = copy.deepcopy(ro.obs[i])
            else:
                ro.obs[i] = copy.deepcopy(obsv)
        ro.obs[:, 20:20+count_dim] = pe
        if i % lookup_frq == 0:
            obs = ro
    return rollouts

def save_stat(success, history, step, path):
    if history is None:
        history = {
            'success_rate':success.mean(),
            'step': np.array(step)
        }
    else:
        history['success_rate'] = np.append(history['success_rate'], success.mean())
        history['step'] = np.append(history['step'], np.array(step))

    with open(path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return history


def get_avr_succ_rew_det(env, learner, epsiodes, path, history, step, seq_len, dense):
    transitions = sample_expert_transitions(
        policy=learner,
        env=env,
        episodes=epsiodes,
        set_deterministic=True
    )
    actions, observations, rewards = parse_sampled_transitions(transitions=transitions, extractor=DummyExtractor(), seq_len=seq_len, device=learner.device, dense=dense)
    success = (rewards.reshape([-1, seq_len]).max(dim=1).values == 1)
    success = success.type(th.float).detach().cpu().numpy()
    rews = rewards.detach().cpu().numpy()
    history = save_stat(success=success, history=history, step=step, path=path)
    return success, rews, history

def get_avr_succ_rew_det_rec(env, learner, epsiodes, path, history, step):
    success = []
    rews = []

    for i in range(epsiodes):
        obs = env.reset()
        done = False
        lstm_states = None
        episode_start_c = 1

        while not done:
            action, lstm_states = learner.predict(observation=obs, deterministic=True, episode_start=np.array([episode_start_c]), state=lstm_states)
            episode_start_c = 0
            obs, rew, done, info = env.step(action)
            rews.append(rew)
            if info[0]['success'] > 0:
                success.append(info[0]['success'])
                break
            if done:
                success.append(0)
                
    success = np.array(success)
    rews = np.array(rews)
    history = save_stat(success=success, history=history, step=step, path=path)
    return success, rews, history



def make_ppo_rec_data_loader(env, vec_expert, n_demonstrations, seq_len, device, dense):
    batch_size=16
    transitions, rollouts = sample_expert_transitions_rollouts(
        vec_expert.predict, env, n_demonstrations)
    actions, observations, rewards = parse_sampled_transitions(transitions=transitions, extractor=DummyExtractor(), seq_len=seq_len, device='cuda', dense=dense)
    inpt_obsv = observations[:,:1].repeat([1, observations.shape[1], 1])
    train_data = DatasetAC(batch_size=batch_size, device=device)
    train_data.add_data(obsv=inpt_obsv, actions=actions, reward=rewards, expert_trjs=rewards.reshape(-1))
    train_data.onyl_positiv = False

    dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return dataloader