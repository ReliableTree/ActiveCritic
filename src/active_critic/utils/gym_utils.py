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


def make_dummy_vec_env(name, seq_len):
    policy_dict = make_policy_dict()

    env_tag = name
    max_episode_steps = seq_len
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
    env._freeze_rand_vec = False
    timelimit = TimeLimit(env=env, max_episode_steps=max_episode_steps)
    dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(timelimit)])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[env_tag][0], env=dv1)
    return dv1, vec_expert

def make_vec_env(env_id, num_cpu, seq_len):
    policy_dict = make_policy_dict()
    def make_env(env_id, rank, seed=0):
        def _init():
            max_episode_steps = seq_len
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_id][1]]()
            env._freeze_rand_vec = False
            timelimit = TimeLimit(env=env, max_episode_steps=max_episode_steps)
            riw = RolloutInfoWrapper(timelimit)
            return riw
        return _init
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    vec_expert = ImitationLearningWrapper(
        policy=policy_dict[env_id][0], env=env)
    return env, vec_expert

def parse_sampled_transitions(transitions, new_epoch, extractor, device='cuda'):
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
            rewards.append(epch_rewards)
            observations.append(epch_observations)
            actions.append(epch_actions)

            epch_actions = []
            epch_observations = []
            epch_rewards = []

        epch_observations.append(current_obs.numpy())
        epch_actions.append(transitions[i]['acts'])
        epch_rewards.append(transitions[i]['infos']['in_place_reward'])

    observations.append(epch_observations)
    actions.append(epch_actions)
    rewards.append(epch_rewards)

    actions = th.tensor(np.array(actions), dtype=th.float, device=device)
    observations = th.tensor(np.array(observations), dtype=th.float, device=device)
    rewards = th.tensor(np.array(rewards), dtype=th.float, device=device)
    return actions, observations, rewards


def sample_expert_transitions(policy, env, episodes):

    expert = policy

    print(f"Sampling expert transitions. {episodes}")
    rollouts = rollout(
        expert,
        env,
        make_sample_until(min_timesteps=None, min_episodes=episodes),
        unwrap=True,
        exclude_infos=False
    )
    return flatten_trajectories(rollouts)


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

def sample_new_episode(policy:ActiveCriticPolicy, env:Env, episodes:int=1, return_gen_trj = False):
        policy.eval()
        policy.reset()
        transitions = sample_expert_transitions(
            policy.predict, env, episodes)
        expected_rewards_after = policy.history.opt_scores[0]
        expected_rewards_before = policy.history.gen_scores[0]
        datas = parse_sampled_transitions(
            transitions=transitions, new_epoch=policy.args_obj.new_epoch, extractor=policy.args_obj.extractor)
        device_data = []
        for data in datas:
            device_data.append(data[:episodes].to(policy.args_obj.device))
        actions, observations, rewards = device_data
        rewards = rewards.unsqueeze(-1)

        if return_gen_trj:
            return actions, policy.history.gen_trj[0][:episodes], observations, rewards, expected_rewards_before[:episodes], expected_rewards_after[:episodes]
        else:
            return actions, observations, rewards, expected_rewards_before[:episodes], expected_rewards_after[:episodes]