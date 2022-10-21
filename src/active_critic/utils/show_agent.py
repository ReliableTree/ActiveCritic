import matplotlib.pyplot as plt
import torch as th
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.utils.dataset import DatasetAC
from active_critic.utils.gym_utils import (DummyExtractor, make_dummy_vec_env,
                                           new_epoch_reach,
                                           parse_sampled_transitions,
                                           sample_expert_transitions,
                                           sample_new_episode)
from active_critic.utils.pytorch_utils import make_part_obs_data, calcMSE
from active_critic.utils.test_utils import setup_ac_reach, setup_ac_reach_op
from gym import Env
import numpy as np
from torch.utils.data.dataloader import DataLoader
from active_critic.server.analyze import make_acl
from active_critic.utils.gym_utils import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, make_policy_dict, ResetCounterWrapper, TimeLimit


def make_env(env_id, seq_len):
    policy_dict = make_policy_dict()
    max_episode_steps = seq_len
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_id][1]]()
    env._freeze_rand_vec = False
    rce = ResetCounterWrapper(env)
    timelimit = TimeLimit(env=rce, max_episode_steps=max_episode_steps)
    return timelimit

def show_agent(model_path):
    device = 'cuda'
    acl, env, expert, seq_len, epsiodes, device = make_acl(device='cuda')
    acl.loadNetworkFromFile(path = model_path, device=device)
    acl.policy.eval()
    acl.policy.reset()
    env = make_env('reach', 100)
    
    obs = env.reset()
    obs = obs.reshape([1,-1])
    for i in range(80):
        actions = acl.policy.predict(obs)
        actions = actions[0]
        obs = obs.reshape([1,-1])

        obs, rew, dones, info = env.step(actions)
        env.render(mode='human')




if __name__ == '__main__':
    model_path = '/home/hendrik/Documents/master_project/LokalData/server/best_validation/'
    show_agent(model_path=model_path)
