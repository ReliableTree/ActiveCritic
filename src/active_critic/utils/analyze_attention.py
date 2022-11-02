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

def plot_attention(model_path):
    device = 'cuda'
    acl, env, expert, seq_len, epsiodes, device = make_acl(device='cuda')
    acl.loadNetworkFromFile(path = model_path, device=device)
    acl.policy.eval()
    acl.policy.reset_epoch()
    obs = env.reset()
    for i in range(40):
        actions = acl.policy.predict(obs)
        obs, rew, dones, info = env.step(actions)
    attention = acl.policy.critic.attention
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('Query')
    ax.set_xlabel('Key')
    cax = ax.matshow(attention[0].to('cpu'), interpolation='nearest')
    fig.colorbar(cax)
    plt.show()

if __name__ == '__main__':
    model_path = '/home/hendrik/Documents/master_project/LokalData/server/best_validation/'
    plot_attention(model_path=model_path)
