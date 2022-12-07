import torch
from active_critic.utils.pytorch_utils import make_part_obs_data
import math


class DatasetAC(torch.utils.data.Dataset):
    def __init__(self, device='cpu'):
        self.device = device
        self.obsv = None
        self.actions = None
        self.reward = None
        self.success = None
        self.onyl_positiv = None
        self.size = None

    def __len__(self):
        if self.obsv is not None:
            if self.onyl_positiv:
                return self.success.sum()
            else:
                return self.obsv.shape[0] * self.obsv.shape[1]
        else:
            return 0

    def set_data(self, obsv: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor):
        self.reward = reward.to(self.device)
        self.obsv = obsv.to(self.device)
        self.actions = actions.to(self.device)
        self.success = self.reward[:,:, -1] == 1

    def add_data(self, obsv: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor, empty_token:float):

        acts, observations, rewards = make_part_obs_data(
            actions=actions, observations=obsv, rewards=reward, empty_token=empty_token)
        actions = acts.reshape([actions.shape[0], -1, actions.shape[1], actions.shape[2]])
        observations = observations.reshape([obsv.shape[0], -1, obsv.shape[1], obsv.shape[2]])
        rewards = rewards.reshape([reward.shape[0], -1, reward.shape[1], reward.shape[2]])

        seq_len = obsv.shape[1]
        
        if self.obsv is None:
            self.set_data(observations, actions, rewards)
        else:
            self.obsv = torch.cat((self.obsv, observations.to(self.device)), dim=0)
            self.actions = torch.cat(
                (self.actions, actions.to(self.device)), dim=0)
            self.reward = torch.cat(
                (self.reward, rewards.to(self.device)), dim=0)

        epochs = self.obsv.shape[0]
        seq_len = self.obsv.shape[1]

        in_buffer = epochs * seq_len

        if in_buffer > self.size:
            max_epochs = math.floor(self.size / seq_len)
            num_ele_delete = epochs - max_epochs
            sorted_args = self.reward[:,0, -1].argsort(dim=0).squeeze()
            keep_indices = sorted_args[num_ele_delete:]

            self.obsv = self.obsv[keep_indices]
            self.actions = self.actions[keep_indices]
            self.reward = self.reward[keep_indices]

        self.success = self.reward[:,:,-1] == 1


    def __getitem__(self, index):
        assert self.onyl_positiv is not None, 'traindata only positiv not set'
        if self.onyl_positiv:
            return self.obsv[self.success][index], self.actions[self.success][index], self.reward[self.success][index]
        else:
            #epochs, seq_len, seq_len, dim
            epochs = self.obsv.shape[0]
            seq_len = self.obsv.shape[1]
            return self.obsv.reshape([epochs*seq_len, seq_len, -1])[index], self.actions.reshape([epochs*seq_len, seq_len, -1])[index], self.reward.reshape([epochs*seq_len, seq_len, -1])[index]
