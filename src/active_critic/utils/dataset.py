import torch


class DatasetAC(torch.utils.data.Dataset):
    def __init__(self, device='cpu'):
        self.device = device
        self.obsv = None
        self.actions = None
        self.reward = None
        self.success = None
        self.onyl_positiv = None

    def __len__(self):
        if self.obsv is not None:
            if self.onyl_positiv:
                return self.success.sum()
            else:
                return len(self.obsv)
        else:
            return 0

    def set_data(self, obsv: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor):
        self.reward = reward.to(self.device)
        self.obsv = obsv.to(self.device)
        self.actions = actions.to(self.device)
        self.success = self.reward[:, -1] == 1

    def add_data(self, obsv: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor, replay_buffer_size:int = 10000000):
        if self.obsv is None:
            self.set_data(obsv, actions, reward)
        else:
            self.obsv = torch.cat((self.obsv, obsv.to(self.device)), dim=0)[:replay_buffer_size]
            self.actions = torch.cat(
                (self.actions, actions.to(self.device)), dim=0)[:replay_buffer_size]
            self.reward = torch.cat(
                (self.reward, reward.to(self.device)), dim=0)[:replay_buffer_size]
            self.success = torch.cat(
                (self.success, (reward[:, -1] == 1).to(self.device)), dim=0)[:replay_buffer_size]

    def __getitem__(self, index):
        assert self.onyl_positiv is not None, 'traindata only positiv not set'
        if self.onyl_positiv:
            return self.obsv[self.success][index], self.actions[self.success][index], self.reward[self.success][index]
        else:
            return self.obsv[index], self.actions[index], self.reward[index]
