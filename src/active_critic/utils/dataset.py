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

    def set_data(self, obsv: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor, expert_trjs: torch.Tensor):
        self.reward = reward.to(self.device)
        self.obsv = obsv.to(self.device)
        self.actions = actions.to(self.device)
        success = self.reward.squeeze().max(-1).values
        self.success = (success == 1)
        self.expert_trjs = expert_trjs.to(self.device)

    def add_data(self, obsv: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor, expert_trjs: torch.Tensor):
        if self.obsv is None:
            self.set_data(obsv=obsv, actions=actions, reward=reward, expert_trjs=expert_trjs)
        else:
            self.obsv = torch.cat((self.obsv, obsv.to(self.device)), dim=0)
            self.actions = torch.cat(
                (self.actions, actions.to(self.device)), dim=0)
            self.reward = torch.cat(
                (self.reward, reward.to(self.device)), dim=0)
            
            success = reward.squeeze().max(-1).values == 1
            self.success = torch.cat(
                (self.success, (success).to(self.device)), dim=0)
            self.expert_trjs = torch.cat((
                self.expert_trjs, expert_trjs.to(self.device)
            ), dim=-1)

    def __getitem__(self, index):
        assert self.onyl_positiv is not None, 'traindata only positiv not set'
        if self.onyl_positiv:
            return self.obsv[self.success][index], self.actions[self.success][index], self.reward[self.success][index], self.expert_trjs[self.success][index]
        else:
            return self.obsv[index], self.actions[index], self.reward[index], self.expert_trjs[index]