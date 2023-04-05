import torch
import math
from active_critic.utils.pytorch_utils import pick_action_from_history
class DatasetAC(torch.utils.data.Dataset):
    def __init__(self, batch_size, device='cpu', max_size = None):
        self.device = device
        self.obsv = None
        self.actions = None
        self.reward = None
        self.success = None
        self.onyl_positiv = None
        self.batch_size = batch_size
        self.max_size = max_size

    def __len__(self):
        if self.obsv is not None:
            if self.onyl_positiv:
                return self.virt_success.sum()
            else:
                return len(self.virt_obsv)
        else:
            return 0

    def set_data(self, obsv: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor, expert_trjs: torch.Tensor, actions_history : torch.Tensor, steps:torch.Tensor):
        self.reward = reward.to(self.device)
        self.obsv = obsv.to(self.device)
        self.actions = actions.to(self.device)
        previous_steps = steps - 1
        previous_steps[previous_steps<0] = 0
        prev_proposed_acts = pick_action_from_history(action_histories=actions_history, steps=previous_steps)
        self.prev_proposed_acts = prev_proposed_acts.to(self.device)
        success = self.reward.reshape([obsv.shape[0], obsv.shape[1]]).max(-1).values
        self.success = (success == 1)
        self.expert_trjs = expert_trjs.to(self.device)
        self.steps = steps.reshape([-1]).to(self.device)
        self.make_virt_data()


    def make_virt_data(self):
        if len(self.obsv) < self.batch_size:
            rep_fac = math.ceil(self.batch_size / len(self.obsv)) + 1
            self.virt_reward = self.reward.repeat([rep_fac, 1, 1])
            self.virt_obsv = self.obsv.repeat([rep_fac, 1, 1])
            self.virt_actions = self.actions.repeat([rep_fac, 1, 1])
            self.virt_success = self.success.repeat([rep_fac])
            self.virt_expert_trjs = self.expert_trjs.repeat([rep_fac])
            self.virt_prev_proposed_acts = self.prev_proposed_acts.repeat([rep_fac, 1, 1])
            self.virt_steps = self.steps.repeat([rep_fac])
        else:
            self.virt_reward = self.reward
            self.virt_obsv = self.obsv
            self.virt_actions = self.actions
            self.virt_success = self.success
            self.virt_expert_trjs = self.expert_trjs
            self.virt_actions_at_time_t = self.prev_proposed_acts
            self.virt_prev_proposed_acts = self.prev_proposed_acts
            self.virt_steps = self.steps




    def add_data(self, 
                 obsv: torch.Tensor, 
                 actions: torch.Tensor, 
                 reward: torch.Tensor, 
                 expert_trjs: torch.Tensor, 
                 actions_history : torch.Tensor,
                 steps : torch.Tensor):
        if self.obsv is None:
            self.set_data(obsv=obsv, actions=actions, reward=reward, expert_trjs=expert_trjs, actions_history=actions_history, steps=steps)
        else:
            self.obsv = torch.cat((self.obsv, obsv.to(self.device)), dim=0)
            self.actions = torch.cat(
                (self.actions, actions.to(self.device)), dim=0)
            self.reward = torch.cat(
                (self.reward, reward.to(self.device)), dim=0)
            
            success = reward.reshape([obsv.shape[0], obsv.shape[1]]).max(-1).values == 1
            self.success = torch.cat(
                (self.success, (success).to(self.device)), dim=0)
            self.expert_trjs = torch.cat((
                self.expert_trjs, expert_trjs.to(self.device)
            ), dim=0)
            previous_steps = steps - 1
            previous_steps[previous_steps<0] = 0
            prev_proposed_acts = pick_action_from_history(action_histories=actions_history, steps=previous_steps)
            self.prev_proposed_acts = torch.cat((
                self.prev_proposed_acts, prev_proposed_acts.to(self.device)
            ), dim=0)
            self.steps = torch.cat((
                self.steps, steps.reshape([-1]).to(self.device)
            ), dim=0)
        self.trunc_data()
        self.make_virt_data()

    def trunc_data(self):
        if self.max_size is not None:
            self.reward = self.reward[-self.max_size:]
            self.obsv = self.obsv[-self.max_size:]
            self.actions = self.actions[-self.max_size:]
            self.success = self.success[-self.max_size:]
            self.expert_trjs = self.expert_trjs[-self.max_size:]
            self.prev_proposed_acts = self.prev_proposed_acts[-self.max_size:]
            self.steps = self.steps[-self.max_size:]

    def __getitem__(self, index):
        assert self.onyl_positiv is not None, 'traindata only positiv not set'
        return (self.virt_obsv[index], 
                self.virt_actions[index], 
                self.virt_reward[index], 
                self.virt_expert_trjs[index],
                self.virt_prev_proposed_acts[index],
                self.virt_steps[index],
                self.virt_obsv[max(index-1, 0)],
                self.virt_success[index])
    
        '''if self.onyl_positiv:
            return (self.virt_obsv[self.virt_success][index], 
                    self.virt_actions[self.virt_success][index], 
                    self.virt_reward[self.virt_success][index], 
                    self.virt_expert_trjs[self.virt_success][index], 
                    self.virt_prev_proposed_acts[self.virt_success][index],
                    self.virt_steps[self.virt_success][index],
                    self.virt_obsv[self.virt_success][max(index-1, 0)])
        else:
            return (self.virt_obsv[index], 
                    self.virt_actions[index], 
                    self.virt_reward[index], 
                    self.virt_expert_trjs[index],
                    self.virt_prev_proposed_acts[index],
                    self.virt_steps[index],
                    self.virt_obsv[max(index-1, 0)])'''

        