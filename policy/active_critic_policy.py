from ast import Str
import copy
from pyclbr import Function
from tkinter.messagebox import NO
from typing import Dict, Optional, Tuple, Union
from matplotlib.pyplot import step

import numpy as np
import torch
from ActiveCritic.model_src.whole_sequence_model import WholeSequenceModel
from ActiveCritic.utils.pytorch_utils import make_partially_observed_seq
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ACPOptResult:
    def __init__(self, gen_trj: torch.Tensor, inpt_trj: torch.Tensor = None, expected_succes_before: torch.Tensor = None, expected_succes_after: torch.Tensor = None) -> None:
        self.gen_trj = gen_trj
        self.inpt_trj = inpt_trj
        self.expected_succes_before = expected_succes_before
        self.expected_succes_after = expected_succes_after


class ActiveCriticPolicySetup:
    def __init__(self) -> None:
        self.extractor: BaseFeaturesExtractor = None
        self.new_epoch: Function = None
        self.optimisation_threshold: float = None
        self.epoch_len: int = None
        self.opt_steps: int = None
        self.device: Str = None
        self.inference_opt_lr: float = None
        self.optimize: bool = None
        self.writer = None
        self.plotter = None


class ActiveCriticPolicy(BaseModel):
    def __init__(
        self,
        observation_space,
        action_space,
        actor: WholeSequenceModel,
        critic: WholeSequenceModel,
        acps: ActiveCriticPolicySetup = None
    ):

        super().__init__(observation_space, action_space)

        self.actor = actor
        self.critic = critic
        self.optim_run = 0
        self.last_update = 0
        self.last_goal = None
        self.current_step = 0
        self.args_obj = acps

    def predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        vec_obsv = self.args_obj.extractor.forward(
            observation).to(self.args_obj.device).unsqueeze(1)

        if (self.last_goal is None) or (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
            self.current_step = 0
            self.last_goal = vec_obsv
            self.current_result = None
            action_seq = None
            self.obs_seq = torch.zeros(
                size=[observation.shape[0], self.args_obj.epoch_len, observation.shape[-1]], device=self.args_obj.device)
        else:
            self.current_step += 1
            action_seq = self.current_result.gen_trj

        self.obs_seq[:, self.current_step:self.current_step+1, :] = vec_obsv

        self.current_result = self.forward(
            observation_seq=self.obs_seq, action_seq=action_seq, optimize=self.args_obj.optimize, current_step=self.current_step)
        return self.current_result.gen_trj[:, self.current_step].cpu().numpy()

    def forward(self, observation_seq: torch.Tensor, action_seq: torch.Tensor, optimize: bool, current_step: int):
        actions = self.actor.forward(observation_seq)
        if action_seq is not None:
            actions = self.proj_actions(
                action_seq, actions, current_step)
        expected_success = self.get_critic_score(
            acts=actions, obs_seq=observation_seq)
        expected_success = expected_success[:,-1]
        if not optimize:
            result = ACPOptResult(
                gen_trj=actions, expected_succes_before=expected_success.detach())
            return result

        else:
            actions, expected_success_opt = self.optimize_act_sequence(
                actions=actions, observations=observation_seq, current_step=current_step)

            return ACPOptResult(
                gen_trj=actions.detach(),
                expected_succes_before=expected_success,
                expected_succes_after=expected_success_opt)

    def optimize_act_sequence(self, actions: torch.Tensor, observations: torch.Tensor, current_step: int):
        optimized_actions = torch.clone(actions.detach())
        optimized_actions.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [optimized_actions], lr=self.args_obj.inference_opt_lr)
        expected_success = torch.zeros(
            size=[actions.shape[0], 1, self.critic.wsms.model_setup.d_output], dtype=torch.float, device=actions.device)
        goal_label = torch.ones(
            size=[actions.shape[0], self.critic.wsms.model_setup.seq_len, self.critic.wsms.model_setup.d_output], dtype=torch.float, device=actions.device)
        step = 0
        if self.critic.model is not None:
            self.critic.model.eval()
        while (not torch.all(expected_success >= self.args_obj.optimisation_threshold)) and (step <= self.args_obj.opt_steps):

            optimized_actions, expected_success = self.inference_opt_step(
                org_actions=actions,
                opt_actions=optimized_actions,
                obs_seq=observations,
                optimizer=optimizer,
                goal_label=goal_label,
                current_step=current_step)
            step += 1

        return optimized_actions, expected_success

    def inference_opt_step(self, org_actions: torch.Tensor, opt_actions: torch.Tensor, obs_seq: torch.Tensor, optimizer: torch.optim.Optimizer, goal_label: torch.Tensor, current_step: int):
        critic_result = self.get_critic_score(
            acts=opt_actions, obs_seq=obs_seq)
        critic_loss = self.critic.loss_fct(
            result=critic_result, label=goal_label)

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()

        actions = self.proj_actions(
            org_actions=org_actions, new_actions=opt_actions, current_step=current_step)
        return actions, critic_result[:,-1]

    def get_critic_score(self, acts, obs_seq):
        critic_input = make_partially_observed_seq(
            obs=obs_seq, acts=acts, seq_len=self.args_obj.epoch_len, act_space=self.action_space)

        critic_result = self.critic.forward(inputs=critic_input) #batch_size, seq_len, 1
        return critic_result

    def proj_actions(self, org_actions: torch.Tensor, new_actions: torch.Tensor, current_step: int):
        with torch.no_grad():
            new_actions[:, :current_step] = org_actions[:, :current_step]
        return new_actions
