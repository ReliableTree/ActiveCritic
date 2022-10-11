from ast import Str
import copy
from pyclbr import Function
from tkinter.messagebox import NO
from typing import Dict, Optional, Tuple, Union
from unittest import result

import numpy as np
from scipy.fftpack import sc_diff
from sklearn.utils import resample
import torch as th
from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from active_critic.utils.pytorch_utils import make_partially_observed_seq
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import pickle

class ACPOptResult:
    def __init__(self, gen_trj: th.Tensor, inpt_trj: th.Tensor = None, expected_succes_before: th.Tensor = None, expected_succes_after: th.Tensor = None) -> None:
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
        self.batch_size: int = None


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
        self.args_obj = acps
        self.register_buffer('gl', th.ones(
            size=[acps.batch_size, acps.epoch_len, critic.wsms.model_setup.d_output], dtype=th.float, device=acps.device))
        self.reset()

    def reset(self):
        self.last_goal = None
        self.current_step = 0
        self.score_history_after = None
        self.score_history_before = None

    def reset_epoch(self, vec_obsv: th.Tensor):
        self.current_step = 0
        self.last_goal = vec_obsv
        self.current_result = None
        self.obs_seq = th.zeros(
            size=[vec_obsv.shape[0], self.args_obj.epoch_len, vec_obsv.shape[-1]], device=self.args_obj.device)
            
        if self.score_history_after is None:
            self.score_history_after = th.zeros(
                size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.wsms.model_setup.d_output], device=self.args_obj.device)
        else:
            self.score_history_after = th.cat((self.score_history_after, th.zeros(
                size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.wsms.model_setup.d_output], device=self.args_obj.device)))

        if self.score_history_before is None:
            self.score_history_before = th.zeros(
                size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.wsms.model_setup.d_output], device=self.args_obj.device)
        else:
            self.score_history_before = th.cat((self.score_history_before, th.zeros(
                size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.wsms.model_setup.d_output], device=self.args_obj.device)))

    def predict(
        self,
        observation: Union[th.Tensor, Dict[str, th.Tensor]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        vec_obsv = self.args_obj.extractor.forward(
            observation).to(self.args_obj.device).unsqueeze(1)

        if (self.last_goal is None) or (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
            self.reset_epoch(vec_obsv=vec_obsv)
            action_seq = None
        else:
            self.current_step += 1
            action_seq = self.current_result.gen_trj

        self.obs_seq[:, self.current_step:self.current_step+1, :] = vec_obsv

        self.current_result = self.forward(
            observation_seq=self.obs_seq, action_seq=action_seq, optimize=self.args_obj.optimize, current_step=self.current_step)
        batch_size = self.current_result.expected_succes_before.shape[0]
        if self.args_obj.optimize:
            self.score_history_after[-batch_size:,
                            self.current_step] = self.current_result.expected_succes_after[:, self.current_step].detach()

        self.score_history_before[-batch_size:,
                        self.current_step] = self.current_result.expected_succes_before[:, self.current_step].detach()

        return self.current_result.gen_trj[:, self.current_step].detach().cpu().numpy()

    def forward(self, observation_seq: th.Tensor, action_seq: th.Tensor, optimize: bool, current_step: int):
        # In inference, we want the maximum eventual reward.
        actor_input = self.get_actor_input(
            obs=observation_seq, actions=action_seq, rew=self.gl[:observation_seq.shape[0]])
        actions = self.actor.forward(actor_input)

        if action_seq is not None:
            actions = self.proj_actions(
                action_seq, actions, current_step)

        critic_input = self.get_critic_input(
            acts=actions, obs_seq=observation_seq)

        expected_success = self.critic.forward(
            inputs=critic_input)  # batch_size, seq_len, 1

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

    def optimize_act_sequence(self, actions: th.Tensor, observations: th.Tensor, current_step: int):
        optimized_actions = th.clone(actions.detach())
        optimized_actions.requires_grad_(True)
        optimizer = th.optim.Adam(
            [optimized_actions], lr=self.args_obj.inference_opt_lr)
        expected_success = th.zeros(
            size=[actions.shape[0], self.critic.wsms.model_setup.seq_len, self.critic.wsms.model_setup.d_output], dtype=th.float, device=actions.device)
        goal_label = self.gl[:actions.shape[0]]
        step = 0
        if self.critic.model is not None:
            self.critic.model.eval()
        while (not th.all(expected_success[:, -1] >= self.args_obj.optimisation_threshold)) and (step <= self.args_obj.opt_steps):

            optimized_actions, expected_success = self.inference_opt_step(
                org_actions=actions,
                opt_actions=optimized_actions,
                obs_seq=observations,
                optimizer=optimizer,
                goal_label=goal_label,
                current_step=current_step)
            step += 1

        return optimized_actions, expected_success

    def inference_opt_step(self, org_actions: th.Tensor, opt_actions: th.Tensor, obs_seq: th.Tensor, optimizer: th.optim.Optimizer, goal_label: th.Tensor, current_step: int):
        critic_inpt = self.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_result = self.critic.forward(inputs=critic_inpt)
        critic_loss = self.critic.loss_fct(
            result=critic_result, label=goal_label)

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()

        actions = self.proj_actions(
            org_actions=org_actions, new_actions=opt_actions, current_step=current_step)
        return actions, critic_result

    def get_critic_input(self, acts, obs_seq):
        critic_input = make_partially_observed_seq(
            obs=obs_seq, acts=acts, seq_len=self.args_obj.epoch_len, act_space=self.action_space)
        return critic_input

    def get_actor_input(self, obs: th.Tensor, actions: th.Tensor, rew: th.Tensor):
        last_reward = rew[:, -1:].repeat([1, obs.shape[1], 1])

        actor_inpt = th.cat((obs, last_reward), dim=-1)
        return actor_inpt

    def proj_actions(self, org_actions: th.Tensor, new_actions: th.Tensor, current_step: int):
        with th.no_grad():
            new_actions[:, :current_step] = org_actions[:, :current_step]
        return new_actions

    def save_policy(self, add, data_path):

        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)

        th.save(self.state_dict(), path_to_file + "/policy_network")
        th.save(self.actor.optimizer.state_dict(), path_to_file + "/optimizer_actor")
        th.save(self.critic.optimizer.state_dict(), path_to_file + "/optimizer_critic")
        with open(path_to_file + '/policy_args.pkl', 'wb') as f:
            pickle.dump(self.args_obj, f)
