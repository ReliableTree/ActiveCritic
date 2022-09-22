from ast import Str
import copy
from pyclbr import Function
from tkinter.messagebox import NO
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from ActiveCritic.model_src.whole_sequence_model import (
    WholeSequenceActor, WholeSequenceCritic, WholeSequenceModelSetup)
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
        self.writer = None
        self.plotter = None


class ActiveCriticPolicy(BaseModel):
    def __init__(
        self,
        observation_space,
        action_space,
        actor: WholeSequenceActor,
        critic: WholeSequenceCritic,
        return_mode=0,
        acps: ActiveCriticPolicySetup = None
    ):

        super().__init__(observation_space, action_space)

        self.actor = actor
        self.critic = critic
        self.return_mode = return_mode
        self.optim_run = 0
        self.max_steps = acps.opt_steps
        self.last_update = 0
        self.last_goal = None
        self.current_step = 0
        self.args_obj = acps
        self.reset_state()


    def reset_state(self):
        self.best_expected_success = None
        self.scores = []

    def predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        vec_obsv = self.args_obj.extractor.forward(
            observation).unsqueeze(1).to(self.args_obj.device)

        if (self.last_goal is None) or (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
            if self.best_expected_success is not None:
                self.scores.append((self.best_expected_success > self.args_obj.optimisation_threshold).type(torch.bool))
            self.current_step = 0
            self.last_goal = vec_obsv
            self.action_seq = None
            self.obs_seq = vec_obsv
        else:
            self.obs_seq = torch.cat((self.obs_seq, vec_obsv), dim=1)
            self.current_step += 1
        model_inpt, self.act_dim = make_partially_observed_seq(
            obs=self.obs_seq, acts=self.action_seq, seq_len=self.args_obj.epoch_len, act_space=self.action_space)
        self.action_seq = self.forward(
            inpt=model_inpt).gen_trj.detach()
        print(f'current step: {self.current_step}')
        return self.action_seq[:, self.current_step].cpu().numpy()


    def forward(self, inpt):
        main_signal_state_dict = copy.deepcopy(
            self.actor.model.state_dict())
        gen_result = self.actor.forward(inpt).gen_trj
        if self.return_mode == 0:
            result = ACPOptResult(gen_trj=gen_result)
            return result
        elif self.return_mode == 1:
            opt_gen_result = torch.clone(gen_result.detach())
            opt_gen_result.requires_grad_(True)

            optimizer = torch.optim.Adam([opt_gen_result], lr=self.lr)
            best_expected_success = None
            best_expected_mean = 0
            best_trj = torch.clone(gen_result)
            step = 0
            if self.critic.model is not None:
                self.critic.model.eval()
            gen_result = gen_result.detach()
            while best_expected_mean < self.args_obj.optimisation_threshold and (step <= self.max_steps):

                critic_input, _ = make_partially_observed_seq(
                    obs=self.obs_seq, acts=opt_gen_result, seq_len=self.args_obj.epoch_len, act_space=self.action_space)

                critic_result = self.critic.forward(inputs=critic_input)

                goal_label = torch.ones_like(critic_result)
                critic_loss = self.critic.loss_fct(
                    inpt=critic_result, success=goal_label)

                if best_expected_success is None:
                    best_expected_success = torch.clone(critic_result)
                    best_trj = opt_gen_result.detach()
                    expected_succes_before = torch.clone(critic_result)
                else:
                    improve_mask = (critic_result >
                                    best_expected_success).reshape(-1)
                    best_expected_success[improve_mask] = critic_result[improve_mask].detach(
                    )
                    best_trj[improve_mask] = opt_gen_result[improve_mask].detach()
                self.best_expected_success = best_expected_success
                best_expected_mean = best_expected_success.mean()

                optimizer.zero_grad()
                critic_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    opt_gen_result[:, :self.current_step,
                                   :] = gen_result[:, :self.current_step, :]

                assert torch.equal(opt_gen_result[:, :self.current_step, :],
                                   gen_result[:, :self.current_step, :]), 'previous actions changed'
                assert not torch.equal(
                    opt_gen_result, gen_result), 'nothing changed'

                self.writer(
                    {str(self.optim_run) + ' in optimisation ': best_expected_mean.detach()}, train=False, step=step)
                mask = torch.zeros(gen_result.size(0), dtype=torch.bool)
                mask[0] = True

                self.plotter(
                    label=gen_result.detach(),
                    trj=gen_result.detach(),
                    inpt=inpt,
                    mask=mask,
                    opt_trj=best_trj.detach(),
                    name='in active critic'
                )
                step += 1
            self.actor.model.load_state_dict(
                main_signal_state_dict, strict=False)
            return ACPOptResult(gen_trj=best_trj.detach(), inpt_trj=gen_result.detach(), expected_succes_before=expected_succes_before, expected_succes_after=best_expected_mean)
