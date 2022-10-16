from pyclbr import Function
from typing import Dict, Optional, Tuple, Union

import numpy as np
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
        self.device: str = None
        self.inference_opt_lr: float = None
        self.optimize: bool = None
        self.batch_size: int = None
        self.stop_opt: bool = None
        self.opt_end: bool = None
        self.optimize_last: bool = False
        self.clip:bool = True


class ActiveCriticPolicyHistory:
    def __init__(self) -> None:
        self.reset()


    def reset(self):
        self.gen_scores = []
        self.opt_scores = []
        self.gen_trj = []


    def new_epoch(self, history:list([th.Tensor]), size:list([int, int, int]), device:str):
        new_field = th.zeros(size=size, device=device)
        if len(history) == 0:
            history.append(new_field)
        else:
            history[0] = th.cat((history[0], new_field))


    def add_value(self, history:list([th.Tensor]), value:th.Tensor, current_step:int):
        history[0][-value.shape[0]:, current_step] = value


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
        self.history = ActiveCriticPolicyHistory()
        self.clip_min = th.tensor(self.action_space.low, device=acps.device)
        self.clip_max = th.tensor(self.action_space.high, device=acps.device)
        self.reset()

    def reset(self):
        self.last_goal = None
        self.current_step = 0
        self.history.reset()

    def reset_epoch(self, vec_obsv: th.Tensor):
        self.current_step = 0
        self.last_goal = vec_obsv
        self.current_result = None

        scores_size = [vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.wsms.model_setup.d_output]
        self.history.new_epoch(self.history.gen_scores, size=scores_size, device=self.args_obj.device)
        self.history.new_epoch(self.history.opt_scores, size=scores_size, device=self.args_obj.device)

        trj_size = [vec_obsv.shape[0], self.args_obj.epoch_len, self.action_space.shape[0]]
        self.history.new_epoch(self.history.gen_trj, size=trj_size, device=self.args_obj.device)
        
        self.obs_seq = th.zeros(
            size=[vec_obsv.shape[0], self.args_obj.epoch_len, vec_obsv.shape[-1]], device=self.args_obj.device)
            

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
            observation_seq=self.obs_seq, action_seq=action_seq, 
            optimize=self.args_obj.optimize, 
            current_step=self.current_step,
            stop_opt=self.args_obj.stop_opt,
            opt_end=self.args_obj.opt_end,
            opt_last=self.args_obj.optimize_last
            )

        if self.args_obj.optimize:
            self.history.add_value(
                history=self.history.opt_scores, 
                value=self.current_result.expected_succes_after[:, self.current_step].detach(), 
                current_step=self.current_step
            )
        self.history.add_value(self.history.gen_scores, value=self.current_result.expected_succes_before[:, self.current_step].detach(), current_step=self.current_step)

        return self.current_result.gen_trj[:, self.current_step].detach().cpu().numpy()

    def forward(self, 
            observation_seq: th.Tensor, 
            action_seq: th.Tensor, 
            optimize: bool, 
            current_step: int,
            stop_opt: bool,
            opt_end: bool,
            opt_last: bool):
        # In inference, we want the maximum eventual reward.
        actor_input = self.get_actor_input(
            obs=observation_seq, actions=action_seq, rew=self.gl[:observation_seq.shape[0]])
        actions = self.actor.forward(actor_input)

        if self.args_obj.clip:
            actions = th.clamp(actions, min=self.clip_min, max=self.clip_max)

        if action_seq is not None:
            actions = self.proj_actions(
                action_seq, actions, current_step)

        self.history.add_value(self.history.gen_trj, actions[:, self.current_step].detach(), current_step=self.current_step)

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
                actions=actions, 
                observations=observation_seq, 
                current_step=current_step,
                stop_opt=stop_opt,
                opt_end=opt_end,
                opt_last=opt_last)

            return ACPOptResult(
                gen_trj=actions.detach(),
                expected_succes_before=expected_success,
                expected_succes_after=expected_success_opt)

    def optimize_act_sequence(self, 
            actions: th.Tensor, 
            observations: th.Tensor, 
            current_step: int, 
            stop_opt:bool, 
            opt_end:bool,
            opt_last:bool):

        optimized_actions = th.clone(actions.detach())
        final_actions = th.clone(optimized_actions)
        optimized_actions.requires_grad_(True)
        optimizer = th.optim.AdamW(
            [optimized_actions], lr=self.args_obj.inference_opt_lr, weight_decay=0)
        expected_success = th.zeros(
            size=[actions.shape[0], self.critic.wsms.model_setup.seq_len, self.critic.wsms.model_setup.d_output], dtype=th.float, device=actions.device)
        final_exp_success = th.clone(expected_success)
        goal_label = self.gl[:actions.shape[0]]
        step = 0
        if self.critic.model is not None:
            self.critic.model.eval()

        while (not th.all(final_exp_success[:, -1] >= self.args_obj.optimisation_threshold)) and (step <= self.args_obj.opt_steps):
            mask = (final_exp_success[:, -1] < self.args_obj.optimisation_threshold).reshape(-1)
            optimized_actions, expected_success = self.inference_opt_step(
                org_actions=actions,
                opt_actions=optimized_actions,
                obs_seq=observations,
                optimizer=optimizer,
                goal_label=goal_label,
                current_step=current_step,
                opt_end=opt_end,
                opt_last=opt_last)
            step += 1

            if stop_opt:
                final_actions[mask] = optimized_actions[mask]
                final_exp_success[mask] = expected_success[mask]
            else:
                final_actions = optimized_actions
                final_exp_success = expected_success

        if self.args_obj.clip:
            with th.no_grad():
                th.clamp(final_actions, min=self.clip_min, max=self.clip_max, out=final_actions)
        return final_actions, final_exp_success

    def inference_opt_step(self, 
            org_actions: th.Tensor, 
            opt_actions: th.Tensor, 
            obs_seq: th.Tensor, 
            optimizer: th.optim.Optimizer, 
            goal_label: th.Tensor, 
            current_step: int, 
            opt_end:bool= False,
            opt_last:bool= False):
        critic_inpt = self.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_result = self.critic.forward(inputs=critic_inpt)
        if opt_end:
            critic_loss = self.critic.loss_fct(
                result=critic_result[:,current_step:], label=goal_label[:,current_step:])
        elif opt_last:
            critic_loss = self.critic.loss_fct(
                result=critic_result[:,-1:], label=goal_label[:,-1:])
        else:
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
            if self.args_obj.clip:
                th.clamp(new_actions, min=self.clip_min, max=self.clip_max, out=new_actions)
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
