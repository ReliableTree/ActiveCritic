from pyclbr import Function
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from active_critic.model_src.state_model import StateModel
from active_critic.utils.pytorch_utils import get_rew_mask, get_seq_end_mask, make_partially_observed_seq
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
        self.clip:bool = True
        self.take_predicted_action = None


class ActiveCriticPolicyHistory:
    def __init__(self) -> None:
        self.reset()


    def reset(self):
        self.scores = []
        self.trj = []
        self.emb = []

        self.goal_scores = []
        self.pred_emb = []
        self.act_emb = []


    def new_epoch(self, history:list([th.Tensor]), size:list([int, int, int, int]), device:str): #batch size, opt step, seq len, score
        new_field = th.zeros(size=size, device=device)
        if len(history) == 0:
            history.append(new_field)
        else:
            history[0] = th.cat((history[0], new_field), dim=0)


    def add_value(self, history:list([th.Tensor]), value:th.Tensor, opt_step:int=0, step:int=None):
        if len(history[0].shape) == 4: #including opt step history
            history[0][-value.shape[0]:, opt_step] = value.detach()
        elif step is None:
            history[0][-value.shape[0]:] = value.detach()
        else:
            history[0][-value.shape[0]:, step:step+1] = value.detach()

    def add_opt_stp_seq(self, history:list([th.Tensor]), value:th.Tensor, opt_step, step):
        #value: batch_size, seq_len, dim
        #history: epochs, opt_step, step, seq_len, dim
        history[0][-value.shape[0]:, opt_step, step] = value.detach()


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
            size=[1000, acps.epoch_len, critic.wsms.model_setup.d_output], dtype=th.float, device=acps.device))
        self.history = ActiveCriticPolicyHistory()
        self.clip_min = th.tensor(self.action_space.low, device=acps.device)
        self.clip_max = th.tensor(self.action_space.high, device=acps.device)
        self.inference = False
        self.reset()

    def reset_models(self):
        self.critic.init_model()
        self.actor.init_model()

    def reset(self):
        self.last_goal = None
        self.current_step = 0
        self.history.reset()

    def reset_epoch(self, vec_obsv:th.Tensor):
        self.current_step = 0
        self.last_goal = vec_obsv[:,0,-3:]

        self.actor.model        

        self.history.new_epoch(history=self.history.trj, size=[vec_obsv.shape[0], self.args_obj.opt_steps + 2, self.args_obj.epoch_len, self.args_obj.epoch_len, self.actor.wsms.model_setup.d_output], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.scores, size=[vec_obsv.shape[0], self.args_obj.opt_steps + 2, self.args_obj.epoch_len, self.args_obj.epoch_len, self.critic.wsms.model_setup.d_output], device=self.args_obj.device)

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

        if (action_seq is None) or (self.inference):
            self.current_result = self.forward(
                observation_seq=self.obs_seq, action_seq=action_seq, 
                optimize=self.args_obj.optimize,
                current_step=self.current_step,
                stop_opt=self.args_obj.stop_opt
                )

        return self.current_result.gen_trj[:, self.current_step].detach().cpu().numpy()

    def forward(self, 
            observation_seq: th.Tensor, 
            action_seq: th.Tensor, 
            optimize: bool, 
            current_step: int,
            stop_opt: bool
            ):
        # In inference, we want the maximum eventual reward.
        actor_input = self.get_actor_input(
            obs=observation_seq, actions=action_seq, rew=self.gl[:observation_seq.shape[0]])
        actions = self.actor.forward(actor_input, offset=0)

        if self.args_obj.clip:
            actions = th.clamp(actions, min=self.clip_min, max=self.clip_max)

        if action_seq is not None:
            batch_size = observation_seq.shape[0]
            steps = th.tensor([current_step], dtype=int, device=observation_seq.device).repeat([batch_size])   

            actions = self.proj_actions(
                org_actions=action_seq, new_actions=actions, steps=steps)
            if self.args_obj.take_predicted_action:
                assert th.all(actions[:, :current_step+1] == action_seq[:, :current_step+1])

        critic_input = self.get_critic_input(
            acts=actions, obs_seq=observation_seq)
        expected_success = self.critic.forward(
            inputs=critic_input, offset=0)  # batch_size, seq_len, 1

        self.history.add_opt_stp_seq(self.history.trj, actions, 0, self.current_step)
        self.history.add_opt_stp_seq(self.history.scores, expected_success, 0, self.current_step)

        if not optimize:
            result = ACPOptResult(
                gen_trj=actions, expected_succes_before=expected_success.detach())
            return result

        else:
            actions, expected_success_opt = self.optimize_act_sequence(
                actions=actions, 
                observations=observation_seq, 
                current_step=current_step,
                stop_opt=stop_opt
                )

            return ACPOptResult(
                gen_trj=actions.detach(),
                expected_succes_before=expected_success,
                expected_succes_after=expected_success_opt)

    def optimize_act_sequence(self, 
            actions: th.Tensor, 
            observations: th.Tensor, 
            current_step: int, 
            stop_opt:bool
            ):
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

        while (not th.all(final_exp_success.max(dim=1)[0] >= self.args_obj.optimisation_threshold)) and (step <= self.args_obj.opt_steps):
            mask = (final_exp_success.max(dim=1)[0] < self.args_obj.optimisation_threshold).reshape(-1)
            optimized_actions, expected_success = self.inference_opt_step(
                org_actions=actions,
                opt_actions=optimized_actions,
                obs_seq=observations,
                optimizer=optimizer,
                goal_label=goal_label,
                current_step=current_step
                )
            step += 1

            if stop_opt:
                final_actions[mask] = optimized_actions[mask]
                final_exp_success[mask] = expected_success[mask]
            else:
                final_actions = optimized_actions
                final_exp_success = expected_success

            self.history.add_opt_stp_seq(self.history.trj, optimized_actions, step, self.current_step)
            self.history.add_opt_stp_seq(self.history.scores, expected_success, step, self.current_step)

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
            current_step: int
            ):
        critic_inpt = self.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_result = self.critic.forward(inputs=critic_inpt, offset=0)

        mask = get_seq_end_mask(critic_result, current_step)
        critic_loss = self.critic.loss_fct(result=critic_result, label=goal_label, mask=mask)

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()

        batch_size = obs_seq.shape[0]
        steps = th.tensor([current_step], dtype=int, device=org_actions.device).repeat([batch_size])
        actions = self.proj_actions(
            org_actions=org_actions, new_actions=opt_actions, current_step=steps)
        if self.args_obj.take_predicted_action:
            assert th.all(actions[:, :current_step+1] == org_actions[:, :current_step+1])
        return actions, critic_result

    def get_critic_input(self, acts, obs_seq):
        critic_input = make_partially_observed_seq(
            obs=obs_seq, acts=acts, seq_len=self.args_obj.epoch_len, act_space=self.action_space)
        return critic_input

    def get_actor_input(self, obs: th.Tensor, actions: th.Tensor, rew: th.Tensor):
        #max_reward, _ = rew.max(dim=1)
        max_reward = rew[:,-1]
        max_reward = max_reward.unsqueeze(1).repeat([1, obs.shape[1], 1])
        
        actor_inpt = th.cat((obs, max_reward), dim=-1)
        return actor_inpt

    def make_step_mask(self, steps, seq_len, device):
        indices_tensor = th.arange(seq_len, device=device).unsqueeze(0)
        mask = indices_tensor < steps[:, None]
        return mask

    def proj_actions(self, org_actions: th.Tensor, new_actions: th.Tensor, steps: int):
            with th.no_grad():
                device = org_actions.device
                if self.args_obj.take_predicted_action:
                    step_input = steps + 1
                    step_input[step_input == 1] = 0
                    mask = self.make_step_mask(steps=step_input, seq_len=org_actions.shape[1], device=device)
                else:
                    mask = self.make_step_mask(steps=steps, seq_len=org_actions.shape[1], device=device)
                new_actions[mask] = org_actions[mask]
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
