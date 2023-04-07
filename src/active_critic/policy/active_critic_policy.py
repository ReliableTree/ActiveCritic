from pyclbr import Function
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from active_critic.model_src.whole_sequence_model import WholeSequenceModel, CriticSequenceModel
from active_critic.utils.pytorch_utils import get_rew_mask, get_seq_end_mask, make_partially_observed_seq
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import pickle
import copy

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
        self.optimizer_mode : str = None
        self.clip:bool = True
        self.buffer_size:int = None

class ActiveCriticPolicyHistory:
    def __init__(self) -> None:
        self.reset()
        self.obsv_buffer = []
        self.act_buffer = []

    def reset(self):
        self.gen_scores = []
        self.opt_scores = []
        self.gen_trj = []


    def new_epoch(self, history:list([th.Tensor]), size:list([int, int, int]), device:str):
        new_field = th.zeros(size=size, device=device)
        if len(history) == 0:
            history.append(new_field)
            history.append(0)
        else:
            history[0] = th.cat((history[0], new_field))


    def add_value(self, history:list([th.Tensor]), value:th.Tensor, current_step:int):
        history[0][-value.shape[0]:, current_step] = value

    def add_buffer_value(self, history:list([th.Tensor]), value:th.Tensor, current_step:int, index:int, training_mode:bool):
        if training_mode:
            history[0][index:index+value.shape[0], current_step] = value

class ActiveCriticPolicy(BaseModel):
    def __init__(
        self,
        observation_space,
        action_space,
        actor: WholeSequenceModel,
        critic: CriticSequenceModel,
        acps: ActiveCriticPolicySetup = None,
        planner: WholeSequenceModel = None,
        write_tboard_scalar = None
    ):

        super().__init__(observation_space, action_space)

        self.actor = actor
        self.critic = critic
        self.planner = planner
        self.args_obj = acps
        self.register_buffer('gl', th.ones(
            size=[100, self.args_obj.epoch_len + 2, 1], dtype=th.float, device=acps.device))
        self.history = ActiveCriticPolicyHistory()
        self.clip_min = th.tensor(self.action_space.low, device=acps.device)
        self.clip_max = th.tensor(self.action_space.high, device=acps.device)
        self.n_inferred = 0
        self.write_tboard_scalar = write_tboard_scalar
        self.reset()

        obsv_buffer_size = [self.args_obj.buffer_size, self.args_obj.epoch_len, self.args_obj.epoch_len, self.observation_space.shape[0]]
        self.history.new_epoch(self.history.obsv_buffer, size=obsv_buffer_size, device=self.args_obj.device)
        acts_buffer_size = [self.args_obj.buffer_size, self.args_obj.epoch_len, self.args_obj.epoch_len, self.action_space.shape[0]]
        self.history.new_epoch(self.history.act_buffer, size=acts_buffer_size, device=self.args_obj.device)


        self.buffer_index = None
        self.buffer_filled_to = 0

        self.training_mode = False
        self.start_training = True

    def reset(self):
        self.last_goal = None
        self.current_step = 0
        self.history.reset()

    def reset_epoch(self, vec_obsv: th.Tensor):
        self.n_inferred += 1
        self.current_step = 0
        self.last_goal = vec_obsv
        self.current_result = None

        scores_size = [vec_obsv.shape[0], 1]
        self.history.new_epoch(self.history.gen_scores, size=scores_size, device=self.args_obj.device)
        self.history.new_epoch(self.history.opt_scores, size=scores_size, device=self.args_obj.device)

        trj_size = [vec_obsv.shape[0], self.args_obj.epoch_len, self.args_obj.epoch_len, self.action_space.shape[0]]
        self.history.new_epoch(self.history.gen_trj, size=trj_size, device=self.args_obj.device)
        
        self.obs_seq = th.zeros(
            size=[vec_obsv.shape[0], self.args_obj.epoch_len, vec_obsv.shape[-1]], device=self.args_obj.device)
        
        if self.training_mode:
            if self.buffer_index is None:
                self.buffer_index = 0
            else:
                self.buffer_index = (self.buffer_index + vec_obsv.shape[0])%(self.args_obj.buffer_size)
                self.buffer_filled_to += vec_obsv.shape[0]
                assert self.buffer_index + vec_obsv.shape[0] <= self.args_obj.buffer_size, 'Buffer size must be modulo 0'
        

    def predict(
        self,
        observation: Union[th.Tensor, Dict[str, th.Tensor]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> th.Tensor:
        vec_obsv = self.args_obj.extractor.forward(
            observation).to(self.args_obj.device).unsqueeze(1)
            
        if (self.last_goal is None) or (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
                print(f'infernce lr in policy: {self.args_obj.inference_opt_lr}')
                self.reset_epoch(vec_obsv=vec_obsv)
                action_seq = None
                #T observations x T actions timesteps
                self.action_history = th.zeros([vec_obsv.shape[0], self.args_obj.epoch_len, self.args_obj.epoch_len, self.action_space.shape[0]], device=self.args_obj.device, dtype=th.float)
        else:
            self.current_step += 1
            if self.start_training:
                action_seq = self.current_result.gen_trj

        if self.start_training:
            self.obs_seq[:, self.current_step:self.current_step+1, :] = vec_obsv
            self.history.add_buffer_value(
                self.history.obsv_buffer, 
                value=th.clone(self.obs_seq.detach()), 
                current_step=self.current_step, 
                index=self.buffer_index,
                training_mode=self.training_mode)


            #self.obs_seq = vec_obsv.repeat([1, self.obs_seq.shape[1], 1]).type(th.float)
            #if action_seq is None:
            self.current_result = self.forward(
                observation_seq=self.obs_seq, action_seq=action_seq, 
                optimize=self.args_obj.optimize, 
                current_step=self.current_step,
                stop_opt=self.args_obj.stop_opt
                )
            self.history.add_buffer_value(
                self.history.act_buffer, 
                value=th.clone(self.current_result.gen_trj.detach()), 
                current_step=self.current_step, 
                index=self.buffer_index,
                training_mode=self.training_mode)
            
            self.action_history[:, self.current_step] = th.clone(self.current_result.gen_trj.detach())

            if self.current_step == 0:
                if self.args_obj.optimize:
                    self.history.add_value(
                        history=self.history.opt_scores, 
                        value=self.current_result.expected_succes_after[:, 0].detach(), 
                        current_step=self.current_step
                    )
                self.history.add_value(self.history.gen_scores, value=self.current_result.expected_succes_before[:, 0].detach(), current_step=self.current_step)
            return self.current_result.gen_trj[:, self.current_step].detach().cpu().numpy()
        else:
            return np.random.rand(vec_obsv.shape[0], self.action_space.shape[0]) * 2 - 1

    def forward(self, 
            observation_seq: th.Tensor, 
            action_seq: th.Tensor, 
            optimize: bool, 
            current_step: int,
            stop_opt: bool
            ):
        
        

        plans = th.zeros([observation_seq.shape[0], observation_seq.shape[1], self.planner.wsms.model_setup.d_output], device=self.args_obj.device, dtype=th.float32)
        actions = self.make_action(action_seq=action_seq, observation_seq=observation_seq, plans=plans, current_step=current_step)

        self.history.add_value(self.history.gen_trj, actions.detach(), current_step=current_step)
        critic_input = self.get_critic_input(obsvs=observation_seq, acts=actions)

        expected_success = self.critic.forward(
            inputs=critic_input).max(dim=1)[0]  # batch_size, 1
        expected_success = expected_success.reshape([-1,1])

        if not optimize:
            result = ACPOptResult(
                gen_trj=actions, 
                expected_succes_before=expected_success.detach(),
                expected_succes_after=expected_success.detach())
            return result
        else:
            #opt_count = int(actions.shape[0]/2)
            batch_size = actions.shape[0]
            '''if (self.buffer_filled_to > 0) and (self.training_mode):
                buffer_actions = self.history.act_buffer[0][:self.buffer_filled_to, self.current_step]
                buffer_obsvs = self.history.obsv_buffer[0][:self.buffer_filled_to, self.current_step]
                actions = th.cat((actions, buffer_actions), dim=0)
                observation_seq = th.cat((observation_seq, buffer_obsvs), dim=0)
                plans = th.zeros([observation_seq.shape[0], observation_seq.shape[1], self.planner.wsms.model_setup.d_output], device=self.args_obj.device, dtype=th.float32)
            '''
            actions_opt, expected_success_opt = self.optimize_act_sequence(
                actions=actions, 
                observations=observation_seq, 
                current_step=current_step,
                plans=plans,
                stop_opt=stop_opt
                )
            #actions[:opt_count + 1] = actions_opt[:opt_count + 1]
            #expected_success_opt[opt_count + 1:] = expected_success[opt_count + 1:]
            actions=actions_opt


            actions = actions[:batch_size]
            expected_success_opt = expected_success_opt[:batch_size]

            return ACPOptResult(
                gen_trj=actions.detach(),
                expected_succes_before=expected_success,
                expected_succes_after=expected_success_opt)

    def make_action(self, action_seq, observation_seq, plans, current_step):
        reward_label = th.ones([observation_seq.shape[0], observation_seq.shape[1], 1], device=observation_seq.device)
        actor_input = self.get_actor_input(plans=plans, obsvs=observation_seq, rewards=reward_label)
        actions = self.actor.forward(actor_input)
        if self.args_obj.clip:
            actions = th.clamp(actions, min=self.clip_min, max=self.clip_max)

        if action_seq is not None:
            actions = self.proj_actions(
                action_seq, actions, current_step)
        return actions
    
    def make_plans(self, acts, obsvs):
        planner_input = self.get_planner_input(acts=acts, obsvs=obsvs)
        plans = self.planner.forward(planner_input)
        #plans = plans.reshape([acts.shape[0], 1, self.planner.wsms.model_setup.d_output]).repeat([1, acts.shape[1], 1])
        return plans

    def optimize_act_sequence(self, 
            actions: th.Tensor, 
            observations: th.Tensor, 
            current_step: int, 
            plans :th.Tensor,
            stop_opt:bool
            ):

        goals = None
        org_actions = th.clone(actions.detach())

        if self.args_obj.optimizer_mode == 'actions':
            optimized_actions = th.clone(actions.detach())
            optimized_actions.requires_grad_(True)
            optimizer = th.optim.AdamW(
                [optimized_actions], lr=self.args_obj.inference_opt_lr, weight_decay=0)
        elif self.args_obj.optimizer_mode == 'actor':
            optimized_actions = self.make_action(action_seq=actions, observation_seq=observations, plans=plans, current_step=current_step).detach()
            actions = actions.detach()
            observations = observations.detach()
            plans = plans.detach()
            optimizer = th.optim.AdamW(
                self.actor.parameters(), lr=self.args_obj.inference_opt_lr, weight_decay=self.actor.wsms.optimizer_kwargs['weight_decay']
                )
        elif self.args_obj.optimizer_mode == 'actor+plan':
            optimized_actions = self.make_action(action_seq=actions, observation_seq=observations, plans=plans, current_step=current_step).detach()
            actions = actions.detach()
            observations = observations.detach()
            plans = self.make_plans(acts=actions, obsvs=observations)

            if current_step == 0:
                self.init_actor = copy.deepcopy(self.actor.state_dict())
                self.init_planner = copy.deepcopy(self.planner.state_dict())

            lr = self.args_obj.inference_opt_lr
            optimizer = th.optim.AdamW(
                [{'params': self.actor.parameters()}, {'params': self.planner.parameters()}],
                lr=lr,
                weight_decay=self.actor.wsms.optimizer_kwargs['weight_decay']
            )

        elif self.args_obj.optimizer_mode == 'plan':
            print('use plan opt mode')
            optimized_actions = self.make_action(action_seq=actions, observation_seq=observations, plans=plans, current_step=current_step).detach()
            actions = actions.detach()
            observations = observations.detach()
            plans = plans.detach()
            plans.requires_grad = True
            optimizer = th.optim.AdamW(
                [plans], lr=self.args_obj.inference_opt_lr, weight_decay=0)
        elif self.args_obj.optimizer_mode == 'goal':
            print('use goal opt mode')
            optimized_actions = self.make_action(action_seq=actions, observation_seq=observations, plans=plans, current_step=current_step).detach()
            actions = actions.detach()
            observations = observations.detach()
            plans = plans.detach()
            goals = th.clone(observations[:, :, -3:])
            goals.requires_grad = True
            optimizer = th.optim.AdamW(
                [goals], lr=self.args_obj.inference_opt_lr, weight_decay=0)
        else:
            print('Choose other optimizer mode')
            1/0
        final_actions = th.clone(optimized_actions)

        expected_success = th.zeros(
            size=[actions.shape[0], 1], dtype=th.float, device=actions.device)
        final_exp_success = th.clone(expected_success)
        if self.critic.wsms.sparse:
            goal_label = self.gl[:actions.shape[0], 0]
            if self.current_step == 0 and self.critic.wsms.sparse:
                print('use sparse critic')
        else:
            goal_label = self.gl[:actions.shape[0], :actions.shape[1]]
        step = 0
        if self.critic.model is not None:
            self.critic.model.eval()
        num_opt_steps = self.args_obj.opt_steps
        '''if self.current_step == 0:
            num_opt_steps = num_opt_steps * 20'''
        while (step <= num_opt_steps):# and (not th.all(final_exp_success.max(dim=1)[0] >= self.args_obj.optimisation_threshold)):
            mask = (final_exp_success.max(dim=1)[0] < self.args_obj.optimisation_threshold).reshape(-1)
            optimized_actions, expected_success, plans = self.inference_opt_step(
                org_actions=org_actions,
                opt_actions=optimized_actions,
                obs_seq=observations,
                optimizer=optimizer,
                goal_label=goal_label,
                current_step=current_step,
                plans=plans,
                goals=goals,
                current_opt_step=step + current_step * self.args_obj.epoch_len
                )
            if self.write_tboard_scalar is not None:
                debug_dict = {
                    f'optimized expected success' : expected_success.detach().mean().cpu()
                }
                self.write_tboard_scalar(debug_dict=debug_dict, train=False, step=step, optimize=True)
            step += 1

            if stop_opt:
                final_actions[mask] = th.clone(optimized_actions[mask]).detach()
                final_exp_success[mask] = th.clone(expected_success.max(dim=1)[0].reshape([-1,1])[mask]).detach()
            else:
                final_actions = optimized_actions
                final_exp_success = expected_success

        if self.args_obj.clip:
            with th.no_grad():
                th.clamp(final_actions, min=self.clip_min, max=self.clip_max, out=final_actions)
        if (self.args_obj.optimizer_mode == 'actor' or self.args_obj.optimizer_mode == 'actor+plan') and self.current_step == self.args_obj.epoch_len - 1:
            self.actor.load_state_dict(self.init_actor)
            self.planner.load_state_dict(self.init_planner)
        return final_actions, final_exp_success

    def inference_opt_step(self, 
            org_actions: th.Tensor, 
            opt_actions: th.Tensor, 
            obs_seq: th.Tensor, 
            optimizer: th.optim.Optimizer, 
            goal_label: th.Tensor, 
            plans:th.Tensor,
            current_step: int,
            goals:th.Tensor,
            current_opt_step
            ):
        if (self.args_obj.optimizer_mode == 'goal'):
            current_obs_seq = th.cat((obs_seq[:, :, :-3], goals), dim=-1)
        else:
            current_obs_seq = obs_seq
            assert goals is None, 'Kinda the wrong mode or smth.'

        if (self.args_obj.optimizer_mode in ['plan', 'goal', 'actor+plan']):
            opt_plan = self.make_plans(opt_actions.detach(), obsvs=obs_seq.detach())
            opt_actions = self.make_action(action_seq=org_actions, observation_seq=current_obs_seq, plans=opt_plan, current_step=current_step)
        elif (self.args_obj.optimizer_mode in ['actions']):
            pass
        else:
            1/0
        critic_inpt = self.get_critic_input(acts=opt_actions, obsvs=obs_seq)
        critic_result = self.critic.forward(inputs=critic_inpt)
        critic_loss = self.critic.loss_fct(result=critic_result, label=goal_label)
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()

        debug_dict = {
            'in optimisation expected success' : critic_result.mean().detach()
        }

        self.write_tboard_scalar(debug_dict=debug_dict, train=False, step=current_opt_step, optimize=True)


        return opt_actions, critic_result, plans
    
    def get_planner_input(self, acts, obsvs):
        return th.cat((acts, obsvs), dim=-1)


    def get_actor_input(self, plans, obsvs, rewards:th.Tensor):
        label = rewards.max(dim=1).values.unsqueeze(1)
        label = label.repeat([1, obsvs.shape[1], 1])
        return th.cat((plans, obsvs, label), dim=-1)

    def get_critic_input(self, obsvs, acts):
        return th.cat((obsvs, acts), dim=-1)


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
