from pyclbr import Function
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from active_critic.model_src.whole_sequence_model import WholeSequenceModel, CriticSequenceModel
from active_critic.utils.pytorch_utils import diff_boundaries, sample_gauss, print_progress, repeat_elements
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import pickle
import copy

class ACPOptResult:
    def __init__(self, opt_trj: th.Tensor, gen_trj: th.Tensor = None, expected_succes_before: th.Tensor = None, expected_succes_after: th.Tensor = None) -> None:
        self.opt_trj = opt_trj
        self.gen_trj = gen_trj
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
        self.use_diff_boundaries = None
        self.variance = None
        self.variance_gamma = None
        self.var_lr = None

class ActiveCriticPolicyHistory:
    def __init__(self) -> None:
        self.reset()
        self.obsv_buffer = []
        self.act_buffer = []

    def reset(self):
        self.gen_scores_hist = []
        self.opt_scores_hist = []

        self.gen_trj_hist = []
        self.opt_trj_hist = []

    def new_epoch(self, history:list([th.Tensor]), size:list([int, int, int]), device:str):
        new_field = th.zeros(size=size, device=device)
        if len(history) == 0:
            history.append(new_field)
            history.append(0)
        else:
            history[0] = th.cat((history[0], new_field))


    def add_value(self, history:list([th.Tensor]), value:th.Tensor, current_step:int):
        history[0][-value.shape[0]:, current_step] = value

    def add_buffer_value(self, history:list([th.Tensor]), value:th.Tensor, current_step:int):
        if current_step == 0:
            history[0][:, current_step] = value
        else:
            complete_history = th.cat((history[0][:, current_step-1, :current_step], value), dim=1)
            history[0][:, current_step] = complete_history


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

    def reset(self):
        self.last_goal = None
        self.current_step = 0
        self.history.reset()

    def reset_epoch(self, vec_obsv: th.Tensor):
        self.n_inferred += 1
        self.current_step = 0
        self.last_goal = vec_obsv
        self.current_result = None

        scores_size = [vec_obsv.shape[0], self.args_obj.epoch_len, self.args_obj.epoch_len, 1]
        self.history.new_epoch(self.history.gen_scores_hist, size=scores_size, device=self.args_obj.device)
        self.history.new_epoch(self.history.opt_scores_hist, size=scores_size, device=self.args_obj.device)

        trj_size = [vec_obsv.shape[0], self.args_obj.epoch_len, self.args_obj.epoch_len, self.action_space.shape[0]]
        self.history.new_epoch(self.history.gen_trj_hist, size=trj_size, device=self.args_obj.device)
        self.history.new_epoch(self.history.opt_trj_hist, size=trj_size, device=self.args_obj.device)
        
        self.obs_seq = -2*th.ones(
            size=[vec_obsv.shape[0], self.args_obj.epoch_len, vec_obsv.shape[-1]], device=self.args_obj.device)
        
        mean_size = [vec_obsv.shape[0], self.args_obj.epoch_len, self.action_space.shape[0]]

        noise_zero_mean =th.zeros(mean_size, device=vec_obsv.device, dtype=th.float)

        noise = sample_gauss(
            noise_zero_mean,
            variance=self.args_obj.variance * th.ones_like(noise_zero_mean)
        )
        self.noise = repeat_elements(noise, 4)
        self.init_noise = th.clone(self.noise.detach())

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
        else:
            self.current_step += 1

        self.obs_seq[:, :1, :] = vec_obsv

        self.current_result = self.forward(
            observation_seq=self.obs_seq, action_seq=None, 
            optimize=self.args_obj.optimize, 
            current_step=self.current_step,
            stop_opt=self.args_obj.stop_opt
            )
        
        self.history.add_buffer_value(
            self.history.gen_trj_hist, 
            value=th.clone(self.current_result.gen_trj.detach()), 
            current_step=self.current_step)
        
        self.history.add_buffer_value(
            self.history.gen_scores_hist,
            value=th.clone(self.current_result.expected_succes_before.detach()),
            current_step=self.current_step
        )

        if self.args_obj.optimize:
            self.history.add_buffer_value(
                history=self.history.opt_trj_hist,
                value=th.clone(self.current_result.opt_trj),
                current_step=self.current_step
            )

            self.history.add_buffer_value(
                history=self.history.opt_scores_hist, 
                value=th.clone(self.current_result.expected_succes_after.detach()),
                current_step=self.current_step
            )

        self.obs_seq = self.obs_seq[:, :-1]

        return self.current_result.opt_trj[:, 0].detach().cpu().numpy()

    def forward(self, 
            observation_seq: th.Tensor, 
            action_seq: th.Tensor, 
            optimize: bool, 
            current_step: int,
            stop_opt: bool
            ):
        
        
        if self.current_step == 0:
            plans = th.zeros([observation_seq.shape[0], observation_seq.shape[1], self.planner.wsms.model_setup.d_output], device=self.args_obj.device, dtype=th.float32)
            actions = self.make_action(observation_seq=observation_seq, plans=plans)
            with th.no_grad():
                plans = self.make_plans(acts=actions, obsvs=observation_seq)
        else:
            app_actions = (self.current_result.opt_trj[:, 1:])
            with th.no_grad():
                plans = self.make_plans(acts=app_actions, obsvs=observation_seq)

        actions = self.make_action(observation_seq=observation_seq, plans=plans)



        critic_input = self.get_critic_input(obsvs=observation_seq, acts=actions)

        expected_success = self.critic.forward(
            inputs=critic_input)
        print_progress(current_step=self.current_step, total_steps=self.args_obj.epoch_len)

        if not optimize:
            result = ACPOptResult(
                opt_trj=actions.detach(), 
                gen_trj=actions.detach(),
                expected_succes_before=expected_success.detach(),
                expected_succes_after=expected_success.detach())
            return result
        else:
            actions_opt, expected_success_opt = self.optimize_act_sequence(
                actions=actions, 
                observations=observation_seq, 
                current_step=current_step,
                plans=plans,
                stop_opt=stop_opt
                )

            return ACPOptResult(
                opt_trj=actions_opt.detach(),
                gen_trj=actions.detach(),
                expected_succes_before=expected_success.detach(),
                expected_succes_after=expected_success_opt.detach())

    def make_action(self, observation_seq, plans):
        reward_label = th.ones([observation_seq.shape[0], observation_seq.shape[1], 1], device=observation_seq.device)
        actor_input = self.get_actor_input(plans=plans, obsvs=observation_seq, rewards=reward_label)
        actions = self.actor.forward(actor_input)
        if self.args_obj.clip:
            actions = th.clamp(actions, min=self.clip_min, max=self.clip_max)

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

        if current_step == 0:
            self.init_actor = copy.deepcopy(self.actor.state_dict())
            self.init_planner = copy.deepcopy(self.planner.state_dict())
        else:
            self.actor.load_state_dict(self.init_actor)
            self.planner.load_state_dict(self.init_planner)

        goals = None
        org_actions = th.clone(actions.detach())
        optimized_actions = th.clone(actions.detach())


        observations = observations.detach()

        lr = self.args_obj.inference_opt_lr
        optimizer = th.optim.AdamW(
            [{'params': self.actor.parameters()}, {'params': self.planner.parameters()}],
            lr=lr,
            weight_decay=self.actor.wsms.optimizer_kwargs['weight_decay']
        )
        self.noise.requires_grad = True
        noise_optimizer = th.optim.AdamW(
            [{'params': self.noise}],
            lr=self.args_obj.var_lr,
            weight_decay=0
        )
        final_actions = th.clone(org_actions)

        expected_success = th.zeros(
            size=[actions.shape[0], actions.shape[1], 1], dtype=th.float, device=actions.device)
        final_exp_success = th.clone(expected_success)

        if self.critic.wsms.sparse:
            goal_label = self.gl[:actions.shape[0], 0]
            if self.current_step == 0 and self.critic.wsms.sparse:
                print('use sparse critic')
        else:
            goal_label = self.gl[:actions.shape[0], :actions.shape[1]]
            if self.current_step == 0:
                print('use dense critic')

        step = 0
        if self.critic.model is not None:
            self.critic.model.eval()
        num_opt_steps = self.args_obj.opt_steps
        '''if self.current_step == 0:
            num_opt_steps = num_opt_steps * 20'''
        while (step <= num_opt_steps):
            mask = (final_exp_success.max(dim=1)[0] < self.args_obj.optimisation_threshold).reshape(-1)
            optimized_actions, expected_success, plans = self.inference_opt_step(
                opt_actions=optimized_actions,
                obs_seq=observations,
                optimizer=optimizer,
                noise_optimizer = noise_optimizer,
                goal_label=goal_label,
                current_step=current_step,
                plans=plans,
                goals=goals,
                current_opt_step=step + current_step * self.args_obj.epoch_len
                )
            if self.write_tboard_scalar is not None:
                debug_dict = {
                    f'optimized expected success' : expected_success.max(dim=1).values.detach().mean().cpu()
                }
                self.write_tboard_scalar(debug_dict=debug_dict, train=False, step=step, optimize=True)
            step += 1

            if stop_opt:
                final_actions[mask] = th.clone(optimized_actions[mask]).detach()
                final_exp_success[mask] = th.clone(expected_success[mask]).detach()
            else:
                final_actions = optimized_actions
                final_exp_success = expected_success



        self.opt_actor = copy.deepcopy(self.actor.state_dict())
        self.opt_planner = copy.deepcopy(self.planner.state_dict())

        self.actor.load_state_dict(self.init_actor)
        self.planner.load_state_dict(self.init_planner)

        self.noise = self.noise.detach()

        final_actions = final_actions + self.noise
        self.noise = self.noise[:, 1:]
        self.init_noise = self.init_noise[:, 1:]
        if self.args_obj.clip:
            with th.no_grad():
                th.clamp(final_actions, min=self.clip_min, max=self.clip_max, out=final_actions)
        return final_actions, final_exp_success

    def inference_opt_step(self, 
            opt_actions: th.Tensor, 
            obs_seq: th.Tensor, 
            optimizer: th.optim.Optimizer, 
            noise_optimizer: th.optim.Optimizer,
            goal_label: th.Tensor, 
            plans:th.Tensor,
            current_step: int,
            goals:th.Tensor,
            current_opt_step
            ):

        current_obs_seq = obs_seq

        opt_plan = self.make_plans(opt_actions.detach(), obsvs=obs_seq.detach())
        opt_actions = self.make_action(observation_seq=current_obs_seq, plans=opt_plan)
        critic_inpt = self.get_critic_input(acts=opt_actions, obsvs=obs_seq)
        critic_result = self.critic.forward(inputs=critic_inpt)

        result_actions = opt_actions.detach() + self.noise
        critic_inpt_noise = self.get_critic_input(acts=result_actions, obsvs=obs_seq)
        critic_result_noise = self.critic.forward(inputs=critic_inpt_noise)

        critic_loss = self.critic.loss_fct(result=critic_result, label=goal_label[:, :critic_result.shape[1]])
        critic_loss_noise = self.critic.loss_fct(result=critic_result_noise, label=goal_label[:, :critic_result.shape[1]])
        critic_sq_loss = self.critic.loss_fct(result=self.noise, label=self.init_noise)

        critic_loss = critic_loss + critic_loss_noise + critic_sq_loss

        if self.args_obj.use_diff_boundaries or True:
            diff_bound_loss_gene = diff_boundaries(
                actions=opt_actions, 
                low=th.tensor(self.action_space.low, device=opt_actions.device), 
                high = th.tensor(self.action_space.high, device=opt_actions.device))
            
            diff_bound_loss_noise = diff_boundaries(
                actions=result_actions, 
                low=th.tensor(self.action_space.low, device=opt_actions.device), 
                high = th.tensor(self.action_space.high, device=opt_actions.device))
            
            critic_loss = critic_loss + diff_bound_loss_gene + diff_bound_loss_noise
        optimizer.zero_grad()
        noise_optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        noise_optimizer.step()

        debug_dict = {
            'in optimisation expected success' : critic_result[:, :self.args_obj.epoch_len - self.current_step].max(dim=1).values.mean().detach(),
            'in optimisation mean noise' : self.noise.detach().mean()
        }

        self.write_tboard_scalar(debug_dict=debug_dict, train=False, step=current_opt_step, optimize=True)


        return opt_actions, critic_result_noise, plans
    
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
            if self.args_obj.clip and False:
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
