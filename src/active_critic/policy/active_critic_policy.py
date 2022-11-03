from inspect import ArgSpec
from re import S
from turtle import right
from typing import Dict, Optional, Tuple, Union
from active_critic.model_src.state_model import StateModel

import numpy as np
import torch as th
from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from active_critic.utils.pytorch_utils import get_seq_end_mask, make_partially_observed_seq, calcMSE
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import pickle

from torch.functional import Tensor

class ACPOptResult:
    def __init__(self, gen_trj: th.Tensor, inpt_trj: th.Tensor = None, expected_succes_before: th.Tensor = None, expected_succes_after: th.Tensor = None) -> None:
        self.gen_trj = gen_trj
        self.inpt_trj = inpt_trj
        self.expected_succes_before = expected_succes_before
        self.expected_succes_after = expected_succes_after


class ActiveCriticPolicySetup:
    def __init__(self) -> None:
        self.extractor: BaseFeaturesExtractor = None
        self.new_epoch = None
        self.optimisation_threshold: float = None
        self.inference_opt_lr: float = None
        self.opt_steps: int = None
        self.optimizer_class:th.optim.Optimizer = None
        self.epoch_len: int = None
        self.device: str = None
        self.optimize: bool = None
        self.batch_size: int = None
        self.pred_mask:th.Tensor = None,
        self.opt_mask:th.Tensor=None,
        self.clip:bool = True


class ActiveCriticPolicyHistory:
    def __init__(self) -> None:
        self.reset()


    def reset(self):
        self.scores = []
        self.gen_trj = []
        self.opt_trj = []
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
            history[0][-value.shape[0]:, opt_step, step] = value[:,step]
        elif step is None:
            history[0][-value.shape[0]:] = value
        else:
            history[0][-value.shape[0]:, step:step+1] = value


class ActiveCriticPolicy(BaseModel):
    def __init__(
        self,
        observation_space,
        action_space,
        actor: StateModel,
        critic: StateModel,
        predictor: Union[WholeSequenceModel, StateModel],
        emitter: StateModel,
        acps: ActiveCriticPolicySetup = None
    ):

        super().__init__(observation_space, action_space)

        self.actor = actor
        self.critic = critic
        self.predicor = predictor
        self.emitter = emitter
        self.args_obj = acps

        self.clip_min = th.tensor(self.action_space.low, device=acps.device, dtype=th.float32)
        self.clip_max = th.tensor(self.action_space.high, device=acps.device, dtype=th.float32)

        self.reset()

    def reset(self):
        self.history = ActiveCriticPolicyHistory()
        self.last_goal = None

    def reset_epoch(self, vec_obsv:th.Tensor):

        self.current_step = 0
        self.last_goal = vec_obsv[:,0,-3:]
        self.goal_label = th.ones([vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.args.arch[-1]])
        self.final_reward = th.ones([vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.args.arch[-1]])
        
        self.history.new_epoch(history=self.history.opt_trj, size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.actor.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.gen_trj, size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.actor.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.scores, size=[vec_obsv.shape[0], self.args_obj.opt_steps, self.args_obj.epoch_len, self.critic.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.pred_emb, size=[vec_obsv.shape[0], self.args_obj.epoch_len -1, self.emitter.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.act_emb, size=[vec_obsv.shape[0], self.args_obj.epoch_len -1, self.emitter.args.arch[-1]], device=self.args_obj.device)
        
        self.current_embeddings = vec_obsv
        self.current_actions = None
        
    def get_critic_input(self, embeddings:th.Tensor, actions:th.Tensor):
        input = th.cat((embeddings, actions), dim=-1)
        return input

    def get_actor_input(self, embeddings:th.Tensor, final_reward:th.Tensor):
        #embessings = [batch, seq, emb]
        #goal_label = [batch, seq, 1]
        return th.cat((embeddings, final_reward), dim=-1)

    def predict_step(self, embeddings:th.Tensor, actions:th.Tensor, mask:th.Tensor = None):
        predictor_input = self.get_critic_input(embeddings=embeddings, actions=actions)
        if type(self.predicor) is WholeSequenceModel:
            next_embeddings = self.predicor.forward(predictor_input, tf_mask=mask)
        else:
            next_embeddings = self.predicor.forward(predictor_input)
        return next_embeddings

    def build_sequence(self, embeddings:th.Tensor, actions:th.Tensor, seq_len:int, mask:th.Tensor):
        while (sl := embeddings.shape[1]) < seq_len + 1:
            if (actions is None) or (actions.shape[1] == embeddings.shape[1] - 1):
                with th.no_grad():
                    actor_inpt = self.get_actor_input(embeddings[:,-1:], self.final_reward[:,-1:])
                    next_actions= self.actor.forward(actor_inpt)
                    if actions is not None:
                        actions = th.cat((actions, next_actions), dim=1)
                    else:
                        actions = next_actions
            if type(self.predicor) is WholeSequenceModel:
                next_embedding = self.predict_step(embeddings=embeddings, actions=actions[:,:embeddings.shape[1]], mask=mask[:sl,:sl])
                embeddings = th.cat((embeddings, next_embedding[:,-1:]), dim=1)
            else:
                next_embedding = self.predict_step(embeddings=embeddings, actions=actions[:,:embeddings.shape[1]])
                embeddings = th.cat((embeddings, next_embedding[:,-1:]), dim=1)
        return embeddings[:,:-1], actions

    def optimize_sequence(self, actions:th.Tensor, seq_embeddings:th.Tensor, pred_mask:th.Tensor, opt_mask:th.Tensor, goal_label:th.Tensor, steps:int, current_step:int):
        if actions is not None:
            actions = actions.detach().clone()
            org_actions = actions.detach().clone()
            actions.requires_grad = True
        else:
            org_actions = None

        seq_embeddings = seq_embeddings.detach().clone()
        if (actions is None) or actions.shape[1] != seq_embeddings.shape[1]:
            _, actions = self.build_sequence(embeddings=seq_embeddings.detach()[:,:current_step+1], actions=actions, seq_len=self.args_obj.epoch_len, mask=pred_mask)
            if self.args_obj.clip:
                actions = th.clamp(actions, min=self.clip_min, max=self.clip_max)
            actions = actions.detach()
            if self.current_step == 0:
                self.history.add_value(history=self.history.gen_trj, value=actions.clone())
            actions.requires_grad = True
        opt_paras = None
        for opt_step in range(steps):
            seq_embeddings, actions = self.build_sequence(embeddings=seq_embeddings.detach()[:,:current_step+1], actions=actions, seq_len=actions.shape[1], mask=pred_mask)
            optimizer = self.args_obj.optimizer_class([actions], lr=self.args_obj.inference_opt_lr)
            critic_input = self.get_critic_input(embeddings=seq_embeddings, actions=actions)

            scores = self.critic.forward(critic_input)
            self.history.add_value(history=self.history.scores, value=scores.detach().clone(), opt_step=opt_step, step=current_step)

            if opt_paras is not None:
                optimizer.load_state_dict(opt_paras)
            else:
                opt_paras = optimizer.state_dict()
            loss = calcMSE(scores[:, opt_mask], goal_label[:, opt_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            opt_paras = optimizer.state_dict()

            if org_actions is not None:
                with th.no_grad():
                    actions[:,:current_step] = org_actions[:,:current_step]
            if self.args_obj.clip:
                with th.no_grad():
                    th.clamp(actions, min=self.clip_min, max=self.clip_max, out=actions)

        return loss.detach(), actions.detach(), seq_embeddings.detach(), scores.detach()
            

    def predict(
        self,
        observation: Union[th.Tensor, Dict[str, th.Tensor]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> th.Tensor:
        vec_obsv = self.args_obj.extractor.forward(observation)
        embedding = self.emitter.forward(vec_obsv)

        
        if (self.last_goal is None) or (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
            if (self.last_goal is not None) and (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
                self.history.add_value(history=self.history.opt_trj, value=self.current_actions)
            self.reset_epoch(vec_obsv=vec_obsv)
            self.current_embeddings = embedding
        else:
            self.current_step += 1
            self.current_embeddings = th.cat((self.current_embeddings, embedding), dim=1)
        if self.args_obj.optimize:
            loss_reward, actions, seq_embeddings, scores = self.optimize_sequence(
                                                    actions=self.current_actions, 
                                                    seq_embeddings=self.current_embeddings, 
                                                    pred_mask = self.args_obj.pred_mask,
                                                    opt_mask = self.args_obj.opt_mask,
                                                    goal_label=self.goal_label,
                                                    steps=self.args_obj.opt_steps,
                                                    current_step=self.current_step)
        if self.current_step == 0:
            self.history.add_value(self.history.pred_emb, seq_embeddings[:,self.current_step+1:self.current_step+2], step=self.current_step)
        elif self.current_step == self.args_obj.epoch_len - 1:
            self.history.add_value(self.history.act_emb, seq_embeddings[:,self.current_step:self.current_step+1], step=self.current_step - 1)
        else:
            self.history.add_value(self.history.pred_emb, seq_embeddings[:,self.current_step+1:self.current_step+2], step=self.current_step)
            self.history.add_value(self.history.act_emb, seq_embeddings[:,self.current_step:self.current_step+1], step=self.current_step - 1)

        self.current_actions = actions[:,:self.current_step+1]
        return actions.cpu().numpy()[:, self.current_step]

    def save_policy(self, add, data_path):

        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)

        th.save(self.state_dict(), path_to_file + "/policy_network")
        th.save(self.actor.optimizer.state_dict(), path_to_file + "/optimizer_actor")
        th.save(self.critic.optimizer.state_dict(), path_to_file + "/optimizer_critic")
        with open(path_to_file + '/policy_args.pkl', 'wb') as f:
            pickle.dump(self.args_obj, f)
