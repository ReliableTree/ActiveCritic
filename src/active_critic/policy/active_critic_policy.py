from inspect import ArgSpec
from re import S
from turtle import right
from typing import Dict, Optional, Tuple, Union
from active_critic.model_src.state_model import StateModel

import numpy as np
import torch as th
from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from active_critic.utils.pytorch_utils import calcMSE, printProgressBar
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
        self.opt_goal:th.Tensor=None,
        self.clip:bool = True
        self.optimize_goal_emb_acts:bool = False
        self.goal_label_multiplier:float = 1


class ActiveCriticPolicyHistory:
    def __init__(self) -> None:
        self.reset()


    def reset(self):
        self.scores = []
        self.goal_scores = []
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
            history[0][-value.shape[0]:, opt_step] = value
        elif step is None:
            history[0][-value.shape[0]:] = value
        else:
            history[0][-value.shape[0]:, step:step+1] = value


class ActiveCriticPolicy(BaseModel):
    def __init__(
        self,
        acps: ActiveCriticPolicySetup,
        observation_space,
        action_space,
        actor: StateModel,
        critic: StateModel,
        predictor: Union[WholeSequenceModel, StateModel],
        emitter: StateModel,
        inverse_critic: StateModel
    ):

        super().__init__(observation_space, action_space)

        self.actor = actor
        self.critic = critic
        self.predicor = predictor
        self.emitter = emitter
        self.args_obj = acps
        self.inverse_critic = inverse_critic

        self.clip_min = th.tensor(self.action_space.low, device=acps.device, dtype=th.float32)
        self.clip_max = th.tensor(self.action_space.high, device=acps.device, dtype=th.float32)

        self.reset()

    def reset(self):
        self.history = ActiveCriticPolicyHistory()
        self.last_goal = None

    def reset_epoch(self, vec_obsv:th.Tensor):
        self.current_step = 0
        self.last_goal = vec_obsv[:,0,-3:]
        self.goal_label = th.ones([vec_obsv.shape[0], self.args_obj.epoch_len, self.critic.args.arch[-1]], device=self.args_obj.device)
        self.goal_label *= self.args_obj.goal_label_multiplier
        
        self.history.new_epoch(history=self.history.opt_trj, size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.actor.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.gen_trj, size=[vec_obsv.shape[0], self.args_obj.epoch_len, self.actor.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.scores, size=[vec_obsv.shape[0], self.args_obj.opt_steps, self.args_obj.epoch_len, self.critic.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.pred_emb, size=[vec_obsv.shape[0], self.args_obj.epoch_len -1, self.emitter.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.act_emb, size=[vec_obsv.shape[0], self.args_obj.epoch_len -1, self.emitter.args.arch[-1]], device=self.args_obj.device)
        self.history.new_epoch(history=self.history.goal_scores, size=[vec_obsv.shape[0], self.args_obj.opt_steps, self.critic.args.arch[-1]], device=self.args_obj.device)

        if self.args_obj.optimize_goal_emb_acts:
            self.goal_emb_acts = self.get_goal_emb_act(
                goal_state=self.last_goal,
                goal_label=self.goal_label,
                lr=self.args_obj.inference_opt_lr,
                opt_steps=self.args_obj.opt_steps
            )
        else:
            #TODO
            #Generalize goal embeddings for whole sequence actor.
            self.goal_emb_acts = self.goal_label[0,-1]

        self.current_embeddings = vec_obsv
        self.current_actions = None
        
    def get_predictor_input(self, embeddings:th.Tensor, actions:th.Tensor):
        input = th.cat((embeddings, actions), dim=-1)
        return input

    def get_critic_input(self, embeddings:th.Tensor, actions:th.Tensor):
        return embeddings

    def get_actor_input(self, embeddings:th.Tensor, goal_emb_acts:th.Tensor):
        #embessings = [batch, seq, emb]
        #goal_embeddings = [batch, 1]
        goal_emb_acts = goal_emb_acts.reshape([1,1,-1]).repeat([embeddings.shape[0], embeddings.shape[1], 1])
        return th.cat((embeddings, goal_emb_acts), dim=-1)


    def get_inverse_critic_input(self, goal_state:th.Tensor, goal_scores:th.Tensor):
        return th.cat((goal_state, goal_scores), dim=-1)


    def predict_step(self, embeddings:th.Tensor, actions:th.Tensor, mask:th.Tensor = None):
        predictor_input = self.get_predictor_input(embeddings=embeddings, actions=actions)
        next_embeddings = self.predicor.forward(predictor_input, tf_mask=mask)
        return next_embeddings

    def project_embeddings(self, embeddings:th.Tensor, goal_state:th.Tensor):
        if goal_state is not None:
            if len(embeddings.shape) == 3:
                with th.no_grad():
                    embeddings[:,:,:goal_state.shape[-1]] = goal_state[:, None]
            else:
                with th.no_grad():
                    embeddings[:,:goal_state.shape[-1]] = goal_state    
        return embeddings

   
    def clip_actions(self, emb_acts:th.Tensor):
        if len(emb_acts.shape) == 3:
            with th.no_grad():
                emb_acts[:,:,-self.actor.args.arch[-1]:] = th.clamp(emb_acts[:,:,-self.actor.args.arch[-1]:], min=self.clip_min, max=self.clip_max)
        else:
            with th.no_grad():
                    emb_acts[:,-self.actor.args.arch[-1]:] = th.clamp(emb_acts[:,-self.actor.args.arch[-1]:], min=self.clip_min, max=self.clip_max)
        return emb_acts

    def get_goal_emb_act(self, 
            goal_state:th.Tensor,
            goal_label:th.Tensor,
            lr:float,
            opt_steps:int):

        inv_crit_inpt = self.get_inverse_critic_input(goal_state=goal_state, goal_scores=goal_label[:,-1])
        goal_emb_act = self.inverse_critic.forward(inpt=inv_crit_inpt)
        goal_emb_act = self.project_embeddings(embeddings=goal_emb_act, goal_state=goal_state)
        goal_emb_act = self.clip_actions(emb_acts=goal_emb_act)
        goal_emb_act = self.optimize_goal_emb_acts(
            goal_emb_acts=goal_emb_act,
            goal_label=goal_label,
            goal_state=goal_state,
            lr=lr,
            opt_steps=opt_steps)

        goal_emb_act = self.project_embeddings(embeddings=goal_emb_act, goal_state=goal_state)
        goal_emb_act = self.clip_actions(emb_acts=goal_emb_act)
        return goal_emb_act.detach()


    def optimize_goal_emb_acts(self, 
            goal_emb_acts:th.Tensor,
            goal_state:th.Tensor, 
            goal_label:th.Tensor,
            lr:float,
            opt_steps:int):
        goal_emb_acts = goal_emb_acts.detach().clone()
        goal_emb_acts.requires_grad = True
        goal_optimizer = th.optim.Adam([goal_emb_acts], lr=lr)
        for i in range(10*opt_steps):
            scores = self.critic.forward(goal_emb_acts)
            loss = calcMSE(scores, goal_label[:, -1])
            goal_optimizer.zero_grad()
            loss.backward()
            goal_optimizer.step()
            self.history.add_value(self.history.goal_scores, scores.detach().clone().unsqueeze(1), step=i)
            goal_emb_acts = self.project_embeddings(embeddings=goal_emb_acts, goal_state=goal_state)
        return goal_emb_acts.detach()


    def build_sequence(
        self,
        embeddings:th.Tensor,
        actions:th.Tensor,
        seq_len:int,
        goal_state:th.Tensor, 
        goal_emb_acts:th.Tensor, 
        tf_mask:th.Tensor,
        actor:StateModel, 
        predictor: Union[WholeSequenceModel, StateModel] ):
        init_embedding = embeddings.detach().clone()
        if actions is not None:
            init_actions = actions.detach().clone()
        else:
            init_actions = None
        
        while embeddings.shape[1] < seq_len:
            embeddings = embeddings.detach()
            if actions is None or actions.shape[1] == embeddings.shape[1]-1:
                actor_inpt = self.get_actor_input(embeddings=embeddings, goal_emb_acts=goal_emb_acts)
                actions = actor.forward(actor_inpt)
                if self.args_obj.clip:
                    with th.no_grad():
                        th.clamp(actions, min=self.clip_min, max=self.clip_max, out=actions)
                if init_actions is None:
                    init_actions = actions.detach().clone()
            if not init_actions.shape[1] == seq_len:
                actions = th.cat((init_actions.clone(), actions[:, init_actions.shape[1]:]), dim=1)
            next_embedings = self.predict_step(embeddings=embeddings, actions=actions[:,:embeddings.shape[1]], mask=tf_mask[:embeddings.shape[1],:embeddings.shape[1]])
            embeddings = th.cat((init_embedding.clone(), next_embedings[:, init_embedding.shape[1]-1:]), dim=1)
        if actions.shape[1] == seq_len - 1:
            actor_inpt = self.get_actor_input(embeddings=embeddings[:,-1:], goal_emb_acts=goal_emb_acts)
            actions = th.cat((actions, actor.forward(actor_inpt)), dim=1)
        if self.args_obj.clip:
            with th.no_grad():
                th.clamp(actions, min=self.clip_min, max=self.clip_max, out=actions)
        return embeddings, actions

    def optimize_sequence(self, 
        actions:th.Tensor, 
        seq_embeddings:th.Tensor, 
        pred_mask:th.Tensor, 
        opt_mask:th.Tensor, 
        goal_label:th.Tensor, 
        goal_emb_acts:th.Tensor,
        steps:int, 
        current_step:int,
        seq_len:int,
        optimizer_class:th.optim.Optimizer,
        lr:float,
        goal_state:th.Tensor = None,
        ):


        if actions is not None:
            actions = actions.detach().clone()
            org_actions = actions.detach().clone()
            actions.requires_grad = True
        else:
            org_actions = None

        seq_embeddings = seq_embeddings.detach().clone()
        if (actions is None) or (actions.shape[1] != seq_embeddings.shape[1]):
            _, actions = self.build_sequence(
                embeddings=seq_embeddings.detach()[:,:current_step+1], 
                actions=actions, 
                seq_len=seq_len, 
                goal_emb_acts=goal_emb_acts,
                goal_state=goal_state,
                tf_mask=pred_mask,
                actor=self.actor,
                predictor=self.predicor)

            if self.args_obj.clip:
                actions = th.clamp(actions, min=self.clip_min, max=self.clip_max)
            actions = actions.detach()
            if self.current_step == 0:
                self.history.add_value(history=self.history.gen_trj, value=actions.clone())
            actions.requires_grad = True

        
        opt_paras = None
        for opt_step in range(steps):
            seq_embeddings, actions = self.build_sequence(
                embeddings=seq_embeddings.detach()[:,:current_step+1], 
                actions=actions, 
                seq_len=seq_len, 
                goal_emb_acts=goal_emb_acts,
                goal_state=goal_state,
                tf_mask=pred_mask,
                actor=self.actor,
                predictor=self.predicor)

            optimizer = optimizer_class([actions], lr=lr)
            critic_inpt = self.get_critic_input(embeddings=seq_embeddings, actions=actions)
            scores = self.critic.forward(critic_inpt)
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
            with th.no_grad():
                th.nan_to_num(actions, out=actions)
            
            seq_embeddings = self.project_embeddings(seq_embeddings, goal_state)

        return loss.detach(), actions.detach(), seq_embeddings.detach(), scores.detach()
            

    def predict(
        self,
        observation: Union[th.Tensor, Dict[str, th.Tensor]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> np.array:
        vec_obsv = self.args_obj.extractor.forward(observation).to(self.args_obj.device)
        embedding = self.emitter.forward(vec_obsv)

        
        if (self.last_goal is None) or (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
            if (self.last_goal is not None) and (self.args_obj.new_epoch(self.last_goal, vec_obsv)):
                self.history.add_value(history=self.history.opt_trj, value=self.current_actions)
            self.reset_epoch(vec_obsv=vec_obsv)
            self.current_embeddings = embedding
        else:
            self.current_step += 1
            self.current_embeddings = th.cat((self.current_embeddings, embedding), dim=1)
        
        printProgressBar(iteration=self.current_step, total=self.args_obj.epoch_len, suffix='Predicting Epsiode')

        if self.args_obj.optimize:
            loss_reward, actions, seq_embeddings, scores = self.optimize_sequence(
                                                    actions=self.current_actions, 
                                                    seq_embeddings=self.current_embeddings, 
                                                    pred_mask = self.args_obj.pred_mask,
                                                    opt_mask = self.args_obj.opt_mask,
                                                    goal_label=self.goal_label,
                                                    goal_emb_acts=self.goal_emb_acts,
                                                    steps=self.args_obj.opt_steps,
                                                    current_step=self.current_step,
                                                    seq_len=self.args_obj.epoch_len,
                                                    optimizer_class=self.args_obj.optimizer_class,
                                                    lr=self.args_obj.inference_opt_lr,
                                                    goal_state=self.last_goal)
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
