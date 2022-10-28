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
        self.epoch_len: int = None
        self.opt_steps: int = None
        self.device: str = None
        self.inference_opt_lr: float = None
        self.optimize: bool = None
        self.batch_size: int = None
        self.stop_opt: bool = None
        self.mask:th.Tensor
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
        actor: StateModel,
        critic: StateModel,
        predictor: WholeSequenceModel,
        emitter: StateModel,
        acps: ActiveCriticPolicySetup = None
    ):

        super().__init__(observation_space, action_space)

        self.actor = actor
        self.critic = critic
        self.predicor = predictor
        self.emitter = emitter
        self.args_obj = acps

        self.clip_min = th.tensor(self.action_space.low, device=acps.device)
        self.clip_max = th.tensor(self.action_space.high, device=acps.device)
        self.reset()

    def reset(self, vec_obsv:th.Tensor = th.ones([1,1,3])):
        self.current_step = 0
        self.last_goal = vec_obsv[:,0,-3:]
        self.current_embeddings = vec_obsv
        self.current_actions = None
        self.goal_label = th.ones([vec_obsv.shape[1], self.args_obj.epoch_len, self.critic.args.arch[-1]])
        self.init_scores = None
        self.opt_scores = None


    def make_critic_input(self, embeddings:th.Tensor, actions:th.Tensor):
        input = th.cat((embeddings, actions), dim=-1)
        return input

    def predict_step(self, embeddings:th.Tensor, actions:th.Tensor, mask:th.Tensor = None):
        predictor_input = self.make_critic_input(embeddings=embeddings, actions=actions)
        next_embeddings = self.predicor.forward(predictor_input, tf_mask=mask)
        return next_embeddings

    def build_sequence(self, embeddings:th.Tensor, actions:th.Tensor, seq_len:int, mask:th.Tensor):
        while (sl := embeddings.shape[1]) < seq_len:
            if (actions is None) or (actions.shape[1] == embeddings.shape[1] - 1):
                next_actions= self.actor.forward(embeddings[:,-1:])
                actions = th.cat((actions, next_actions), dim=1)
            next_embedding = self.predict_step(embeddings=embeddings, actions=actions[:,:embeddings.shape[1]], mask=mask[:sl,:sl])
            embeddings = th.cat((embeddings, next_embedding[:,-1:]), dim=1)

        return embeddings, actions

    def optimize_sequence(self, actions:th.Tensor, seq_embeddings:th.Tensor, mask:th.Tensor, goal_label:th.Tensor, steps:int, current_step:int):
        actions = actions.detach().clone()
        org_actions = actions.detach().clone()

        actions.requires_grad = True
        seq_embeddings = seq_embeddings.detach().clone()
        org_embeddings = seq_embeddings.detach().clone()
        
        optimizer = th.optim.Adam([actions], lr=1e-2)
        opt_paras = optimizer.state_dict()
        for i in range(steps):
            actions = actions.detach().clone()
            actions.requires_grad = True
            optimizer = th.optim.Adam([actions], lr=1e-2)
            seq_embeddings, seq_actions = self.build_sequence(embeddings=seq_embeddings.detach()[:,:current_step], actions=actions, seq_len=actions.shape[1], mask=mask, detach=False)
            critic_input = self.make_critic_input(embeddings=seq_embeddings, actions=actions)

            scores = self.critic.forward(critic_input)
            optimizer.load_state_dict(opt_paras)

            loss_reward = calcMSE(scores[:, current_step:], goal_label[:, current_step:])
            loss = loss_reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            opt_paras = optimizer.state_dict()
            with th.no_grad():
                actions[:,:current_step] = org_actions[:,:current_step]

        return loss_reward, actions, seq_embeddings, scores
            

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
            self.reset(vec_obsv=vec_obsv)
            self.current_embeddings = embedding
        else:
            self.current_step += 1
            self.current_embeddings = th.cat((self.current_embeddings, embedding), dim=1)
        
        next_action = self.actor.forward(embedding)
        if self.current_actions is None:
            self.current_actions = next_action
        else:
            self.current_actions = th.cat((self.current_actions, next_action), dim=1)
        
        if self.args_obj.optimize:
            loss_reward, actions, seq_embeddings     = self.optimize_sequence(
                                                                            actions=self.current_actions, 
                                                                            seq_embeddings=self.current_embeddings, 
                                                                            mask = self.args_obj.mask,
                                                                            goal_label=self.goal_label,
                                                                            steps=self.args_obj.opt_steps,
                                                                            current_step=self.current_step)

        return actions[:, self.current_step]

    def save_policy(self, add, data_path):

        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)

        th.save(self.state_dict(), path_to_file + "/policy_network")
        th.save(self.actor.optimizer.state_dict(), path_to_file + "/optimizer_actor")
        th.save(self.critic.optimizer.state_dict(), path_to_file + "/optimizer_critic")
        with open(path_to_file + '/policy_args.pkl', 'wb') as f:
            pickle.dump(self.args_obj, f)
