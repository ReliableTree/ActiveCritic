from abc import ABC, abstractstaticmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th


class ActiveCriticLearnerArgs:
    def __init__(self) -> None:
        self.extractor:BaseFeaturesExtractor = None
        self.logname:str = None
        self.tboard:bool = None
        self.imitation_phase:bool = None
        self.data_path:str = None
        self.device:str = None
        self.batch_size:int = None
        self.validation_episodes:int = None
        self.training_epsiodes:int = None
        self.val_every:int= None
        self.add_data_every:int = None
        self.actor_threshold:float = None
        self.critic_threshold:float = None
        self.min_critic_threshold:float = None
        self.num_cpu:int = None
        self.patients: int = None
        self.validation_rep:int = None
        self.total_training_epsiodes:int =None
        self.num_expert_demos:int = None
        self.start_critic:int = None
        self.make_graphs:bool = None
        self.dense:bool = None
        self.max_epoch_steps = None
        self.explore_until = None
        self.use_pred_loss = None
        self.explore_cautious_until = None