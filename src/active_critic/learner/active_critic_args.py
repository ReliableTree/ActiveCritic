from abc import ABC, abstractstaticmethod
from tkinter.messagebox import NO
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
        self.predictor_threshold:float = None
        self.gen_scores_threshold:float = None
        self.loss_auto_predictor_threshold:float = None
        self.num_cpu:int = None
        self.use_pain:bool = None
        self.patients:int = None
