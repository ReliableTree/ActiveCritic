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
