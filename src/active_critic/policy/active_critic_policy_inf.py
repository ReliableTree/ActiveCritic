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
from active_critic.policy.active_critic_policy import *

class ActiveCriticPolicyInf(ActiveCriticPolicy):
    
    def __init__(self, observation_space, action_space, actor: WholeSequenceModel, critic: WholeSequenceModel, acps: ActiveCriticPolicySetup = None):
        super().__init__(observation_space, action_space, actor, critic, acps)

    