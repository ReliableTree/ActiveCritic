from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional, Union
from active_critic.policy.active_critic_policy import ActiveCriticPolicySetup
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch

class ActiveCriticLearnerArgs:
    def __init__(self) -> None:
        self.extractor:BaseFeaturesExtractor = None
        self.logname:str = None
        self.tboard:bool = None
        self.imitation_phase:bool = None
        self.data_path:str = None
        self.device:str = None


class ActiveCriticArgs1:
    def __init__(self) -> None:
        pass


    def set_critic_optimisation_threshold(self, threshold:float):
        self.optimisation_threshold = threshold

    def set_quick_eval_epochs(self, quick_eval_epochs:int):
        self.quick_eval_epochs = quick_eval_epochs
    
    def set_network_setup(self, network_setup):
        self.network_setup = network_setup

    def set_epoch_len(self, epoch_len:int):
        self.network_setup.critic_nn.seq_len = epoch_len
        self.network_setup.actor_nn.seq_len = epoch_len
        self.epoch_len = epoch_len

    def set_new_epoch(self, new_epoch):
        self.new_epoch = new_epoch

    def set_feature_extractor(self, extractor:BaseFeaturesExtractor):
        self.extractor = extractor

    def set_data_path(self, path:str, model_path:Optional[str] = None):
        self.data_path = path
        if model_path is not None:
            self.model_path = path + model_path
        else:
            self.model_path = None

    def set_log_name(self, logname:str):
        self.logname = logname

    def set_critic_search_lr(self, cslr:float):
        self.cslr = cslr

    def set_meta_optimizer_lr(self, mo_lr:float):
        self.network_setup.critic_nn.lr = mo_lr

    def set_lr(self, lr:float):
        self.network_setup.actor_nn.lr = lr

    def set_demonstrations(self, demonstrations:list):
        self.demonstrations = demonstrations

    def set_device(self, device:str):
        self.device = device
        self.network_setup.actor_nn.device = device
        self.network_setup.critic_nn.device = device

    def set_tboard(self, tboard:bool):
        self.tboard = tboard

    def set_batchsize(self, batch_size:int):
        self.batch_size = batch_size

    def set_n_steps(self, n_steps:int):
        self.n_steps = n_steps

    def set_imitation_phase(self, imitation_phase:bool):
        self.imitation_phase=imitation_phase

    def set_weight_decay(self, weight_decay:float):
        self.weight_decay = weight_decay

    def set_eval_epochs(self, epochs:int):
        self.eval_epochs = epochs

    def set_opt_steps(self, opt_steps:int):
        self.opt_steps = opt_steps

    def set_complete_modulo(self, complete:int):
        self.complete_modulo = complete

    def set_observable(self, observable:bool):
        self.observable = observable