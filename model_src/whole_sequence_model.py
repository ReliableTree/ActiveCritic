from abc import ABC, abstractclassmethod, abstractmethod
import typing
import torch as th
import torch.nn as nn
from ActiveCritic.model_src.transformer import TransformerModel, ModelSetup
from ActiveCritic.utils.pytorch_utils import calcMSE


class WholeSequenceModelSetup:
    def __init__(self) -> None:
        self.model_setup: ModelSetup = None
        self.optimizer_class:th.optim.Optimizer = None
        self.optimizer_kwargs = {}
        self.lr:float = None
        self.name:str = None

class WholeSequenceModel(nn.Module, ABC):
    def __init__(self, wsms: WholeSequenceModelSetup) -> None:
        super().__init__()
        self.wsms = wsms
        self.model: TransformerModel = None
        self.optimizer: th.optim.Optimizer = None

    def init_model(self):
        self.model = None
        self.optimizer = None

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        if (self.model is None):
            self.wsms.model_setup.ntoken = inputs.size(-1)
            self.model = TransformerModel(
                model_setup=self.wsms.model_setup).to(inputs.device)
        result = self.model.forward(inputs)
        if self.optimizer is None:
            self.optimizer = self.wsms.optimizer_class(
                self.model.parameters(), self.wsms.lr, **self.wsms.optimizer_kwargs)
        return result

    def optimizer_step(self, data: typing.Tuple[th.Tensor, th.Tensor, th.Tensor], prefix='') -> typing.Dict:
        inputs, label, success = data
        assert th.all(success), 'wrong trajectories input to Actor Model'
        results = self.forward(inputs=inputs)
        loss = self.loss_fct(result=results, label=label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        debug_dict = {
            'Trajectory Loss ' + prefix: loss.detach()
        }

        return debug_dict

    def loss_fct(self, result:th.Tensor, label:th.Tensor) -> th.Tensor:
        loss = calcMSE(result, label)
        return loss
