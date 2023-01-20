from abc import ABC, abstractstaticmethod
import typing
import torch as th
import torch.nn as nn
from active_critic.model_src.transformer import TransformerModel, ModelSetup
from active_critic.utils.pytorch_utils import calcMSE


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

    def optimizer_step(self, inputs:th.Tensor, label:th.Tensor, prefix='', mask:th.Tensor=None) -> typing.Dict:
        result = self.forward(inputs=inputs)
        if mask is not None:
            if (mask.shape[0] != result.shape[0]) and (result.shape[0] == 1):
                mask = mask.unsqueeze(0)
            loss = self.loss_fct(result=result[mask], label=label[mask])
        else:
            loss = self.loss_fct(result=result, label=label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        debug_dict = {
            'Loss ' + prefix: loss.detach()
        }

        return debug_dict

    def loss_fct(self, result:th.Tensor, label:th.Tensor, mask:th.Tensor = None) -> th.Tensor:
        if mask is not None:
            loss = calcMSE(result[mask], label[mask])
        else:
            loss = calcMSE(result, label)
        return loss
