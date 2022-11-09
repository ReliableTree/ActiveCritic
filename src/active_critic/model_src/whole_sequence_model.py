from abc import ABC, abstractstaticmethod
from doctest import master
import typing
import torch as th
import torch.nn as nn
from active_critic.model_src.transformer import TransformerModel, ModelSetup
from active_critic.utils.pytorch_utils import calcMSE


class WholeSequenceModelArgs:
    def __init__(self) -> None:
        self.model_setup: ModelSetup = None
        self.optimizer_class:th.optim.Optimizer = None
        self.optimizer_kwargs = {}
        self.lr:float = None
        self.name:str = None

class WholeSequenceModel(nn.Module, ABC):
    def __init__(self, args: WholeSequenceModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model: TransformerModel = None
        self.optimizer: th.optim.Optimizer = None

    def init_model(self):
        self.model = None
        self.optimizer = None

    def forward(self, inputs: th.Tensor, tf_mask:th.Tensor=None) -> th.Tensor:
        if (self.model is None):
            self.args.model_setup.ntoken = inputs.size(-1)
            self.model = TransformerModel(
                model_setup=self.args.model_setup).to(inputs.device)
        result = self.model.forward(inputs, mask=tf_mask)
        if self.optimizer is None:
            self.optimizer = self.args.optimizer_class(
                self.model.parameters(), self.args.lr, **self.args.optimizer_kwargs)
        return result

    def optimizer_step(self, inputs:th.Tensor, label:th.Tensor, prefix='', mask:th.Tensor=None) -> typing.Dict:
        loss = self.calc_loss(inpt=inputs, label=label, mask=mask)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        debug_dict = {
            'Loss ' + prefix: loss.detach()
        }

        return debug_dict

    def calc_loss(self, inpt:th.Tensor, label:th.Tensor, mask:th.Tensor = None, tf_mask:th.Tensor=None) -> th.Tensor:
        result = self.forward(inputs=inpt, tf_mask=tf_mask)
        if mask is not None:
            loss = calcMSE(result[mask], label[mask])
        else:
            loss = calcMSE(result, label)
        return loss
