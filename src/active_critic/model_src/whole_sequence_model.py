from abc import ABC, abstractstaticmethod
import typing
import torch as th
import torch.nn as nn
from active_critic.model_src.transformer import TransformerModel, ModelSetup
from active_critic.utils.pytorch_utils import calcMSE
import torchvision as tv

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

    def forward(self, inputs: th.Tensor, attention_mask:th.Tensor=None) -> th.Tensor:
        if (self.model is None):
            self.wsms.model_setup.ntoken = inputs.size(-1)
            self.model = TransformerModel(
                model_setup=self.wsms.model_setup).to(inputs.device)

        result = self.model.forward(inputs, mask=attention_mask)
        if self.optimizer is None:
            self.optimizer = self.wsms.optimizer_class(
                self.model.parameters(), self.wsms.lr, **self.wsms.optimizer_kwargs)
            #self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, int(100000/32), gamma=0.97)

        return result

    def optimizer_step(self, inputs:th.Tensor, label:th.Tensor, prefix='', result_mask:th.Tensor=None, attention_mask:th.Tensor=None) -> typing.Dict:

        result = self.forward(inputs=inputs, attention_mask=attention_mask)
        if result_mask is not None:
            if (result_mask.shape[0] != result.shape[0]) and (result.shape[0] == 1):
                result_mask = result_mask.unsqueeze(0)
            try:
                loss = self.loss_fct(result=result[result_mask], label=label[result_mask])
            except:
                print(f'result_mask: {result_mask}')
                print(f'result: {result.shape}')
                print(f'label: {label.shape}')
                print(f'result_mask: {result_mask.shape}')
        else:
            loss = self.loss_fct(result=result, label=label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        debug_dict = {
            'Loss ' + prefix: loss.detach(),
        }

        return debug_dict

    def loss_fct(self, result:th.Tensor, label:th.Tensor, mask:th.Tensor = None) -> th.Tensor:
        if mask is not None:
            loss = calcMSE(result[mask], label[mask])
        else:
            loss = calcMSE(result, label)
        return loss

class CriticSequenceModel(WholeSequenceModel):
    def __init__(self, wsms: WholeSequenceModelSetup) -> None:
        super().__init__(wsms)
        self.mlp = None

        self.sm = th.nn.Sigmoid()

    def forward(self, inputs: th.Tensor, attention_mask:th.Tensor=None) -> th.Tensor:
        inputs_dynamics = inputs[...,:-3]
        inputs_goal = inputs[...,-3:]
        dynamic_results = super().forward(inputs_dynamics, attention_mask=attention_mask)
        if self.mlp is None:
            self.mlp = tv.ops.MLP(in_channels=3+dynamic_results.shape[-1], hidden_channels=[64, 64, 1], activation_layer=th.nn.ReLU).to(inputs.device)
        mlp_inpt = th.cat((dynamic_results, inputs_goal), dim=-1)
        result = self.mlp.forward(mlp_inpt)
        #pre_sm = dynamic_results[:, :1]
        #result = self.sm(pre_sm)
        return result
