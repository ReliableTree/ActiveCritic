from abc import abstractclassmethod
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


class WholeSequenceModel(nn.Module):
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
            self.model = self.wsms.model_setup.model_class(
                model_setup=self.wsms.model_setup).to(inputs.device)
        result = self.model.forward(inputs)
        if self.optimizer is None:
            self.optimizer = self.wsms.optimizer_class(
                self.model.parameters(), self.wsms.lr, **self.wsms.optimizer_kwargs)
        return result

    @abstractclassmethod
    def optimizer_step(self, data: typing.Tuple[th.Tensor, th.Tensor, th.Tensor]) -> typing.Dict:
        raise NotImplementedError

    @abstractclassmethod 
    def loss_fct(self, result:th.Tensor, label:th.Tensor) -> th.Tensor:
        raise NotImplementedError


class WholeSequenceActor(WholeSequenceModel):
    def __init__(self, wsms: WholeSequenceModelSetup):
        super().__init__(wsms)

    def loss_fct(self, result:th.Tensor, label:th.Tensor) -> th.Tensor:
        trj_loss = calcMSE(result, label)
        return trj_loss

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


class WholeSequenceCritic(WholeSequenceModel):
    def __init__(self, wsms: WholeSequenceModelSetup):
        super().__init__(wsms)

    def init_model(self):
        super().init_model()

    def loss_fct(self, result:th.Tensor, label:th.Tensor):
        label = label.reshape(-1).type(th.float)
        result = result.reshape(-1).type(th.float)
        loss = (result - label)**2
        label = label.type(th.bool)
        label_sum = label.sum()
        if label_sum > 0:
            loss_positive = loss[label].mean()
        else:
            loss_positive = th.zeros(1, device=result.device).mean()
        if label_sum < len(label):
            loss_negative = loss[~label].mean()
        else:
            loss_negative = th.zeros(1, device=result.device).mean()
        loss = loss.mean()
        return loss, loss_positive, loss_negative

    def optimizer_step(self, data: typing.Tuple[th.Tensor, th.Tensor, th.Tensor]) -> typing.Dict:
        debug_dict = {}
        inputs, label, success = data
        scores = self.forward(inputs=inputs)
        loss, loss_positive, loss_negative = self.loss_fct(
            result=scores, label=success)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        debug_dict['critic loss'] = loss.detach()
        debug_dict['critic loss positive'] = loss_positive.detach()
        debug_dict['critic loss negative'] = loss_negative.detach()

        return debug_dict
