from abc import ABC, abstractstaticmethod
import typing
import torch as th
import torch.nn as nn
from active_critic.model_src.transformer import TransformerModel, ModelSetup
from active_critic.utils.pytorch_utils import calcMSE, part_goal_obs
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

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        if (self.model is None):
            self.wsms.model_setup.ntoken = inputs.size(-1)
            self.model = TransformerModel(
                model_setup=self.wsms.model_setup).to(inputs.device)
        result = self.model.forward(inputs)
        if self.optimizer is None:
            self.optimizer = self.wsms.optimizer_class(
                self.parameters(), self.wsms.lr, **self.wsms.optimizer_kwargs)
            self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, int(100000/32), gamma=1)

        return result

    def optimizer_step(self, inputs:th.Tensor, label:th.Tensor, prefix='', mask:th.Tensor=None) -> typing.Dict:
        pass
    
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
        self.scheduler.step()

        debug_dict = {
            'Loss ' + prefix: loss.detach(),
            'Learning Rate' + prefix: th.tensor(self.scheduler.get_last_lr())
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
        self.result_decoder = None
        self.sm = th.nn.Sigmoid()

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        goal, obsv = part_goal_obs(inputs)
        if self.result_decoder is None:
            #self.result_decoder = nn.Linear(self.wsms.model_setup.d_output * self.wsms.model_setup.seq_len, 1, device=inputs.device)
            self.result_decoder = tv.ops.MLP(
                in_channels=self.wsms.model_setup.d_output * self.wsms.model_setup.seq_len, 
                hidden_channels=[200, 200, 200, 1]).to(inputs.device)
            
            self.goal_decoder = tv.ops.MLP(
                in_channels=self.wsms.model_setup.d_output + 3, 
                hidden_channels=[200, 200, 200, 1]).to(inputs.device)

        trans_result = super().forward(obsv)
        trans_result = th.cat((trans_result, goal), dim=-1)
        trans_goal_result = self.goal_decoder.forward(trans_result)
        pre_sm = self.result_decoder.forward(trans_goal_result.reshape([trans_result.shape[0], -1]))

        #result = self.sm(pre_sm)
        return pre_sm, trans_goal_result
    
class ActorSequenceModel(WholeSequenceModel):
    def optimizer_step(self, inputs: th.Tensor, label: th.Tensor, prefix='', mask: th.Tensor = None, critc = CriticSequenceModel) -> typing.Dict:
        result, proxy_result = self.forward(inputs=inputs)
        critic_inpt = critc.inp
        if mask is not None:
            if (mask.shape[0] != result.shape[0]) and (result.shape[0] == 1):
                mask = mask.unsqueeze(0)
            loss = self.loss_fct(result=result[mask], label=label[mask])
            proxy_loss = self.loss_fct(result=proxy_result[mask], label=proxy[mask])
        else:
            loss = self.loss_fct(result=result, label=label)
            proxy_loss = self.loss_fct(result=proxy_result, label=proxy)

        total_loss = proxy_loss + loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        debug_dict = {
            'Loss ' + prefix: loss.detach(),
            'Proxy Loss ': proxy_loss.detach(), 
            'Learning Rate' + prefix: th.tensor(self.scheduler.get_last_lr())
        }

        return debug_dict
    
 

