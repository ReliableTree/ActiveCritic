import torch as th
import torch.nn as nn
from active_critic.utils.pytorch_utils import calcMSE

class StateModelArgs:
    def __init__(self) -> None:
        self.arch:list = None
        self.lr:float = None
        self.device:str = None


class MLPNetwork(th.nn.Module):
    def __init__(self, arch:list[int]) -> None:
        super().__init__()
        self.layers = th.nn.Sequential()
        for i in range(len(arch)-2):
            self.layers.append(th.nn.Linear(arch[i], arch[i+1]))
            self.layers.append(th.nn.ReLU())
        self.layers.append(th.nn.Linear(arch[-2], arch[-1]))

    def forward(self, inpt:th.Tensor) -> th.Tensor:

        return self.layers.forward(inpt)

class StateModel(nn.Module):
    def __init__(self, args:StateModelArgs) -> None:
        super().__init__()
        self.args = args
        self.is_init = False

    def reset(self):
        self.model = MLPNetwork(arch=self.args.arch).to(self.args.device)
        self.optimizer = th.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0)
                
    def forward(self, inpt):
        if not self.is_init:
            self.args.arch = [inpt.shape[-1]] + self.args.arch
            self.reset()
            self.is_init = True
        return self.model.forward(inpt)

    def calc_loss(self, inpt, label):
        result = self.forward(inpt=inpt)
        loss = calcMSE(result, label)
        return loss