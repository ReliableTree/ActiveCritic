import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import gym
import torch as th
from active_critic.utils.gym_utils import make_dummy_vec_env, sample_expert_transitions_rollouts, parse_sampled_transitions, DummyExtractor
from active_critic.utils.pytorch_utils import calcMSE
from active_critic.utils.dataset import DatasetAC
from torch.utils.data.dataloader import DataLoader


class Rec_PPO_BC:
    def __init__(self, model:RecurrentPPO, dataloader:DatasetAC, device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.policy = model.policy

    def train(self, n_epochs, verbose = False, bc_mult=1):
        self.model.policy.train()
        for epoch in range(n_epochs):
            for obs, act, rew, _ in self.dataloader:
                batch_size = obs.shape[0]
                lstm = self.model.policy.lstm_actor
                self.model.policy.train()

                single_hidden_state_shape = (lstm.num_layers, batch_size, lstm.hidden_size)
                # hidden and cell states for actor and critic
                lstm_states = RNNStates(
                    (
                        th.zeros(single_hidden_state_shape).to(self.model.device),
                        th.zeros(single_hidden_state_shape).to(self.model.device),
                    ),
                    (
                        th.zeros(single_hidden_state_shape).to(self.model.device),
                        th.zeros(single_hidden_state_shape).to(self.model.device),
                    ),
                )

                actions_res, values, log_prob, rnn_states = self.model.policy.forward(obs = obs, lstm_states=lstm_states, episode_starts = th.zeros([obs.shape[0] * obs.shape[1]], device=self.device), deterministic=False)
                self.model.policy.optimizer.zero_grad()
                loss = calcMSE(actions_res.reshape(-1), act.reshape(-1))
                loss = bc_mult*loss
                loss.backward()
                self.model.policy.optimizer.step()
                if verbose and epoch % 2000 == 0:
                    print(f'loss: {loss}')