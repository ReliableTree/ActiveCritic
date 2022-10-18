import os
import pickle
import sys
import time
from distutils.log import debug

import tensorflow as tf
import torch as th
import torch.nn as nn
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.dataset import DatasetAC
from active_critic.utils.gym_utils import sample_new_episode
from active_critic.utils.pytorch_utils import calcMSE, get_rew_mask, make_part_obs_data
from active_critic.utils.tboard_graphs import TBoardGraphs
from gym.envs.mujoco import MujocoEnv
from torch.utils.data.dataloader import DataLoader
import numpy as np


class ACLScores:
    def __init__(self) -> None:
        self.mean_actor = [float('inf')]
        self.mean_critic = [float('inf')]
        self.mean_reward = [0]

    def update_min_score(self, old_score, new_score):
        new_min = old_score[0] > new_score
        if new_min:
            old_score[0] = new_score
        return new_min

    def update_max_score(self, old_score, new_score):
        new_max = old_score[0] < new_score
        if old_score[0] < new_score:
            old_score[0] = new_score
        return new_max


class ActiveCriticLearner(nn.Module):
    def __init__(self,
                 ac_policy: ActiveCriticPolicy,
                 env: MujocoEnv,
                 eval_env: MujocoEnv,
                 network_args_obj: ActiveCriticLearnerArgs = None
                 ):
        super().__init__()
        self.network_args = network_args_obj
        self.env = env
        self.eval_env = eval_env
        self.policy = ac_policy
        self.extractor = network_args_obj.extractor
        self.logname = network_args_obj.logname

        self.scores = ACLScores()

        if network_args_obj.tboard:
            self.tboard = TBoardGraphs(
                self.logname, data_path=network_args_obj.data_path)
        self.global_step = 0

        self.train_data = DatasetAC(device='cpu')
        self.train_data.onyl_positiv = False

    def setDatasets(self, train_data: DatasetAC):
        self.train_data = train_data
        if len(train_data) > 0:
            self.train_loader = DataLoader(
                dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def add_data(self, actions: th.Tensor, observations: th.Tensor, rewards: th.Tensor):
        acts, obsv, rews = make_part_obs_data(
            actions=actions, observations=observations, rewards=rewards)
        self.train_data.add_data(obsv=obsv.to(
            'cpu'), actions=acts.to('cpu'), reward=rews.to('cpu'))
        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def actor_step(self, data, loss_actor):
        obsv, actions, reward = data
        actor_input = self.policy.get_actor_input(
            obs=obsv, actions=actions, rew=reward)
        mask = get_rew_mask(reward)
        debug_dict = self.policy.actor.optimizer_step(
            inputs=actor_input, label=actions, mask=mask)
        if loss_actor is None:
            loss_actor = debug_dict['Loss '].unsqueeze(0)
        else:
            loss_actor = th.cat(
                (loss_actor, debug_dict['Loss '].unsqueeze(0)), dim=0)
        return loss_actor

    def critic_step(self, data, loss_critic):
        obsv, actions, reward = data
        critic_inpt = self.policy.get_critic_input(acts=actions, obs_seq=obsv)
        mask = get_rew_mask(reward)

        debug_dict = self.policy.critic.optimizer_step(
            inputs=critic_inpt, label=reward, mask=mask)
        if loss_critic is None:
            loss_critic = debug_dict['Loss '].unsqueeze(0)
        else:
            loss_critic = th.cat(
                (loss_critic, debug_dict['Loss '].unsqueeze(0)), dim=0)
        return loss_critic

    def add_training_data(self):
        h = time.perf_counter()
        actions, observations, rewards, _, _ = sample_new_episode(
            policy=self.policy,
            env=self.env,
            device=self.network_args.device,
            episodes=self.network_args.training_epsiodes)

        debug_dict = {
            'Training epoch time': th.tensor(time.perf_counter() - h)
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=True)
        self.add_data(
            actions=actions,
            observations=observations,
            rewards=rewards
        )

    def train_step(self, train_loader, actor_step, critic_step, loss_actor, loss_critic):
        for data in train_loader:
            device_data = []
            for dat in data:
                device_data.append(dat.to(self.network_args.device))
            loss_actor = actor_step(device_data, loss_actor)
            loss_critic = critic_step(device_data, loss_critic)
        return loss_actor, loss_critic

    def train(self, epochs):
        next_val = self.network_args.val_every
        next_add = 0
        for epoch in range(epochs):
            if (not self.network_args.imitation_phase) and (self.global_step >= next_add):
                next_add += self.network_args.add_data_every
                self.add_training_data()

            self.policy.eval()
            loss_actor = None
            loss_critic = None
            mean_actor = float('inf')
            mean_critic = float('inf')

            while (mean_actor > self.network_args.actor_threshold) or (mean_critic > self.network_args.critic_threshold):

                loss_actor, loss_critic = self.train_step(
                    train_loader=self.train_loader,
                    actor_step=self.actor_step,
                    critic_step=self.critic_step,
                    loss_actor=loss_actor,
                    loss_critic=loss_critic
                )

                mean_actor = loss_actor.mean()
                mean_critic = loss_critic.mean()

                self.scores.update_min_score(
                    self.scores.mean_critic, mean_critic)
                self.scores.update_min_score(
                    self.scores.mean_actor, mean_actor)

                debug_dict = {
                    'Loss Actor': mean_actor,
                    'Loss Critic': mean_critic
                }
                self.write_tboard_scalar(debug_dict=debug_dict, train=True)
                self.global_step += len(self.train_data)
            if epoch >= next_val:
                next_val = epoch + self.network_args.val_every
                if self.network_args.tboard:
                    self.run_validation()

    def write_tboard_scalar(self, debug_dict, train, step=None):
        if step is None:
            step = self.global_step

        if self.network_args.tboard:
            for para, value in debug_dict.items():
                value = value.detach().to('cpu')
                if train:
                    self.tboard.addTrainScalar(para, value, step)
                else:
                    self.tboard.addValidationScalar(para, value, step)

    def run_validation(self):
        h = time.perf_counter()
        opt_actions, gen_actions, observations, rewards, expected_rewards_before, expected_rewards_after = sample_new_episode(
            policy=self.policy,
            env=self.eval_env,
            device=self.network_args.device,
            episodes=self.network_args.validation_episodes,
            return_gen_trj=True)
        debug_dict = {
            'Validation epoch time': th.tensor(time.perf_counter() - h)
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=False)

        for i in range(min(opt_actions.shape[0], 4)):
            self.createGraphs([gen_actions[i], opt_actions[i]], ['Generated Actions ' + str(i), 'Opimized Actions '+str(i)], plot_name='Trajectories')
            self.createGraphs([rewards[i], self.policy.history.opt_scores[0][i], self.policy.history.gen_scores[0][i]], 
                                ['GT Reward ' + str(i), 'Expected Optimized Reward '+str(i), 'Expected Generated Reward '+str(i)], plot_name='Rewards')

        last_reward, _ = rewards.max(dim=1)
        best_model = self.scores.update_max_score(
            self.scores.mean_reward, last_reward.mean())
        if best_model:
            self.saveNetworkToFile(add='best_validation', data_path=os.path.join(
                self.network_args.data_path, self.logname))
        last_expected_rewards_before = expected_rewards_before[:, -1]
        last_expected_reward_after = expected_rewards_after[:, -1]
        self.analyze_critic_scores(
            last_reward, last_expected_rewards_before, '')
        self.analyze_critic_scores(
            last_reward, last_expected_reward_after, ' optimized')
        success = (last_reward == 1)
        success = success.type(th.float)
        debug_dict = {
            'Success Rate': success.mean(),
            'Reward': last_reward.mean(),
            'Training Epochs': th.tensor(int(len(self.train_data)/self.policy.args_obj.epoch_len))
        }
        print(f'Success Rate: {success.mean()}')
        print(f'Reward: {last_reward.mean()}')
        print(
            f'training samples: {int(len(self.train_data) / self.policy.args_obj.epoch_len)}')
        self.write_tboard_scalar(debug_dict=debug_dict, train=False)


    def analyze_critic_scores(self, reward: th.Tensor, expected_reward: th.Tensor, add: str):
        success = reward == 1
        expected_success = expected_reward >= 0.95
        success = success.reshape(-1)
        fail = ~success
        expected_success = expected_success.reshape(-1)
        expected_fail = ~ expected_success

        expected_success_float = expected_success.type(th.float)
        expected_fail_float = expected_fail.type(th.float)
        fail_float = fail.type(th.float).reshape(-1)
        success_float = success.type(th.float).reshape(-1)

        tp = (expected_success_float * success_float)[success == 1].mean()
        if success_float.sum() == 0:
            tp = th.tensor(0)
        fp = (expected_success_float * fail_float)[fail == 1].mean()
        tn = (expected_fail_float * fail_float)[fail == 1].mean()
        fn = (expected_fail_float * success_float)[success == 1].mean()

        debug_dict = {}

        debug_dict['true positive' + add] = tp
        debug_dict['false positive' + add] = fp
        debug_dict['true negative' + add] = tn
        debug_dict['false negative' + add] = fn
        debug_dict['critic success' +
                   add] = (expected_success == success).type(th.float).mean()
        debug_dict['critic expected reward' + add] = expected_reward.mean()
        debug_dict['critic reward' + add] = reward.mean()
        debug_dict['critic expected success' +
                   add] = expected_success.type(th.float).mean()
        debug_dict['critic L2 error reward' +
                   add] = calcMSE(reward, expected_reward)

        self.write_tboard_scalar(debug_dict=debug_dict, train=False)

    def plot_with_mask(self, label, trj, inpt, mask, name, opt_trj=None):
        if mask.sum() > 0:
            label = label[mask][0]
            trj = trj[mask][0]
            inpt = inpt[mask][0, 0]
            if opt_trj is not None:
                opt_trj = opt_trj[mask][0]
            self.createGraphsMW(d_in=1, d_out=label, result=trj, toy=False,
                                inpt=inpt, name=name, opt_trj=opt_trj, window=0)


    def createGraphs(self, trjs:list([th.tensor]), trj_names:list([str]), plot_name:str):
        np_trjs = []
        trj_colors = ['forestgreen', 'orange', 'pink']
        for trj in trjs:
            np_trjs.append(trj.detach().cpu().numpy())
        self.tboard.plot_graph(trjs=np_trjs, trj_names=trj_names, trj_colors=trj_colors, plot_name=plot_name, step=self.global_step)


    def saveNetworkToFile(self, add, data_path):

        path_to_file = os.path.join(data_path, add)
        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)

        print(path_to_file)

        th.save(self.state_dict(), path_to_file + "/policy_network")
        th.save(self.policy.actor.optimizer.state_dict(),
                path_to_file + "/optimizer_actor")
        th.save(self.policy.critic.optimizer.state_dict(),
                path_to_file + "/optimizer_critic")
        th.save(th.tensor(self.global_step),
                path_to_file + "/global_step")
        with open(path_to_file + '/scores.pkl', 'wb') as f:
            pickle.dump(self.scores, f)

        th.save(self.train_data, path_to_file+'/train')

    def loadNetworkFromFile(self, path, device='cuda'):
        optimize = self.policy.args_obj.optimize
        self.policy.args_obj.optimize = False
        sample_new_episode(
            policy=self.policy,
            env=self.env,
            episodes=1,
            device=self.network_args.device)
        self.load_state_dict(th.load(
            path + "policy_network", map_location=device))
        self.policy.actor.optimizer.load_state_dict(
            th.load(path + "/optimizer_actor"))
        self.policy.critic.optimizer.load_state_dict(
            th.load(path + "/optimizer_critic"))
        self.global_step = int(th.load(path+'/global_step'))
        self.setDatasets(train_data=th.load(
            path+'/train', map_location=device))
        with open(path + '/scores.pkl', 'rb') as f:
            self.scores = pickle.load(f)
        self.policy.args_obj.optimize = optimize

