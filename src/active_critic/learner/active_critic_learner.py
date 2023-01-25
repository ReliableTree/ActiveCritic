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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_size_list)):
            self.hidden_layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
            self.hidden_layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_size_list[-1], output_size)

    def forward(self, x):
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i+1](x)
        return self.output_layer(x)



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
        self.last_scores = None

        self.last_trj = None
        self.last_trj_training = None
        

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

        b, _ = th.max(reward, dim=1)
        successfull_trj = (b == 1).squeeze()

        if successfull_trj.sum() > 0:
            obsv = obsv[successfull_trj]
            actions = actions[successfull_trj]
            reward = reward[successfull_trj]

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

    def add_training_data(self, policy=None, episodes = 1):
        if policy is None:
            policy = self.policy
            policy.eval()
        h = time.perf_counter()
        actions, observations, rewards, _, expected_rewards = sample_new_episode(
            policy=policy,
            env=self.env,
            extractor=self.network_args.extractor,
            device=self.network_args.device,
            episodes=episodes)

        last_reward, _ = rewards.max(dim=1)
        print(f'Mean Reward: {last_reward.mean()}')

        if self.last_trj_training is not None:
            self.compare_expecations(self.last_trj_training, 'Training')
        self.last_trj_training = [
            observations[:1],
            actions[:1],
            rewards[:1],
            expected_rewards[:1]
        ]

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
        next_add = self.network_args.add_data_every
        for epoch in range(epochs):
            if epoch >= next_val:
                next_val = epoch + self.network_args.val_every
                if self.network_args.tboard:
                    print('_____________________________________________________________')
                    self.run_validation(optimize=True)
                    self.run_validation(optimize=False)


            if (not self.network_args.imitation_phase) and (epoch >= next_add):
                next_add += self.network_args.add_data_every
                self.add_training_data(episodes=self.network_args.training_epsiodes)

            self.policy.train()

            max_actor = float('inf')
            max_critic = float('inf')

            while (max_actor > self.network_args.actor_threshold) or (max_critic > self.network_args.critic_threshold):
                loss_actor = None
                loss_critic = None

                loss_actor, loss_critic = self.train_step(
                    train_loader=self.train_loader,
                    actor_step=self.actor_step,
                    critic_step=self.critic_step,
                    loss_actor=loss_actor,
                    loss_critic=loss_critic
                )

                max_actor = th.max(loss_actor)
                max_critic = th.max(loss_critic)

                self.scores.update_min_score(
                    self.scores.mean_critic, max_critic)
                self.scores.update_min_score(
                    self.scores.mean_actor, max_actor)

                debug_dict = {
                    'Loss Actor': max_actor,
                    'Loss Critic': max_critic,
                    'Examples': th.tensor(int(len(self.train_data) / self.policy.args_obj.epoch_len))
                }
                self.write_tboard_scalar(debug_dict=debug_dict, train=True, step=self.global_step)
                self.global_step += len(self.train_data)


    def write_tboard_scalar(self, debug_dict, train, step=None):
        if step is None:
            step = int(len(self.train_data) / self.policy.args_obj.epoch_len)

        if self.network_args.tboard:
            for para, value in debug_dict.items():
                value = value.detach().to('cpu')
                if train:
                    self.tboard.addTrainScalar(para, value, step)
                else:
                    self.tboard.addValidationScalar(para, value, step)

    def compare_expecations(self, trj, post_fix):
        last_obsv = trj[0]
        last_actions = trj[1]
        last_rewards = trj[2]
        last_expected_reward = trj[3]
        critic_input = self.policy.get_critic_input(acts=last_actions, obs_seq=last_obsv)
        expected_reward = self.policy.critic.forward(critic_input)
        self.createGraphs([last_rewards[0], last_expected_reward[0], expected_reward[0]], ['Last Rewards', 'Last Expected Rewards', 'Current Expectation'], 'Compare Learn Critic ' + post_fix)

    def run_validation(self, optimize):
        if optimize:
            fix = ' optimize'
            if self.last_trj is not None:
                self.compare_expecations(self.last_trj, 'Validation')
        else:
            fix = ' w/0 optimize'
            
        pre_opt = self.policy.args_obj.optimize
        self.policy.args_obj.optimize = optimize

        h = time.perf_counter()
        opt_actions, gen_actions, observations, rewards, expected_rewards_before, expected_rewards_after = sample_new_episode(
            policy=self.policy,
            env=self.eval_env,
            extractor=self.network_args.extractor,
            device=self.network_args.device,
            episodes=self.network_args.validation_episodes,
            return_gen_trj=True)
        if optimize:
            self.last_trj = [
                observations[:1],
                opt_actions[:1],
                rewards[:1],
                expected_rewards_after[:1]
            ]
        debug_dict = {
            'Validation epoch time '+fix : th.tensor(time.perf_counter() - h)
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=False)

        for i in range(min(opt_actions.shape[0], 4)):
            self.createGraphs([gen_actions[i], opt_actions[i]], ['Generated Actions', 'Opimized Actions'+str(i)], plot_name='Trajectories ' + str(i) + fix)
            self.createGraphs([rewards[i], self.policy.history.opt_scores[0][i], self.policy.history.gen_scores[0][i]], 
                                ['GT Reward ' + str(i), 'Expected Optimized Reward', 'Expected Generated Reward'], plot_name='Rewards '+str(i) + fix)

        last_reward, _ = rewards.max(dim=1)
        if (self.last_scores is None) or (self.last_scores < last_reward.mean()):
            self.policy.args_obj.opt_steps *= 1.1
            self.last_scores = last_reward.mean()
            print(f'new opt_steps = {self.policy.args_obj.opt_steps}')
        best_model = self.scores.update_max_score(
            self.scores.mean_reward, last_reward.mean())
        if best_model:
            self.saveNetworkToFile(add='best_validation', data_path=os.path.join(
                self.network_args.data_path, self.logname))
        last_expected_rewards_before, _ = expected_rewards_before.max(dim=1)
        last_expected_reward_after, _ = expected_rewards_after.max(dim=1)
        self.analyze_critic_scores(
            last_reward, last_expected_rewards_before,  fix)
        self.analyze_critic_scores(
            last_reward, last_expected_reward_after, ' optimized'+ fix)
        success = (last_reward == 1)
        success = success.type(th.float)
        debug_dict = {
            'Success Rate' + fix: success.mean(),
            'Reward' + fix: last_reward.mean(),
            'Training Epochs' + fix: th.tensor(int(len(self.train_data)/self.policy.args_obj.epoch_len))
        }
        print(f'Success Rate: {success.mean()}' + fix)
        print(f'Reward: {last_reward.mean()}' + fix)
        print(
            f'training samples: {int(len(self.train_data) / self.policy.args_obj.epoch_len)}' + fix)
        self.write_tboard_scalar(debug_dict=debug_dict, train=False)
        self.policy.args_obj.optimize = pre_opt


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
            extractor=self.network_args.extractor,
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

