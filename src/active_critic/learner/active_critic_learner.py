import os
import pickle
import sys
import time
from distutils.log import debug

import tensorflow as tf
import torch as th
import torch.nn as nn
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy, ActiveCriticPolicyHistory
from active_critic.utils.dataset import DatasetAC
from active_critic.utils.gym_utils import sample_new_episode
from active_critic.utils.pytorch_utils import calcMSE, get_rew_mask
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
        self.train_data.size = network_args_obj.buffer_size
        self.train_data.onyl_positiv = False
        self.train_data.size = self.network_args.buffer_size

        self.last_observation = None
        self.last_action = None
        self.last_reward = None

        self.num_sampled_episodes = 0

        self.current_patients = network_args_obj.patients

    def setDatasets(self, train_data: DatasetAC):
        self.train_data = train_data
        self.train_data.size = self.network_args.buffer_size
        if len(train_data) > 0:
            self.train_loader = DataLoader(
                dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def plot_history(self, history:ActiveCriticPolicyHistory, rewards:th.Tensor, prefix:str, num_timesteps:int = 3):
        for epoch in range(min(history.trj[0].shape[0], 4)):
            for time_step in range(0, history.trj[0].shape[2], int(history.trj[0].shape[2]/num_timesteps)):
                #history: [epochs, opt_step, act_step, pred_step, dim]
                gen_actions = history.trj[0][epoch, 0, time_step]
                opt_actions = history.trj[0][epoch, -1, time_step]
                self.createGraphs([gen_actions, opt_actions], ['Generated Actions', 'Opimized Actions'], plot_name=f'{prefix} Trajectories Epoch {epoch} Step {time_step}')
                
                pred_gen_rew = history.scores[0][epoch, 0, time_step]
                pred_opt_rew = history.scores[0][epoch, -1, time_step]
                gt_rewards = rewards[epoch]
                self.createGraphs([pred_gen_rew, pred_opt_rew, gt_rewards], ['Pred Gen Rewards', 'Pred Opt Rewards', 'GT Rewards'], plot_name=f'{prefix} Rewards Epoch {epoch} Step {time_step}')
                
    def add_data(self, actions: th.Tensor, observations: th.Tensor, rewards: th.Tensor):
        self.train_data.add_data(obsv=observations.to(
            'cpu'), actions=actions.to('cpu'), reward=rewards.to('cpu'))
        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def actor_step(self, data, loss_actor, offset):
        obsv, actions, reward = data
        actions = th.clamp(actions, min=-1, max=1)
        actor_input = self.policy.get_actor_input(
            obs=obsv, actions=actions, rew=reward)
        #morgen?
        mask = th.ones_like(reward, dtype=th.bool).squeeze(-1)
        mask[:,-1] = 1
        debug_dict = self.policy.actor.optimizer_step(
            inputs=actor_input, label=actions, mask=mask, offset=offset)
        if loss_actor is None:
            loss_actor = debug_dict['Loss '].unsqueeze(0)
        else:
            loss_actor = th.cat(
                (loss_actor, debug_dict['Loss '].unsqueeze(0)), dim=0)
        return loss_actor

    def pain_boundaries(self, actions:th.Tensor, min_bound:float, max_bound:float):
        pain = (th.exp((actions[actions < min_bound] - min_bound)**2)).sum().nan_to_num()
        pain += (th.exp((actions[actions > max_bound] - max_bound)**2)).sum().nan_to_num()
        pain = pain / actions.numel()
        pain = pain * self.policy.args_obj.epoch_len
        return pain

    def critic_step(self, data, loss_critic, offset):
        obsv, actions, reward = data
        critic_inpt = self.policy.get_critic_input(acts=actions, obs_seq=obsv)
        mask = get_rew_mask(reward)

        #morgen?
        mask = th.zeros_like(reward, dtype=th.bool).squeeze(-1)
        mask[:,-1] = 1
        #mask = mask.squeeze()
        assert mask.sum() == reward.shape[0], 'mask wrong calculated'

        scores = self.policy.critic.forward(inputs=critic_inpt, offset=offset)
        individual_loss = (scores[mask].reshape(-1) - reward[mask].reshape(-1))**2
        individual_side_loss = (scores.reshape(-1) - reward.reshape(-1))**2
        individual_loss.reshape([1,-1])
        loss = individual_loss.mean() + individual_side_loss.mean()
        self.policy.critic.optimizer.zero_grad()
        loss.backward()
        self.policy.critic.optimizer.step()

        if loss_critic is None:
            loss_critic = individual_loss.detach()
        else:
            loss_critic = th.cat(
                (loss_critic, individual_loss.detach()), dim=0)
        return loss_critic

    def causal_step(self, data, loss_causal, offset):
        obsv, actions, reward = data

        initial_sequence_mask = th.all(th.all(obsv[:, 1:] == 0, dim=-1), dim=-1)
        steps = obsv.shape[1] - th.all(obsv[:, 1:] == 0, dim=-1).sum(dim=-1)
        weights = 1 / (steps)
        obsv = obsv[initial_sequence_mask]

        actions = actions[initial_sequence_mask]
        reward = reward[initial_sequence_mask]
        if initial_sequence_mask.sum() == 0:
            return loss_causal
        else:
            actor_input = self.policy.get_actor_input(
                obs=obsv, actions=actions, rew=th.ones_like(reward))
            optimized_actions = self.policy.actor.forward(actor_input, offset=offset)
            #optimized_actions.retain_grad()

            #optimized_actions = self.policy.proj_actions(org_actions=actions, new_actions=optimized_actions, steps=steps-1)
            critic_input = self.policy.get_critic_input(acts=optimized_actions, obs_seq=obsv)

            #morgen angucken
            mask = th.zeros_like(reward)
            mask[:,-1] = 1
            mask = mask.type(th.bool)

            scores = self.policy.critic.forward(critic_input, offset=offset)
            assert mask.sum() == reward.shape[0], 'mask wrong calculated'
            assert scores[mask].numel() == mask.sum(), 'mask wrong applied'

            individual_loss = (scores[mask].reshape(-1) - th.ones_like(scores[mask].reshape(-1)))**2
            scores_shape_before = list(individual_loss.shape)

            #individual_loss = individual_loss * weights
            assert list(individual_loss.shape) == scores_shape_before, 'scores weights wrong.'
            individual_loss.reshape([1,-1])
            loss = individual_loss.mean()

            pain = self.pain_boundaries(actions=optimized_actions, min_bound=-1, max_bound=1)
            loss = loss + pain
            #careful
            self.policy.critic.optimizer.zero_grad()
            self.policy.actor.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            self.policy.critic.optimizer.step()
            self.policy.actor.optimizer.step()

            if loss_causal is None:
                loss_causal = individual_loss.detach()#[initial_sequence_mask]
            else:
                loss_causal = th.cat(
                    (loss_causal, individual_loss.detach()), dim=0)#[initial_sequence_mask]), dim=0)
            

            return loss_causal


    def add_training_data(self):
        h = time.perf_counter()
        self.policy.inference = False
        self.num_sampled_episodes += self.network_args.training_epsiodes
        actions, observations, rewards = sample_new_episode(
            policy=self.policy,
            env=self.env,
            device=self.network_args.device,
            episodes=self.network_args.training_epsiodes)
        print(f'Training Rewards: {rewards.mean()}')
        debug_dict = {
            'Training epoch time': th.tensor(time.perf_counter() - h),
            'Training Rewards' : rewards.detach().mean()
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=True)
        self.add_data(
            actions=actions,
            observations=observations,
            rewards=rewards
        )

        if self.last_observation is None:
            self.last_observation = observations
            self.last_observation[:,1:] = 0
            self.last_action = actions
            self.last_reward = rewards
        else:
            with th.no_grad():
                new_actions_inputs = self.policy.get_actor_input(self.last_observation, None, th.ones_like(rewards))
                new_actions = self.policy.actor.forward(inputs=new_actions_inputs, offset=0)
                new_score_inputs = self.policy.get_critic_input(acts=new_actions, obs_seq=self.last_observation)
                new_scores = self.policy.critic.forward(inputs=new_score_inputs, offset=0)
                new_old_scores_input = self.policy.get_critic_input(self.last_action, self.last_observation)
                new_old_scores = self.policy.critic.forward(inputs=new_old_scores_input, offset=0)

            self.createGraphs(trjs=[self.last_reward[0], new_scores[0], new_old_scores[0]], trj_names=['GT Reward', 'New Trj Score', 'Old Trj Score'], plot_name='Trajectory Scores Improvement')
            self.createGraphs(trjs=[self.last_action[0], new_actions[0]], trj_names=['Old Actions', 'New Actions'], plot_name='Trajectory Improvement')
            self.last_observation = observations
            self.last_observation[:,1:] = 0
            self.last_action = actions
            self.last_reward = rewards
        try:
            self.plot_history(self.policy.history, rewards=rewards, prefix='Train ', num_timesteps=10)
        except:
            print('Training FAIL')


    def train_step(self, train_loader, actor_step, critic_step, causal_step):
        loss_actor = None
        loss_critic = None
        loss_causal = None
        
        for data in train_loader:
            device_data = []
            for dat in data:
                device_data.append(dat.to(self.network_args.device))
            offset = th.randint(low=0, high=device_data[0].shape[1], size=[1], device=self.network_args.device)
            loss_actor = actor_step(device_data, loss_actor, offset=offset)
            loss_critic = critic_step(device_data, loss_critic, offset=offset)
            loss_causal = causal_step(device_data, loss_causal, offset=offset)
        return loss_actor, loss_critic, loss_causal

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
            loss_causal = None
            max_causal = float('inf')
            max_critic = float('inf')

            while (max_causal > self.network_args.causal_threshold) or (max_critic > self.network_args.critic_threshold):
                loss_actor, loss_critic, loss_causal = self.train_step(
                    train_loader=self.train_loader,
                    actor_step=self.actor_step,
                    critic_step=self.critic_step,
                    causal_step=self.causal_step
                )
                max_critic = loss_critic.max()
                max_causal = loss_causal.max()

                debug_dict = {
                    'Loss Actor': loss_actor.mean(),
                    'Loss Critic': max_critic,
                    'Loss Causal': max_causal
                }
                self.write_tboard_scalar(debug_dict=debug_dict, train=True, step=self.global_step)
                self.global_step += len(self.train_data)
                self.current_patients -= len(self.train_data)
                if self.current_patients <= 0:
                    self.network_args.patients *= 2
                    self.current_patients = self.network_args.patients
                    self.policy.reset_models()

                    loss_actor = None
                    loss_critic = None
                    loss_causal = None
                    max_causal = float('inf')
                    max_critic = float('inf')

            self.current_patients = self.network_args.patients

            if epoch >= next_val:
                next_val = epoch + self.network_args.val_every
                if self.network_args.tboard:
                    self.run_validation()

    def write_tboard_scalar(self, debug_dict, train, step=None):
        if step is None:
            step = self.num_sampled_episodes

        if self.network_args.tboard:
            for para, value in debug_dict.items():
                value = value.detach().to('cpu')
                if train:
                    self.tboard.addTrainScalar(para, value, step)
                else:
                    self.tboard.addValidationScalar(para, value, step)

    def run_validation(self):
        h = time.perf_counter()
        self.policy.inference = False
        opt_actions, observations, rewards = sample_new_episode(
            policy=self.policy,
            env=self.eval_env,
            device=self.network_args.device,
            episodes=self.network_args.validation_episodes,
            return_gen_trj=True)
        debug_dict = {
            'Validation epoch time': th.tensor(time.perf_counter() - h)
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=False)

        try:
            self.plot_history(self.policy.history, rewards=rewards, prefix='Validation ', num_timesteps=10)
        except:
            'Validation FAIL'

        last_reward, _ = rewards.max(dim=1)

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
            f'training samples: {self.num_sampled_episodes}')
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
            th.load(path + "/optimizer_actor", map_location=device))
        self.policy.critic.optimizer.load_state_dict(
            th.load(path + "/optimizer_critic", map_location=device))
        self.global_step = int(th.load(path+'/global_step'))
        self.setDatasets(train_data=th.load(
            path+'/train', map_location=device))
        '''with open(path + '/scores.pkl', 'rb') as f:
            self.scores = pickle.load(f, device=device)'''
        self.policy.args_obj.optimize = optimize

