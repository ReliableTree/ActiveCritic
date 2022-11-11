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
        self.training_samples = 0

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
        self.train_data.add_data(obsv=observations, actions=actions, reward=rewards)
        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def predicotr_step(self, data):
        obsv, actions, reward = data
        embeddings = self.policy.emitter.forward(obsv)
        inpt_embeddings = embeddings.clone()
        auto_loss = None
        gen_critic_loss = None
        for step in range(obsv.shape[1] - 1):
            embeddings = embeddings.detach()
            pred_embeddings, gen_actions = self.policy.build_sequence(
                actor=self.policy.actor, 
                predictor=self.policy.predicor, 
                seq_len=self.policy.args_obj.epoch_len, 
                embeddings=inpt_embeddings[:,:step+1], 
                tf_mask=self.policy.args_obj.pred_mask,
                actions=actions[:,:step],
                goal_state=None,
                goal_emb_acts=self.policy.goal_label[:,-1])

            loss_predictor = calcMSE(embeddings, pred_embeddings) 
            gen_critic_inputs = self.policy.get_critic_input(pred_embeddings, gen_actions)
            gen_loss_critic = self.policy.critic.calc_loss(inpt=gen_critic_inputs, label=self.policy.goal_label, mask=self.policy.args_obj.opt_mask, print_fn=False)
            

            self.policy.actor.optimizer.zero_grad()
            self.policy.predicor.optimizer.zero_grad()

            gen_loss_mean = gen_loss_critic.mean()
            if gen_critic_loss is None:
                gen_critic_loss = gen_loss_mean.detach()
            else:
                gen_critic_loss += gen_loss_mean.detach()
            
            loss_auto_predictor_mean = loss_predictor.mean()
            if auto_loss is None:
                auto_loss = loss_auto_predictor_mean.detach()
            else:
                auto_loss += loss_auto_predictor_mean.detach()

            loss = gen_loss_mean + loss_auto_predictor_mean
            
            if self.network_args.use_pain:
                pain = self.pain_boundaries(gen_actions, -1, 1)
                loss = loss + pain

            loss.backward()

            self.policy.actor.optimizer.step()
            self.policy.predicor.optimizer.step()

        auto_loss /= step
        gen_critic_loss /= step

        return auto_loss, gen_critic_loss




    def model_step(self, data, losses):
        obsv, actions, reward = data
        embeddings = self.policy.emitter.forward(obsv)

        critic_input = self.policy.get_critic_input(embeddings=embeddings, actions=actions)
        loss_critic = self.policy.critic.calc_loss(inpt=critic_input, label=reward)

        predictor_input = self.policy.get_predictor_input(embeddings=embeddings, actions=actions)

        loss_predictor = self.policy.predicor.calc_loss(inpt=predictor_input[:,:-1], label=embeddings[:,1:], tf_mask=self.policy.args_obj.pred_mask[:-1, :-1])

        
        
        self.policy.emitter.optimizer.zero_grad()
        self.policy.critic.optimizer.zero_grad()
        self.policy.predicor.optimizer.zero_grad()

        loss = loss_critic + loss_predictor
        loss.backward()

        self.policy.emitter.optimizer.step()
        self.policy.critic.optimizer.step()
        self.policy.predicor.optimizer.step()

        loss_auto_predictor_mean, gen_loss_mean = self.predicotr_step(data=data)



        if losses is None:
            losses = [
                loss_critic.unsqueeze(0),
                loss_predictor.unsqueeze(0),
                loss_auto_predictor_mean.unsqueeze(0),
                gen_loss_mean.unsqueeze(0)
            ]
        else:
            losses[0] = th.cat((losses[0], loss_critic.unsqueeze(0)), dim=0)
            losses[1] = th.cat((losses[1], loss_predictor.unsqueeze(0)), dim=0)
            losses[2] = th.cat((losses[2], loss_auto_predictor_mean.unsqueeze(0)), dim=0)
            losses[3] = th.cat((losses[3], gen_loss_mean.unsqueeze(0)), dim=0)
        return losses


    def add_training_data(self):
        self.training_samples += 1
        h = time.perf_counter()
        self.policy.reset()
        actions, observations, rewards= sample_new_episode(
            policy=self.policy,
            env=self.env,
            device=self.network_args.device,
            episodes=self.network_args.training_epsiodes)

        debug_dict = {
            'Training epoch time': th.tensor(time.perf_counter() - h)
        }
        print(f'Training Reward: {rewards[:,-1]}')
        self.plot_history(self.policy.history, rewards=rewards, prefix='Training', num_timesteps=3)
        self.write_tboard_scalar(debug_dict=debug_dict, train=True, step=self.training_samples)
        self.add_data(
            actions=actions,
            observations=observations,
            rewards=rewards
        )

    def train_step(self, train_loader):
        losses = None
        for data in train_loader:
            device_data = []
            for dat in data:
                device_data.append(dat.to(self.network_args.device))
            losses = self.model_step(device_data, losses)
        return losses

    def pain_boundaries(self, actions:th.Tensor, min_bound:float, max_bound:float):
        pain = ((actions[actions < min_bound] - min_bound)**2).sum().nan_to_num()
        pain += ((actions[actions > max_bound] - max_bound)**2).sum().nan_to_num()
        pain = pain / actions.numel()
        return pain

    def train(self, epochs):
        next_val = self.network_args.val_every
        next_add = 0
        for epoch in range(epochs):
            if (not self.network_args.imitation_phase) and (epoch >= next_add):
                next_add = epoch + self.network_args.add_data_every
                self.add_training_data()

            self.policy.eval()
            mean_actor = float('inf')
            mean_critic = float('inf')
            mean_predictor = float('inf')
            mean_gen_score = float('inf')

            while (mean_critic > self.network_args.critic_threshold)\
                or (mean_predictor > self.network_args.predictor_threshold):

                losses = self.train_step(train_loader=self.train_loader)
                mean_critic = losses[0].mean()
                mean_predictor = losses[1].mean()
                loss_auto_predictor_mean = losses[2].mean()
                gen_loss_mean = losses[3].mean()

                self.scores.update_min_score(
                    self.scores.mean_critic, mean_critic)
                self.scores.update_min_score(
                    self.scores.mean_actor, mean_actor)

                debug_dict = {
                    'Loss Critic': mean_critic,
                    'Loss Predictor': mean_predictor,
                    'Loss Gen Score': gen_loss_mean,
                    'Loss Autoregressive Predictor': loss_auto_predictor_mean
                }
                self.write_tboard_scalar(debug_dict=debug_dict, train=True, step=self.global_step)
                self.global_step += len(self.train_data)
            if epoch >= next_val:
                next_val = epoch + self.network_args.val_every
                if self.network_args.tboard:
                    self.run_validation(step = epoch)

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
                
            pred_emb = history.emb[0][epoch, 0, 0]
            act_emb = history.emb[0][epoch, 0, -1]
            self.createGraphs(trjs=[pred_emb, act_emb], trj_names=['Pred Embeddings', 'Actual Embeddins'], plot_name=f'{prefix} Embeddings Epoch {epoch}')

    def run_validation(self, step = None):
        h = time.perf_counter()
        actions, observations, rewards = sample_new_episode(
            policy=self.policy,
            env=self.eval_env,
            device=self.network_args.device,
            episodes=self.network_args.validation_episodes
            )

        debug_dict = {
            'Validation epoch time': th.tensor(time.perf_counter() - h),
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=False, step=step)

        self.plot_history(self.policy.history, rewards=rewards, prefix='Validation', num_timesteps=3)


        last_reward = rewards[:,-1]
        best_model = self.scores.update_max_score(
            self.scores.mean_reward, last_reward.mean())
        if best_model:
            self.saveNetworkToFile(add='best_validation', data_path=os.path.join(
                self.network_args.data_path, self.logname))
        '''last_expected_rewards_before = expected_rewards_before[:,-1]
        last_expected_reward_after = expected_rewards_after[:,-1]
        self.analyze_critic_scores(
            last_reward, last_expected_rewards_before, '')
        self.analyze_critic_scores(
            last_reward, last_expected_reward_after, ' optimized')'''
        success = (last_reward == 1)
        success = success.type(th.float)
        debug_dict = {
            'Success Rate': success.mean(),
            'Reward': last_reward.mean(),
            'Training Epochs': th.tensor(int(len(self.train_data)/self.policy.args_obj.epoch_len)),
        }
        print(f'Success Rate: {success.mean()}')
        print(f'Reward: {last_reward.mean()}')
        print(f'training samples: {self.training_samples}')
        self.write_tboard_scalar(debug_dict=debug_dict, train=False, step=self.training_samples)


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

        self.write_tboard_scalar(debug_dict=debug_dict, train=False, step=self.training_samples)


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

