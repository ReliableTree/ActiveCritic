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
from active_critic.utils.pytorch_utils import calcMSE, get_rew_mask, make_part_obs_data, count_parameters
from active_critic.utils.tboard_graphs import TBoardGraphs
from gym.envs.mujoco import MujocoEnv
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pickle
import copy
import math

class ACLScores:
    def __init__(self) -> None:
        self.mean_actor = [float('inf')]
        self.mean_critic = [float('inf')]
        self.mean_reward = [0]

    def reset_min_score(self, score):
        score[0] = float('inf')

    def update_min_score(self, old_score, new_score):
        new_min = old_score[0] > new_score
        if new_min:
            old_score[0] = new_score
        return new_min

    def update_max_score(self, old_score, new_score):
        new_max = old_score[0] <= new_score
        if new_max:
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
        self.best_success = -1
        self.train_critic = self.network_args.start_critic
        self.virtual_step = 0

        self.NPSuccessRate = None
        self.NPExpectedSuccessRate = None
        self.NPExpectedSuccessRateOptimized = None 
        

        if network_args_obj.tboard:
            self.tboard = TBoardGraphs(
                self.logname, data_path=network_args_obj.data_path)
            self.tboard_opt = TBoardGraphs(
                self.logname + ' optimized', data_path=network_args_obj.data_path)
        self.global_step = 0

        self.train_data = DatasetAC(batch_size=self.network_args.batch_size, device='cpu', max_size=None)
        self.train_data.onyl_positiv = False
        self.exp_dict_opt = None
        self.exp_dict = None

        self.set_best_actor = False

        self.next_critic_init = None

        self.first_switch = False

        self.inter_path = os.path.join(self.network_args.data_path, self.logname, 'inter_models/')
        if not os.path.exists(self.inter_path):
            os.makedirs(self.inter_path)

    def setDatasets(self, train_data: DatasetAC):
        self.train_data = train_data
        if len(train_data) > 0:
            self.train_loader = DataLoader(
                dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def add_data(self, actions: th.Tensor, observations: th.Tensor, rewards: th.Tensor, expert_trjs:th.Tensor, action_history:th.Tensor):
        acts, obsv, rews, steps, exp_trjs = make_part_obs_data(
            actions=actions, observations=observations, rewards=rewards, expert_trjs=expert_trjs)

        self.train_data.add_data(
            obsv=obsv, 
            actions=acts, 
            reward=rews,
            expert_trjs=exp_trjs,
            actions_history=action_history,
            steps=steps
            )
        
        print(f'llen after add: {self.train_data.obsv.shape}')

        self.train_data.onyl_positiv = False
        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)
    
    def add_training_data(self, policy=None, episodes = 1, seq_len = None):
        if policy is None:
            policy = self.policy
            policy.eval()
            opt_before = self.policy.args_obj.optimize
            self.policy.args_obj.optimize = (
                self.policy.args_obj.optimize and 
                self.network_args.start_critic and 
                (self.get_num_pos_samples()>self.network_args.explore_cautious_until))
            print(f'policy oprimisation mode: {self.policy.args_obj.optimize}')
            iterations = math.ceil(episodes/self.env.num_envs)
            policy.training_mode = True

        else:
            opt_before = None

        h = time.perf_counter()
        add_results = []
        for iteration in range(iterations):
            result = sample_new_episode(
                policy=policy,
                env=self.env,
                dense=self.network_args.dense,
                extractor=self.network_args.extractor,
                device=self.network_args.device,
                episodes=self.env.num_envs,
                seq_len=seq_len,
                start_training=self.get_num_training_samples()>self.network_args.explore_until)
            result_array = []
            for res in result:
                result_array.append(res)
            add_results.append(result_array)
        
        for element in add_results[1:]:
            for i, part in enumerate(element):
                add_results[0][i] = th.cat((add_results[0][i], part), dim=0)
        actions, observations, rewards, _, expected_rewards, action_history = add_results[0]
        assert actions.shape[0] >= episodes, 'ASDSAD'
        actions = actions[:episodes]
        observations = observations[:episodes]
        rewards = rewards[:episodes]
        expected_rewards = expected_rewards[:episodes]
        action_history = action_history[:episodes]
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
        success = rewards.squeeze().max(-1).values
        success = (success == 1).type(th.float).mean()

        sparse_reward, _ = rewards.max(dim=1)

        print(f'last rewards: {rewards.mean()}')
        print(f'max last rewards: {sparse_reward.mean()}')

        print(f'last success: {success}')
        print(f'self.last_scores: {self.last_scores}')
        expert_trjs = th.zeros([episodes], dtype=th.bool, device=actions.device)
        self.add_data(
            actions=actions,
            observations=observations,
            rewards=rewards,
            expert_trjs=expert_trjs,
            action_history=action_history
        )
        if opt_before is not None:
            self.policy.args_obj.optimize = opt_before

        if self.get_num_training_samples() < self.network_args.explore_until:
            self.policy.actor.init_model()
            self.policy.critic.init_model()
            self.policy.planner.init_model()
            self.policy.args_obj.inference_opt_lr = 1e-6
            print('__________________________________reinit model_________________________________')
        
        if self.network_args.explore_cautious_until < self.get_num_pos_samples():
            self.policy.args_obj.inference_opt_lr = 1e-5
            print('higher lr in optimizer')
        else:
            self.policy.actor.init_model()
            self.policy.critic.init_model()
            self.policy.planner.init_model()
            self.policy.args_obj.inference_opt_lr = 1e-6
            print('__________________________________reinit model_________________________________')




    def actor_step(self, data, loss_actor):
        obsv, actions, reward, expert_trjs, _, _, _, success = data
        if self.train_data.onyl_positiv:
            if success.sum() == 0:
                return loss_actor
            obsv = obsv[success]
            actions = actions[success]
            reward = reward[success]
            expert_trjs = expert_trjs[success]

        plans = self.policy.make_plans(acts=actions, obsvs=obsv)
        plans[expert_trjs] = 0

        actor_input = self.policy.get_actor_input(plans=plans, obsvs=obsv, rewards=reward)
        actor_result = self.policy.actor.forward(actor_input)
        loss = calcMSE(actor_result, actions)
        self.policy.actor.optimizer.zero_grad()
        self.policy.planner.optimizer.zero_grad()
        loss.backward()
        self.policy.actor.optimizer.step()
        self.policy.planner.optimizer.step()
        loss = loss.detach()

        if loss_actor is None:
            loss_actor = loss.unsqueeze(0)
        else:
            loss_actor = th.cat(
                (loss_actor, loss.unsqueeze(0)), dim=0)
        self.write_tboard_scalar(debug_dict={'lr actor': th.tensor(self.policy.actor.scheduler.get_last_lr()).mean()}, train=True)
        return loss_actor
    
    def critic_step(self, data, loss_critic, loss_prediction):

        obsv, actions, reward, expert_trjs, prev_proposed_actions, steps, prev_observation, _ = data

        if self.policy.critic.wsms.sparse:
            label = self.make_critic_score(reward)
        else:
            label = reward


        critic_input = self.policy.get_critic_input(obsvs=obsv, acts=actions)
        critic_result = self.policy.critic.forward(critic_input)
        reward_loss, l2_dist = calcMSE(critic_result, label, return_tensor=True)

        if self.network_args.use_pred_loss:
            prediction_mask = steps > 0
            critic_predicted_input_prev = self.policy.get_critic_input(obsvs=prev_observation[prediction_mask], acts=prev_proposed_actions[prediction_mask])
            critic_pred_result_prev = self.policy.critic.forward(critic_predicted_input_prev)

            critic_predicted_input_current = self.policy.get_critic_input(obsvs=obsv[prediction_mask], acts=prev_proposed_actions[prediction_mask])
            critic_pred_result_current = self.policy.critic.forward(critic_predicted_input_current)

            pred_loss, l2_pred = calcMSE(critic_pred_result_current, critic_pred_result_prev, return_tensor=True)

            loss = reward_loss + pred_loss
        else:
            l2_pred = th.zeros_like(l2_dist)
            loss = reward_loss


        self.policy.critic.optimizer.zero_grad()
        loss.backward()
        self.policy.critic.optimizer.step()
        loss = loss.detach()
        if loss_critic is None:
            loss_critic = l2_dist.reshape([-1])
            loss_prediction = l2_pred.reshape([-1])
        else:
            loss_critic = th.cat(
                (loss_critic, l2_dist.reshape([-1])), dim=0)
            loss_prediction = th.cat(
                (loss_prediction, l2_pred.reshape([-1])), dim=0)
        self.write_tboard_scalar(debug_dict={'lr actor': th.tensor(self.policy.critic.scheduler.get_last_lr()).mean()}, train=True)
        return loss_critic, loss_prediction


    def train_step(self, train_loader, loss_actor, loss_critic, train_critic, loss_prediction):

        self.train_data.onyl_positiv = False
        local_step = 0
        for data in train_loader:
            device_data = []
            for dat in data:
                device_data.append(dat.to(self.network_args.device))
            local_step += dat.shape[0]
            self.train_data.onyl_positiv = (self.policy.critic.wsms.sparse or (self.train_data.success.sum() > 0))
            self.train_data.onyl_positiv = True
            loss_actor = self.actor_step(device_data, loss_actor)
            self.train_data.onyl_positiv = False
            if train_critic:
                loss_critic, loss_prediction = self.critic_step(device_data, loss_critic, loss_prediction)
            if local_step > self.network_args.max_epoch_steps:
                print('max steps in learning reached')
                return loss_actor, loss_critic, loss_prediction

        return loss_actor, loss_critic, loss_prediction

    def get_num_training_samples(self):
        return self.virtual_step
    
    def get_num_pos_samples(self):
        if self.train_data.success is None:
            return 0
        return int(self.train_data.success.sum() / self.policy.args_obj.epoch_len)

    def train(self, epochs):
        next_val = 0
        next_add = 0
        for epoch in range(epochs):
            if self.global_step >= next_val:
                next_val = self.global_step + self.network_args.val_every
                print(f'current run stats: {self.logname}')
                if self.network_args.tboard:
                    print('_____________________________________________________________')
                    th.save(self.policy.actor.state_dict(),self.inter_path +  'actor_before')
                    th.save(self.policy.planner.state_dict(),self.inter_path +  'planner_before')

                    if self.set_best_actor:
                        self.policy.actor.load_state_dict(th.load(self.inter_path + 'best_actor'))
                        self.policy.planner.load_state_dict(th.load(self.inter_path + 'best_planner'))

                    self.policy.eval()
                    self.run_validation(optimize=True)
                    self.run_validation(optimize=False)
                    print(f'self.get_num_training_samples(): {self.get_num_training_samples()}')
                    print(f'self.network_args.total_training_epsiodes: {self.network_args.total_training_epsiodes}')
                    self.policy.actor.load_state_dict(th.load(self.inter_path + 'actor_before'), strict=False)
                    self.policy.planner.load_state_dict(th.load(self.inter_path + 'planner_before'), strict=False)
                    self.scores.reset_min_score(self.scores.mean_actor)

                    if self.get_num_training_samples()>= self.network_args.total_training_epsiodes:
                        return None


            if (not self.network_args.imitation_phase) and (self.global_step >= next_add):
                next_add = self.global_step + self.network_args.add_data_every
                th.save(self.policy.actor.state_dict(),self.inter_path +  'actor_before')
                th.save(self.policy.planner.state_dict(),self.inter_path +  'planner_before')
                if self.set_best_actor:
                    self.policy.actor.load_state_dict(th.load(self.inter_path + 'best_actor'))
                    self.policy.planner.load_state_dict(th.load(self.inter_path + 'best_planner'))
                if (self.train_data.success is not None) and (self.train_data.success.sum() == 0):
                    self.policy.actor.init_model()
                    self.policy.planner.init_model()
                    print(f'_________________________reinit actor__________________________________________')
                self.add_training_data(episodes=self.network_args.training_epsiodes)
                self.policy.actor.load_state_dict(th.load(self.inter_path + 'actor_before'), strict=False)
                self.policy.planner.load_state_dict(th.load(self.inter_path + 'planner_before'), strict=False)

                self.virtual_step += self.network_args.training_epsiodes

                if self.next_critic_init is None:
                    self.next_critic_init = self.get_num_training_samples() * 4
                if (self.get_num_training_samples() > self.next_critic_init) and (self.get_num_training_samples() < 100):
                    self.policy.critic.init_model()
                    self.policy.actor.init_model()
                    self.policy.planner.init_model()
                    self.next_critic_init = 4 * self.get_num_training_samples()
                    print('______________________________init models___________________________________')

            elif (self.global_step >= next_add):
                next_add = self.global_step + self.network_args.add_data_every
                self.virtual_step += self.network_args.training_epsiodes
            if (self.network_args.imitation_phase) and (epoch >= next_add):
                next_add += self.network_args.add_data_every
                self.network_args.total_training_epsiodes -= self.network_args.training_epsiodes
                print(f'training now: {self.network_args.training_epsiodes}')
                print(f'self.network_args.total_training_epsiodes: {self.network_args.total_training_epsiodes}')

            if self.get_num_pos_samples() > self.network_args.explore_cautious_until or True:
                self.policy.train()

                max_actor = float('inf')
                mean_critic = float('inf')

                loss_actor = None
                loss_critic = None
                loss_prediction = None

                loss_actor, loss_critic, loss_prediction = self.train_step(
                    train_loader=self.train_loader,
                    loss_actor=loss_actor,
                    loss_critic=loss_critic,
                    train_critic=self.train_critic,
                    loss_prediction=loss_prediction
                )

                new_min = self.scores.update_min_score(
                    self.scores.mean_actor, max_actor)
                if loss_actor is not None:
                    max_actor = th.max(loss_actor)
                else:
                    max_actor = None

                
                if loss_critic is not None:
                    mean_critic = th.mean(loss_critic)
                    mean_prediction = th.mean(loss_prediction)
                    self.scores.update_min_score(
                    self.scores.mean_critic, mean_critic)
                    self.train_critic = (mean_critic>self.network_args.min_critic_threshold)
                else:
                    mean_critic = None

                

                if new_min:
                    th.save(self.policy.actor.state_dict(),self.inter_path +  'best_actor')
                    th.save(self.policy.planner.state_dict(),self.inter_path +  'best_planner')
                    self.set_best_actor = True

                reward = self.train_data.reward
                b, _ = th.max(reward, dim=1)
                successfull_trj = (b == 1)
                positive_examples = th.tensor(int(successfull_trj.sum()/self.policy.args_obj.epoch_len))

                debug_dict = {
                    'Examples': th.tensor(int(len(self.train_data.obsv)/self.policy.args_obj.epoch_len)),
                    'Positive Examples': positive_examples
                }
                if mean_critic is not None:
                    debug_dict['Loss Critic'] = mean_critic
                    debug_dict['Loss Prediction'] = mean_prediction
                else:
                    mean_critic = 0

                if max_actor is not None:
                    debug_dict['Loss Actor'] = max_actor
                else:
                    max_actor = 0

                self.write_tboard_scalar(debug_dict=debug_dict, train=True, step=self.global_step)
                self.train_data.onyl_positiv = False
                self.global_step += len(self.train_data)
                '''if current_patients <= 0:
                    self.policy.critic.init_model()
                    print('reinit critic')
                    self.network_args.patients *= 2'''
            else:
                self.global_step = min(next_add, next_val)



    def write_tboard_scalar(self, debug_dict, train, step=None, optimize=True):
        if step is None:
            step = self.get_num_training_samples()

        if self.network_args.tboard:
            for para, value in debug_dict.items():
                value = value.detach().to('cpu')
                if train:
                    if optimize:
                        self.tboard_opt.addTrainScalar(para, value, step)
                    else:
                        self.tboard.addTrainScalar(para, value, step)
                else:
                    if optimize:
                        self.tboard_opt.addValidationScalar(para, value, step)
                    else:
                        self.tboard.addValidationScalar(para, value, step)

    def compare_expecations(self, trj, post_fix):
        last_obsv = trj[0]
        last_actions = trj[1]
        last_rewards = trj[2]
        last_expected_reward = trj[3]
        critic_input = self.policy.get_critic_input(obsvs=last_obsv, acts=last_actions)

        expected_reward = self.policy.critic.forward(critic_input).reshape([-1, 1, 1 ])


        label = self.make_critic_score(last_rewards)
        last_expected_label = self.make_critic_score(last_expected_reward)
        self.createGraphs([label[0], last_expected_label[0], expected_reward[0]], ['Last Rewards', 'Last Expected Rewards', 'Current Expectation'], 'Compare Learn Critic ' + post_fix)

    def make_critic_score(self, rewards):
        labels = rewards.max(1).values.squeeze().reshape([-1, 1, 1]) == 1
        labels = labels.type(th.float)
        return labels
    
    def save_stat(self, success, rewards, expected_success, opt_exp, exp_dict):
        if exp_dict is None:
            exp_dict = {
            'success_rate':success.mean().cpu().numpy(),
            'expected_success' : expected_success.mean().cpu().numpy(),
            'rewards': rewards.unsqueeze(0).detach().cpu().numpy(),
            'step':np.array(self.get_num_training_samples())
            }
            if opt_exp is not None:
                exp_dict['optimized_expected'] =  opt_exp.mean().cpu().numpy()

        else:
            exp_dict['success_rate'] = np.append(exp_dict['success_rate'], success.mean().cpu().numpy())
            exp_dict['expected_success'] = np.append(exp_dict['expected_success'], expected_success.mean().cpu().numpy())
            exp_dict['step'] = np.append(exp_dict['step'], np.array(self.get_num_training_samples()))
            exp_dict['rewards'] = np.append(exp_dict['rewards'], np.array(rewards.unsqueeze(0).detach().cpu().numpy()), axis=0)

            if opt_exp is not None:
                exp_dict['optimized_expected'] = np.append(exp_dict['optimized_expected'], opt_exp.mean().cpu().numpy())

        path_to_stat = os.path.join(self.network_args.data_path, self.network_args.logname)

        if not os.path.exists(path_to_stat):
            os.makedirs(path_to_stat)

        add = ''
        if opt_exp is not None:
            add = 'optimized'

        with open(path_to_stat + '/stats'+add, 'wb') as handle:
            pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return exp_dict


    def run_validation(self, optimize):
        self.policy.training_mode = False

        if optimize:
            fix = ' optimize'
            if self.last_trj is not None:
                self.compare_expecations(self.last_trj, 'Validation')
            '''opt_steps_before = self.policy.args_obj.opt_steps
            if self.train_data.success is not None:
                self.policy.args_obj.opt_steps = min(opt_steps_before, 10 * self.train_data.success.sum())
            else:
                self.policy.args_obj.opt_steps = 0'''
            print(f'self.policy.args_obj.opt_steps: {self.policy.args_obj.opt_steps}')
        else:
            fix = ''
            
        pre_opt = self.policy.args_obj.optimize
        self.policy.args_obj.optimize = optimize

        h = time.perf_counter()
        rewards_cumm = None

        for i in range(self.network_args.validation_rep):
            opt_actions, gen_actions, observations, rewards_run, expected_rewards_before, expected_rewards_after, _ = sample_new_episode(
                policy=self.policy,
                env=self.eval_env,
                dense=self.network_args.dense,
                extractor=self.network_args.extractor,
                device=self.network_args.device,
                episodes=self.network_args.validation_episodes,
                return_gen_trj=True,
                start_training=self.get_num_training_samples()> self.network_args.explore_until)
            if rewards_cumm is None:
                rewards_cumm = rewards_run
            else:
                rewards_cumm = th.cat((rewards_cumm, rewards_run))
        if optimize:
            self.last_trj = [
                observations[:1],
                opt_actions[:1],
                rewards_run[:1],
                expected_rewards_after[:1]
            ]
        debug_dict = {
            'Validation epoch time '+fix : th.tensor(time.perf_counter() - h)
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=False, optimize=optimize)

        for i in range(min(opt_actions.shape[0], 4)):
            self.createGraphs([gen_actions[i, 0], opt_actions[i]], ['Generated Actions', 'Opimized Actions'+str(i)], plot_name='Trajectories ' + str(i) + fix)
            labels = self.make_critic_score(rewards=rewards_run)
            self.createGraphs([labels[i].reshape([-1, 1]), self.policy.history.opt_scores[0][i].reshape([-1, 1]), self.policy.history.gen_scores[0][i].reshape([-1, 1])], 
                                ['GT Reward ' + str(i), 'Expected Optimized Reward', 'Expected Generated Reward'], plot_name='Rewards '+str(i) + fix)

        last_sparse_reward, _ = rewards_run.max(dim=1)
        sparse_reward, _ = rewards_cumm.max(dim=1)

        best_model = self.scores.update_max_score(
            self.scores.mean_reward, sparse_reward.mean())
            

        last_expected_rewards_before, _ = expected_rewards_before.max(dim=1)
        last_expected_reward_after, _ = expected_rewards_after.max(dim=1)
            

        self.analyze_critic_scores(
            last_sparse_reward, last_expected_rewards_before,  fix)
        self.analyze_critic_scores(
            last_sparse_reward, last_expected_reward_after, ' optimized'+ fix)
        success = (sparse_reward == 1)
        success = success.type(th.float)

        debug_dict = {
            'Success Rate': success.mean(),
            'Reward': sparse_reward.mean(),
            'Training Epochs': th.tensor(self.get_num_training_samples())
        }

        if not optimize:
            if (self.best_success >= 0) and (self.best_success > success.mean()):
                self.network_args.start_critic = True

            if self.network_args.start_critic:
                self.train_critic = True

            if success.mean() > self.best_success:
                self.best_success = success.mean()


        print(f'Success Rate: {success.mean()}' + fix)
        print(f'Reward: {sparse_reward.mean()}' + fix)
        try:
            print(
                f'training samples: {int(len(self.train_data.obsv))}' + fix)
            print(f'positive training samples: {int(self.train_data.success.sum()/self.policy.args_obj.epoch_len)}' + fix)
        except:
            pass
        self.write_tboard_scalar(debug_dict=debug_dict, train=False, optimize=optimize, step=self.get_num_training_samples())

        self.policy.args_obj.optimize = pre_opt

        if optimize:
            exp_after = expected_rewards_after
            exp_dict = self.exp_dict_opt
        else:
            exp_after = None
            exp_dict = self.exp_dict

        exp_dict = self.save_stat(success=success, rewards=rewards_cumm, expected_success=expected_rewards_before, opt_exp=exp_after, exp_dict=exp_dict)

        if optimize:
            self.exp_dict_opt = exp_dict
            #self.policy.args_obj.opt_steps = opt_steps_before
            print(f'self.policy.args_obj.opt_steps after: {self.policy.args_obj.opt_steps}')
        else:
            self.exp_dict = exp_dict



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
        if self.network_args.make_graphs:
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
            dense=self.network_args.dense,
            extractor=self.network_args.extractor,
            episodes=1,
            device=self.network_args.device,
            start_training=True)
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

