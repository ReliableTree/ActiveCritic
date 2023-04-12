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
import os

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

        self.train_data = DatasetAC(batch_size=self.network_args.batch_size, device=self.network_args.device)
        self.train_data.onyl_positiv = False
        self.exp_dict_opt = None
        self.exp_dict = None

        self.set_best_actor = False
        self.set_best_critic = False


        self.next_critic_init = None

        self.inter_path = os.path.join(self.network_args.data_path, self.logname, 'inter_models/')
        if not os.path.exists(self.inter_path):
            os.makedirs(self.inter_path)

    def setDatasets(self, train_data: DatasetAC):
        self.train_data = train_data
        if len(train_data) > 0:
            self.train_loader = DataLoader(
                dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def add_data(self, actions: th.Tensor, observations: th.Tensor, rewards: th.Tensor, expert_trjs:th.Tensor):
        acts, obsv, rews = make_part_obs_data(
            actions=actions, observations=observations, rewards=rewards)
        self.train_data.add_data(
            obsv=obsv, 
            actions=acts, 
            reward=rews,
            expert_trjs=expert_trjs
            )
        self.train_data.onyl_positiv = False
        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)
    

    def actor_step(self, data, loss_actor):
        obsv, actions, reward, expert_trjs = data
        plans = self.policy.make_plans(acts=actions, obsvs=obsv)

        app_obsv = th.cat((obsv, obsv[expert_trjs]), dim=0)
        app_actions = th.cat((actions, actions[expert_trjs]), dim=0)
        app_plans = th.cat((plans, plans[expert_trjs]), dim=0)

        app_plans[:plans.shape[0]][expert_trjs] = 0
        actor_input = self.policy.get_actor_input(plans=app_plans, obsvs=app_obsv)
        actor_result = self.policy.actor.forward(actor_input)
        loss = calcMSE(actor_result, app_actions)
        loss = loss + ((app_plans)**2).mean() * self.network_args.plan_decay
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
    
    def critic_step(self, data, loss_critic):
        obsv, actions, reward, expert_trjs = data

        critic_input = self.policy.get_critic_input(obsvs=obsv, acts=actions)
        critic_result = self.policy.critic.forward(critic_input)
        label = self.make_critic_score(reward)
        loss, l2_dist = calcMSE(critic_result, label, return_tensor=True)
        self.policy.critic.optimizer.zero_grad()
        loss.backward()
        self.policy.critic.optimizer.step()
        loss = loss.detach()
        if loss_critic is None:
            loss_critic = l2_dist.reshape([-1])
        else:
            loss_critic = th.cat(
                (loss_critic, l2_dist.reshape([-1])), dim=0)
        self.write_tboard_scalar(debug_dict={'lr actor': th.tensor(self.policy.critic.scheduler.get_last_lr()).mean()}, train=True)
        return loss_critic

    def add_training_data(self, policy=None, episodes = 1, seq_len = None):
        if policy is None:
            self.policy.train_inference = self.network_args.train_inference
            policy = self.policy
            policy.eval()
            opt_before = self.policy.args_obj.optimize
            self.policy.args_obj.optimize = (self.policy.args_obj.optimize and self.network_args.start_critic)
            iterations = math.ceil(episodes/self.env.num_envs)
        else:
            opt_before = None

        h = time.perf_counter()
        add_results = []
        start_training = self.get_num_training_samples()>self.network_args.start_training
        for iteration in range(iterations):
            result = sample_new_episode(
                policy=policy,
                env=self.env,
                extractor=self.network_args.extractor,
                device=self.network_args.device,
                episodes=self.env.num_envs,
                seq_len=seq_len,
                start_training=start_training,
                set_deterministic=False)

            result_array = []
            for res in result:
                result_array.append(res)
            add_results.append(result_array)
        
        for element in add_results[1:]:
            for i, part in enumerate(element):
                add_results[0][i] = th.cat((add_results[0][i], part), dim=0)
        actions, observations, rewards, _, expected_rewards = add_results[0]
        assert actions.shape[0] >= episodes, 'ASDSAD'
        actions = actions[:episodes]
        observations = observations[:episodes]
        rewards = rewards[:episodes]
        expected_rewards = expected_rewards[:episodes]
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

        print(f'success training: {success}')
        expert_trjs = th.zeros([episodes], dtype=th.bool, device=actions.device)
        self.add_data(
            actions=actions,
            observations=observations,
            rewards=rewards,
            expert_trjs=expert_trjs
        )
        if opt_before is not None:
            self.policy.args_obj.optimize = opt_before

        if (success.sum() == 0) and (((self.train_data.success is None) or (self.train_data.success.sum() == 0))):
            self.policy.actor.init_model()
            self.policy.critic.init_model()
            self.policy.planner.init_model()
            self.set_best_actor = False
            self.set_best_critic = False

            print('reinit because no positive data')


    def train_step(self, train_loader, actor_step, critic_step, loss_actor, loss_critic, train_critic):
        self.train_data.onyl_positiv = True
        if len(self.train_data) > 0:
            for data in train_loader:
                device_data = []
                for dat in data:
                    device_data.append(dat.to(self.network_args.device))
                loss_actor = actor_step(device_data, loss_actor)

        if train_critic:
            self.train_data.onyl_positiv = False
            for data in train_loader:
                device_data = []
                for dat in data:
                    device_data.append(dat.to(self.network_args.device))
                loss_critic = critic_step(device_data, loss_critic)

        return loss_actor, loss_critic

    def get_num_training_samples(self):
        #return int(len(self.train_data) - self.network_args.num_expert_demos)
        return self.virtual_step

    def train(self, epochs):
        next_val = 0
        next_add = 0
        for epoch in range(epochs):
            if self.global_step >= next_val:
                next_val = self.global_step + self.network_args.val_every
                if self.network_args.tboard:
                    
                    print('_____________________________________________________________')
                    th.save(self.policy.actor.state_dict(), self.inter_path + 'actor_before'+self.logname)
                    th.save(self.policy.planner.state_dict(), self.inter_path + 'planner_before'+self.logname)

                    if self.set_best_actor:
                        self.policy.actor.load_state_dict(th.load(self.inter_path + 'best_actor'+self.logname))
                        self.policy.planner.load_state_dict(th.load(self.inter_path + 'best_planner'+self.logname))
                        self.saveNetworkToFile(add=self.logname, data_path=self.network_args.data_path)
                    if self.set_best_critic:
                        self.policy.critic.load_state_dict(th.load(self.inter_path + 'best_critic'+self.logname))


                    self.policy.eval()
                    self.run_validation(optimize=True)
                    self.run_validation(optimize=False)
                    print(f'self.get_num_training_samples(): {self.get_num_training_samples()}')
                    self.policy.actor.load_state_dict(th.load(self.inter_path + 'actor_before'+self.logname), strict=False)
                    self.policy.planner.load_state_dict(th.load(self.inter_path + 'planner_before'+self.logname), strict=False)
                    self.scores.reset_min_score(self.scores.mean_actor)
                    self.scores.reset_min_score(self.scores.mean_critic)


                    if self.get_num_training_samples()>= self.network_args.total_training_epsiodes:
                        return None


            if (not self.network_args.imitation_phase) and (self.global_step >= next_add):
                next_add = self.global_step + self.network_args.add_data_every
                th.save(self.policy.actor.state_dict(), self.inter_path + 'actor_before'+self.logname)
                th.save(self.policy.planner.state_dict(), self.inter_path + 'planner_before'+self.logname)
                if self.set_best_actor:
                    self.policy.actor.load_state_dict(th.load(self.inter_path + 'best_actor'+self.logname))
                    self.policy.planner.load_state_dict(th.load(self.inter_path + 'best_planner'+self.logname))
                if self.set_best_critic:
                    self.policy.critic.load_state_dict(th.load(self.inter_path + 'best_critic'+self.logname))

                self.add_training_data(episodes=self.network_args.training_epsiodes)
                self.policy.actor.load_state_dict(th.load(self.inter_path + 'actor_before'+self.logname), strict=False)
                self.policy.planner.load_state_dict(th.load(self.inter_path + 'planner_before'+self.logname), strict=False)

                self.virtual_step += self.network_args.training_epsiodes

                if self.next_critic_init is None:
                    self.next_critic_init = int(self.train_data.success.sum()) * 10
                if self.train_data.success.sum() > self.next_critic_init:
                    self.policy.critic.init_model()
                    self.policy.planner.init_model()
                    self.next_critic_init = self.train_data.success.sum() * 10   
                    self.scores.reset_min_score(self.scores.mean_critic)
                    self.scores.reset_min_score(self.scores.mean_actor)
                    print('_____________________________________________________________init____________________________________________________')
                self.train_critic = True

            elif (self.global_step >= next_add):
                next_add = self.global_step + self.network_args.add_data_every
                self.virtual_step += self.network_args.training_epsiodes
            if (self.network_args.imitation_phase) and (epoch >= next_add):
                next_add += self.network_args.add_data_every
                self.network_args.total_training_epsiodes -= self.network_args.training_epsiodes
                print(f'training now: {self.network_args.training_epsiodes}')
                print(f'self.network_args.total_training_epsiodes: {self.network_args.total_training_epsiodes}')

            self.policy.train()

            max_actor = float('inf')
            mean_critic = float('inf')
            current_patients = self.network_args.patients
            #while ((max_actor > self.network_args.actor_threshold) or (max_critic > self.network_args.critic_threshold and (not self.network_args.imitation_phase))) and current_patients > 0:
            current_patients -= len(self.train_data)

            loss_actor = None
            loss_critic = None

            loss_actor, loss_critic = self.train_step(
                train_loader=self.train_loader,
                actor_step=self.actor_step,
                critic_step=self.critic_step,
                loss_actor=loss_actor,
                loss_critic=loss_critic,
                train_critic=self.train_critic
            )


            if loss_actor is not None:
                max_actor = th.max(loss_actor)
                new_min = self.scores.update_min_score(
                    self.scores.mean_actor, max_actor)
            else:
                max_actor = None
                new_min = False
            
            if loss_critic is not None:
                mean_critic = th.mean(loss_critic)
                new_min_critic = self.scores.update_min_score(
                self.scores.mean_critic, mean_critic)
                self.train_critic = (mean_critic>self.network_args.min_critic_threshold)
            else:
                mean_critic = None
                new_min_critic = False
            

            if new_min:
                th.save(self.policy.actor.state_dict(), self.inter_path + 'best_actor'+self.logname)
                th.save(self.policy.planner.state_dict(), self.inter_path + 'best_planner'+self.logname)
                self.set_best_actor = True
            if new_min_critic:
                th.save(self.policy.critic.state_dict(), self.inter_path + 'best_critic'+self.logname)
                self.set_best_critic = True 
            reward = self.train_data.reward
            b, _ = th.max(reward, dim=1)
            successfull_trj = (b == 1)
            positive_examples = successfull_trj.sum()

            debug_dict = {
                'Examples': th.tensor(int(len(self.train_data.obsv))),
                'Positive Examples': positive_examples
            }
            if mean_critic is not None:
                debug_dict['Loss Critic'] = mean_critic
            else:
                mean_critic = 0

            if max_actor is not None:
                debug_dict['Loss Actor'] = max_actor
            else:
                max_actor = 0

            self.write_tboard_scalar(debug_dict=debug_dict, train=True, step=self.global_step)
            self.global_step += max(int(self.train_data.virt_success.sum().detach().cpu()), self.network_args.batch_size)
            '''if current_patients <= 0:
                self.policy.critic.init_model()
                print('reinit critic')
                self.network_args.patients *= 2'''



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
    
    def save_stat(self, success, rewards, expected_success, opt_exp, exp_dict, gen_actions, opt_actions):
        if exp_dict is None:
            exp_dict = {
            'success_rate':success.mean().cpu().numpy(),
            'expected_success' : expected_success.mean().cpu().numpy(),
            'rewards': rewards.unsqueeze(0).detach().cpu().numpy(),
            'gen_actions': gen_actions.unsqueeze(0).detach().cpu().numpy(),
            'opt_actions' : opt_actions.unsqueeze(0).detach().cpu().numpy(),
            'step':np.array(self.get_num_training_samples())
            }
            if opt_exp is not None:
                exp_dict['optimized_expected'] =  opt_exp.mean().cpu().numpy()
            print(f'save stats gen: {exp_dict["gen_actions"].shape}')
            print(f'save stats opt_actions: {exp_dict["opt_actions"].shape}')

        else:
            exp_dict['success_rate'] = np.append(exp_dict['success_rate'], success.mean().cpu().numpy())
            exp_dict['expected_success'] = np.append(exp_dict['expected_success'], expected_success.mean().cpu().numpy())
            exp_dict['step'] = np.append(exp_dict['step'], np.array(self.get_num_training_samples()))
            exp_dict['rewards'] = np.append(exp_dict['rewards'], np.array(rewards.unsqueeze(0).detach().cpu().numpy()), axis=0)
            exp_dict['gen_actions'] = np.append(exp_dict['gen_actions'], np.array(gen_actions.unsqueeze(0).detach().cpu().numpy()), axis=0)
            exp_dict['opt_actions'] = np.append(exp_dict['opt_actions'], np.array(opt_actions.unsqueeze(0).detach().cpu().numpy()), axis=0)
            print(f'save stats gen: {exp_dict["gen_actions"].shape}')
            print(f'save stats opt_actions: {exp_dict["opt_actions"].shape}')

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
        self.policy.train_inference = False
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
        start_training = self.get_num_training_samples()>self.network_args.start_training

        for i in range(self.network_args.validation_rep):
            opt_actions, gen_actions, observations, rewards_run, expected_rewards_before, expected_rewards_after = sample_new_episode(
                policy=self.policy,
                env=self.eval_env,
                extractor=self.network_args.extractor,
                device=self.network_args.device,
                episodes=self.network_args.validation_episodes,
                return_gen_trj=True,
                start_training=start_training,
                set_deterministic=False)
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
            self.createGraphs([gen_actions[i,0], opt_actions[i]], ['Generated Actions', 'Opimized Actions'+str(i)], plot_name='Trajectories ' + str(i) + fix)
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
            print(f'positive training samples: {int(self.train_data.success.sum())}' + fix)
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


        exp_dict = self.save_stat(
            success=success, 
            rewards=rewards_cumm, 
            expected_success=expected_rewards_before, 
            opt_exp=exp_after, 
            exp_dict=exp_dict,
            gen_actions=gen_actions[0,0],
            opt_actions=opt_actions[0])

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
        th.save(self.policy.planner.optimizer.state_dict(),
                path_to_file + "/optimizer_planner")
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
            device=self.network_args.device,
            set_deterministic=False)

        self.load_state_dict(th.load(
            path + "policy_network", map_location=device))
        self.policy.actor.optimizer.load_state_dict(
            th.load(path + "/optimizer_actor"))
        self.policy.critic.optimizer.load_state_dict(
            th.load(path + "/optimizer_critic"))
        self.policy.planner.optimizer.load_state_dict(
            th.load(path + "/optimizer_critic"))
        self.global_step = int(th.load(path+'/global_step'))
        self.setDatasets(train_data=th.load(
            path+'/train', map_location=device))
        with open(path + '/scores.pkl', 'rb') as f:
            self.scores = pickle.load(f)
        self.policy.args_obj.optimize = optimize

