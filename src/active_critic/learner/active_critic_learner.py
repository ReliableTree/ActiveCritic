from email import policy
from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from cv2 import MSER
from gym.envs.mujoco import MujocoEnv
from matplotlib.pyplot import polar
from sklearn.utils import resample
import torch.nn as nn
import torch as th
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.tboard_graphs import TBoardGraphs
from active_critic.utils.dataset import DatasetAC
from torch.utils.data.dataloader import DataLoader
from active_critic.utils.gym_utils import sample_new_episode
from active_critic.utils.pytorch_utils import make_part_obs_data, calcMSE
import tensorflow as tf

class ACLScores:
    def __init__(self) -> None:
        self.mean_actor = [float('inf')]
        self.mean_critic = [float('inf')]

    def update_min_score(self, old_score, new_score):
        if old_score[0] > new_score:
            old_score[0] = new_score

class ActiveCriticLearner(nn.Module):
    def __init__(self,
                 ac_policy:ActiveCriticPolicy,
                 env: MujocoEnv,
                 network_args_obj: ActiveCriticLearnerArgs = None
                 ):
        super().__init__()
        self.network_args = network_args_obj
        self.env = env
        self.policy = ac_policy
        self.extractor = network_args_obj.extractor
        self.logname = network_args_obj.logname

        self.scores = ACLScores()

        if network_args_obj.tboard:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
            self.tboard = TBoardGraphs(
                self.logname, data_path=network_args_obj.data_path)
        self.global_step = 0

        self.train_data = DatasetAC()
        self.train_data.onyl_positiv = False

    def setDatasets(self, train_data: DatasetAC):
        self.train_data = train_data
        if len(train_data) > 0:
            self.train_loader = DataLoader(
                dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)

    def add_data(self, actions:th.Tensor, observations:th.Tensor, rewards:th.Tensor):
        acts, obsv, rews = make_part_obs_data(actions=actions, observations=observations, rewards=rewards)
        self.train_data.add_data(obsv=obsv, actions=acts, reward=rews)
        self.train_loader = DataLoader(
                dataset=self.train_data, batch_size=self.network_args.batch_size, shuffle=True)


    def actor_step(self, data, loss_actor):
        obsv, actions, reward = data
        actor_input = self.policy.get_actor_input(obs=obsv, actions=actions, rew=reward)
        debug_dict = self.policy.actor.optimizer_step(inputs=actor_input, label=actions)
        if loss_actor is None:
            loss_actor = debug_dict['Loss '].unsqueeze(0)
        else:
            loss_actor = th.cat((loss_actor, debug_dict['Loss '].unsqueeze(0)), dim=0)
        return loss_actor

    def critic_step(self, data, loss_critic):
        obsv, actions, reward = data
        critic_inpt = self.policy.get_critic_input(acts=actions, obs_seq=obsv)
        debug_dict = self.policy.critic.optimizer_step(inputs=critic_inpt, label=reward)
        if loss_critic is None:
            loss_critic = debug_dict['Loss '].unsqueeze(0)
        else:
            loss_critic = th.cat((loss_critic, debug_dict['Loss '].unsqueeze(0)), dim=0)
        return loss_critic

    def add_training_data(self):
        actions, observations, rewards, _, _ = sample_new_episode(
            policy=self.policy,
            env=self.env,
            episodes=1)
        self.add_data(
            actions=actions,
            observations=observations,
            rewards=rewards
        )


    def train(self, epochs):
        for epoch in range(epochs):
            if not self.network_args.imitation_phase:
                self.add_training_data()

            self.policy.train()                
            loss_actor = None
            loss_critic = None
            for data in self.train_loader:
                loss_actor = self.actor_step(data, loss_actor)                
                loss_critic = self.critic_step(data, loss_critic)                
                self.global_step += 1

            mean_actor = loss_actor.mean()
            mean_critic = loss_critic.mean()
            
            self.scores.update_min_score(self.scores.mean_critic, mean_critic)
            self.scores.update_min_score(self.scores.mean_actor, mean_actor)
            
            debug_dict = {
                'Loss Actor': mean_actor,
                'Loss Critic': mean_critic
            }
            self.write_tboard_scalar(debug_dict=debug_dict, train=True)
            if (epoch+1)%self.network_args.val_every == 0:
                self.run_validation() 


    def write_tboard_scalar(self, debug_dict, train, step=None):
        if step is None:
            step = self.global_step
        if self.network_args.tboard:
            for para, value in debug_dict.items():
                value = value.to('cpu')
                if train:
                    self.tboard.addTrainScalar(para, value, step)
                else:
                    self.tboard.addValidationScalar(para, value, step)

    def run_validation(self):
        actions, observations, rewards, expected_rewards_before, expected_rewards_after = sample_new_episode(
            policy=self.policy,
            env=self.env,
            episodes=self.network_args.validation_episodes)

        last_reward = rewards[:,-1]
        last_expected_rewards_before = expected_rewards_before[:, -1]
        last_expected_reward_after = expected_rewards_after[:, -1]
        self.analyze_critic_scores(last_reward, last_expected_rewards_before, '')
        self.analyze_critic_scores(last_reward, last_expected_reward_after, ' optimized')
        success = (last_reward == 1)
        success = success.type(th.float)
        debug_dict = {
            'Success Rate' : success.mean(),
            'Reward': last_reward.mean()
        }
        self.write_tboard_scalar(debug_dict=debug_dict, train=False)



    def torch2tf(self, inpt):
        if inpt is not None:
            return tf.convert_to_tensor(inpt.detach().cpu().numpy())
        else:
            return None

    def tf2torch(self, inpt):
        if inpt is not None:
            return th.tensor(inpt.numpy(), device=self.device)
        else:
            return None

    def analyze_critic_scores(self, reward:th.Tensor, expected_reward:th.Tensor, add:str):
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
        debug_dict['critic success' + add] = (expected_success == success).type(th.float).mean()
        debug_dict['critic expected success' + add] = expected_success.type(th.float).mean()
        debug_dict['critic L2 error reward' + add] = calcMSE(reward, expected_reward)

        self.write_tboard_scalar(debug_dict=debug_dict, train=False)

    def runValidation(self, complete=False):
        self.policy.eval()
        if complete:
            num_envs = self.network_args.eval_epochs
            print("Running full validation...")

        else:
            num_envs = self.network_args.quick_eval_epochs

        self.policy.return_mode = 0

        actions, observations, success, expected_rewards = self.sample_new_episode(
            episodes=num_envs, add_data=False)
        data_gen = (actions, observations, success.type(torch.bool))
        mean_success = success.mean()
        fail = ~success.type(torch.bool)
        print(f'mean success before: {mean_success}')
        debug_dict = {'success rate generated': mean_success}
        self.write_tboard_scalar(debug_dict=debug_dict, train=False)

        self.analyze_critic_scores(success, expected_rewards)

        if self.add_data:
            self.policy.return_mode = 1
            actions_opt, observations_opt, success_opt, expected_rewards_opt = self.sample_new_episode(
                episodes=num_envs, add_data=False)
            self.analyze_critic_scores(success_opt, expected_rewards_opt)

        fail_opt = ~success_opt.type(torch.bool)

        self.write_tboard_scalar(
            {'num optimisation steps': torch.tensor(self.policy.max_steps)}, train=False)

        mean_success_opt = (success_opt.type(torch.float)).mean()
        print(f'mean success after: {mean_success_opt}')
        debug_dict = {}
        debug_dict['success rate optimized'] = mean_success_opt
        debug_dict['improved success rate'] = mean_success_opt - \
            mean_success
        if mean_success_opt - mean_success > self.best_improved_success:
            self.best_improved_success = mean_success_opt - mean_success
            self.saveNetworkToFile(
                add=self.logname + "/best_improved/", data_path=self.data_path)

        num_improved = (success_opt * fail).type(torch.float).mean()
        num_deproved = (success * fail_opt).type(torch.float).mean()
        debug_dict['rel number improved'] = num_improved
        debug_dict['rel number failed'] = num_deproved

        self.write_tboard_scalar(
            debug_dict=debug_dict, train=not complete, step=self.global_step)

        if self.use_tboard:
            success = success.type(torch.bool)
            success_opt = success_opt.type(torch.bool)
            self.plot_with_mask(
                label=actions, trj=actions, inpt=observations, mask=success, name='success')

            fail = ~success
            fail_opt = ~success_opt
            self.plot_with_mask(label=actions, trj=actions,
                                inpt=observations, mask=fail, name='fail')

            fail_to_success = success_opt & fail
            self.plot_with_mask(label=actions, trj=actions, inpt=observations,
                                mask=fail_to_success, name='fail to success', opt_trj=actions_opt)

            success_to_fail = success & fail_opt
            self.plot_with_mask(label=actions, trj=actions, inpt=observations,
                                mask=success_to_fail, name='success to fail', opt_trj=actions_opt)

            fail_to_fail = fail & fail_opt
            self.plot_with_mask(label=actions, trj=actions, inpt=observations,
                                mask=fail_to_fail, name='fail to fail', opt_trj=actions_opt)

        if self.use_tboard:
            self.saveNetworkToFile(
                add=self.logname + "/last/", data_path=self.data_path)




    def plot_with_mask(self, label, trj, inpt, mask, name, opt_trj=None):
        if mask.sum() > 0:
            label = label[mask][0]
            trj = trj[mask][0]
            inpt = inpt[mask][0, 0]
            if opt_trj is not None:
                opt_trj = opt_trj[mask][0]
            self.createGraphsMW(d_in=1, d_out=label, result=trj, toy=False,
                                inpt=inpt, name=name, opt_trj=opt_trj, window=0)

    def loadingBar(self, count, total, size, addition="", end=False):
        if total == 0:
            percent = 0
        else:
            percent = float(count) / float(total)
        full = int(percent * size)
        fill = size - full
        print("\r  {:5d}/{:5d} [".format(count, total) +
              "#" * full + " " * fill + "] " + addition, end="")
        if end:
            print("")
        sys.stdout.flush()

    def createGraphsMW(self, d_in, d_out, result, save=False, name_plot='', epoch=0, toy=True, inpt=None, name='Trajectory', opt_trj=None, window=0):
        target_trj = d_out
        gen_trj = result

        path_to_plots = self.data_path + "/plots/" + \
            str(self.logname) + '/' + str(epoch) + '/'

        tol_neg = None
        tol_pos = None
        self.tboard.plotDMPTrajectory(target_trj, gen_trj, torch.zeros_like(gen_trj),
                                      None, None, None, stepid=self.global_step, save=save, name_plot=name_plot, path=path_to_plots,
                                      tol_neg=tol_neg, tol_pos=tol_pos, inpt=inpt, name=name, opt_gen_trj=opt_trj, window=window)

    def saveNetworkToFile(self, add, data_path):
        import os
        import pickle

        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not path.exists(path_to_file):
            makedirs(path_to_file)

        torch.save(self.state_dict(), path_to_file + "policy_network")
        torch.save(self.tailor_modules[0].model.state_dict(
        ), path_to_file + "tailor_network")
        torch.save(self.optimizer.state_dict(), path_to_file + "optimizer")
        torch.save(self.tailor_modules[0].meta_optimizer.state_dict(
        ), path_to_file + "tailor_optimizer")
        torch.save(torch.tensor(self.global_step),
                   path_to_file + "global_step")

        with open(path_to_file + 'model_setup.pkl', 'wb') as f:
            pickle.dump(self.network_args, f)

        torch.save(self.train_loader, path_to_file+'train')

    def loadNetworkFromFile(self, path, device='cuda'):
        self.load_state_dict(torch.load(
            path + "policy_network", map_location=device))
        self.tailor_modules[0].model.load_state_dict(
            torch.load(path + "tailor_network", map_location=device))
        self.optimizer.load_state_dict(torch.load(path + "optimizer"))
        self.tailor_modules[0].meta_optimizer.load_state_dict(
            torch.load(path + "tailor_optimizer", map_location=device))

        self.global_step = int(torch.load(path+'global_step'))

        self.train_loader = torch.load(path+'train')
