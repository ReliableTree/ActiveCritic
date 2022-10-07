import imp
from gym.envs.mujoco import MujocoEnv
import torch.nn as nn
import torch as th
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.tboard_graphs import TBoardGraphs
from active_critic.utils.dataset import DatasetAC
from torch.utils.data.dataloader import DataLoader

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

        self.total_steps = 0
        self.max_success_rate = 0
        self.extractor = network_args_obj.extractor

        self.logname = network_args_obj.logname

        if network_args_obj.tboard:
            self.tboard = TBoardGraphs(
                self.logname, data_path=network_args_obj.data_path)

        self.imitation_phase = not network_args_obj.imitation_phase

        self.train_data = DatasetAC(device=network_args_obj.device)

        self.train_loader = None
        self.val_loader = None

    def setDatasets(self, train_data: DatasetAC, batch_size=32):
        if len(train_data) > 0:
            self.train_data = train_data
            self.train_loader = DataLoader(
                dataset=train_data, batch_size=batch_size, shuffle=True)

    def train(self, epochs):
        self.expert_examples_len = len(self.train_data)
        print(f'inital num examples: {self.expert_examples_len}')
        model_step = 0
        for epoch in range(epochs):
            self.policy.eval()
            if self.add_data:
                self.sample_new_episode(episodes=1, add_data=True)
            self.policy.train()
            while model_step < self.network_args.n_steps:
                model_step += len(self.train_data)
                self.policy.train()

                if not self.init_train:
                    self.train_data.onyl_positiv = False
                    for data in self.train_loader:
                        debug_dict = self.policy.critic.optimizer_step(data)
                        self.write_tboard_scalar(
                            debug_dict=debug_dict, train=True)
                        self.global_step += 1

                if self.train_data.success.sum() > 0:
                    self.train_data.onyl_positiv = True
                    for data in self.train_loader:
                        debug_dict = self.policy.actor.optimizer_step(data)
                        self.write_tboard_scalar(
                            debug_dict=debug_dict, train=True)
                        self.global_step += 1

            model_step = 0
            self.num_vals += 1
            complete = (self.num_vals % self.network_args.complete_modulo == 0)
            print(f'logname: {self.logname}')
            self.runValidation(complete=complete)

    def sample_new_episode(self, episodes=1, add_data=True):
        self.policy.eval()
        self.policy.reset_state()
        transitions = sample_expert_transitions(
            self.policy, self.env, episodes)
        expected_rewards = torch.tensor(*self.policy.scores)
        datas = parse_sampled_transitions(
            transitions=transitions, new_epoch=self.network_args.new_epoch, extractor=self.extractor)
        device_data = []
        for data in datas:
            device_data.append(data.to(self.network_args.device))
        actions, observations, rewards = device_data
        if add_data:
            self.add_data_to_loader(inpt_obs_opt=observations, trajectories_opt=actions,
                                    success_opt=rewards, ftrjs_opt=actions, episodes=episodes)
        return actions, observations, rewards, expected_rewards

    def torch2tf(self, inpt):
        if inpt is not None:
            return tf.convert_to_tensor(inpt.detach().cpu().numpy())
        else:
            return None

    def tf2torch(self, inpt):
        if inpt is not None:
            return torch.tensor(inpt.numpy(), device=self.device)
        else:
            return None

    def analyze_critic_scores(self, success, expected_success):
        success.reshape(-1)
        fail = ~success
        expected_success = expected_success.reshape(-1)
        expected_fail = ~ expected_success

        expected_success_float = expected_success.type(torch.float)
        expected_fail_float = expected_fail.type(torch.float)
        fail_float = fail.type(torch.float).reshape(-1)
        success_float = success.type(torch.float).reshape(-1)

        tp = (expected_success_float * success_float)[success == 1].mean()
        if success_float.sum() == 0:
            tp = torch.tensor(0)
        fp = (expected_success_float * fail_float)[fail == 1].mean()
        tn = (expected_fail_float * fail_float)[fail == 1].mean()
        fn = (expected_fail_float * success_float)[success == 1].mean()

        if self.policy.return_mode == 0:
            add = ' '
        elif self.policy.return_mode == 1:
            add = ' optimized '
        elif self.policy.return_mode == 2:
            add = ' label '

        debug_dict = {}

        debug_dict['true positive' + add] = tp
        debug_dict['false positive' + add] = fp
        debug_dict['true negative' + add] = tn
        debug_dict['false negative' + add] = fn
        debug_dict['tailor success' +
                   add] = (expected_success == success).type(torch.float).mean()
        debug_dict['tailor expected success' +
                   add] = (expected_success).type(torch.float).mean()

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

    def add_data_to_loader(self, inpt_obs_opt, trajectories_opt, success_opt, ftrjs_opt, episodes):
        if self.add_data:

            print(f'inpt_obs_opt: {inpt_obs_opt.shape}')
            print(f'trajectories_opt: {trajectories_opt.shape}')
            print(f'success_opt: {success_opt}')
            print(f'ftrjs_opt: {ftrjs_opt.shape}')
            print(f'episodes: {episodes}')

            inpt_obs_opt = inpt_obs_opt.to(self.network_args.device)
            trajectories_opt = trajectories_opt.to(self.network_args.device)
            success_opt = success_opt.to(self.network_args.device)
            ftrjs_opt = ftrjs_opt.to(self.network_args.device)

            success_opt = success_opt.type(torch.bool)
            self.train_data.add_data(
                obsv=inpt_obs_opt[:episodes], actions=trajectories_opt[:episodes], success=success_opt)

            self.write_tboard_scalar(
                {'num examples': torch.tensor(len(self.train_data))}, train=False)
            self.init_train = ~self.train_data.success.sum() == 0
            self.train_loader = DataLoader(
                self.train_data, batch_size=32, shuffle=True)
            self.val_loader = DataLoader(
                self.val_data, batch_size=32, shuffle=True)
        print(f'num examples: {len(self.train_data)}')
        print(f'num demonstrations: {self.train_data.success.sum()}')

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
