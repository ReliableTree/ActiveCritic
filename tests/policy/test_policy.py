import unittest

import torch as th
import numpy as np

from active_critic.model_src.whole_sequence_model import WholeSequenceModel, OptimizeMaximumCritic, OptimizeEndCritic
from active_critic.policy.active_critic_policy import ActiveCriticPolicy, ACPOptEnd
from active_critic.utils.test_utils import (make_acps, make_obs_act_space,
                                            make_wsm_setup)
from active_critic.utils.gym_utils import (DummyExtractor, make_dummy_vec_env,
                                           new_epoch_pap,
                                           new_epoch_reach)

from active_critic.utils.gym_utils import make_policy_dict, new_epoch_reach, make_dummy_vec_env, sample_expert_transitions, parse_sampled_transitions

class TestPolicy(unittest.TestCase):

    def setup_ac(self):
        seq_len = 50
        d_output = 2
        obs_dim = 3
        batch_size = 2
        wsm_actor_setup = make_wsm_setup(
            seq_len=seq_len, d_output=d_output)
        wsm_critic_setup = make_wsm_setup(
            seq_len=seq_len, d_output=1)
        acps = make_acps(
            seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_pap)
        obs_space, acts_space = make_obs_act_space(
            obs_dim=obs_dim, action_dim=d_output)
        actor = WholeSequenceModel(wsm_actor_setup)
        critic = WholeSequenceModel(wsm_critic_setup)
        ac = ActiveCriticPolicy(observation_space=obs_space, action_space=acts_space,
                                actor=actor, critic=critic, acps=acps)
        return ac, acps, d_output, obs_dim, batch_size

    def setup_ac_reach(self):
        seq_len = 50
        env, gt_policy = make_dummy_vec_env('reach', seq_len=seq_len)
        d_output = env.action_space.shape[0]
        wsm_actor_setup = make_wsm_setup(
            seq_len=seq_len, d_output=d_output)
        wsm_critic_setup = make_wsm_setup(
            seq_len=seq_len, d_output=1)
        acps = make_acps(
            seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach)
        actor = WholeSequenceModel(wsm_actor_setup)
        critic = WholeSequenceModel(wsm_critic_setup)
        ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                                actor=actor, critic=critic, acps=acps)
        return ac, acps, env, gt_policy

    def setup_opt_end(self):
        seq_len = 50
        d_output = 2
        obs_dim = 3
        batch_size = 2
        wsm_actor_setup = make_wsm_setup(
            seq_len=seq_len, d_output=d_output)
        wsm_critic_setup = make_wsm_setup(
            seq_len=seq_len, d_output=1)
        acps = make_acps(
            seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_pap)
        acps.opt_steps = 2
        obs_space, acts_space = make_obs_act_space(
            obs_dim=obs_dim, action_dim=d_output)
        actor = WholeSequenceModel(wsm_actor_setup)
        critic = OptimizeEndCritic(wsms=wsm_critic_setup)
        ac = ACPOptEnd(observation_space=obs_space, action_space=acts_space,
                                actor=actor, critic=critic, acps=acps)
        return ac, acps, d_output, obs_dim, batch_size

    def setup_opt_max(self):
        seq_len = 50
        d_output = 2
        obs_dim = 3
        batch_size = 2
        wsm_actor_setup = make_wsm_setup(
            seq_len=seq_len, d_output=d_output)
        wsm_critic_setup = make_wsm_setup(
            seq_len=seq_len, d_output=1)
        acps = make_acps(
            seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_pap)
        obs_space, acts_space = make_obs_act_space(
            obs_dim=obs_dim, action_dim=d_output)
        actor = WholeSequenceModel(wsm_actor_setup)
        critic = OptimizeMaximumCritic(wsms=wsm_critic_setup)
        ac = ActiveCriticPolicy(observation_space=obs_space, action_space=acts_space,
                                actor=actor, critic=critic, acps=acps)
        return ac, acps, d_output, obs_dim, batch_size


    def test_policy_output_shape(self):
        ac, acps, d_output, obs_dim, batch_size = self.setup_ac()

        obs = th.ones([batch_size, obs_dim], dtype=th.float, device='cuda')
        action = ac.predict(obs)
        expected_shape = [batch_size, d_output]
        self.assertTrue(list(action.shape) == expected_shape,
                        'AC output wrong shape for first observation, no action.')

    def test_policy_opt_step(self):
        th.manual_seed(0)
        ac, acps, act_dim, obs_dim, batch_size = self.setup_ac()

        current_step = 1
        org_actions = th.ones([batch_size, acps.epoch_len, act_dim],
                              device=acps.device, dtype=th.float, requires_grad=False)
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                               device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                              device=acps.device, dtype=th.float, requires_grad=False)
        org_obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                  device=acps.device, dtype=th.float, requires_grad=False)
        optimizer = th.optim.Adam([opt_actions], lr=1e-1)
        goal_label = th.ones([batch_size, acps.epoch_len, 1],
                             device=acps.device, dtype=th.float)

        actions, critic_result = ac.inference_opt_step(
            org_actions=org_actions, opt_actions=opt_actions, obs_seq=obs_seq, optimizer=optimizer,
            goal_label=goal_label, current_step=current_step)

        last_critic_result = th.clone(critic_result.detach())

        for i in range(1):
            actions, critic_result = ac.inference_opt_step(
                org_actions=org_actions, opt_actions=opt_actions, obs_seq=obs_seq, optimizer=optimizer,
                goal_label=goal_label, current_step=current_step)
            self.assertTrue(th.equal(
                org_actions[:, :current_step], actions[:, :current_step]), 'org_actions were overwritten')
            self.assertFalse(th.equal(
                opt_actions[:, current_step:], org_actions[:, current_step:]), 'opt actions did not change')
            self.assertTrue(
                th.all(critic_result.mean(dim=[1, 2]) > last_critic_result.mean(dim=[1, 2])), 'optimisation does not work.')
            last_critic_result = th.clone(critic_result.detach())

    def test_seq_optimizer(self):
        th.manual_seed(0)

        current_step = 1
        ac, acps, act_dim, obs_dim, batch_size = self.setup_ac()

        org_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                               device=acps.device, dtype=th.float, requires_grad=False)
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                               device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                              device=acps.device, dtype=th.float, requires_grad=False)
        org_obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                  device=acps.device, dtype=th.float, requires_grad=False)

        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=False)
        self.assertTrue(th.equal(
            org_actions[:, :current_step], actions[:, :current_step]), 'org_actions were overwritten')

        self.assertFalse(th.equal(actions[:, current_step:], org_actions[:,
                                                                         current_step:]),
                         'seq optimisation did not change the actions')
        self.assertTrue(th.all(
            expected_success[:, -1] >= ac.args_obj.optimisation_threshold), 'optimisation does not work.')
        self.assertTrue(th.equal(obs_seq, org_obs_seq),
                        'Observations were changed.')

    def test_prediction(self):
        th.manual_seed(0)
        ac, acps, env, _ = self.setup_ac_reach()
        obsv = env.reset()
        last_obsv = th.tensor(obsv)
        all_taken_actions = []
        all_observations = [obsv]
        for i in range(50):
            action = ac.predict(obsv)
            all_taken_actions.append(action)
            obsv, rew, dones, info = env.step(action)
            all_observations.append(obsv)
            self.assertTrue(len(th.nonzero(
                ac.obs_seq[:, ac.current_step + 1:])) == 0, 'unobserved seq is not 0')
            if new_epoch_reach(last_obsv, th.tensor(obsv)):
                self.assertTrue(
                    ac.current_step == ac.args_obj.epoch_len - 1, 'Steps are counted wrong')
                ata = th.tensor(all_taken_actions).transpose(0, 1)
                self.assertTrue(th.equal(ata.to('cuda'), ac.current_result.gen_trj),
                                'Actual action sequence differs from saved action sequence. Maybe problem with backprop?')
                aob = th.tensor(all_observations).transpose(0, 1)[:, :50]
                self.assertTrue(th.equal(aob.to('cuda'), ac.obs_seq),
                                'Observation sequence was overridden')

                self.assertTrue(ac.current_result.expected_succes_before.mean() < ac.current_result.expected_succes_after.mean(),
                                'In inference optimisation went wrong.')

    def test_score_history(self):
        th.manual_seed(0)
        ac, acps, env, _ = self.setup_ac_reach()
        obsv = env.reset()
        all_taken_actions = []
        all_observations = [obsv]
        all_scores_after = []
        all_scores_before = []
        ac.reset()
        epsiodes = 2

        for i in range(epsiodes*ac.args_obj.epoch_len):
            action = ac.predict(obsv)
            all_taken_actions.append(action)
            obsv, rew, dones, info = env.step(action)
            all_observations.append(obsv)
            all_scores_after.append(ac.current_result.expected_succes_after)
            all_scores_before.append(ac.current_result.expected_succes_before)
            self.assertTrue(len(th.nonzero(ac.obs_seq[:, ac.current_step+1:])) == 0, 'Unobserved observations are set in ac.obs_seq.')
            if (i+1) % 5 == 0:
                all_observations = [obsv]

        all_scores_after_th = th.tensor(np.array([s.detach().cpu().numpy() for s in all_scores_after]).reshape(
            [epsiodes, ac.args_obj.epoch_len, ac.args_obj.epoch_len, 1]), device=ac.args_obj.device)
        all_scores_before_th = th.tensor(np.array([s.detach().cpu().numpy() for s in all_scores_before]).reshape(
            [epsiodes, ac.args_obj.epoch_len, ac.args_obj.epoch_len, 1]), device=ac.args_obj.device)

        ac, acps, env, _ = self.setup_ac_reach()
        ac.args_obj.optimize = False
        obsv = env.reset()
        all_taken_actions = []
        all_observations = [obsv]
        all_scores_after = []
        all_scores_before = []
        ac.reset()
        epsiodes = 2

        for i in range(epsiodes*ac.args_obj.epoch_len):
            action = ac.predict(obsv)
            all_taken_actions.append(action)
            obsv, rew, dones, info = env.step(action)


        all_actions_th = th.tensor(np.array(all_taken_actions).reshape(
            [epsiodes, ac.args_obj.epoch_len, 4]), device=ac.args_obj.device)
        
        for i in range(epsiodes):
            for j in range(all_scores_after_th.shape[1]):
                self.assertTrue(th.equal(
                    all_actions_th[i, j], ac.history.gen_trj[0][i, j]), 'Trajectory history of ac is corrupted.')
                

        for i in range(epsiodes*ac.args_obj.epoch_len):
            action = ac.predict(obsv)
            all_taken_actions.append(action)
            obsv, rew, dones, info = env.step(action)
            all_observations.append(obsv)
            all_scores_after.append(ac.current_result.expected_succes_after)
            self.assertTrue(len(th.nonzero(ac.obs_seq[:, ac.current_step+1:])) == 0)
            if (i+1) % 5 == 0:
                all_taken_actions = []
                all_observations = [obsv]

        self.assertTrue(
            ac.history.opt_scores[0].shape[0] == 2*epsiodes, 'Scores after are not properly appended.')
        self.assertTrue(
            ac.history.gen_scores[0].shape[0] == 2*epsiodes, 'Scores before are not properly appended.')

        all_taken_actions = []
        all_observations = [obsv]
        all_scores_after = []
        ac.reset()
        epsiodes = 2
        for i in range(epsiodes*ac.args_obj.epoch_len):
            action = ac.predict(obsv)
            all_taken_actions.append(action)
            obsv, rew, dones, info = env.step(action)
            all_observations.append(obsv)
            all_scores_after.append(ac.current_result.expected_succes_after)
            self.assertTrue(len(th.nonzero(ac.obs_seq[:, ac.current_step+1:])) == 0)
            if (i+1) % 5 == 0:
                all_taken_actions = []
                all_observations = [obsv]
        self.assertTrue(
            ac.history.opt_scores[0].shape[0] == 2, 'Epochs reset did not work.')
        self.assertTrue(
            ac.history.gen_scores[0].shape[0] == 2, 'Epochs reset did not work.')

    def test_early_stopping(self):
        th.manual_seed(1)
        current_step = 1

        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        ac.args_obj.clip = False
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        actions, expected_success_nonstop = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=False)

        th.manual_seed(1)
        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        ac.args_obj.clip = False
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=True)
        self.assertTrue((expected_success[:,-1] < expected_success_nonstop[:,-1]).sum() > 0)
        self.assertTrue((expected_success[:,-1] < expected_success_nonstop[:,-1]).sum() < len(expected_success))

    def test_opt_end(self):
        th.manual_seed(1)
        current_step = 1

        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        actions, expected_success_nonstop = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=False, opt_end=False, opt_last=False)

        th.manual_seed(1)
        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=False, opt_end=True, opt_last=False)

        self.assertTrue((max(expected_success[:,-1]) > max(expected_success_nonstop[:,-1])))

    def test_clip(self):
        th.manual_seed(1)
        current_step = 1
        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=False)

        maxim, _ = th.max(actions, dim=1)
        minim, _ = th.min(actions, dim=1)
        self.assertTrue(th.all(maxim <= th.tensor(ac.action_space.high, device=maxim.device)))
        self.assertTrue(th.all(minim >= th.tensor(ac.action_space.low, device=maxim.device)))


    def test_opt_end(self):
        th.manual_seed(1)
        current_step = 1

        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        actions, expected_success_end = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=False, opt_end=True, opt_last=False)

        th.manual_seed(1)
        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=False, opt_end=False, opt_last=True)
        self.assertTrue(th.all(expected_success[:,-1] > expected_success_end[:,-1]))

    def test_opt_max(self):
        th.manual_seed(3)
        current_step = 1

        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        ac.args_obj.opt_steps = 2

        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        self.assertTrue(th.all(critic_scores < 1))
        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=True)

        th.manual_seed(3)
        current_step = 1

        ac, acps, act_dim, obs_dim, batch_size =  self.setup_opt_max()
        ac.args_obj.opt_steps = 2

        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        self.assertTrue(th.all(critic_scores < 1))

        actions, expected_success_max = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=True)

        max_n, _ = expected_success.max(dim=1)
        max_m, _ = expected_success_max.max(dim=1)
        self.assertTrue(th.all(max_m > max_n))

    def test_opt_end(self):
        th.manual_seed(3)
        current_step = 30

        ac, acps, act_dim, obs_dim, batch_size =  self.setup_ac()
        ac.args_obj.opt_steps = 2

        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        self.assertTrue(th.all(critic_scores < 1))
        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=True)
        end_wo = expected_success[:,current_step:].mean()
        all_wo = expected_success.mean()

        th.manual_seed(3)
        current_step = 30

        ac, acps, act_dim, obs_dim, batch_size =  self.setup_opt_end()
        ac.args_obj.opt_steps = 2

        opt_actions = th.zeros([batch_size, acps.epoch_len, act_dim],
                                device=acps.device, dtype=th.float, requires_grad=True)
        obs_seq = 2 * th.ones([batch_size, current_step + 1, obs_dim],
                                device=acps.device, dtype=th.float, requires_grad=False)
        obs_seq[0] *= 2
        critic_input = ac.get_critic_input(acts=opt_actions, obs_seq=obs_seq)
        critic_scores = ac.critic.forward(critic_input)
        self.assertTrue(th.all(critic_scores < 1))
        actions, expected_success = ac.optimize_act_sequence(
            actions=opt_actions, observations=obs_seq, current_step=current_step, stop_opt=True)

        end_w = expected_success[:,current_step:].mean()
        all_w = expected_success.mean()

        self.assertTrue(th.all(end_w > end_wo))
        self.assertTrue(th.all(all_w < all_wo))

if __name__ == '__main__':
    unittest.main()
    #to = TestPolicy()
    #to.test_opt_max()