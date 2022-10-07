import unittest

import torch as th

from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.test_utils import (make_acps, make_obs_act_space,
                                    make_wsm_setup)
from active_critic.utils.gym_utils import (DummyExtractor, make_dummy_vec_env,
                             new_epoch_pap,
                             new_epoch_reach)


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
        d_result = 1
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
        return ac, acps, env

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
        goal_label = th.ones([batch_size, acps.epoch_len, 1], device=acps.device, dtype=th.float)

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
                th.all(critic_result.mean(dim=[1,2]) > last_critic_result.mean(dim=[1,2])), 'optimisation does not work.')
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
            actions=opt_actions, observations=obs_seq, current_step=current_step)
        self.assertTrue(th.equal(
            org_actions[:, :current_step], actions[:, :current_step]), 'org_actions were overwritten')

        self.assertFalse(th.equal(actions[:, current_step:], org_actions[:,
                                                             current_step:]),
                         'seq optimisation did not change the actions')
        self.assertTrue(th.all(
            expected_success[:,-1] >= ac.args_obj.optimisation_threshold), 'optimisation does not work.')
        self.assertTrue(th.equal(obs_seq, org_obs_seq),
                        'Observations were changed.')

    def test_prediction(self):
        th.manual_seed(0)
        ac, acps, env = self.setup_ac_reach()
        obsv = env.reset()
        last_obsv = th.tensor(obsv)
        all_taken_actions = []
        all_observations = [obsv]
        for i in range(50):
            action = ac.predict(obsv)
            all_taken_actions.append(action)
            obsv, rew, dones, info = env.step(action)
            all_observations.append(obsv)
            self.assertTrue(len(th.nonzero(ac.obs_seq[:, ac.current_step + 1:])) == 0, 'unobserved seq is not 0')
            if new_epoch_reach(last_obsv, th.tensor(obsv)):
                self.assertTrue(ac.current_step == ac.args_obj.epoch_len - 1, 'Steps are counted wrong')
                ata = th.tensor(all_taken_actions).transpose(0, 1)
                self.assertTrue(th.equal(ata.to('cuda'), ac.current_result.gen_trj),
                                'Actual action sequence differs from saved action sequence. Maybe problem with backprop?')
                aob = th.tensor(all_observations).transpose(0, 1)[:, :50]
                self.assertTrue(th.equal(aob.to('cuda'), ac.obs_seq), 'Observation sequence was overridden')

                self.assertTrue(ac.current_result.expected_succes_before.mean() < ac.current_result.expected_succes_after.mean(),
                                'In inference optimisation went wrong.')


if __name__ == '__main__':
    unittest.main()
