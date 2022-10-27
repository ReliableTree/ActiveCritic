import unittest

import torch as th
import numpy as np

from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from active_critic.model_src.state_model import StateModel, StateModelArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.test_utils import (make_acps, make_obs_act_space,
                                            make_wsm_setup)
from active_critic.utils.gym_utils import (DummyExtractor, make_dummy_vec_env,
                                           new_epoch_pap,
                                           new_epoch_reach)

from active_critic.utils.gym_utils import make_policy_dict, new_epoch_reach, make_dummy_vec_env, sample_expert_transitions, parse_sampled_transitions
from active_critic.utils.pytorch_utils import build_tf_horizon_mask

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


    def setup_opt_state(device='cpu'):
        seq_len = 6
        action_dim = 2
        obs_dim = 3
        batch_size = 2
        embed_dim = 4
        lr = 1e-3

        actor_args = StateModelArgs()
        actor_args.arch = [20, action_dim]
        actor_args.device = device
        actor_args.lr = lr
        actor = StateModel(args=actor_args)

        critic_args = StateModelArgs()
        critic_args.arch = [10, 1]
        critic_args.device = device
        critic_args.lr = lr
        critic = StateModel(args=critic_args)

        emitter_args = StateModelArgs()
        emitter_args.arch = [20, embed_dim]
        emitter_args.device = device
        emitter_args.lr = lr
        emitter = StateModel(args=emitter_args)

        predictor_args = make_wsm_setup(
            seq_len=seq_len, d_output=embed_dim, device=device)
        predictor_args.model_setup.d_hid = 200
        predictor_args.model_setup.d_model = 200
        predictor_args.model_setup.nlayers = 1
        predictor = WholeSequenceModel(args=predictor_args)


        acps = make_acps(
            seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_pap, device=device)
        acps.opt_steps = 2
        obs_space, acts_space = make_obs_act_space(
            obs_dim=obs_dim, action_dim=action_dim)
        ac = ActiveCriticPolicy(observation_space=obs_space, 
                                action_space=acts_space,
                                actor=actor,
                                critic=critic,
                                predictor=predictor,
                                emitter=emitter,
                                acps=acps)
        return ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len


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

    def test_grad_whole_sequence_one_step(self):
        th.manual_seed(0)
        device = 'cuda'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = self.setup_opt_state(device=device)
        horizon = 0
        embeddings = th.ones([batch_size, 1, embed_dim], device=device)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        goal_embeddings = th.ones_like(embeddings)
        action_optim = th.optim.Adam([actions], lr=1e-1)
        opt_paras = action_optim.state_dict()

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings = ac.build_sequence(embeddings=embeddings, actions=actions, seq_len=seq_len, mask=mask, detach=True)
        opt_actions = actions.detach()
        opt_actions.requires_grad = True
        next_embedding = ac.predict_step(embeddings=seq_embeddings.detach(), actions=opt_actions, mask=mask)
        seq_embedding = th.cat((embeddings[:,:1], next_embedding[:,:-1]), dim=1)
        loss = ((seq_embedding - th.ones_like(seq_embedding))**2).mean()
        loss.backward()
        assert (opt_actions.grad != 0).sum() == opt_actions[:,:-1].numel()

    def test_grad_var_horizon(self):
        th.manual_seed(0)
        device = 'cuda'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = self.setup_opt_state(device=device)
        horizon = 2
        embeddings = th.ones([batch_size, 1, embed_dim], device=device)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        goal_embeddings = th.ones_like(embeddings)
        action_optim = th.optim.Adam([actions], lr=1e-1)
        opt_paras = action_optim.state_dict()

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings = ac.build_sequence(embeddings=embeddings, actions=actions, seq_len=seq_len, mask=mask, detach=True)
        opt_actions = actions.detach()
        opt_actions.requires_grad = True
        next_embedding = ac.predict_step(embeddings=seq_embeddings.detach(), actions=opt_actions, mask=mask)
        seq_embedding = th.cat((embeddings[:,:1], next_embedding[:,:-1]), dim=1)
        loss = ((seq_embedding[:,-1] - th.ones_like(seq_embedding[:,-1] ))**2).mean()
        loss.backward()
        assert (opt_actions.grad != 0).sum() == (min(seq_len-1, 1+horizon) * action_dim*batch_size)


if __name__ == '__main__':
    unittest.main()
    #to = TestPolicy()
    #to.test_opt_max()