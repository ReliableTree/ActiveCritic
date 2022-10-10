import unittest

import torch as th
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.utils.dataset import DatasetAC
from active_critic.utils.gym_utils import (DummyExtractor, make_dummy_vec_env,
                                           new_epoch_reach,
                                           parse_sampled_transitions,
                                           sample_expert_transitions,
                                           sample_new_episode)
from active_critic.utils.pytorch_utils import make_part_obs_data
from active_critic.utils.test_utils import setup_ac_reach
from gym import Env

th.manual_seed(0)


class TestUtils(unittest.TestCase):
    def make_acl(self):
        device = 'cuda'
        acla = ActiveCriticLearnerArgs()
        acla.data_path = '/home/hendrik/Documents/master_project/LokalData/TransformerImitationLearning/'
        acla.device = device
        acla.extractor = DummyExtractor()
        acla.imitation_phase = True
        acla.logname = 'test_acl'
        acla.tboard = False
        acla.batch_size = 32
        acla.val_every = 10000
        acla.validation_episodes = 5
        seq_len = 5
        epsiodes = 2
        ac, acps, env = setup_ac_reach(seq_len=seq_len)
        acl = ActiveCriticLearner(ac_policy=ac, env=env, network_args_obj=acla)
        env, expert = make_dummy_vec_env(name='reach', seq_len=seq_len)
        return acl, env, expert, seq_len, epsiodes, device

    def test_make_part_seq_with_td(self):
        acl, env, expert, seq_len, epsiodes, device = self.make_acl()
        transitions = sample_expert_transitions(
            policy=expert.predict, env=env, episodes=epsiodes)
        exp_actions, exp_observations, exp_rewards = parse_sampled_transitions(
            transitions=transitions, new_epoch=new_epoch_reach, extractor=DummyExtractor(), device=device)

        part_acts, part_obsv, part_rews = make_part_obs_data(
            actions=exp_actions, observations=exp_observations, rewards=exp_rewards)

        for i in range(seq_len*epsiodes):
            org_index = int(i/seq_len)
            part_obs = th.clone(exp_observations[org_index])
            part_obs[i % seq_len + 1:] = 0
            self.assertTrue(
                th.equal(part_acts[i], exp_actions[org_index]), 'make_part_obs_data corrupted')
            self.assertTrue(
                th.equal(part_obsv[i], part_obs), 'make_part_obs_data corrupted')

    def test_sample_new_epsiode(self):
        acl, env, expert, seq_len, epsiodes, device = self.make_acl()

        actions, observations, rewards,expected_rewards_before, expected_rewards_after = sample_new_episode(
            policy=acl.policy,
            env=env,
            episodes=1,
        )
        self.assertTrue(list(actions.shape) == [
                        1, seq_len, env.action_space.shape[0]])
        self.assertTrue(list(observations.shape) == [
                        1, seq_len, env.observation_space.shape[0]])
        self.assertTrue(list(rewards.shape) == [1, seq_len, 1])
        self.assertTrue(list(expected_rewards_after.shape) == [1, seq_len, 1])

    def test_convergence(self):
        acl, env, expert, seq_len, epsiodes, device = self.make_acl()
        transitions = sample_expert_transitions(
            policy=expert.predict, env=env, episodes=epsiodes)
        exp_actions, exp_observations, exp_rewards = parse_sampled_transitions(
            transitions=transitions, new_epoch=new_epoch_reach, extractor=DummyExtractor(), device=device)
        part_acts, part_obsv, part_rews = make_part_obs_data(
            actions=exp_actions, observations=exp_observations, rewards=exp_rewards)

        imitation_data = DatasetAC(device='cuda')
        imitation_data.onyl_positiv = False
        imitation_data.add_data(
            obsv=part_obsv, actions=part_acts, reward=part_rews)

        self.assertTrue(len(imitation_data) == epsiodes * seq_len)
        self.assertTrue(list(imitation_data.actions.shape) == [
                        epsiodes * seq_len, seq_len, env.action_space.shape[0]])
        self.assertTrue(list(imitation_data.obsv.shape) == [
                        epsiodes * seq_len, seq_len, env.observation_space.shape[0]])
        self.assertTrue(list(imitation_data.reward.shape)
                        == [epsiodes * seq_len, seq_len, 1])
        imitation_data.onyl_positiv = True
        self.assertTrue(len(imitation_data) == 0)

        imitation_data.onyl_positiv = False
        dataloader = th.utils.data.DataLoader(
            imitation_data, batch_size=2*len(imitation_data))
        for data in dataloader:
            dobsv, dact, drews = data
        self.assertTrue(th.equal(dact, part_acts))
        self.assertTrue(th.equal(dobsv, part_obsv))
        self.assertTrue(th.equal(drews, part_rews))

        imitation_data.add_data(
            obsv=part_obsv, actions=part_acts, reward=part_rews)
        imitation_data.onyl_positiv = False
        dataloader = th.utils.data.DataLoader(
            imitation_data, batch_size=2*len(imitation_data))
        for data in dataloader:
            dobsv, dact, drews = data
        self.assertTrue(list(dobsv.shape) == [
                        2*epsiodes * seq_len, seq_len, env.observation_space.shape[0]])
        self.assertTrue(list(dact.shape) == [
                        2*epsiodes * seq_len, seq_len, env.action_space.shape[0]])
        self.assertTrue(list(drews.shape) == [
                        2*epsiodes * seq_len, seq_len, 1])

        acl.setDatasets(train_data=imitation_data)

        acl.train(epochs=1000)
        self.assertTrue(acl.scores.mean_actor[0] < 1e-3)
        self.assertTrue(acl.scores.mean_critic[0] < 1e-3)

        lenght_before = 0
        for data in acl.train_loader:
            obsv, actions, reward = data
            lenght_before += len(obsv)

        acl.add_training_data()
        lenght_after = 0
        for data in acl.train_loader:
            obsv, actions, reward = data
            lenght_after += len(obsv)          
        self.assertTrue(lenght_before + acl.policy.args_obj.epoch_len == lenght_after, 'data has not ben added to externally set training data set.')



    def test_ac_score(self):
        acl_scores = ACLScores()
        acl_scores.update_min_score(acl_scores.mean_actor, 0)
        self.assertTrue(acl_scores.mean_actor == [0])

    def test_add_training_data(self):
        acl, env, expert, seq_len, epsiodes, device = self.make_acl()

        acl.add_training_data()
        length = 0
        for data in acl.train_loader:
            obsv, actions, reward = data
            length += len(obsv)
        self.assertTrue(acl.policy.args_obj.epoch_len == length)



if __name__ == '__main__':
    unittest.main()
