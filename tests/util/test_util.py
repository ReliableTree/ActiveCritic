import unittest

import gym
import numpy as np
import torch as th
from metaworld.envs import \
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from active_critic.utils.gym_utils import make_policy_dict, new_epoch_reach, make_dummy_vec_env, sample_transitions, parse_sampled_transitions_legacy, parse_sampled_transitions, make_vec_env
from active_critic.utils.pytorch_utils import make_partially_observed_seq, make_part_obs_data, make_inf_seq
from gym.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from active_critic.utils.gym_utils import DummyExtractor
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.utils.test_utils import setup_ac_reach
from active_critic.learner.active_critic_learner import ActiveCriticLearner
from active_critic.utils.gym_utils import sample_new_episode


class TestUtils(unittest.TestCase):

    def test_make_partially_observed_seq(self):
        seq_len = 3
        d_out = 2
        d_in = 3
        acts_array_low = [0]*d_out
        acts_array_high = [1]*d_out
        action_space = gym.spaces.box.Box(
            np.array(acts_array_low), np.array(acts_array_high), (d_out,), float)
        obs = th.ones([2, 1, d_in], dtype=th.float, device='cuda')
        pos = make_partially_observed_seq(
            obs=obs, acts=None, seq_len=seq_len, act_space=action_space)
        expected_shape = list(obs.shape)
        expected_shape[1] = seq_len
        expected_shape[2] += action_space.shape[0]

        self.assertTrue( list(
            pos.shape) == expected_shape, f'Output shape is not as expected. Expected: {expected_shape}, but got {pos.shape}')
        self.assertTrue( th.all(pos[:, :, obs.shape[-1]:] ==
                      0), 'No action input, but some action fields are not zero.')
        self.assertTrue( th.equal(pos[:, :obs.shape[1], :obs.shape[2]],
                        obs), 'Observation not correctly inserted into sequence.')

        current_obs_len = 2
        obs = th.ones([2, current_obs_len, d_in],
                      dtype=th.float, device='cuda')
        acts = 2*th.ones([2, current_obs_len-1, d_out],
                         dtype=th.float, device='cuda')
        pos = make_partially_observed_seq(
            obs=obs, acts=acts, seq_len=seq_len, act_space=action_space)
        expected_shape = list(obs.shape)
        expected_shape[1] = seq_len
        expected_shape[2] += action_space.shape[0]
        self.assertTrue(list(
            pos.shape) == expected_shape, f'Output shape is not as expected. Expected: {expected_shape}, but got {pos.shape}')
        self.assertTrue( th.all(pos[:, acts.shape[1], obs.shape[-1]:] ==
                      0), 'Action that werent input are displayed.')
        self.assertTrue( th.equal(pos[:, :obs.shape[1], :obs.shape[2]],
                        obs), 'Observation not correctly inserted into sequence.')
        self.assertTrue( th.equal(pos[:, :acts.shape[1], obs.shape[2]:],
                        acts), 'Not all actions have been inserted correctly.')

    def test_new_epoch(self):
        policy_dict = make_policy_dict()
        env_tag = 'reach'
        max_episode_steps = 5
        re = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[policy_dict[env_tag][1]]()
        re._freeze_rand_vec = False
        timelimit = TimeLimit(env=re, max_episode_steps=max_episode_steps)
        dv1 = DummyVecEnv([lambda: RolloutInfoWrapper(timelimit)])
        de = DummyExtractor()
        obs1 = de.forward(dv1.reset())
        self.assertFalse(new_epoch_reach(obs1, obs1), 'new_epoch cant diff the same input.')

        obs2_np, _,_,_ = dv1.step(np.array([[0,1,0,0]]))
        obs2 = de.forward(obs2_np)
        
        self.assertFalse(th.equal(obs2, obs1), 'new observation after step is same as old one.') 
        self.assertFalse(new_epoch_reach(obs1, obs2), 'Same epoch, different observation. Should not be a new epoch.') 
        new_obs = de.forward(dv1.reset())
        self.assertTrue(new_epoch_reach(obs1, new_obs), 'New epoch was not recognized.')

    def test_collect_new_transitions(self):
        seq_len = 100
        episodes = 2
        name = 'reach'
        env, exp = make_dummy_vec_env(name=name, seq_len=seq_len)
        transitions = sample_transitions(policy=exp.predict, env=env, episodes=episodes)
        actions, observations, rewards = parse_sampled_transitions(transitions=transitions, seq_len=seq_len, extractor=DummyExtractor())
        
        
        self.assertTrue( list(actions.shape) == [episodes, seq_len, env.action_space.shape[0]])
        self.assertTrue(list(observations.shape) == [episodes, seq_len, env.observation_space.shape[0]])
        self.assertTrue(list(rewards.shape) == [episodes, seq_len,1])
        self.assertTrue(th.all(rewards[:,-1] == 1))

    def test_part_observ_seq(self):
        epsiodes = 2
        seq_len = 5
        env, expert = make_dummy_vec_env(name='reach', seq_len=seq_len)
        transitions = sample_transitions(policy=expert.predict, env=env, episodes=epsiodes)
        actions, observations, rewards = parse_sampled_transitions(transitions=transitions, extractor=DummyExtractor(), seq_len=seq_len)
        acts, obsv, rews = make_part_obs_data(actions=actions, observations=observations, rewards=rewards)
        print(f'acts.shape: {acts.shape}')
        print([epsiodes*seq_len, seq_len, env.action_space.shape[0]])
        self.assertTrue( list(acts.shape) == [epsiodes*seq_len, seq_len, env.action_space.shape[0]])
        self.assertTrue( list(obsv.shape) == [epsiodes*seq_len, seq_len, env.observation_space.shape[0]])
        self.assertTrue( list(rews.shape) == [epsiodes*seq_len, seq_len, 1])

        for i in range(seq_len*epsiodes):
            org_index = int(i/seq_len)
            part_obs = th.clone(observations[org_index])
            part_obs[i%seq_len + 1 :] = 0
            self.assertTrue( th.equal(obsv[i], part_obs))
            self.assertTrue( th.equal(acts[i], actions[org_index]))

    def test_sample_new_episode(self):
        th.manual_seed(0)
        device = 'cpu'
        acla = ActiveCriticLearnerArgs()
        acla.data_path = '/home/hendrik/Documents/master_project/LokalData/TransformerImitationLearning/'
        acla.device = device
        acla.extractor = DummyExtractor()
        acla.imitation_phase = False
        acla.logname = 'test_acl'
        acla.tboard = False
        epsiodes = 2
        seq_len = 5
        ac, acps, env = setup_ac_reach()
        acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=env, network_args_obj=acla)
        env, expert = make_dummy_vec_env(name='reach', seq_len=seq_len)
        actions, observations, rewards, expected_rewards_before, expected_rewards_after = sample_new_episode(
            policy=ac,
            env=env,
            device=device,
            episodes=epsiodes)
            
        exp_act_shp = [epsiodes, seq_len, env.action_space.shape[0]]
        self.assertTrue( list(actions.shape) == exp_act_shp)

        exp_obs_shp = [epsiodes, seq_len, env.observation_space.shape[0]]
        self.assertTrue( list(observations.shape) == exp_obs_shp)

        exp_rew_shp = [epsiodes, seq_len, 1]
        self.assertTrue( list(rewards.shape) == exp_rew_shp)

        exp_exp_rew_shp = [epsiodes, seq_len, ac.critic.wsms.model_setup.d_output]
        self.assertTrue( list(expected_rewards_before.shape) == exp_exp_rew_shp)

        ac.args_obj.optimize = False
        for i in range(4): #rng is not save
            actions, gen_act, observations, rewards, expected_rewards_before, expected_rewards_after = sample_new_episode(
                policy=ac,
                env=env,
                episodes=epsiodes,
                device=device,
                return_gen_trj=True)
            self.assertTrue(th.equal(actions, gen_act), 'Gen Actions output wrong.')

    def test_make_inf_seq(self):
        obs_len = 6
        batch_size = 2
        dim = 1
        seq_len = 3
        inp_seq = th.arange(obs_len).reshape([1,obs_len,1]).repeat([batch_size,1,dim])

        inf_seq = make_inf_seq(inp_seq,seq_len)
        self.assertTrue(list(inf_seq.shape) == [batch_size*(obs_len-seq_len+int(seq_len/2)+1), seq_len, dim])

        obs_len = 7
        batch_size = 2
        dim = 1
        seq_len = 3
        inp_seq = th.arange(obs_len).reshape([1,obs_len,1]).repeat([batch_size,1,dim])

        inf_seq = make_inf_seq(inp_seq, seq_len)
        self.assertTrue(list(inf_seq.shape) == [batch_size*(obs_len-seq_len+int(seq_len/2)+1), seq_len, dim])

        obs_len = 7
        batch_size = 2
        dim = 1
        seq_len = 4
        inp_seq = th.arange(obs_len).reshape([1,obs_len,1]).repeat([batch_size,1,dim])

        inf_seq = make_inf_seq(inp_seq, seq_len)
        self.assertTrue(list(inf_seq.shape) == [batch_size*(obs_len-seq_len+int(seq_len/2)+1), seq_len, dim])

    def test_parse_sampled_seq(self):
        seq_len = 100
        episodes = 3
        name = 'reach'
        env, exp = make_vec_env(env_id='reach', num_cpu=1, seq_len=100)
        transitions = sample_transitions(policy=exp.predict, env=env, episodes=episodes)
        actions_leg, observations_leg, rewards_leg = parse_sampled_transitions_legacy(transitions=transitions, new_epoch=new_epoch_reach, seq_len=seq_len, extractor=DummyExtractor())
        actions, observations, rewards = parse_sampled_transitions(transitions=transitions, seq_len=seq_len, extractor=DummyExtractor())
        
        self.assertTrue(len(transitions) < episodes*seq_len)

        self.assertTrue(th.equal(actions, actions_leg))
        self.assertTrue(th.equal(observations, observations_leg))
        self.assertTrue(th.equal(rewards, rewards_leg))

        epochs = [0]
        for i, step in enumerate(transitions):
            if step['dones']:
                epochs.append(i)
        epochs = th.tensor(epochs)
        episodes_len = th.diff(epochs)

        for i, reward in enumerate(rewards):
            self.assertTrue(th.all(reward[int(episodes_len[i]+1):] == -1))


if __name__ == '__main__':
    #unittest.main()
    to = TestUtils()
    to.test_parse_sampled_seq()