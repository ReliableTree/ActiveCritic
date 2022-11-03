import unittest

import torch as th
import numpy as np

from active_critic.model_src.whole_sequence_model import *
from active_critic.policy.active_critic_policy import *
from active_critic.model_src.state_model import StateModel, StateModelArgs

from active_critic.utils.test_utils import (make_obs_act_space,
                                            make_wsm_setup)
from active_critic.utils.gym_utils import (DummyExtractor, make_dummy_vec_env,
                                           new_epoch_pap,
                                           new_epoch_reach)
from active_critic.utils.pytorch_utils import build_tf_horizon_mask

def make_acps(seq_len, new_epoch, batch_size = 32, device='cpu', horizon = 0):
    acps = ActiveCriticPolicySetup()
    acps.device=device
    acps.epoch_len=seq_len
    acps.extractor=DummyExtractor()
    acps.new_epoch=new_epoch
    acps.opt_steps=100
    acps.inference_opt_lr = 1e-1
    acps.optimizer_class = th.optim.Adam
    acps.optimize = True
    acps.batch_size = batch_size
    acps.pred_mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
    acps.opt_mask = th.zeros([seq_len, 1], device=device, dtype=bool)
    acps.opt_mask[:,-1] = 1
    acps.clip = False
    acps.opt_goal = True
    return acps

class DummyHistory:
    def __init__(self) -> None:
        self.reset()


    def reset(self):
        self.scores = []
        self.goal_scores = []
        self.gen_trj = []
        self.opt_trj = []
        self.pred_emb = []
        self.act_emb = []


    def new_epoch(self, history:list([th.Tensor]), size:list([int, int, int, int]), device:str): #batch size, opt step, seq len, score
        pass


    def add_value(self, history:list([th.Tensor]), value:th.Tensor, opt_step:int=0, step:int=0):
        pass

def setup_opt_state(device='cuda', embed_dim = 4, action_dim=2):
    seq_len = 6
    obs_dim = 3
    batch_size = 2
    embed_dim = embed_dim
    lr = 1e-3

    actor_args = StateModelArgs()
    actor_args.arch = [20, 20, action_dim]
    actor_args.device = device
    actor_args.lr = lr
    actor = StateModel(args=actor_args)

    critic_args = StateModelArgs()
    critic_args.arch = [20, 20, 1]
    critic_args.device = device
    critic_args.lr = lr
    critic = StateModel(args=critic_args)

    inv_critic_args = StateModelArgs()
    inv_critic_args.arch = [20, 20, embed_dim+action_dim]
    inv_critic_args.device = device
    inv_critic_args.lr = lr
    inv_critic = StateModel(args=inv_critic_args)

    emitter_args = StateModelArgs()
    emitter_args.arch = [20, 20, embed_dim]
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
        seq_len=seq_len, new_epoch=new_epoch_pap, device=device, batch_size=batch_size)
    acps.opt_steps = 2
    obs_space, acts_space = make_obs_act_space(
        obs_dim=obs_dim, action_dim=action_dim)
    ac = ActiveCriticPolicy(observation_space=obs_space, 
                            action_space=acts_space,
                            actor=actor,
                            critic=critic,
                            predictor=predictor,
                            emitter=emitter,
                            inverse_critic=inv_critic,
                            acps=acps)
    return ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len

class TestPolicy(unittest.TestCase):

    def test_embeddings_projection(self):
        th.manual_seed(0)
        device = 'cpu'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device)
        acps.opt_steps = 2
        goal_dim = 1
        change_dim = 2
        acps.inference_opt_lr = 1
        observations = th.ones([batch_size, 3, obs_dim], dtype=float)
        observations[:,:,:change_dim] = 0
        org_obs = observations.detach().clone()
        observations.requires_grad = True
        observations.mean().backward()
        goal = th.ones([batch_size, goal_dim])
        proj_obs = ac.project_embeddings(embeddings=observations, goal_state=goal)

        self.assertTrue(th.all(proj_obs[:,:,:goal_dim] == goal[:, None]))
        self.assertTrue(th.all(proj_obs[:,:,goal_dim:] == org_obs[:,:,goal_dim:]))
        self.assertTrue(th.all(proj_obs.grad != 0))

    def optimize_goal_embeddings(self):
        th.manual_seed(0)
        device = 'cpu'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device)
        acps.opt_steps = 2
        goal_dim = 1

        embeddings = th.ones([batch_size, seq_len, embed_dim], dtype=th.float32)
        actions = th.ones([batch_size, seq_len, action_dim], dtype=th.float32)
        goal_state = th.zeros([batch_size, goal_dim], dtype=th.float32)
        goal_label = th.ones([batch_size, seq_len, 1])
        opt_steps = 100
        lr = 1

        inv_inpt = ac.get_inverse_critic_input(goal_state=goal_state, goal_scores=goal_label[:,-1])

        self.assertTrue(th.all(inv_inpt[:,:goal_dim] == goal_state))
        self.assertTrue(th.all(inv_inpt[:,goal_dim:] == goal_label[:,-1]))

        goal_embeddings = ac.inverse_critic.forward(inv_inpt)
        goal_embeddings = ac.project_embeddings(embeddings=goal_embeddings, goal_state=goal_state)
        expected_goal_shape = [batch_size, embed_dim]

        self.assertTrue(list(goal_embeddings.shape) == expected_goal_shape)
        self.assertTrue(th.all(goal_embeddings[:, :goal_dim] == goal_state))

        lr = 1
        opt_steps = 100
        ac.history = DummyHistory()
        opt_goal_embeddings = ac.optimize_goal_emb_acts(
            goal_emb_acts=goal_embeddings,
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )
        self.assertTrue(list(goal_embeddings.shape) == expected_goal_shape)
        self.assertTrue(th.all(goal_embeddings[:, :goal_dim] == goal_state))
        old_critic_scores = ac.critic.forward(goal_embeddings)
        new_critic_scores = ac.critic.forward(opt_goal_embeddings)
        self.assertTrue(th.all(new_critic_scores > old_critic_scores))
        ac_opt_goal_embeddings = ac.get_goal_emb_act(
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )

        self.assertTrue(th.equal(ac_opt_goal_embeddings, opt_goal_embeddings))


    def test_actions_grad(self):
        th.manual_seed(0)
        device = 'cpu'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device)
        ac.history = DummyHistory()

        horizon = 0
        embeddings = th.ones([batch_size, 1, embed_dim], device=device, requires_grad=True)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        goal_dim = 1
        goal_state = th.zeros([batch_size, goal_dim], dtype=th.float32)
        goal_label = th.ones([batch_size, seq_len, 1])
        opt_steps = 100
        lr = 1

        goal_embeddings = ac.get_goal_emb_act(
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings, actions = ac.build_sequence(
            embeddings=embeddings,
            actions=actions,
            seq_len=seq_len,
            goal_state=goal_state,
            goal_emb_acts=goal_embeddings,
            mask=mask
        )
        loss = ((seq_embeddings[:,-1] - th.ones_like(seq_embeddings[:,-1] ))**2).mean()
        loss.backward()
        self.assertTrue((actions.grad!=0).sum() == actions.numel() - action_dim*batch_size, 'Actions grad not as expected.')

    def test_predict_step_grad(self):
        th.manual_seed(0)
        device = 'cpu'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device)
        ac.history = DummyHistory()

        horizon = 0
        goal_dim = 1
        goal_state = th.zeros([batch_size, goal_dim], dtype=th.float32)
        goal_label = th.ones([batch_size, seq_len, 1])
        opt_steps = 100
        lr = 1

        goal_embeddings = ac.get_goal_emb_act(
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )

        embeddings = th.ones([batch_size, 1, embed_dim], device=device, requires_grad=True)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        action_optim = th.optim.Adam([actions], lr=1e-1)
        opt_paras = action_optim.state_dict()

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings, na = ac.build_sequence(
            embeddings=embeddings,
            actions=actions,
            seq_len=seq_len,
            goal_state=goal_state,
            goal_emb_acts=goal_embeddings,
            mask=mask
        )
        opt_actions = actions.detach()
        opt_actions.requires_grad = True
        next_embedding = ac.predict_step(embeddings=seq_embeddings.detach(), actions=opt_actions, mask=mask)
        seq_embedding = th.cat((embeddings[:,:1], next_embedding[:,:-1]), dim=1)
        loss = ((seq_embedding[:,-1] - th.ones_like(seq_embedding[:,-1] ))**2).mean()
        loss.backward()
        self.assertTrue((opt_actions.grad != 0).sum() == batch_size*action_dim, 'Prediction model can look beyond horizon.')

    def test_optimize_1_step(self):
        th.manual_seed(0)
        device = 'cpu'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device)
        ac.history = DummyHistory()

        horizon = 0
        goal_dim = 1
        goal_state = th.zeros([batch_size, goal_dim], dtype=th.float32)
        goal_label = th.ones([batch_size, seq_len, 1])
        opt_steps = 100
        lr = 1
        optimizer_class = th.optim.Adam

        goal_embeddings = ac.get_goal_emb_act(
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )

        embeddings = th.ones([batch_size, 1, embed_dim], device=device, requires_grad=True)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        action_optim = th.optim.Adam([actions], lr=1e-1)
        opt_paras = action_optim.state_dict()

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings, na = ac.build_sequence(
            embeddings=embeddings,
            actions=actions,
            seq_len=seq_len,
            goal_state=goal_state,
            goal_emb_acts=goal_embeddings,
            mask=mask
        )

        steps = 1
        current_step = 2

        pred_mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        opt_mask = th.zeros([seq_len, 1], dtype=th.bool)
        opt_mask[-current_step:] = 1

        loss_reward, actions, seq_embeddings, scores = ac.optimize_sequence(
            actions=actions,
            seq_embeddings=seq_embeddings,
            pred_mask=pred_mask,
            opt_mask=opt_mask,
            goal_label=goal_label,
            goal_emb_acts=goal_embeddings,
            steps=steps,
            current_step=current_step,
            seq_len=seq_len,
            optimizer_class=optimizer_class,
            lr=lr,
            goal_state=goal_state
        )

        self.assertTrue(loss_reward > 1e-1, 'Sequence should not be an optimal solution after noe step.')

    def test_optimize_n_step(self):
        th.manual_seed(0)
        device = 'cpu'
        embed_dim = 100
        action_dim = 100
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device, embed_dim=embed_dim, action_dim=action_dim)
        ac.history = DummyHistory()
        acps.clip = False

        horizon = 0
        goal_dim = 1
        goal_state = th.zeros([batch_size, goal_dim], dtype=th.float32)
        goal_label = th.ones([batch_size, seq_len, 1])
        opt_steps = 10
        lr = 1
        optimizer_class = th.optim.Adam

        goal_embeddings = ac.get_goal_emb_act(
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )

        embeddings = th.ones([batch_size, 1, embed_dim], device=device, requires_grad=True)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        org_actions = actions.detach().clone()
        action_optim = th.optim.Adam([actions], lr=1e-1)
        opt_paras = action_optim.state_dict()

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings, na = ac.build_sequence(
            embeddings=embeddings,
            actions=actions,
            seq_len=seq_len,
            goal_state=goal_state,
            goal_emb_acts=goal_embeddings,
            mask=mask
        )
        org_seq_embeddings = seq_embeddings.detach().clone()

        steps = 500
        lr = 1e-2
        current_step = 1

        pred_mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        opt_mask = th.zeros([seq_len, 1], dtype=th.bool)
        opt_mask[-current_step:] = 1

        loss_reward, actions, seq_embeddings, scores = ac.optimize_sequence(
            actions=actions,
            seq_embeddings=seq_embeddings,
            pred_mask=pred_mask,
            opt_mask=opt_mask,
            goal_label=goal_label,
            goal_emb_acts=goal_embeddings,
            steps=steps,
            current_step=current_step,
            seq_len=seq_len,
            optimizer_class=optimizer_class,
            lr=lr,
            goal_state=goal_state
        )
        self.assertTrue(loss_reward < 1e-4, 'Sequence optimisation failed.')

        self.assertTrue(th.equal(actions[:, :current_step], org_actions[:,:current_step]), 'Actions were overridden')
        self.assertTrue(not th.equal(actions[:,current_step:], org_actions[:,current_step:]), 'No new actions were chosen.')

    def test_whole_vs_part_seq_optimizer(self):
        th.manual_seed(0)
        device = 'cpu'
        embed_dim = 100
        action_dim = 100
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device, embed_dim=embed_dim, action_dim=action_dim)
        ac.history = DummyHistory()
        acps.clip = False

        horizon = 0
        goal_dim = 1
        goal_state = th.zeros([batch_size, goal_dim], dtype=th.float32)
        goal_label = th.ones([batch_size, seq_len, 1])
        opt_steps = 10
        lr = 1
        optimizer_class = th.optim.Adam

        goal_embeddings = ac.get_goal_emb_act(
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )

        embeddings = th.ones([batch_size, 1, embed_dim], device=device, requires_grad=True)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        org_actions = actions.detach().clone()
        action_optim = th.optim.Adam([actions], lr=1e-1)
        opt_paras = action_optim.state_dict()

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings, na = ac.build_sequence(
            embeddings=embeddings,
            actions=actions,
            seq_len=seq_len,
            goal_state=goal_state,
            goal_emb_acts=goal_embeddings,
            mask=mask
        )
        org_seq_embeddings = seq_embeddings.detach().clone()

        steps = 100
        lr = 1e-1
        current_step = 0

        pred_mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        opt_mask = th.zeros([seq_len, 1], dtype=th.bool)
        opt_mask[-current_step:] = 1

        loss_reward_full, actions, seq_embeddings, scores = ac.optimize_sequence(
            actions=actions,
            seq_embeddings=seq_embeddings,
            pred_mask=pred_mask,
            opt_mask=opt_mask,
            goal_label=goal_label,
            goal_emb_acts=goal_embeddings,
            steps=steps,
            current_step=current_step,
            seq_len=seq_len,
            optimizer_class=optimizer_class,
            lr=lr,
            goal_state=goal_state
        )

        th.manual_seed(0)
        device = 'cpu'
        embed_dim = 100
        action_dim = 100
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device, embed_dim=embed_dim, action_dim=action_dim)
        ac.history = DummyHistory()
        acps.clip = False

        horizon = 0
        goal_dim = 1
        goal_state = th.zeros([batch_size, goal_dim], dtype=th.float32)
        goal_label = th.ones([batch_size, seq_len, 1])
        opt_steps = 10
        lr = 1
        optimizer_class = th.optim.Adam

        goal_embeddings = ac.get_goal_emb_act(
            goal_state=goal_state,
            goal_label=goal_label,
            lr=lr,
            opt_steps=opt_steps
        )

        embeddings = th.ones([batch_size, 1, embed_dim], device=device, requires_grad=True)
        actions = th.ones([batch_size, seq_len, action_dim], device=device, requires_grad=True)
        org_actions = actions.detach().clone()
        action_optim = th.optim.Adam([actions], lr=1e-1)
        opt_paras = action_optim.state_dict()

        mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        seq_embeddings, na = ac.build_sequence(
            embeddings=embeddings,
            actions=actions,
            seq_len=seq_len,
            goal_state=goal_state,
            goal_emb_acts=goal_embeddings,
            mask=mask
        )
        org_seq_embeddings = seq_embeddings.detach().clone()

        steps = 100
        lr = 1e-1
        current_step = 3

        pred_mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
        opt_mask = th.zeros([seq_len, 1], dtype=th.bool)
        opt_mask[-current_step:] = 1

        loss_reward_part, actions, seq_embeddings, scores = ac.optimize_sequence(
            actions=actions,
            seq_embeddings=seq_embeddings,
            pred_mask=pred_mask,
            opt_mask=opt_mask,
            goal_label=goal_label,
            goal_emb_acts=goal_embeddings,
            steps=steps,
            current_step=current_step,
            seq_len=seq_len,
            optimizer_class=optimizer_class,
            lr=lr,
            goal_state=goal_state
        )

        self.assertTrue(loss_reward_part > loss_reward_full, 'Optimizing the last part of the sequence is better then optmizing the whole sequence.')


    def test_score_history(self):
        th.manual_seed(0)
        device = 'cpu'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device)
        acps.opt_steps = 100
        acps.inference_opt_lr = 1
        observation = th.ones([batch_size, 1, obs_dim])
        action = ac.predict(observation=observation)
        self.assertTrue(th.all((ac.history.scores[0][:,-1,-1] - ac.history.scores[0][:,0,-1]) >= 0), 'Scores are not increasing.')

    def test_history_shape(self):
        th.manual_seed(0)
        device = 'cpu'
        ac, acps, action_dim, obs_dim, batch_size, embed_dim, seq_len = setup_opt_state(device=device)
        acps.opt_steps = 10
        acps.inference_opt_lr = 1
        observation = th.ones([batch_size, 1, obs_dim])
        action = th.tensor(ac.predict(observation=observation))
        all_actions = action.clone().unsqueeze(1)
        all_embeddings = ac.current_embeddings.clone()
        expected_action_shape = [batch_size, 1, action_dim]
        expected_embedding_shape = [batch_size, 1, embed_dim]
        self.assertTrue(list(all_actions.shape) == expected_action_shape, 'Current actions in AC not in shape.')
        self.assertTrue(list(all_embeddings.shape) == expected_embedding_shape, 'Current embeddings in AC not in shape.')
        for i in range(seq_len-1):
            action = th.tensor(ac.predict(observation=observation))
            self.assertTrue(th.equal(ac.current_actions[:,:-1], all_actions), 'Current actions are changed post.')
            self.assertTrue(th.equal(ac.current_embeddings[:,:-1], all_embeddings), 'Current embeddings are changed post.')
            all_actions = th.cat((all_actions, action.clone().unsqueeze(1)), dim=1)
            all_embeddings = ac.current_embeddings.clone()

        observation = 2*th.ones([batch_size, 1, obs_dim])
        action = th.tensor(ac.predict(observation=observation))

        all_actions_2 = action.clone().unsqueeze(1)
        for i in range(seq_len-1):
            action = th.tensor(ac.predict(observation=observation))
            all_actions_2 = th.cat((all_actions_2, action.clone().unsqueeze(1)), dim=1)

        observation = 3*th.ones([batch_size, 1, obs_dim])
        action = th.tensor(ac.predict(observation=observation))


        expected_action_shape = [batch_size, seq_len, action_dim]
        expected_embedding_shape = [batch_size, seq_len, embed_dim]
        self.assertTrue(list(all_actions_2.shape) == expected_action_shape, 'New Epoch did not work as intended.')

        expected_trj_history_shape = [3*batch_size, seq_len, action_dim]
        expected_scores_history_shape = [3*batch_size, ac.args_obj.opt_steps, seq_len, 1]


        self.assertTrue(list(ac.history.gen_trj[0].shape) == expected_trj_history_shape)
        self.assertTrue(list(ac.history.opt_trj[0].shape) == expected_trj_history_shape)
        self.assertTrue(list(ac.history.scores[0].shape) == expected_scores_history_shape)

        self.assertTrue(th.equal(ac.history.opt_trj[0][:batch_size], all_actions))
        self.assertTrue(th.equal(ac.history.opt_trj[0][batch_size:2*batch_size], all_actions_2))


if __name__ == '__main__':
    unittest.main()
    #to = TestPolicy()
    #to.test_optimize_n_step()
    #to.test_actions_grad()
    #to.test_history_shape()
    #to.test_whole_vs_part_seq_optimizer()
    #to.test_history_shape()