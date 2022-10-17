from math import fabs
import gym
import numpy as np
import torch as th
from metaworld.envs import \
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from active_critic.model_src.transformer import (ModelSetup, generate_square_subsequent_mask)
from active_critic.model_src.whole_sequence_model import (WholeSequenceModelSetup, WholeSequenceModel)
from active_critic.policy.active_critic_policy import ActiveCriticPolicySetup, ActiveCriticPolicy
from active_critic.utils.gym_utils import (DummyExtractor, new_epoch_reach)
from active_critic.utils.gym_utils import make_dummy_vec_env, make_vec_env


def make_seq_encoding_data(batch_size, seq_len, ntoken, d_out, device = 'cuda'):
    inpt_seq = th.ones([batch_size,seq_len,ntoken], dtype=th.float, device=device)
    outpt_seq = th.ones([batch_size,seq_len,d_out], dtype=th.float, device=device)
    outpt_seq[:,::2] = 0
    return inpt_seq, outpt_seq

def make_mask_data(batch_size, seq_len, ntoken, device = 'cuda'):
    mask = generate_square_subsequent_mask(seq_len).to(device)
    inpt_seq = th.ones([batch_size,seq_len,ntoken], dtype=th.float, device=device)
    inpt_seq[0,-1,0] = 0
    outpt_seq = th.ones_like(inpt_seq)
    outpt_seq[0] = 0
    return inpt_seq, outpt_seq, mask

def make_critic_data(batch_size, seq_len, ntoken, device = 'cuda'):
    inpt_seq = th.ones([batch_size,seq_len,ntoken], dtype=th.float, device=device)
    inpt_seq[0,-1,0] = 0
    outpt_seq = th.ones([batch_size,1], dtype=th.float, device=device)
    outpt_seq[0] = 0
    return inpt_seq, outpt_seq

def make_policy_obs_action(seq_len, ntoken, d_out, diff_ele, device = 'cuda'):
    o1 = th.zeros([1, seq_len, ntoken], dtype=th.float, device=device)
    o2 = th.zeros_like(o1)
    o2[0, diff_ele] = 1

    a1 = th.zeros(1, seq_len, d_out, dtype=th.float, device=device)
    a2 = th.zeros_like(a1)
    a2[:, diff_ele:] = 1
    o = th.cat((o1, o2), dim=0)
    a = th.cat((a1, a2), dim=0)
    
    return o, a

def make_wsm_setup(seq_len, d_output, device='cuda'):
    wsm = WholeSequenceModelSetup()
    wsm.model_setup = ModelSetup()
    seq_len = seq_len
    d_output = d_output
    wsm.model_setup.d_output = d_output
    wsm.model_setup.nhead = 1
    wsm.model_setup.d_hid = 10
    wsm.model_setup.d_model = 10
    wsm.model_setup.nlayers = 2
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout = 0
    wsm.lr = 1e-3
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.Adam
    wsm.optimizer_kwargs = {}
    return wsm

def make_obs_act_space(obs_dim, action_dim):
    obs_array_low = [0]*obs_dim
    obs_array_high = [1]*obs_dim
    action_low = [0]*action_dim
    action_high = [1]*action_dim
    observation_space = gym.spaces.box.Box(
        np.array(obs_array_low), np.array(obs_array_high), (obs_dim,), float)
    # next state (4)
    action_space = gym.spaces.box.Box(
        np.array(action_low), np.array(action_high), (action_dim,), float)
    return observation_space, action_space

def make_acps(seq_len, extractor, new_epoch, batch_size = 32, device='cuda'):
    acps = ActiveCriticPolicySetup()
    acps.device=device
    acps.epoch_len=seq_len
    acps.extractor=extractor
    acps.new_epoch=new_epoch
    acps.opt_steps=100
    acps.optimisation_threshold=0.5
    acps.inference_opt_lr = 1e-1
    acps.optimize = True
    acps.batch_size = batch_size
    acps.stop_opt = False
    acps.opt_end = False
    acps.optimize_last = False
    return acps

def setup_ac_reach(seq_len = 5, device='cuda'):
    seq_len = seq_len
    env, gt_policy = make_dummy_vec_env('reach', seq_len=seq_len)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = WholeSequenceModel(wsm_critic_setup)
    ac = ActiveCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env

def setup_ac_reach_op(seq_len, device):
    seq_len = seq_len
    env, expert = make_vec_env('reach',seq_len=seq_len, num_cpu=1)
    d_output = env.action_space.shape[0]
    wsm_actor_setup = make_wsm_setup(
        seq_len=seq_len, d_output=d_output, device=device)
    wsm_critic_setup = make_wsm_setup(
        seq_len=seq_len, d_output=1, device=device)
    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device)
    actor = WholeSequenceModel(wsm_actor_setup)
    critic = OptimizeEndCritic(wsm_critic_setup)
    ac = ACPOptEnd(observation_space=env.observation_space, action_space=env.action_space,
                            actor=actor, critic=critic, acps=acps)
    return ac, acps, env, expert