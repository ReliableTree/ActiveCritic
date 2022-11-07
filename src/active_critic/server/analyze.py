import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_vec_env, DummyExtractor, new_epoch_reach
from active_critic.utils.pytorch_utils import build_tf_horizon_mask
from active_critic.utils.dataset import DatasetAC
from active_critic.policy.active_critic_policy import ActiveCriticPolicySetup, ActiveCriticPolicy
from active_critic.model_src.state_model import *


from gym import Env
th.manual_seed(0)

def make_acps(seq_len, extractor, new_epoch, batch_size = 2, device='cpu', horizon = 0):
    acps = ActiveCriticPolicySetup()
    acps.device=device
    acps.epoch_len=seq_len
    acps.extractor=extractor
    acps.new_epoch=new_epoch
    acps.opt_steps=100
    acps.inference_opt_lr = 1e-1
    acps.optimizer_class = th.optim.Adam
    acps.optimize = True
    acps.batch_size = batch_size
    acps.pred_mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
    acps.opt_mask = th.zeros([seq_len, 1], device=device, dtype=bool)
    acps.opt_mask[:,-1] = 1
    acps.opt_goal = True
    return acps

def setup_opt_state(batch_size, seq_len, device='cpu'):
    num_cpu = 1
    env, expert = make_vec_env('reach', num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    embed_dim = 20
    lr = 1e-3

    actor_args = StateModelArgs()
    actor_args.arch = [200, 200, env.action_space.shape[0]]
    actor_args.device = device
    actor_args.lr = lr
    actor = StateModel(args=actor_args)

    critic_args = StateModelArgs()
    critic_args.arch = [200, 200, 1]
    critic_args.device = device
    critic_args.lr = lr
    critic = StateModel(args=critic_args)

    inv_critic_args = StateModelArgs()
    inv_critic_args.arch = [200, 200, embed_dim + env.action_space.shape[0]]
    inv_critic_args.device = device
    inv_critic_args.lr = lr
    inv_critic = StateModel(args=inv_critic_args)

    emitter_args = StateModelArgs()
    emitter_args.arch = [200, 200, embed_dim]
    emitter_args.device = device
    emitter_args.lr = lr
    emitter = StateModel(args=emitter_args)

    predictor_args = StateModelArgs()
    predictor_args.arch = [200, 200, embed_dim]
    predictor_args.device = device
    predictor_args.lr = lr
    predictor = StateModel(args=emitter_args)


    acps = make_acps(
        seq_len=seq_len, extractor=DummyExtractor(), new_epoch=new_epoch_reach, device=device, batch_size=batch_size)
    acps.clip = True
    ac = ActiveCriticPolicy(observation_space=env.observation_space, 
                            action_space=env.action_space,
                            actor=actor,
                            critic=critic,
                            predictor=predictor,
                            emitter=emitter,
                            inverse_critic=inv_critic,
                            acps=acps)
    return ac, acps, batch_size, seq_len, env, expert


def make_acl(device):
    device = device
    acla = ActiveCriticLearnerArgs()
    #acla.data_path = '/home/hendrik/Documents/master_project/LokalData/TransformerImitationLearning/'
    acla.data_path = '/data/bing/hendrik/'
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = False
    acla.logname = 'reach_plot_predictive_embedding_sparse'
    acla.tboard = True
    acla.batch_size = 32
    acla.validation_episodes = 5
    acla.training_epsiodes = 1
    acla.actor_threshold = 1e-2
    acla.critic_threshold = 1e-2
    acla.predictor_threshold = 1e-2
    acla.num_cpu = 5

    batch_size = 32
    seq_len = 100
    ac, acps, batch_size, seq_len, env, expert= setup_opt_state(device=device, batch_size=batch_size, seq_len=seq_len)
    
    acps.opt_steps = 20
    acla.val_every = 10
    acla.add_data_every = 1

    

    eval_env, expert = make_vec_env('reach', num_cpu=acla.num_cpu, seq_len=seq_len)
    acl = ActiveCriticLearner(ac_policy=ac, env=env, eval_env=eval_env, network_args_obj=acla)
    return acl, env, expert, seq_len, device


def run_experiment_analyze(device):
    acl, env, expert, seq_len, device = make_acl(device=device)
    acl.train(epochs=100000)

if __name__ == '__main__':
    run_experiment_analyze(device='cuda')
