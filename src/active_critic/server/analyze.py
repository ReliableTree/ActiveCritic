import torch as th
from active_critic.learner.active_critic_learner import ActiveCriticLearner, ACLScores
from active_critic.learner.active_critic_args import ActiveCriticLearnerArgs
from active_critic.policy.active_critic_policy import ActiveCriticPolicy
from active_critic.utils.gym_utils import make_vec_env, DummyExtractor, new_epoch_reach, sample_expert_transitions, parse_sampled_transitions
from active_critic.utils.pytorch_utils import build_tf_horizon_mask
from active_critic.utils.dataset import DatasetAC
from active_critic.policy.active_critic_policy import ActiveCriticPolicySetup, ActiveCriticPolicy
from active_critic.model_src.state_model import *
from active_critic.model_src.whole_sequence_model import WholeSequenceModel, WholeSequenceModelArgs
from active_critic.model_src.transformer import ModelSetup


from gym import Env
th.manual_seed(0)

class DummyStateModel(StateModel):
    def __init__(self, args: StateModelArgs) -> None:
        super().__init__(args)

    def forward(self, inpt):
        super().forward(inpt)
        return inpt

def make_wsm_setup(seq_len, d_output, device='cpu'):
    wsm = WholeSequenceModelArgs()
    wsm.model_setup = ModelSetup()
    seq_len = seq_len
    d_output = d_output
    wsm.model_setup.d_output = d_output
    wsm.model_setup.nhead = 1
    wsm.model_setup.d_hid = 512
    wsm.model_setup.d_model = 512
    wsm.model_setup.nlayers = 3
    wsm.model_setup.seq_len = seq_len
    wsm.model_setup.dropout = 0
    wsm.lr = 1e-3
    wsm.model_setup.device = device
    wsm.optimizer_class = th.optim.Adam
    wsm.optimizer_kwargs = {}
    return wsm

def make_acps(seq_len, extractor, new_epoch, batch_size, device, horizon):
    acps = ActiveCriticPolicySetup()
    acps.device=device
    acps.epoch_len=seq_len
    acps.extractor=extractor
    acps.new_epoch=new_epoch
    acps.opt_steps=100
    acps.inference_opt_lr = 1e-2
    acps.optimizer_class = th.optim.SGD
    acps.optimize = True
    acps.batch_size = batch_size
    acps.pred_mask = build_tf_horizon_mask(seq_len=seq_len, horizon=horizon, device=device)
    acps.opt_mask = th.zeros([seq_len], device=device, dtype=bool)
    acps.opt_mask[-1] = 1
    acps.opt_goal = True
    acps.optimize_goal_emb_acts = False
    acps.goal_label_multiplier = 1
    return acps

def setup_opt_state(batch_size, seq_len, device='cpu'):
    num_cpu = 1
    env, expert = make_vec_env('reach', num_cpu, seq_len=seq_len)
    d_output = env.action_space.shape[0]
    embed_dim = 39
    lr = 5e-4

    actor_args = StateModelArgs()
    actor_args.arch = [512,512, env.action_space.shape[0]]
    actor_args.device = device
    actor_args.lr = lr
    actor = StateModel(args=actor_args)

    critic_args = StateModelArgs()
    critic_args.arch = [512,512, 1]
    critic_args.device = device
    critic_args.lr = lr
    critic = StateModel(args=critic_args)

    inv_critic_args = StateModelArgs()
    inv_critic_args.arch = [200, embed_dim + env.action_space.shape[0]]
    inv_critic_args.device = device
    inv_critic_args.lr = lr
    inv_critic = StateModel(args=inv_critic_args)

    emitter_args = StateModelArgs()
    emitter_args.arch = [200, 200, embed_dim]
    emitter_args.device = device
    emitter_args.lr = lr
    #emitter = StateModel(args=emitter_args)
    emitter = DummyStateModel(args=emitter_args)

    predictor_args = make_wsm_setup(
    seq_len=seq_len, d_output=embed_dim, device=device)
    predictor_args.model_setup.d_hid = 2048
    predictor_args.model_setup.d_model = 2048
    predictor_args.model_setup.nlayers = 4
    predictor_args.model_setup.nhead = 16
    predictor_args.lr = 5e-4
    predictor = WholeSequenceModel(args=predictor_args)

    horizon = seq_len


    acps = make_acps(
        seq_len=seq_len, 
        extractor=DummyExtractor(), 
        new_epoch=new_epoch_reach, 
        device=device, 
        batch_size=batch_size,
        horizon=horizon)
    acps.clip = False
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
    #acla.data_path = '/home/hendrik/Documents/master_project/LokalData/WSM/'
    acla.data_path = '/data/bing/hendrik/'
    acla.device = device
    acla.extractor = DummyExtractor()
    acla.imitation_phase = False
    acla.logname = 'push data trough 40'
    acla.tboard = True
    acla.batch_size = 32
    acla.validation_episodes = 40
    acla.training_epsiodes = 1
    acla.actor_threshold = 1e-2
    acla.critic_threshold = 1e-3
    acla.predictor_threshold = 1e-2
    acla.gen_scores_threshold = 1
    acla.loss_auto_predictor_threshold = 1e-3
    acla.num_cpu = acla.validation_episodes
    acla.use_pain = True
    acla.patients = 1000
    
    seq_len = 100
    ac, acps, batch_size, seq_len, env, expert= setup_opt_state(device=device, batch_size=acla.batch_size, seq_len=seq_len)
    
    acps.opt_steps = 0
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
