import argparse
from active_critic.server import *
from active_critic.server.analyze import run_experiment_analyze
from active_critic.server.init_reach import run_experiment_init_reach
from active_critic.server.reach_clip_only_last import run_experiment_reach_last
from active_critic.server.continuous import run_reach_learn_mdp

if __name__ == '__main__':
    path = '/data/bing/hendrik/'
    #path = '/home/hendrik/Documents/master_project/LokalData/'
    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=str,
                    help='Choose free GPU')
    args = parser.parse_args()
    run_reach_learn_mdp(path=path, device=args.device)
