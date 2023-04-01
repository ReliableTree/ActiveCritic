import argparse
from active_critic.server import *
from active_critic.server.analyze import run_eval_stats_env
from active_critic.server.init_reach import run_experiment_init_reach
from active_critic.server.reach_clip_only_last import run_experiment_reach_last

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str,
                    help='Choose free GPU')
    parser.add_argument('-demos', type=int,
                    help='demos')
    parser.add_argument('-wd', type=float,
                    help='weight decay')
    args = parser.parse_args()
    #run_experiment_analyze(device=args.device)
    #run_eval_stats(device=args.device, demos=args.demos, weight_decay=args.wd)
    run_eval_stats_env(device=args.device, weight_decay=1e-2)