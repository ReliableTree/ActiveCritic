import argparse
from active_critic.server.analyze import run_experiment_analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=str,
                    help='Choose free GPU')
    args = parser.parse_args()
    run_experiment_analyze(device=args.device)