from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    TQC_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-25/'
    include_TQC = ['TQC_GAIL_pickplace_lr_1e-07_demonstrations_4', 'learner']
    exclude_TQC = []

    PPO_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-24/'
    include_PPO = ['PPO_GAIL_pickplace_lr_1e-05_demonstrations_4', 'learner']
    exclude_PPO = []

    make_plot(
        paths=[TQC_path, PPO_path], 
        includes=[include_TQC, include_PPO], 
        excludes=[exclude_TQC, exclude_PPO],
        names=['PPO', 'TQC', 'AC', 'AC non'],
        plot_name='Reach Environment',
        save_path = None,
        plot_closest=False
        )