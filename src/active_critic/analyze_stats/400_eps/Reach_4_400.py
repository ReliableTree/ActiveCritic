from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    bl_path_PPO = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-24'

    include_bl_PPO = ['learner', 'PPO_GAIL_reach_lr_1e-05_demonstrations_4']
    exclude_bl_PPO = ['range(0, 3)']

    bl_path_TQC = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-25'
    include_bl_TQC = ['learner', 'TQC_GAIL_reach_lr_1e-07_demonstrations_4']
    exclude_bl_TQC = []

    ac_path = '/data/bing/hendrik/AC_var_2023-03-24'

    include_AC = ['reachincr incr tiny planner trainin eps: 400 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.1, val_every: 6000', 
                'statsoptimized']
    exclude_AC = []


    RPPO_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-25/'
    include_RPPO = ['RPPO_reach_lr_1e-06_demonstrations_4', 'learner']
    exclude_RPPO = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Reach 4 400/'


    make_plot(
        paths=[ac_path, bl_path_PPO, bl_path_TQC, RPPO_path], 
        includes=[include_AC, include_bl_PPO, include_bl_TQC, include_RPPO], 
        excludes=[exclude_AC, exclude_bl_PPO, exclude_bl_TQC, exclude_RPPO],
        names=['AC', 'PPO + BC + GAIL', 'TQC + BC + GAIL', 'RPPO + BC'],
        plot_name='Reach',
        save_path = None,
        plot_closest=True
        )