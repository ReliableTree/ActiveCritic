from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    bl_PPO_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-24'
    bl_TQC_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-25'
    AC_reach_path = '/data/bing/hendrik/AC_var_2023-03-24/'

    include_PPO_reach = ['learner', 'PPO_GAIL_reach_lr_1e-05_demonstrations_4']
    exclude_PPO_reach = []

    include_TQC_reach = ['learner', 'TQC_GAIL_reach', 'demonstrations_4']
    exclude_TQC_reach = ['5e-07_demonstrations_4_id_2']

    include_RPPO_reach = ['learner', 'RPPO_reach_lr_1e-06_demonstrations_4']
    exclude_RPPO_reach = ['id_3']

    include_AC_reach = ['reachincr incr tiny planner trainin eps: 400 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.1, val_every: 6000', 'statsoptimized']
    exclude_AC_reach = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Reach 4 400/'


    make_plot(
        paths=[AC_reach_path, bl_PPO_path, bl_TQC_path, bl_TQC_path], 
        includes=[include_AC_reach, include_PPO_reach, include_TQC_reach, include_RPPO_reach], 
        excludes=[exclude_AC_reach, exclude_PPO_reach, exclude_TQC_reach, exclude_RPPO_reach],
        names=['AC', 'PPO + BC + GAIL', 'TQC + BC + GAIL', 'RPPO + BC'],
        plot_name='Reach Environment',
        save_path = save_path,
        plot_closest=False
        )