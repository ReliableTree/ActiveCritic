from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    pp_AC_path = '/data/bing/hendrik/AC_var_2023-03-24/'

    include_AC_pp = ['windowopenincr incr tiny planner trainin eps: 400 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.1, val_every: 6000', 'statsoptimized']
    exclude_AC_pp = []

    bl_path_PPO = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-24'
    include_PPO = ['PPO_GAIL_windowopen_lr_1e-05_demonstrations_4', 'learner']
    exculde_PPO = []

    bl_path_TQC = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-25'
    include_TQC = ['TQC_GAIL_windowopen_lr_1e-07_demonstrations_4', 'learner']
    exculde_TQC = []

    bl_path_RPPO = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-25'
    inlcude_RPPO = ['RPPO_windowopen_lr_1e-06_demonstrations_4', 'learner']
    exculde_RPPO = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/WindowOpen 4 400/'

    make_plot(
        paths=[pp_AC_path, bl_path_PPO, bl_path_TQC, bl_path_RPPO], 
        includes=[include_AC_pp, include_PPO, include_TQC, inlcude_RPPO], 
        excludes=[exclude_AC_pp, exculde_PPO, exculde_TQC, exculde_RPPO],
        names=['AC', 'PPO + BC + GAIL', 'TQC + BC + GAIL', 'RPPO + BC'],
        plot_name='Window Open',
        save_path = None,
        plot_closest=False
        )