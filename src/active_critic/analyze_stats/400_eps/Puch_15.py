from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':

    bl_path_RPPO_TQC = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-27'

    include_RPPO = ['RPPO_push_lr_1e-05_demonstrations_15', 'learner']
    exclude_RPPO = []

    include_TQC = ['TQC_GAIL_push_lr_1e-07_demonstrations_15', 'learner']
    exclude_TQC = []

    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/AC/AC_var_2023-03-25/'

    include_AC = ['push 1e-4 opr lr trainin eps: 400 opt mode: actor+plan demonstrations: 15, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.01, val_every: 6000', 'statsoptimized']
    exclude_AC = []

    PPO_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-26/'
    include_PPO = ['PPO_GAIL_push_lr_1e-05_demonstrations_15', 'learner']
    exclude_PPO = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Push 15 400/'


    make_plot(
        paths=[AC_path, PPO_path, bl_path_RPPO_TQC, bl_path_RPPO_TQC], 
        includes=[include_AC, include_PPO, include_TQC, include_RPPO], 
        excludes=[exclude_AC, exclude_PPO, exclude_TQC, exclude_RPPO],
        names=['AC', 'PPO + BC + GAIL', 'TQC + BC + GAIL', 'RPPO + BC'],
        plot_name='Push',
        save_path = save_path,
        plot_closest=False,
        mean=True
        )