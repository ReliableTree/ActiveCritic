from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/AC/AC_var_2023-02-23'
    bl_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-02-22/'
    bl_RPPO_batch = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-21'

    include_ac = ['push demonstrations: 10, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.01, val_every: 5000', 'stats']
    exclude_ac = ['optimized']



    include_bl_PPO = ['demonstrations_10', 'learner', 'push', 'PPO', '0.0001']
    exclude_bl = []

    include_bl_TQC = ['TQC_GAIL_push_lr_1e-07_demonstrations_10', 'learner']
    exclude_bl = []

    include_bl_RPPO = ['RPPO_push_lr_1e-06_demonstrations_10_id', 'learner']
    exclude_bl = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Push 15/'

    make_plot(
        paths=[AC_path, bl_path, bl_path, bl_RPPO_batch], 
        includes=[include_ac, include_bl_PPO, include_bl_TQC, include_bl_RPPO], 
        excludes=[exclude_ac, exclude_bl, exclude_bl, exclude_bl],
        names=['AC', 'PPO + BC + GAIL', 'TQC + BC + GAIL', 'RPPO + BC'],
        plot_name='Push Environment',
        save_path = save_path,
        plot_closest=True
        )