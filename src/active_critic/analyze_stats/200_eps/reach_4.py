from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/AC/AC_var_2023-03-07'
    bl_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-07'

    include_ac_reach_4_ref = ['reach', 'stats', 'False', 'demonstrations: 4']
    exclude_ac_imitation = ['optimized']

    include_PPO_reach_4 = ['reach', 'demonstrations_4', 'PPO', 'learner_stats_gail']
    exclude_bl_PPO = ['RPPO']

    include_RPPO_reach_4 = ['reach', 'demonstrations_4', 'RPPO', 'learner_stats_rec_PPO_stepsize_10']
    exclude_bl_RPPO = []

    include_TQC_reach_4 = ['reach', 'demonstrations_4', 'TQC', 'learner_stats_gail_stepsize_10']
    exclude_bl_RPPO = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Reach 4 200/'

    make_plot(
        paths=[AC_path, bl_path, bl_path, bl_path], 
        includes=[include_ac_reach_4_ref, include_PPO_reach_4, include_TQC_reach_4, include_RPPO_reach_4], 
        excludes=[exclude_ac_imitation, exclude_bl_PPO, exclude_bl_RPPO, exclude_bl_RPPO],
        names=['AC', 'PPO + BC + GAIL', 'TQC + BC + GAIL', 'RPPO + BC'],
        
        plot_name='Reach Environment',
        save_path = save_path,
        plot_closest=True
        )