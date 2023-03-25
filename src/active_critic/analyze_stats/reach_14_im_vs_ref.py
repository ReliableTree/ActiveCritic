from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/AC/AC_var_2023-03-07'
    bl_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-03-07'

    include_ac_reach_4_im = ['reach', 'stats', 'True', 'demonstrations: 4']
    exclude_ac_imitation = ['optimized']

    include_ac_reach_4_ref = ['reach', 'stats', 'False', 'demonstrations: 4']
    exclude_ac_imitation = ['optimized']

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Reach Imi vs. Ref 15/'

    make_plot(
        paths=[AC_path, AC_path], 
        includes=[include_ac_reach_4_im, include_ac_reach_4_ref], 
        excludes=[exclude_ac_imitation, exclude_ac_imitation],
        names=['AC Imitation Only', 'AC Reinforcement'],
        plot_name='Reach Environment',
        save_path = save_path,
        plot_closest=True
        )