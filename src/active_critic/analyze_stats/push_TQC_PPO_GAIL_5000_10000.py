from active_critic.analyze_stats.analyze_stats import make_plot

def run_exp():
    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/AC/AC_var_2023-02-23'
    bl_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-02-22/'

    demonstrations = 10

    include_ac = ['push', f'demonstrations: {demonstrations}','True', 'stats', '10000']
    exclude_ac = ['optimize']

    include_ac2 = ['push', f'demonstrations: {demonstrations}','False', 'stats', '10000']
    exclude_ac2 = ['optimize']

    include_ac3 = ['push', f'demonstrations: {demonstrations}','False', 'stats', '5000']
    exclude_ac3 = ['optimize']

    include_bl = [f'demonstrations_{demonstrations}', 'learner', 'push', 'PPO', '0.0001']
    exclude_bl = []

    include_bl_tqc = [f'demonstrations_{demonstrations}', 'learner', 'push', 'TQC', '1e-07']
    exclude_bl = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Push 10 lookup 5000 vs 10000 PPO + TQC Rinforcement/'

    make_plot(
        paths=[AC_path, AC_path, bl_path, bl_path], 
        includes=[include_ac2, include_ac3, include_bl, include_bl_tqc], 
        excludes=[exclude_ac, exclude_ac2, exclude_ac3, [], []],
        names=['AC 10000', 'AC 5000', 'PPO + GAIL', 'TQC + GAIL'],
        plot_name='Push Environment',
        save_path = save_path
        )