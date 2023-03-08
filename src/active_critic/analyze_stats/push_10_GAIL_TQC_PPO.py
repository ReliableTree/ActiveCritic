from active_critic.analyze_stats.analyze_stats import make_plot

def run_exp():
    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/AC/AC_var_2023-02-23'
    bl_path = '/home/hendrik/Documents/master_project/LokalData/server/baselines/Baselines_Stats_GAIL_2023-02-22/'

    demonstrations = 10

    include_ac = ['push', f'demonstrations: {demonstrations}','True', 'stats', '10000']
    exclude_ac = ['optimize']

    include_ac3 = ['push', f'demonstrations: {demonstrations}','False', 'stats', '5000']
    exclude_ac3 = ['optimize']

    include_bl001 = [f'demonstrations_{demonstrations}', 'learner', 'push', 'PPO', '0.001']
    exclude_bl = []

    include_bl_0001 = [f'demonstrations_{demonstrations}', 'learner', 'push', 'PPO', '0.0001']
    exclude_bl = []

    include_bl_tqc07 = [f'demonstrations_{demonstrations}', 'learner', 'push', 'TQC', '1e-07']
    exclude_bl = []

    include_bl_tqc06 = [f'demonstrations_{demonstrations}', 'learner', 'push', 'TQC', '1e-06']
    exclude_bl = []

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Push 10 AC vs GAIl PPO vs GAIL TQC/'

    make_plot(
        paths=[AC_path, bl_path, bl_path, bl_path, bl_path], 
        includes=[include_ac3, include_bl001, include_bl_0001, include_bl_tqc07, include_bl_tqc06], 
        excludes=[exclude_ac, exclude_bl, exclude_ac3, [], []],
        names=['AC', 'PPO + GAIL lr 1e-3', 'PPO + GAIL lr 1e-4', 'TQC + GAIL lr 1e-6', 'TQC + GAIL lr 1e-7'],
        plot_name='Push Environment',
        save_path = save_path
        )