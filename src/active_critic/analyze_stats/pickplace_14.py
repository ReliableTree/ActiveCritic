from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':

    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/pickplace 14/AC/'
    PPOGail = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/pickplace 14/PPO GAIL'
    TQCPath = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/pickplace 14/TQC GAIL'
    RPPOpath = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/pickplace 14/RPPO'

    include_ac = ['stats']
    exclude_ac = ['optimize']

    include_PPOGail = ['learner']
    exclude_bl = []

    include_TQC = ['learner', 'lr_1e-07']
    exclude_bl_TQC = []

    include_RPPO = ['learner']
    exclude_RPPO = ['id_2']

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/PickPlace 15/'


    make_plot(
        paths=[AC_path, PPOGail, TQCPath, RPPOpath], 
        includes=[include_ac, include_PPOGail, include_TQC, include_RPPO], 
        excludes=[exclude_ac, exclude_bl, exclude_bl_TQC, exclude_RPPO],
        names=['AC', 'PPO + BC + GAIL', 'TQC + BC + GAIL', 'RPPO + BC'],
        plot_name='Pick and Place Environment',
        save_path = save_path,
        plot_closest=True
        )