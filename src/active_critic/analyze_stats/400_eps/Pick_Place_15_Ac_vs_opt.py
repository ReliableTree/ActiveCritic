from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    AC_path = '/home/hendrik/Documents/master_project/LokalData/server/AC/AC_var_2023-03-25/'

    include_AC = ['pickplace 1e-4 opr lr trainin eps: 400 opt mode: actor+plan demonstrations: 15, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.01, val_every: 6000', 'statsoptimized']
    exclude_AC = []

    include_AC_non = ['pickplace 1e-4 opr lr trainin eps: 400 opt mode: actor+plan demonstrations: 15, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.01, val_every: 6000', 'stats']
    exclude_AC_non = ['optimized']

    make_plot(
        paths=[AC_path, AC_path], 
        includes=[include_AC, include_AC_non], 
        excludes=[exclude_AC,exclude_AC_non],
        names=['AC optimized', 'AC non optimized'],
        plot_name='Pick and Place',
        save_path = None,
        plot_closest=False
        )