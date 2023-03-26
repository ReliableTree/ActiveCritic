from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    AC_path = '/data/bing/hendrik/AC_var_2023-03-24/'
    include_AC = ['push linear critic tiny planner trainin eps: 405 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 15, min critic: 5e-05, wd: 0.1, val_every: 8000', 'statsoptimized']
    exclude_AC = []

    AC_path_non = '/data/bing/hendrik/AC_var_2023-03-24/'
    include_AC_non = ['push linear critic tiny planner trainin eps: 405 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 15, min critic: 5e-05, wd: 0.1, val_every: 8000', 'stats']
    exclude_AC_non = ['optimized']

    make_plot(
        paths=[AC_path, AC_path_non], 
        includes=[include_AC, include_AC_non], 
        excludes=[exclude_AC, exclude_AC_non],
        names=['AC optimized', 'AC non optimized'],
        plot_name='Reach Environment',
        save_path = None,
        plot_closest=False
        )
