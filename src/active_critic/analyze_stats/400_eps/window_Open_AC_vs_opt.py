from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    AC_path = '/data/bing/hendrik/AC_var_2023-03-24/'

    include_AC = ['windowopenincr incr tiny planner trainin eps: 400 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.1, val_every: 6000', 'statsoptimized']
    exclude = []

    include_AC_non = ['windowopenincr incr tiny planner trainin eps: 400 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.1, val_every: 6000', 'stats']
    exclude_non = ['optimized']
    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Window_Open 4 400 AC vs opt/'

    make_plot(
        paths=[AC_path, AC_path], 
        includes=[include_AC, include_AC_non], 
        excludes=[exclude, exclude_non],
        names=['AC optimized', 'AC non optimized'],
        plot_name='Window Open',
        save_path = save_path,
        plot_closest=True
        )