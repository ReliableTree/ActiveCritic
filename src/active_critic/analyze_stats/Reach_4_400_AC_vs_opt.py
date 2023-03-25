from active_critic.analyze_stats.analyze_stats import make_plot


if __name__ == '__main__':
    AC_reach_path = '/data/bing/hendrik/AC_var_2023-03-24/'

    include_AC_reach = ['reachincr incr tiny planner trainin eps: 400 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.1, val_every: 6000', 'statsoptimized']
    exclude_AC_reach = []

    include_AC_reach_non = ['reachincr incr tiny planner trainin eps: 400 opt mode: actor+plan demonstrations: 4, im_ph:False, training_episodes: 10, min critic: 5e-05, wd: 0.1, val_every: 6000', 'stats']
    exclude_AC_reach_non = ['optimized']

    save_path = '/home/hendrik/Documents/master_project/LokalData/server/Stats for MA/Reach 4 400 AC vs opt/'


    make_plot(
        paths=[AC_reach_path, AC_reach_path], 
        includes=[include_AC_reach, include_AC_reach_non], 
        excludes=[exclude_AC_reach, exclude_AC_reach_non],
        names=['AC', 'AC non optimized'],
        plot_name='Reach Environment',
        save_path = save_path,
        plot_closest=False
        )