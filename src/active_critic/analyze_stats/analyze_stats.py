import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def parse_data(paths, find_closest):
    result_dict = {}
    dicts = []
    for path in paths:
        with open(path, 'rb') as f:
            dicts.append(pickle.load(f))
    if find_closest:
        inter_dicts = []
        min_time_frame = float('inf')
        p_timesteps = None
        for dict in dicts:
            time_frame = dict['step'][-1] - dict['step'][0]
            if time_frame < min_time_frame:
                min_time_frame = time_frame
                p_timesteps = dict['step']

        for dict in dicts:
            inter_dict = {}
            timesteps = dict['step']
            dist_timesteps = (timesteps[None,:] - p_timesteps[:, None])**2
            ind_new_timesteps = np.argmin(dist_timesteps, axis=1)
            for key in dict:
                inter_dict[key] = dict[key][ind_new_timesteps]
            inter_dicts.append(inter_dict)
    else:
        inter_dicts = dicts
    for dict in inter_dicts:
        for key in dict:
            if (key != 'gen_actions') and (key != 'opt_actions'):
                next_entrance = dict[key].reshape([1, -1])
            else:
                next_entrance = dict[key]
            if key in result_dict:
                result_dict[key] = np.append(result_dict[key], next_entrance, axis=0)
            else:
                result_dict[key] = next_entrance
    return result_dict

def file_crawler(path, substrings, exclude=[]):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            if all(s in file_path for s in substrings) and not any(e in file_path for e in exclude):
                result.append(file_path)
    print(f'for path: {path}: {len(result)}')
    return result

def plot_experiment_data(
        timesteps, 
        experiments, 
        names, 
        plot_name, 
        mean,  
        path=None, 
        plot_closest=False, 
        loc=None, 
        total_plot_points=1, 
        mark_every = None,
        font_size = 14,
        legend_font_size=14,
        first_different_color = False,
        plot_legend=True):
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    # create figure and axis objects
    fig, ax = plt.subplots()

    # find the experiment with the smallest number of timesteps n_timesteps with timestep values p_timesteps
    min_timesteps = min([len(t) for t in timesteps])
    p_timesteps = timesteps[[len(t) for t in timesteps].index(min_timesteps)]
    # for all other experiments, find the n_timesteps timesteps, that are the closest to p_timesteps

    new_experiments = []
    if plot_closest:
        for exp in range(len(experiments)):
            if timesteps[exp][0] != 0:
                experiments[exp] = np.concatenate((np.zeros_like(experiments[exp][:,:1]), experiments[exp]), axis=1)
                timesteps[exp] = np.concatenate((np.zeros_like(timesteps[exp][:1]), timesteps[exp]), axis=0)
            dist_timesteps = (timesteps[exp][None,:] - p_timesteps[:, None])**2
            ind_new_timesteps = np.argmin(dist_timesteps, axis=1)
            new_experiments.append(experiments[exp][:, ind_new_timesteps])
    else:
        new_experiments = experiments
            

    # loop over experiments
    for i, experiment in enumerate(new_experiments):
        # calculate mean and standard deviation of each time step for this experiment
        mean_data = np.mean(experiment, axis=0)
        std_data = np.std(experiment, axis=0)

        if mean:
            std_data = 1 / np.sqrt(experiment.shape[0]) * std_data

        # plot mean data as a line and shade area between ±1 standard deviation
        if plot_closest:
            if (mark_every is not None) and (total_plot_points is not None):
                print(f'mark_every: {mark_every},  total_plot_points: {total_plot_points}')
                1/0
            if total_plot_points is not None:
                mark_every_calc = math.ceil(len(p_timesteps) / total_plot_points)
            else:
                mark_every_calc = mark_every
            p_timesteps_steps = np.arange(p_timesteps.shape[0])

            # plot the experiments at those timesteps
            use_steps = p_timesteps_steps % mark_every_calc == 0
            if first_different_color:
                ax.plot(p_timesteps[use_steps], mean_data[use_steps], label=names[i], color=colors[i])
                ax.fill_between(p_timesteps, np.maximum(mean_data-std_data, 0), np.minimum(mean_data+std_data, 1), alpha=0.3, color=colors[i])
            else:
                ax.plot(p_timesteps[use_steps], mean_data[use_steps], label=names[i])
                ax.fill_between(p_timesteps, np.maximum(mean_data-std_data, 0), np.minimum(mean_data+std_data, 1), alpha=0.3)
        else:
            if mark_every is not None and total_plot_points is not None:
                print(f'mark_every: {mark_every},  total_plot_points: {total_plot_points}')
                1/0
            if total_plot_points is not None:
                mark_every_calc = math.ceil(len(timesteps[i]) / total_plot_points)
            else:
                mark_every_calc = mark_every
            timesteps_steps = np.arange(timesteps[i].shape[0])

            use_steps = timesteps_steps % mark_every_calc == 0
            if first_different_color:
                ax.plot(timesteps[i][use_steps], mean_data[use_steps], label=names[i], markevery=1, color=colors[i])
                ax.fill_between(timesteps[i],np.maximum(mean_data-std_data, 0), np.minimum(mean_data+std_data, 1), alpha=0.3, color=colors[i])
            else:
                ax.plot(timesteps[i][use_steps], mean_data[use_steps], label=names[i], markevery=1)
                ax.fill_between(timesteps[i],np.maximum(mean_data-std_data, 0), np.minimum(mean_data+std_data, 1), alpha=0.3)

    # add labels, title, and legend to the plot
    ax.set_xlabel('Sampled Trajectories', fontsize=font_size)
    ax.set_ylabel('Success Rate', fontsize=font_size)
    ax.set_title(plot_name, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    if plot_legend:
        if loc is None:
            ax.legend(fontsize=legend_font_size)
        else:
            ax.legend(loc = loc, fontsize=legend_font_size)
    if path is not None:
        # create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # save the plot
        plt.savefig(os.path.join(path, plot_name + '.png'), bbox_inches='tight')
def make_plot(
        paths, 
        includes, 
        excludes, 
        names, 
        plot_name, 
        save_path = None, 
        plot_closest=False, 
        mean = True, 
        find_closest = False, 
        loc=None,
        total_plot_points=1,
        mark_every = None,
        font_size = 14,
        legend_font_size=14,
        first_different_color=False,
        plot_legend=True):
    abs_file_path_list = []
    
    for i in range(len(paths)):
        abs_file_path_list.append(file_crawler(path=paths[i], substrings=includes[i], exclude=excludes[i]))
    dicts = []
    
    for result in abs_file_path_list:
        dicts.append(parse_data(paths=result, find_closest=find_closest))

    print(dicts)

    plot_experiment_data(
        timesteps=[result_dict['step'][0] for result_dict in dicts], 
        experiments=[result_dict['success_rate'] for result_dict in dicts],
        names=names,
        plot_name=plot_name,
        path=save_path,
        plot_closest=plot_closest,
        mean=mean,
        loc=loc,
        total_plot_points=total_plot_points,
        mark_every=mark_every,
        font_size=font_size,
        legend_font_size=legend_font_size,
        first_different_color=first_different_color,
        plot_legend=plot_legend
        )
    
def plot_actions(
        paths, 
        includes, 
        excludes, 
        experiment_num, 
        time_step, 
        save_path, 
        title, 
        legend_fontsize=12, 
        label_fontsize=14, 
        xlabel="Time Step", 
        ylabel="Action",
        tick_size = 1):
    abs_file_path_list = []

    for i in range(len(paths)):
        abs_file_path_list.append(file_crawler(path=paths[i], substrings=includes[i], exclude=excludes[i]))
    dict_list = []

    for result in abs_file_path_list:
        dict_list.append(parse_data(paths=result, find_closest=False))
    gen_actions = dict_list[experiment_num]["gen_actions"]
    opt_actions = dict_list[experiment_num]["opt_actions"]
    
    T, d_a = gen_actions.shape[1:]
    
    fig, ax = plt.subplots(2, 2, sharex=True)

    for i in range(2):
        for j in range(2):
            index = i*2 + j
            
            gen_action = gen_actions[time_step, :, index]
            opt_action = opt_actions[time_step, :, index]
            
            ax[i, j].plot(range(T), gen_action, label="Generated")
            ax[i, j].plot(range(T), opt_action, label="Optimized")
            ax[i, j].set_ylabel(f"{ylabel} {index}", fontsize=label_fontsize)
            ax[i, j].tick_params(axis="both", which="major", labelsize=tick_size)
            if j == 0:
                ax[i, j].yaxis.set_label_coords(-0.2, 0.5)
            else:
                ax[i, j].yaxis.set_label_coords(-0.2, 0.5)
            
            if i == 1:
                ax[i, j].set_xlabel(xlabel, fontsize=label_fontsize)

    plt.suptitle(title, fontsize=label_fontsize, y=1)

    fig.legend(["Generated", "Optimized"], loc="lower center", ncol=2, fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, -0.15))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, title), bbox_inches='tight')