import torch as th
import matplotlib.pyplot as plt
import numpy as np

'''trj1 = th.rand([10, 4])
trj2 = th.rand([10, 4])
for sp in range(4):
    idx = sp // 3
    idy = sp  % 3
    ax[idx,idy].clear()
    l1, = ax[idx,idy].plot(range(trj1.shape[0]), trj1[:,sp], alpha=0.75, color='orangered', label='tr1' + str(sp))
    l2, = ax[idx,idy].plot(range(trj2.shape[0]), trj2[:,sp], alpha=0.75, color='green', label='tr2' + str(sp))
fig.legend(handles=[l1, l2])
plt.show()'''

def plot_graph(trjs:list([np.array]), trj_names:list([str]), trj_colors:list([str]), plot_name:str):
    fig, ax = plt.subplots(2,2)
    for sp in range(trjs[0].shape[-1]):
        idx = sp // 2
        idy = sp  % 2
        
        ls = []

        ax[idx, idy].clear()


        for trj_num in range(len(trjs)):
            l, = ax[idx, idy].plot(range(trjs[trj_num].shape[0]), trjs[trj_num][:,sp], color=trj_colors[trj_num], label=trj_names[trj_num])
            ls.append(l)
    legend = fig.legend(handles=ls)
    legend.remove()
    plt.show()

if __name__ == '__main__':
    trjs = np.random.rand(2, 10, 4)
    names = ['t1', 't2']
    colors = ['blue', 'orangered']
    #plot_graph(trjs=trjs, trj_names=names, trj_colors=colors, plot_name='')
    plot_graph(trjs=trjs[:1,:,:1], trj_names=names[:1], trj_colors=colors[:1], plot_name='')
