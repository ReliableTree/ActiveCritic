import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from hashids import Hashids
import os
import torch as th

class TBoardGraphs():
    def __init__(self, logname= None, data_path = None):
        tf.autograph.set_verbosity(0)
        if logname is not None:
            self.__hashids           = Hashids()
            #self.logdir              = "Data/TBoardLog/" + logname + "/"
            self.logdir              = os.path.join(data_path, "gboard/" + logname + "/")
            self.__tboard_train      = tf.summary.create_file_writer(self.logdir + "train/")
            self.__tboard_validation = tf.summary.create_file_writer(self.logdir + "validate/")
            #self.voice               = Voice(path=data_path)
        self.fig, self.ax = plt.subplots(2,2)
        self.legend = None

    def startDebugger(self):
        tf.summary.trace_on(graph=True, profiler=True)
    
    def stopDebugger(self):
        with self.__tboard_validation.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=self.logdir)

    def finishFigure(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def addTrainScalar(self, name, value, stepid):
        with self.__tboard_train.as_default():

            tfvalue = self.torch2tf(value)
            tf.summary.scalar(name, tfvalue, step=stepid)
            self.__tboard_train.flush()

    def addValidationScalar(self, name, value, stepid):
        with self.__tboard_validation.as_default():
            tfvalue = self.torch2tf(value)
            tf.summary.scalar(name, tfvalue, step=stepid)
            self.__tboard_validation.flush()


    def torch2tf(self, inpt):
        if inpt is not None:
            return tf.convert_to_tensor(inpt.detach().cpu().numpy())
        else:
            return inpt

    def plot_graph(self, trjs:list([np.array]), trj_names:list([str]), trj_colors:list([str]), plot_name:str, step:int):
        fig, ax = self.fig, self.ax

        for sp in range(4):
            idx = sp // 2
            idy = sp  % 2
            ax[idx, idy].clear()
        
        for sp in range(trjs[0].shape[-1]):
            idx = sp // 2
            idy = sp  % 2
            
            ls = []

            for trj_num in range(len(trjs)):
                l, = ax[idx, idy].plot(range(trjs[trj_num].shape[0]), trjs[trj_num][:,sp], color=trj_colors[trj_num], label=trj_names[trj_num])
                ls.append(l)

        if self.legend is not None:
            self.legend.remove()
        self.legend = fig.legend(handles=ls)
        result = np.expand_dims(self.finishFigure(fig), 0)
        with self.__tboard_validation.as_default():
            tf.summary.image(plot_name, data=result, step=step)


    def plotDMPTrajectory(self, y_true, y_pred, y_pred_std = None, phase= None, \
        dt= None, p_dt= None, stepid= None, name = "Trajectory", save = False, \
            name_plot = None, path=None, tol_neg = None, tol_pos=None, inpt = None, opt_gen_trj=None, window = 0,\
                ):
        tf_y_true = self.torch2tf(y_true)
        tf_y_pred = self.torch2tf(y_pred)
        tf_phase = self.torch2tf(phase)
        if inpt is not None:
            tf_inpt = self.torch2tf(inpt)
        if p_dt is not None:
            tf_dt = self.torch2tf(dt)
            tf_p_dt = self.torch2tf(p_dt)
        if opt_gen_trj is not None:
            tf_opt_gen_trj = self.torch2tf(opt_gen_trj)
            tf_opt_gen_trj = tf_opt_gen_trj.numpy()

        tf_y_true      = tf_y_true.numpy()
        tf_y_pred      = tf_y_pred.numpy()
        if inpt is not None:
            tf_inpt        = tf_inpt.numpy()
        if tf_phase is not None:
            tf_phase       = tf_phase.numpy()

        if p_dt is not None:
            tf_dt          = tf_dt.numpy() * 350.0
            tf_p_dt        = tf_p_dt.numpy()
        trj_len      = tf_y_true.shape[0]
        
        #fig, ax = plt.subplots(3,3)
        fig, ax = self.fig, self.ax
        #fig.set_size_inches(9, 9)
        if tol_neg is not None:
            neg_inpt = tf_y_true + tol_neg[None,:].cpu().numpy()
            pos_inpt = tf_y_true + tol_pos[None,:].cpu().numpy()
        for sp in range(len(tf_y_true[0])):
            idx = sp // 2
            idy = sp  % 2
            ax[idx,idy].clear()

            # GT Trajectory:
            if tol_neg is not None:
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), neg_inpt[:,sp], alpha=0.75, color='orangered')
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), pos_inpt[:,sp], alpha=0.75, color='orangered')
            ax[idx,idy].plot(range(trj_len), tf_y_true[:,sp],   alpha=1.0, color='forestgreen')            
            ax[idx,idy].plot(range(tf_y_pred.shape[0]), tf_y_pred[:,sp], alpha=0.75, color='mediumslateblue')
            if opt_gen_trj is not None:
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), tf_opt_gen_trj[:,sp], alpha=0.75, color='lightseagreen')
                diff_vec = tf_opt_gen_trj - tf_y_pred
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), diff_vec[:,sp], alpha=0.75, color='pink')

            #ax[idx,idy].errorbar(range(tf_y_pred.shape[0]), tf_y_pred[:,sp], xerr=None, yerr=None, alpha=0.25, fmt='none', color='mediumslateblue')
            #ax[idx,idy].set_ylim([-0.1, 1.1])
            if p_dt is not None:
                ax[idx,idy].plot([tf_dt, tf_dt], [0.0,1.0], linestyle=":", color='forestgreen')

        if inpt is not None:
            ax[-1,-1].clear()
            ax[-1,-1].plot(range(inpt.shape[-1]), tf_inpt,   alpha=1.0, color='forestgreen')     
        
        if tf_phase is not None:
            ax[2,2].clear()
            ax[2,2].plot(range(tf_y_pred.shape[0]), tf_phase, color='orange')
        if p_dt is not None:
            ax[2,2].plot([tf_dt, tf_dt], [0.0,1.0], linestyle=":", color='forestgreen')
            ax[2,2].plot([tf_p_dt*350.0, tf_p_dt*350.0], [0.0,1.0], linestyle=":", color='mediumslateblue')
            ax[2,2].set_ylim([-0.1, 1.1])

        result = np.expand_dims(self.finishFigure(fig), 0)
        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name_plot + '.png')
        #fig.clear()
        #plt.close()
        if not save:
            with self.__tboard_validation.as_default():
                tf.summary.image(name, data=result, step=stepid)