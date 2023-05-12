
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# smooth curves based on moving window
# def smooth(y, window=10):
#     box = np.ones(window)/window
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth

def smooth(y, window=11, poly_order=3):
    y_smooth = savgol_filter(y, window, poly_order)
    return y_smooth


# get statistics of saved arrays
def get_stat(*args, window=10):
    """
    each input argument is in shape (100, 10)
    """
    means, stds = [], []
    for k in args:
        mean = smooth(np.mean(k, 1), window=window)
        std = smooth(np.std(k, 1), window=window)
        means.append(mean)
        stds.append(std)
    return means, stds


def plot_avg_ret(*args, baselines=None, fig_name=None, **kwargs):

    
    sac_means = kwargs['sacs']['means']
    sac_stds = kwargs['sacs']['stds']
        
    # params
    fmt = kwargs['fmt']
    labels = kwargs['labels']
    colors = kwargs['colors']
    num_sub_plots = len(sac_means)   # number of subplots = number of tasks 
    capsize = 3.0
    line_width = 2.0

    plt.figure(figsize=(21, 3))
    plt.rcParams['axes.linewidth'] = 2.0 

    for k in range(num_sub_plots):
        plt.subplot(1, num_sub_plots, k+1)
        x = [j for j in range(len(sac_means[0]))]

        # plot algo stat
        for m in range(len(args)):
                plt.errorbar(x[::5], args[m]['means'][k][::5], yerr=args[m]['stds'][k][::5], fmt=fmt[m], capsize=capsize, color=colors[m], label=labels[m])
                plt.plot(x[::5], args[m]['means'][k][::5], lw=line_width, color=colors[m])

        if baselines is not None:
            plt.hlines(baselines[k], 0, len(x), lw=line_width, color='black', linestyles='--', label='zero-shot')


        # plot sac baseline stat
        plt.errorbar(x[::5], sac_means[k][::5], yerr=sac_stds[k][::5], fmt=fmt[-1], capsize=capsize, color=colors[-1], label=labels[-1])
        plt.plot(x[::5], sac_means[k][::5], lw=line_width, color=colors[-1])
        
        # plot properties
        plt.legend(loc='lower right')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel(r'timesteps', fontsize=18)
        plt.ylabel(r'$\rho_t$', fontsize=30)
        plt.legend(loc='lower right', fontsize=14)
        plt.ylim([-1780, 1])

    if fig_name is not None:
        plt.savefig(f'figures/{fig_name}.png', bbox_inches="tight")

        


def plot_adv(*args, fig_name=None, **kwargs):
    means = kwargs['tau_stat']['means']
    stds = kwargs['tau_stat']['stds']

    labels = kwargs['labels']
    colors = kwargs['colors']
    line_width = 3.0
    error_line_width = 2.0
    markersize = 5.0

    plt.figure(figsize=(21, 3))

    # subplot 1 ---> this is the plot for: tau vs task similarity 
    plt.subplot(1, 3, 1)
    for m in range(len(means)):
        plt.plot(means[m][::5], marker='o', markersize=markersize, lw=line_width, color=colors[m], label=labels[m])            
        plt.legend(loc='upper right', fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(r'timesteps', fontsize=18)
        plt.ylabel(r'$\tau_t$', fontsize=30)

    # subplot 2 ---> this is the plot for beta vs task similarity 
    plt.subplot(1, 3, 2)
    plt.hlines(1.0, 0, len(args[0][100:-100]), color='black', lw=line_width, linestyles='--')
    for m in range(3): 
        plt.plot(args[m][100:-100], label=labels[m], color=colors[m])
    plt.legend(loc='upper right', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'timesteps', fontsize=18)
    plt.ylabel(r'$\tau_t$', fontsize=30)

    # subplot 3 ---> this is the plot for beta vs task similarity 
    plt.subplot(1, 3, 3)
    plt.hlines(1.0, 0, len(args[3][100:-100]), color='black', lw=line_width, linestyles='--')
    for m in range(3): 
        plt.plot(args[3+m][100:-100], label=labels[m], color=colors[m])
        plt.legend(loc='upper right', fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(r'timesteps', fontsize=18)
        plt.ylabel(r'$\tau_t$', fontsize=30)

    if fig_name is not None:
        plt.savefig(f'figures/{fig_name}.png', bbox_inches="tight")
