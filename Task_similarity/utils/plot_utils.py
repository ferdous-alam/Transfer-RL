import numpy as np
import matplotlib.pyplot as plt


def plot_task_similarity_score(means, stds, fig_name='test'):
    x_axis = [k for k in range(len(means))]
    plt.figure(figsize=(10, 8))
    plt.errorbar(x_axis, means, lw=2.0, linestyle='--', yerr=stds, 
                ecolor='black', color='black', capsize=10.0, capthick=2.0, 
                markeredgewidth=2.0)
    plt.scatter(x_axis, means, s=100)
    plt.xlabel('tasks', fontsize=18)
    plt.ylabel('similarity', fontsize=18)
    plt.xticks(x_axis, [f'$sim(T_0, T_{k})$' for k in range(len(means))], fontsize=8)    
    plt.savefig(f'figures/{fig_name}.png', dpi=300)
    plt.show()
    

