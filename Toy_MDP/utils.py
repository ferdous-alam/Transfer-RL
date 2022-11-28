import numpy as np
import matplotlib.pyplot as plt



def plot_q_values(Q_trained, q1, q2, q3, q4, q5, q6, fname):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    linewidth = 2.0
    plt.figure(figsize=(10, 8))
    plt.plot(q1, label=r'$Q(s_2, a_l)$', lw=linewidth)
    plt.plot(q2, label=r'$Q(s_2, a_r)$', lw=linewidth)
    plt.plot(q3, label=r'$Q(s_0, a_l)$', lw=linewidth)
    plt.plot(q4, label=r'$Q(s_0, a_r)$', lw=linewidth)
    plt.plot(q5, label=r'$Q(s_1, a_l)$', lw=linewidth)
    plt.plot(q6, label=r'$Q(s_1, a_r)$', lw=linewidth)
    plt.xlabel(r'episode', fontsize=20)
    plt.ylabel(r'action-value', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f'figures/q_values_{fname}.png')