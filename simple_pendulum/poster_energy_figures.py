"""visualization.py"""
import os
import platform
pc = platform.system()
w = False
if pc == 'Windows':
    import sys
    sys.path.insert(1, os.getcwd())
    w = True
    
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import cm, projections
import matplotlib as mpl
from matplotlib.ticker import LinearLocator

from utils.config import CONF

def plot_energy(energy, config: CONF, name: str = 'insertname'):
    """Surface plot of the energy function"""

    #Preparing datapoints
    q_plot = np.arange(-config.q_lim + config.q_star, config.q_star + config.q_lim, config.delta)
    p_plot = np.arange(-config.p_lim + config.p_star, config.p_star + config.p_lim, config.delta)
    
    q_plot, p_plot = np.meshgrid(q_plot, p_plot)
    q = np.expand_dims(q_plot.flatten(), axis=-1)
    p = np.expand_dims(p_plot.flatten(), axis=-1)

    energy_values = energy(tf.concat((q, p), axis = 1), config)
    max_energy = max(energy_values)
    energy_values = tf.reshape(energy_values, q_plot.shape)

    #Plotting surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(q_plot, p_plot, energy_values, linewidth = 0, antialiased = False, label = name, color = '#eb9234')
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", color='#eb9234', marker='o')
    ax.scatter(config.q_star, config.p_star, 0, marker='o', c='red')

    #Customizing axis.
    ax.set_xlim(-config.q_lim + config.q_star, config.q_star + config.q_lim)
    ax.set_ylim(-config.p_lim + config.p_star, config.p_star + config.p_lim)
    ax.set_zlim(-max_energy, max_energy)
    ax.zaxis.set_major_locator(LinearLocator(10))

    #Adding labels
    plt.xlabel("q")
    plt.ylabel("p")
    ax.legend([fake2Dline], [name], numpoints=1, loc='upper left', fontsize=13)
    plt.title(name + " energy plot", fontsize=13)

    #Show plot
    # plt.show()


def plot_nn_energy(energy, neuralnet, config: CONF, name: str = 'insertname'):
    """Surface plot of the energy function"""
    config = config.model

    #Preparing datapoints
    q_plot = np.arange(-config.q_lim + config.q_star, config.q_star + config.q_lim, config.delta, dtype=np.float32)
    p_plot = np.arange(-config.p_lim + config.p_star, config.p_star + config.p_lim, config.delta, dtype=np.float32)
    
    q_plot, p_plot = np.meshgrid(q_plot, p_plot)
    q = np.expand_dims(q_plot.flatten(), axis=-1)
    p = np.expand_dims(p_plot.flatten(), axis=-1)

    energy_values = energy(tf.concat((q, p), axis = 1), neuralnet)
    max_energy = max(energy_values)
    energy_values = tf.reshape(energy_values, q_plot.shape)

    #Plotting surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(q_plot, p_plot, energy_values, linewidth = 0, antialiased = False, label = name, color = '#eb9234')
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", color='#eb9234', marker='o')
    ax.scatter(config.q_star, config.p_star, 0, marker='o', c='red')

    #Customizing axis.
    ax.set_xlim(-config.q_lim + config.q_star, config.q_star + config.q_lim)
    ax.set_ylim(-config.p_lim + config.p_star, config.p_star + config.p_lim)
    ax.set_zlim(-max_energy, max_energy)
    ax.zaxis.set_major_locator(LinearLocator(10))

    #Adding labels
    plt.xlabel("q")
    plt.ylabel("p")
    ax.legend([fake2Dline], [name], numpoints=1, loc='upper left', fontsize=13)
    plt.title(name + " energy plot", fontsize=13)

    #Show plot
    # plt.show()


def plot_multiple_energy(energy1, energy2, energy3, neuralnet, config: CONF, name1: str = 'insertname1', name2: str = 'insertname2', name3: str = 'insertname3'):
    """Surface plot of the energy function"""
    config = config.model

    #Preparing datapoints
    q_plot = np.arange(-config.q_lim + config.q_star, config.q_star + config.q_lim, config.delta, dtype=np.float32)
    p_plot = np.arange(-config.p_lim + config.p_star, config.p_star + config.p_lim, config.delta, dtype=np.float32)
    
    q_plot, p_plot = np.meshgrid(q_plot, p_plot)
    q = np.expand_dims(q_plot.flatten(), axis=-1)
    p = np.expand_dims(p_plot.flatten(), axis=-1)

    energy_values1 = energy1(tf.concat((q, p), axis = 1), neuralnet)
    energy_values2 = energy2(tf.concat((q, p), axis = 1), neuralnet)
    energy_values3 = energy3(tf.concat((q, p), axis = 1), neuralnet)
    max_energy = max(energy_values3)
    energy_values1 = tf.reshape(energy_values1, q_plot.shape)
    energy_values2 = tf.reshape(energy_values2, q_plot.shape)
    energy_values3 = tf.reshape(energy_values3, q_plot.shape)

    #Plotting surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(q_plot, p_plot, energy_values1, linewidth = 0, alpha = 0.15, antialiased = False, label = name1, color = '#225375', zorder=0)
    surf = ax.plot_surface(q_plot, p_plot, energy_values2, linewidth = 0, alpha = 0.15, antialiased = False, label = name2, color = '#eb9234', zorder=1)
    surf = ax.plot_surface(q_plot, p_plot, energy_values3, linewidth = 0, antialiased = False, label = name3, cmap = cm.coolwarm, zorder=2)
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", color='#225375', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", color='#eb9234', marker='o')
    fake2Dline3 = mpl.lines.Line2D([0], [0], linestyle="none", color='#f57373', marker='o')
    ax.scatter(config.q_star, config.p_star, 0, marker='o', c='red')

    #Customizing axis.
    ax.set_xlim(-config.q_lim + config.q_star, config.q_star + config.q_lim)
    ax.set_ylim(-config.p_lim + config.p_star, config.p_star + config.p_lim)
    ax.set_zlim(-max_energy, max_energy)
    ax.zaxis.set_major_locator(LinearLocator(10))

    #Adding labels
    plt.xlabel("q")
    plt.ylabel("p")
    ax.legend([fake2Dline1, fake2Dline2, fake2Dline3], [name1, name2, name3], numpoints=1, loc='upper left', fontsize=13)
    plt.title("Multiple energy plot", fontsize=13)

    #Show plot
    # plt.show()



def plot_energy_poster(energy1, energy2, energy3, neuralnet, config: CONF, name1: str = 'insertname1', name2: str = 'insertname2', name3: str = 'insertname3'):
    """Surface plot of the energy function"""
    config = config.model

    #Preparing datapoints
    q_plot = np.arange(-config.q_lim + config.q_star, config.q_star + config.q_lim, config.delta, dtype=np.float32)
    p_plot = np.arange(-config.p_lim + config.p_star, config.p_star + config.p_lim, config.delta, dtype=np.float32)
    
    q_plot, p_plot = np.meshgrid(q_plot, p_plot)
    q = np.expand_dims(q_plot.flatten(), axis=-1)
    p = np.expand_dims(p_plot.flatten(), axis=-1)

    energy_values1 = energy1(tf.concat((q, p), axis = 1), neuralnet)
    energy_values2 = energy2(tf.concat((q, p), axis = 1), neuralnet)
    energy_values3 = energy3(tf.concat((q, p), axis = 1), neuralnet)

    max_energy = max(energy_values3)
    energy_values1 = tf.reshape(energy_values1, q_plot.shape)
    energy_values2 = tf.reshape(energy_values2, q_plot.shape)
    energy_values3 = tf.reshape(energy_values3, q_plot.shape)


    #Plotting surface
    fig, axs = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(12, 5))
    axs[0].plot_surface(q_plot, p_plot, energy_values1, linewidth = 0, alpha = 1., antialiased = False, label = name1, color = '#1911df', zorder=0)
    # leg = axs[0].legend(loc=0, ncol=2, shadow=True, fancybox=True)
    axs[1].plot_surface(q_plot, p_plot, energy_values1, linewidth = 0, alpha = 0.5, antialiased = False, label = name1, color = '#1911df', zorder=0)
    axs[1].plot_surface(q_plot, p_plot, energy_values2, linewidth = 0, alpha = 0.5, antialiased = False, label = name2, color = '#f3874a', zorder=10)
    # leg = axs[1].legend(loc=0, ncol=2, shadow=True, fancybox=True)
    axs[2].plot_surface(q_plot, p_plot, energy_values3, linewidth = 0, alpha = 1.0 , antialiased = False, label = name3, cmap = cm.plasma, zorder=-10000)
    # leg = axs[2].legend(loc=0, ncol=2, shadow=True, fancybox=True)

    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", color='#1911df', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", color='#f3874a', marker='o')
    fake2Dline3 = mpl.lines.Line2D([0], [0], linestyle="none", color='#f34a58', marker='o')

    for i in range(3):
        axs[i].set_axis_off()
        axs[i].view_init(elev=5., azim=-52)   


    axs[0].legend([fake2Dline1], [name1], numpoints=1, loc='upper right', fontsize=13, frameon=False, bbox_to_anchor=(1.04, 0.85))
    axs[1].legend([fake2Dline1, fake2Dline2], [name1, name2], numpoints=1, loc='upper right', fontsize=13, frameon=False, bbox_to_anchor=(1.04, 0.85))
    axs[2].legend([fake2Dline3], [name3], numpoints=1, loc='upper right', fontsize=13, frameon=False, bbox_to_anchor=(1.04, 0.85))

    return fig, axs


def plot_energy_poster2(energy1, energy2, neuralnet, config: CONF, name1: str = 'insertname1',
                       name2: str = 'insertname2', name3: str = 'insertname3'):
    """Surface plot of the energy function"""
    config = config.model

    # Preparing datapoints
    q_plot = np.arange(-config.q_lim + config.q_star, config.q_star + config.q_lim, config.delta, dtype=np.float32)
    p_plot = np.arange(-config.p_lim + config.p_star, config.p_star + config.p_lim, config.delta, dtype=np.float32)

    q_plot, p_plot = np.meshgrid(q_plot, p_plot)
    q = np.expand_dims(q_plot.flatten(), axis=-1)
    p = np.expand_dims(p_plot.flatten(), axis=-1)

    energy_values1 = energy1(tf.concat((q, p), axis=1), neuralnet)
    energy_values2 = energy2(tf.concat((q, p), axis=1), neuralnet)

    max_energy = max(energy_values2)
    energy_values1 = tf.reshape(energy_values1, q_plot.shape)
    energy_values2 = tf.reshape(energy_values2, q_plot.shape)

    # Plotting surface
    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))
    axs[0].plot_surface(q_plot, p_plot, energy_values1, linewidth=0, alpha=1., antialiased=False, label=name1,
                        color='#1911df', zorder=0)
    # leg = axs[1].legend(loc=0, ncol=2, shadow=True, fancybox=True)
    axs[1].plot_surface(q_plot, p_plot, energy_values2, linewidth=0, alpha=1.0, antialiased=False, label=name2,
                        cmap=cm.plasma, zorder=-10000)
    # leg = axs[2].legend(loc=0, ncol=2, shadow=True, fancybox=True)

    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", color='#1911df', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", color='#f34a58', marker='o')

    for i in range(2):
        # axs[i].set_axis_off()
        axs[i].view_init(elev=5., azim=-70)
        axs[i].set_xlabel('$q$', labelpad=10)
        axs[i].set_ylabel('$p$', labelpad=10)


    axs[0].legend([fake2Dline1], [name1], numpoints=1, loc='upper right', fontsize=13, frameon=False,
                  bbox_to_anchor=(1.04, 0.85))
    axs[1].legend([fake2Dline2], [name2], numpoints=1, loc='upper right', fontsize=13, frameon=False,
                  bbox_to_anchor=(1.04, 0.85))

    return fig, axs

def plot_error(training_loss, validation_loss, config:CONF):
    fig, ax = plt.subplots()
    line1 = ax.plot(training_loss, label = 'Training loss')
    line2 = ax.plot(validation_loss, label = 'Validation loss')
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=13)

    plt.xlabel("Epoch * {}".format(config.neuralnet.print_period), fontsize=13)
    plt.ylabel('Error', fontsize=13)
    plt.title('Training/Validation Error', fontsize=13)
    
    #Show plot
    # plt.show()


if __name__ == '__main__':
    # Plot figure for poster
    from Simplependulum import *
    from utils.neuralnet import *
    print("Simple pendulum")

    config  = CONF(seed=123)
    config.model = Simplependulum(x_star=[1.3,0.])
    config.model.set_ja(0.)
    config.model.analytical = True
    config.neuralnet = NN(epochs= 15000, nn_width=60)
    # config.neuralnet = NN(epochs= 10000, activation = lambda x: x**2)
    config.neuralnet.epsilon = 0.1

    x_train, x_test = config.model.data_gen_uniform()
    # n_samples = config.neuralnet.model.count_params()
    # x_train, x_test = config.model.generate_data_random(n_samples, config.seed)

    # tf.print("Residuals before training: ")
    # loss_fn(x_test[0], config, residuals = True)
    if not config.model.analytical:
        t_loss, v_loss = train_fn(x_train, x_test, config)
        plot_error(t_loss, v_loss, config)

    config.model.q_lim = 8
    config.model.delta = 0.1
    # plot_nn_energy(config.model.ha_fn, config.neuralnet, config, name='$H_a(x)$')

    # Plot energy poster:
    flag = False
    if flag:
        fig, axs = plot_energy_poster(config.model.h_fn, config.model.ha_fn, config.model.hd_fn, config.neuralnet, config, name1='$H(x)$', name2='$H_a(x)$', name3='$H_d(x)$')

        axs[2].scatter(config.model.q_star, config.model.p_star, 0., marker='*', c='red', edgecolors='#1911df',linewidths=1., zorder=10000, s=220)
        axs[2].text(config.model.q_star+0.6, config.model.p_star+0.6, -7.5, '$x^\star$', fontsize=15)
        axs[2].computed_zorder = False

    # Plot energy poster2:
    fig, axs = plot_energy_poster2(config.model.h_fn, config.model.hd_fn, config.neuralnet, config, name1='$H(x)$', name2='$H_a(x)$', name3='$H_d(x)$')
    for i in range(2):
        axs[i].scatter(config.model.q_star, config.model.p_star, 0., marker='*', c='red', edgecolors='#1911df',
                       linewidths=1., zorder=10000, s=220)
        axs[i].computed_zorder = False

    axs[0].text(config.model.q_star + 0.6, config.model.p_star + 0.6, -2.8, '$x^\star$', fontsize=15)
    axs[1].text(config.model.q_star + 0.6, config.model.p_star + 0.6, -7.5, '$x^\star$', fontsize=15)

    image_format = 'svg'
    image_name = 'energy_poster2.svg'
    plt.tight_layout()
    if w:
        fig.savefig('c:\\Users\\ssanc\\Documents\\GitHub\\Total-Energy-Shaping-Neural-IDAPBC\\figures\\test'+image_name, format = image_format, dpi=1200, transparent = True)
    else:
        fig.savefig('../figures/'+image_name, format = image_format, dpi=1200, transparent = True)
    plt.show()