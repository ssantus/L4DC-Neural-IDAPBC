"""visualization.py"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import cm, projections
import matplotlib as mpl
from matplotlib.ticker import LinearLocator

from config import CONF

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
    plt.show()


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
    surf = ax.plot_surface(q_plot, p_plot, energy_values1, linewidth = 0, alpha = 0.15, antialiased = False, label = name1, color = '#225375')
    surf = ax.plot_surface(q_plot, p_plot, energy_values2, linewidth = 0, alpha = 0.15, antialiased = False, label = name2, color = '#eb9234')
    surf = ax.plot_surface(q_plot, p_plot, energy_values3, linewidth = 0, antialiased = False, label = name3, cmap = cm.coolwarm)
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