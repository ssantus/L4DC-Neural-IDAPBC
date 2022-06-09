"""config.py"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class CONF():
    def __init__(self, seed=123) -> None:
        self.set_seed(seed)
        self.model = None
        self.neuralnet = None

        # Plot config
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.rcParams['font.size'] = 13
        plt.rcParams['font.weight'] = 'normal'

    def set_seed(self, seed):
        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

