"""config.py"""
import tensorflow as tf
import numpy as np

class CONF():
    def __init__(self, seed=123) -> None:
        self.set_seed(seed)
        self.model = None
        self.neuralnet = None

    def set_seed(self, seed):
        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

