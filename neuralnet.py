"""neuralnet.py"""
import tensorflow as tf
import numpy as np


class NN(tf.keras.Model):

    def __init__(self, dim_in = 2, dim_out = 1, nn_width = 12, nn_depth = 1, activation = 'tanh', epochs = 5000) -> None:
        super(NN, self).__init__()

        # NN options
        self.nn_input_dim = dim_in
        self.set_nn_arch(nn_width, nn_depth)
        self.nn_output_dim = dim_out
        self.nn_activation = activation

        self.init_network()
        self.build((None, self.nn_input_dim))
        self.summary()

        # Training options
        self.epsilon = 0.
        self.epochs = epochs
        self.lr = 0.001 #learning rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
        self.train_tracker = tf.keras.metrics.Mean(name = 'train_tracker')
        self.validation_tracker = tf.keras.metrics.Mean(name = 'validation_tracker')
        self.print_period = 100
        self.print_residuals = True

    def set_nn_arch(self, nn_width:int, nn_depth:int):
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.nn_layers = [self.nn_width]*self.nn_depth

    def init_network(self):
        initializer = tf.keras.initializers.GlorotUniform()
        self.model = tf.keras.Sequential()
        for neurons in self.nn_layers:
            self.model.add(tf.keras.layers.Dense(neurons, activation = self.nn_activation, kernel_initializer = initializer))
        self.model.add(tf.keras.layers.Dense(self.nn_output_dim))

    def call(self, x):
        return self.model(x)