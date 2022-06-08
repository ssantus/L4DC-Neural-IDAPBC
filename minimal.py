import math
import tensorflow as tf
import numpy as np
from neuralnet import NN
from config import CONF
from visualization import * 
from training import *
from models import *


if __name__ == "__main__":
    simplependulum = False
    doublependulum = not simplependulum

    if simplependulum:
        print("Simple pendulum")

        config  = CONF(seed=123)
        config.model = Simplependulum(x_star=[1.3,0.])
        config.model.set_ja(0.)
        config.model.analytical = False
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

        plot_nn_energy(config.model.ha_fn, config.neuralnet, config)
        plot_multiple_energy(config.model.h_fn, config.model.ha_fn, config.model.hd_fn, config.neuralnet, config)
        plt.show()

    elif doublependulum:
        print("Double pendulum")
    else:
        print('CHOOSE A SYSTEM')
    tf.print('debug here')