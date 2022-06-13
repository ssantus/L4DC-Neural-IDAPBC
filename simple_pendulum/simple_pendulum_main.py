"""main.py"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.visualization import *
from simple_pendulum_training import *
from simple_pendulum_solver import *
from Simplependulum import *


if __name__ == "__main__":
    print("Simple pendulum")
    # Setup
    config  = CONF(seed=123)
    config.model = Simplependulum(x_star=[1.5,0.])
    config.model.set_ja(0.)
    config.model.set_ra(0.7)
    config.model.analytical = False
    config.neuralnet = NN(epochs= 15000, nn_width=60)
    config.neuralnet.epsilon = 0.1

    # Data
    x_train, x_test = config.model.data_gen_uniform()

    # Train
    train = False
    if not config.model.analytical:
        name = 'simple_pendulum_width{}_depth{}'.format(config.neuralnet.nn_width, config.neuralnet.nn_depth)
        directory_name = 'weights/L4DC/'+name+'.npy'

        if train == True:
            t_loss, v_loss = train_fn(x_train, x_test, config)
            # np.save(directory_name, config.neuralnet.get_weights(), allow_pickle=True)
            plot_error(t_loss, v_loss, config)
        else:
            try:
                weights = np.load(directory_name, allow_pickle = True)
                print("Setting weights from: " + directory_name)
                config.neuralnet.set_weights(weights)
            except:
                print("Make sure that the weights you are trying to load exist and that the size fits with the current NN.")


    # Solve
    # 1. Plot energy functions
    plot_nn_energy(config.model.ha_fn, config.neuralnet, config, name='$H_a(\\theta;x)$')
    plot_multiple_energy(config.model.h_fn, config.model.ha_fn, config.model.hd_fn, config.neuralnet, config, name1 = '$H(x)$', name2 = '$H_a(\\theta;x)$', name3 = '$H_d(\\theta;x)$')

    # 2. Time response
    solver = Timeresponse()
    solver.t_final = 15
    n_trajectories = 15

    fig = solver.ivp_multiple_solve('hd', config, n_trajectories)
    plt.tight_layout()
    image_format = 'svg'
    image_name = name + '_tresponse.svg'
    # fig.savefig('figures/'+image_name, format = image_format, dpi=1200, transparent = True)
    plt.show()

