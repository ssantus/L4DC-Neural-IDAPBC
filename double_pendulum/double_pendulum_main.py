import os
import platform
pc = platform.system()
w = False
if pc == 'Windows':
    import sys
    sys.path.insert(1, os.getcwd())
    w = True
    
from utils.visualization import *
from double_pendulum_training import *
from double_pendulum_solver import *
from Doublependulum import *


if __name__ == "__main__":
    print("Double pendulum")

    config  = CONF(seed=123)
    config.model = Doublependulum(x_star=[1.5,1.5,0.,0.])
    config.model.set_ja(0., 0.)
    config.model.set_ra(1.5, 1.5)
    config.model.analytical = False
    # config.neuralnet = NN(epochs= 15000, nn_width=20, nn_depth= 3, dim_in=config.model.dim_in)
    config.neuralnet = NN(epochs= 3000, nn_width=60, dim_in=config.model.dim_in, activation = lambda x: x**2)
    config.neuralnet.epsilon = 0.1

    x_train, x_test = config.model.data_gen_uniform()
    # n_samples = config.neuralnet.model.count_params()
    # x_train, x_test = config.model.generate_data_random(n_samples, config.seed)
    
    train = False
    # tf.print("Residuals before training: ")
    # loss_fn(x_test[0], config, residuals = True)
    if not config.model.analytical:
        if w:
            directory_name = 'c:\\Users\\ssanc\\Documents\\GitHub\\Total-Energy-Shaping-Neural-IDAPBC\\weights\\L4DC\\double_pendulum_width{}_depth{}.npy'.format(config.neuralnet.nn_width, config.neuralnet.nn_depth)
        else:
            directory_name = '../weights/L4DC/double_pendulum_width{}_depth{}.npy'.format(config.neuralnet.nn_width, config.neuralnet.nn_depth)

        if train == True:
            t_loss, v_loss = train_fn(x_train, x_test, config)
            np.save(directory_name, config.neuralnet.get_weights(), allow_pickle=True)
            plot_error(t_loss, v_loss, config)
        else:
            try:
                weights = np.load(directory_name, allow_pickle = True)
                print("Setting weights from: " + directory_name)
                config.neuralnet.set_weights(weights)
            except:
                print("Make sure that the weights you are trying to load exist and that the size fits with the current NN.")

    # plot_nn_energy(config.model.ha_fn, config.neuralnet, config)
    # plot_multiple_energy(config.model.h_fn, config.model.ha_fn, config.model.hd_fn, config.neuralnet, config)

    solver = Timeresponse()
    solver.t_final = 15
    n_trajectories = 15
    # solution = solver.ivp_solve('h', config)
    # solution = solver.ivp_solve('hd', config)
    fig = solver.ivp_multiple_solve('hd', config, n_trajectories)
    plt.tight_layout()
    image_format = 'svg'
    image_name = 'double_pendulum_width{}_depth{}_tresponse.svg'.format(config.neuralnet.nn_width, config.neuralnet.nn_depth)
    fig.savefig('figures/'+image_name, format = image_format, dpi=1200, transparent = True)
    plt.show()

