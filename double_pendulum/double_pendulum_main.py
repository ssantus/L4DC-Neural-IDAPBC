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
    train = False

    config  = CONF(seed=123)
    config.model = Doublependulum(x_star=[1.5,1.5,0.,0.])
    config.model.set_ja(0., 0.)
    config.model.set_ra(0.5, 0.5)
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

    x = tf.constant([0.,0.,0.,0.], shape=(1,4))
    q1, q2, p1, p2 = tf.split(x, 4, axis=1)
    m1, m2, g, l1, l2 = config.model.parameters
    M = tf.constant(
        [[tf.ones_like(q1)*(m1 + m2) * l1 ** 2, m2 * l1 * l2 * tf.cos(q1 - q2)], [m2 * l1 * l2 * tf.cos(q1 - q2), tf.ones_like(q1)*m2 * l2 ** 2]],
        shape=(1, 4, 4))
    tf.print(M)
    tf.print(config.model.h_fn(x))
    solver = Timeresponse()
    solver.t_final = 15
    n_trajectories = 30
    solution = solver.ivp_solve('h', config)
    solution = solver.ivp_solve('hd', config)
    # fig = solver.ivp_multiple_solve('hd', config, n_trajectories)
    # plt.tight_layout()
    # image_format = 'svg'
    # image_name = 'simple_pendulum_tresponse.svg'
    # fig.savefig('figures/'+image_name, format = image_format, dpi=1200, transparent = True)
    plt.show()

