"""double_pendulum_main.py"""
from utils.visualization import *
from double_pendulum_training import *
from double_pendulum_solver import *
from Doublependulum import *


if __name__ == "__main__":
    print("Double pendulum")
    # Setup
    config  = CONF(seed=123)
    config.model = Doublependulum(x_star=[np.pi,1.5,0.,0.])
    config.model.set_ja(0., 0.)
    config.model.set_ra(1.5, 1.5)
    config.model.analytical = False
    config.neuralnet = NN(epochs= 5000, nn_width=60, dim_in=config.model.dim_in, activation = lambda x: x**2)
    config.neuralnet.epsilon = 0.1

    # Data
    x_train, x_test = config.model.data_gen_uniform()
    
    # Train
    train = False
    if not config.model.analytical:
        name = 'double_pendulum_width{}_depth{}_2'.format(config.neuralnet.nn_width, config.neuralnet.nn_depth)
        directory_name = 'weights/L4DC/'+name+'.npy'

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


    # Solve
    solver = Timeresponse()
    solver.t_final = 15
    n_trajectories = 15

    fig = solver.ivp_multiple_solve('hd', config, n_trajectories)
    plt.tight_layout()
    image_format = 'svg'
    image_name = name+'_tresponse.svg'.format(config.neuralnet.nn_width, config.neuralnet.nn_depth)
    fig.savefig('figures/'+image_name, format = image_format, dpi=1200, transparent = True)
    plt.show()

