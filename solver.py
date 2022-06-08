"""solver.py"""
import tensorflow as tf
from scipy.integrate import solve_ivp
import sys
import numpy as np
from config import CONF
from models import *
import matplotlib.pyplot as plt


class Timeresponse():
    def __init__(self, t_start = 0., t_final = 4., resolution = 10, x0 = [np.pi/2, 0.], method = 'LSODA', rtol = 1e-2):
        self.t_start = t_start
        self.t_final = t_final
        self.resolution = resolution #number of steps per second
        self.x0 = x0
        self.rtol = rtol
        self.method = method

    def ph_dynamics(self, t, x, energy_fn, config: CONF):
        """
        Compute the porh hamiltonian gradients in the form:
        dot(x) = (J-R) gradH_x + g(x)*u
        """
        x = tf.constant(x, shape = (1,len(x)), dtype = tf.float32)
        with tf.GradientTape() as gradient:
            gradient.watch(x)
            h_val = energy_fn(x, config)
        gradH_x = gradient.gradient(h_val, x)
        J = tf.constant([[0., 1.], [-1., 0.]])
        R = tf.constant([[0.,0.],[0.,0.5]])
        xdot = tf.linalg.matvec((J-R), gradH_x)
        return xdot.numpy()[0]
    
    def ivp_solve(self, energy_fn, config: CONF):
        """
        Solve initial value problem
        """
        t_eval = np.linspace(self.t_start, self.t_final, int(self.resolution*(self.t_final- self.t_start)))
        try:
            sol = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(energy_fn, config), t_span = [self.t_start, self.t_final], y0 = self.x0, t_eval = t_eval, rtol = self.rtol)
            return sol  
        except:
            print("A neural network for the auxiliary energy function needs to be defined, please use set_nn(your_nn)")
            sys.exit()
            
    

if __name__ == '__main__':
    solver = Timeresponse()
    solver.t_final = 10
    config  = CONF(seed=123)
    config.model = Simplependulum(x_star=[1.3,0.])
    config.model.set_ja(0.)
    config.model.analytical = False
    config.neuralnet = NN(epochs= 15000, nn_width=60)
    # config = CONF()
    # LOAD WEIGHTS
    # solution = solver.ivp_solve(config.model.hd_fn, config)
    # hd_fn = lambda x: config.model.hd_fn(x, nn=config.neuralnet)
    solution = solver.ivp_solve(hd_fn, config)
    q, p = np.split(solution.y, 2)
    q = q.flatten()
    p = p.flatten()
    t = solution.t

    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(10, 4))
    axs[0].plot(t, q, label="$q(t)$")
    leg = axs[0].legend(loc=0, ncol=2, shadow=True, fancybox=True)
    axs[1].plot(t, p, label="$p(t)$")
    leg = axs[1].legend(loc=0, ncol=2, shadow=True, fancybox=True)
    fig.suptitle('IDA-PBC Open-loop dynamics', fontsize=12, y=1.)
    plt.show()

