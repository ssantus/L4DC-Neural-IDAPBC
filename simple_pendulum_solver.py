"""solver.py"""
import tensorflow as tf
from scipy.integrate import solve_ivp
import sys
import numpy as np
from config import CONF
from models import *
import matplotlib.pyplot as plt
from functools import partial

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
            h_val = energy_fn(x, config.neuralnet)
        gradH_x = gradient.gradient(h_val, x)
        J = config.model.jd#tf.constant([[0., 1.], [-1., 0.]])
        R = config.model.rd#tf.constant([[0.,0.],[0.,0.5]])
        xdot = tf.linalg.matvec((J-R), gradH_x)
        return xdot.numpy()[0]
    
    def ivp_solve(self, energy_fn:str, config: CONF):
        """
        Solve initial value problem
        """
        t_eval = np.linspace(self.t_start, self.t_final, int(self.resolution*(self.t_final- self.t_start)))

        fig, axs = plt.subplots(2, 1, sharex='all', figsize=(10, 4))

        if energy_fn == 'h':
            response = 'Open-loop response'
            fig.suptitle('IDA-PBC '+ response, fontsize=13, y=1.)
            print(response)
            try:
                # solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.h_fn, config), t_span = [self.t_start, self.t_final], y0 = self.x0, t_eval = t_eval, rtol = self.rtol)
                solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.h_fn, config), t_span = [self.t_start, self.t_final], y0 = self.x0, t_eval = t_eval, rtol = self.rtol)
            except:
                print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                sys.exit()
        elif energy_fn == 'hd':
            response = 'Closed-loop response'
            fig.suptitle('IDA-PBC '+ response, fontsize=13, y=1.)
            print(response)
            try:
                solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.hd_fn, config), t_span = [self.t_start, self.t_final], y0 = self.x0, t_eval = t_eval, rtol = self.rtol)
            except:
                print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                sys.exit()

        q, p = np.split(solution.y, 2)
        q = q.flatten()
        p = p.flatten()
        t = solution.t

        axs[0].plot(t, q, label="$q(t)$")
        leg = axs[0].legend(loc=0, ncol=2, shadow=True, fancybox=True)
        axs[1].plot(t, p, label="$p(t)$")
        leg = axs[1].legend(loc=0, ncol=2, shadow=True, fancybox=True)

        return solution
            
    def ivp_multiple_solve(self, energy_fn:str, config: CONF, n_trajectories):
        """
        Solve initial value problem
        """
        t_eval = np.linspace(self.t_start, self.t_final, int(self.resolution*(self.t_final- self.t_start)))

        fig, ax = plt.subplots(figsize=(10, 4))
        q0_start = config.model.q_star-np.pi/2
        q0_final = config.model.q_star+np.pi/2
        q0s = np.linspace(q0_start, q0_final, n_trajectories)

        for q0 in q0s: 
            if energy_fn == 'h':
                response = 'Open-loop response'
                fig.suptitle('IDA-PBC '+ response, fontsize=13, y=1.)
                # print(response)
                try:
                    solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.h_fn, config), t_span = [self.t_start, self.t_final], y0 = [q0, 0.], t_eval = t_eval, rtol = self.rtol)
                except:
                    print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                    sys.exit()
            elif energy_fn == 'hd':
                response = 'Closed-loop response'
                fig.suptitle('IDA-PBC '+ response, fontsize=13, y=1.)
                # print(response)
                try:
                    solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.hd_fn, config), t_span = [self.t_start, self.t_final], y0 = [q0, 0.], t_eval = t_eval, rtol = self.rtol)
                except:
                    print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                    sys.exit()

            q, p = np.split(solution.y, 2)
            q = q.flatten()
            p = p.flatten()
            t = solution.t

            line, = ax.plot(t, q, color = '#1911df')
        line.set_label("$q(t)$")
        leg = ax.legend(loc=0, ncol=2, shadow=True, fancybox=True, fontsize=13)
        plt.xticks(fontsize= 12)
        plt.yticks(fontsize= 12)
        ax.set_xlabel('$t$'+ '  ' +'$[s]$', fontsize= 13)
        ax.set_ylabel('$\Theta$'+ '  ' +'$[rad]$', fontsize= 13)
        plt.xticks(fontsize= 11)

               

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

