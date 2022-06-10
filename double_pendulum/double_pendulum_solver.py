"""solver.py"""
import os
import platform
pc = platform.system()
w = False
if pc == 'Windows':
    import sys
    sys.path.insert(1, os.getcwd())
    w = True

from scipy.integrate import solve_ivp
import sys
from utils.config import CONF
from Doublependulum import *
import matplotlib.pyplot as plt


class Timeresponse():
    def __init__(self, t_start = 0., t_final = 4., resolution = 10, x0 = [np.pi/2, np.pi/2, 0., 0.], method = 'LSODA', rtol = 1e-2):
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
        J = config.model.jd
        R = config.model.rd
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
            fig.suptitle('IDA-PBC '+ response, y=1.)
            print(response)
            try:
                # solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.h_fn, config), t_span = [self.t_start, self.t_final], y0 = self.x0, t_eval = t_eval, rtol = self.rtol)
                solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.h_fn, config), t_span = [self.t_start, self.t_final], y0 = self.x0, t_eval = t_eval, rtol = self.rtol)
            except:
                print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                sys.exit()
        elif energy_fn == 'hd':
            response = 'Closed-loop response'
            fig.suptitle('IDA-PBC '+ response, y=1.)
            print(response)
            try:
                solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.hd_fn, config), t_span = [self.t_start, self.t_final], y0 = self.x0, t_eval = t_eval, rtol = self.rtol)
            except:
                print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                sys.exit()

        q1, q2, p1, p2 = np.split(solution.y, 4)
        q1 = q1.flatten()
        q2 = q2.flatten()
        p1 = p1.flatten()
        p2 = p2.flatten()
        t = solution.t

        axs[0].plot(t, q1, label="$q_1(t)$")
        leg = axs[0].legend(loc=0, ncol=2, shadow=True, fancybox=True)
        axs[1].plot(t, q2, label="$q_2(t)$")
        leg = axs[1].legend(loc=0, ncol=2, shadow=True, fancybox=True)

        return solution
    #
    # def ivp_multiple_solve(self, energy_fn:str, config: CONF, n_trajectories):
    #     """
    #     Solve initial value problem
    #     """
    #     print("Solving initial value problem for {} trajectories".format(n_trajectories))
    #     t_eval = np.linspace(self.t_start, self.t_final, int(self.resolution*(self.t_final- self.t_start)))
    #
    #     fig, axs = plt.subplots(1,2, figsize=(10, 4))
    #     q0_start = config.model.q_star-np.pi/2
    #     q0_final = config.model.q_star+np.pi/2
    #     q0s = np.linspace(q0_start, q0_final, n_trajectories)
    #
    #     for q0 in q0s:
    #         if energy_fn == 'h':
    #             response = 'Open-loop response'
    #             fig.suptitle('IDA-PBC '+ response, y=1.)
    #             # print(response)
    #             try:
    #                 solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.h_fn, config), t_span = [self.t_start, self.t_final], y0 = [q0, 0.], t_eval = t_eval, rtol = self.rtol)
    #             except:
    #                 print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
    #                 sys.exit()
    #         elif energy_fn == 'hd':
    #             response = 'Closed-loop response'
    #             fig.suptitle('Neural IDA-PBC '+ response, y=1.)
    #             # print(response)
    #             try:
    #                 solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.hd_fn, config), t_span = [self.t_start, self.t_final], y0 = [q0, 0.], t_eval = t_eval, rtol = self.rtol)
    #             except:
    #                 print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
    #                 sys.exit()
    #
    #         q, p = np.split(solution.y, 2)
    #         q = q.flatten()
    #         p = p.flatten()
    #         t = solution.t
    #         lineq, = axs[0].plot(t, q, color = '#1911df')
    #
    #         q = np.expand_dims(q.flatten(), axis=-1).astype(np.float32)
    #         p = np.expand_dims(p.flatten(), axis=-1).astype(np.float32)
    #         x = tf.concat((q, p), axis = 1)
    #         u = config.model.u_fn(x, config.neuralnet)
    #         lineu, = axs[1].plot(t, u, color = '#1911df')
    #
    #     lineq.set_label("$q(t)$")
    #     lineu.set_label("$u(t)$")
    #
    #     axs[0].set_ylabel('$\\theta$'+ '  ' +'$[rad]$')
    #     axs[1].set_ylabel('$\\tau$'+ '  ' +'$[N\cdot rad]$')
    #
    #     for i in range(2):
    #         leg = axs[i].legend(loc=1, ncol=2, shadow=True, fancybox=True)
    #         axs[i].set_xlabel('$t$'+ '  ' +'$[s]$')
    #         axs[i].grid(True, ls=':')
    #
    #     textstr = '\n'.join((
    #         r'$R=0, R_a=(%.1f)$' % (config.model.ra[0,1,1],),
    #         r'$x^\star=(%.1f,0,0)$' % (config.model.q_star)))
    #     # these are matplotlib.patch.Patch properties
    #     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #     # place a text box in upper left in axes coords
    #     axs[0].text(0.63, 0.20, textstr, transform=axs[0].transAxes,
    #             verticalalignment='top', bbox=props)
    #
    #
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     return fig


if __name__ == '__main__':
    solver = Timeresponse()
    solver.t_final = 10
    config  = CONF(seed=123)
    config.model = Doublependulum(x_star=[0.,1.,0.,0.])
    config.model.set_ja(0.,0.)
    config.model.analytical = False
    config.neuralnet = NN(epochs= 15000, nn_width=60)
    # config = CONF()
    # LOAD WEIGHTS
    # solution = solver.ivp_solve(config.model.hd_fn, config)
    # hd_fn = lambda x: config.model.hd_fn(x, nn=config.neuralnet)
    x = [[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
    x = tf.constant(x, shape = (len(x),len(x[0])), dtype = tf.float32)
    q1, q2, p1, p2 = tf.split(x, 4, axis=1)
    m1, m2, g, l1, l2 = config.model.parameters
    el11 = tf.ones(shape = (q1.shape[0], q1.shape[1]))*(m1+m2)*l1**2
    el12 = m2*l1*l2*tf.cos(q1-q2)
    el21 = m2*l1*l2*tf.cos(q1-q2)
    el22 = tf.ones(shape = (q1.shape[0], q1.shape[1]))*m2*l2**2
    tf.print("element 1,1: ",el11)
    tf.print("element 1,2: ",el12)
    tf.print("element 2,1: ",el21)
    tf.print("element 2,2: ",el22)
    M = tf.constant([el11, el12, el21, el22])
    # M = tf.constant([[tf.ones(shape = (q1.shape[0], q1.shape[1]))*(m1+m2)*l1**2, m2*l1*l2*tf.cos(q1-q2)], [m2*l1*l2*tf.cos(q1-q2), tf.ones(shape = (q1.shape[0], q1.shape[1]))*m2*l2**2]], shape=(q1.shape[0], 4, 4))
    tf.print('Inertia matrix: ', M)
    sol = solver.ph_dynamics(0, x, config.model.h_fn, config)
    solution = solver.ivp_solve('h', config)
    q1, q2, p1, p2 = np.split(solution.y, 4)
    q1 = q1.flatten()
    q2 = q2.flatten()
    p1 = p1.flatten()
    p2 = p2.flatten()
    t = solution.t

    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(10, 4))
    axs[0].plot(t, q, label="$q(t)$")
    leg = axs[0].legend(loc=0, ncol=2, shadow=True, fancybox=True)
    axs[1].plot(t, p, label="$p(t)$")
    leg = axs[1].legend(loc=0, ncol=2, shadow=True, fancybox=True)
    fig.suptitle('IDA-PBC Open-loop dynamics', fontsize=12, y=1.)
    plt.show()

