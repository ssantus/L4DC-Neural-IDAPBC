"""solver.py"""
from scipy.integrate import solve_ivp
import sys
from utils.config import CONF
from Doublependulum import *
import matplotlib.pyplot as plt


class Timeresponse():
    """Contains all the necessary to compute a time response simulation"""
    def __init__(self, t_start = 0., t_final = 4., resolution = 10, x0 = [np.pi/2, np.pi/2, 0., 0.], method = 'LSODA', rtol = 1e-2):
        self.t_start = t_start
        self.t_final = t_final
        self.resolution = resolution #number of steps per second
        self.x0 = x0
        self.rtol = rtol
        self.method = method


    def ph_dynamics(self, t, x, energy_fn, config: CONF):
        """
        Compute the port hamiltonian gradients in the form:
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
        energy_fn: string that can be set to 'h' or 'hd' for open or closed loop response respectively
        """
        t_eval = np.linspace(self.t_start, self.t_final, int(self.resolution*(self.t_final- self.t_start)))

        fig, axs = plt.subplots(2, 1, sharex='all', figsize=(10, 4))

        if energy_fn == 'h':
            response = 'Open-loop response'
            fig.suptitle('IDA-PBC '+ response, y=1.)
            print(response)
            try:
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

        # Plotting response
        q1, q2, p1, p2 = np.split(solution.y, 4)
        q1 = q1.flatten()
        q2 = q2.flatten()
        p1 = p1.flatten()
        p2 = p2.flatten()
        t = solution.t
        
        axs[0].plot(t, q1, label="$q_1(t)$")
        axs[0].plot(t, q2, label="$q_2(t)$")
        leg = axs[0].legend(loc=0, ncol=2, shadow=True, fancybox=True)
        axs[1].plot(t, p1, label="$p_1(t)$")
        axs[1].plot(t, p2, label="$p_2(t)$")
        leg = axs[1].legend(loc=0, ncol=2, shadow=True, fancybox=True)

        return solution
    

    def ivp_multiple_solve(self, energy_fn:str, config: CONF, n_trajectories):
        """
        Solve initial value problem for multiple initial conditions, works the same way as ivp_solve
        """
        print("Solving initial value problem for {} trajectories".format(n_trajectories))
        t_eval = np.linspace(self.t_start, self.t_final, int(self.resolution*(self.t_final- self.t_start)))
    
        # Initial conditions
        fig, axs = plt.subplots(1,2, figsize=(10, 4))
        q0_start = config.model.q1_star-np.pi/2
        q0_final = config.model.q1_star+np.pi/2
        q0s = np.linspace(q0_start, q0_final, n_trajectories)
    
        for q0 in q0s:
            if energy_fn == 'h':
                response = 'Open-loop response'
                fig.suptitle('Double Pendulum IDA-PBC '+ response, y=1.)
                try:
                    solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.h_fn, config), t_span = [self.t_start, self.t_final], y0 = [q0, q0, 0., 0.], t_eval = t_eval, rtol = self.rtol)
                except:
                    print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                    sys.exit()
            elif energy_fn == 'hd':
                response = 'Closed-loop response'
                fig.suptitle('Double Pendulum Neural IDA-PBC '+ response, y=1.)
                try:
                    solution = solve_ivp(fun = self.ph_dynamics, method = self.method, args=(config.model.hd_fn, config), t_span = [self.t_start, self.t_final], y0 = [q0, q0, 0., 0.], t_eval = t_eval, rtol = self.rtol)
                except:
                    print("Please define a neural network for the auxiliary energy function, use set_nn(your_nn)")
                    sys.exit()
    
            # Plotting
            q1, q2, p1, p2 = np.split(solution.y, 4)
            q1 = q1.flatten()
            q2 = q2.flatten()
            p1 = p1.flatten()
            p2 = p2.flatten()
            t = solution.t
            lineq1, = axs[0].plot(t, q1, linestyle = '-.',color = '#1911df')
            lineq2, = axs[0].plot(t, q2, linestyle = '-.', color = '#f3874a')
    
            q1 = np.expand_dims(q1.flatten(), axis=-1).astype(np.float32)
            q2 = np.expand_dims(q2.flatten(), axis=-1).astype(np.float32)
            p1 = np.expand_dims(p1.flatten(), axis=-1).astype(np.float32)
            p2 = np.expand_dims(p2.flatten(), axis=-1).astype(np.float32)
            x = tf.concat((q1,q2,p1,p2), axis = 1)
            u1, u2 = config.model.u_fn(x, config.neuralnet)
            lineu1, = axs[1].plot(t, u1, linestyle = '-.', color = '#1911df')
            lineu2, = axs[1].plot(t, u2, linestyle = '-.', color = '#f3874a')
    
        lineq1.set_label("$q1(t)$")
        lineq2.set_label("$q2(t)$")
        lineu1.set_label("$u1(t)$")
        lineu2.set_label("$u2(t)$")
    
        axs[0].set_ylabel('$\\theta$'+ '  ' +'$[rad]$')
        axs[1].set_ylabel('$\\tau$'+ '  ' +'$[N\cdot rad]$')
    
        for i in range(2):
            leg = axs[i].legend(loc=1, ncol=2, shadow=True, fancybox=True)
            axs[i].set_xlabel('$t$'+ '  ' +'$[s]$')
            axs[i].grid(True, ls=':')
    
        textstr = '\n'.join((
            r'$R=0, R_a=diag(%.1f)$' % (config.model.ra[0,2,2]),
            r'$x^\star=(%.1f,%.1f,0,0)$' % (config.model.q1_star,config.model.q2_star)))
        # matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Place the text box in the figure
        axs[0].text(0.52, 0.20, textstr, transform=axs[0].transAxes,
                verticalalignment='top', bbox=props)
    
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        return fig


if __name__ == '__main__':
    pass
