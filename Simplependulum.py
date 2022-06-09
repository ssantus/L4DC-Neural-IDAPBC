from utils.neuralnet import NN
import tensorflow as tf
import numpy as np


class Simplependulum():
    def __init__(self, parameters = [1., 9.81, 1.], x_star = [0., 0.], j = [[0., 1.],[-1., 0.]], r = [[0., 0.],[0., 0.]], g=[0.,1.], g_perp=[1., 0.]):
        # System definition
        self.h_dim_in = 2 # number of arguments in energy function
        self.h_dim_out = 1 # energy is a scalar function
        self.parameters = parameters # m , g, l
        self.set_x_star(x_star)
        self.j = tf.constant(j, shape = (1,2,2))
        self.r = tf.constant(r, shape = (1,2,2))
        self.jd = self.j
        self.rd = self.r
        self.g = tf.constant(g, shape = (1,2))
        self.g_perp = tf.constant(g_perp, shape = (1,2))
        self.analytical = False # set to true to use analytical Ha

        # Data sample options
        self.q_lim = 4.
        self.p_lim = 4.
        self.delta = 0.5
        self.split = 0.7
        self.batch_size = 256

    def set_x_star(self, x_star):
        self.x_star = tf.constant(x_star, shape=(1,2))
        self.q_star, self.p_star = self.x_star[0]

    def set_ja(self, ja: float):
        self.ja = tf.constant([[0., ja],[-ja, 0.]], shape=(1,2,2))
        self.jd = self.j + self.ja

    def set_ra(self, ra: float):
        self.ra = tf.constant([[0., 0.],[0., ra]], shape=(1,2,2))
        self.rd = self.r + self.ra

    def set_nn(self, nn: NN):
        self.nn = nn

    @tf.function
    def h_fn(self, x, nn = None):
        """Energy function of simple pendulum"""
        q, p = tf.split(x, 2, axis=1)
        m, g, l = self.parameters
        return (1/(2*m*l**2))*p**2 + m*g*l*(1-tf.cos(q))

    @tf.function
    def ha_fn(self, x, nn = None):
        """Auxiliary energy neural form for simple pendulum"""
        if self.analytical == True:
            q, p = tf.split(x, 2, axis=1)
            m, g, l = self.parameters
            return - m*g*l*(1-tf.cos(q)) + (q-self.q_star)**2
        elif nn == None:
            print('Here')
            raise Exception('NN empty')
        else:
            return nn(x)

    @tf.function
    def hd_fn(self, x, nn = None):
        """Energy neural form for simple pendulum"""
        return self.h_fn(x) + self.ha_fn(x, nn)

    @tf.function
    def u_fn(self, x, nn=None):
        with tf.GradientTape(persistent=True) as gradient:
            gradient.watch(x)
            h_val = self.h_fn(x,nn)
            hd_val = self.hd_fn(x, nn)
        gradH_x = gradient.gradient(h_val, x)
        J_R = self.j-self.r

        gradHd_x = gradient.gradient(hd_val, x)
        Jd_Rd = self.jd-self.rd

        u = tf.linalg.matvec(Jd_Rd, gradHd_x) - tf.linalg.matvec(J_R, gradH_x)
        return u[:,1]

    def data_gen_uniform(self):
        #Training data
        q = np.arange(-self.q_lim + self.q_star, self.q_star + self.q_lim, self.delta, dtype=np.float32)
        p = np.arange(-self.p_lim + self.p_star, self.p_star + self.p_lim, self.delta, dtype=np.float32)
        
        q, p = np.meshgrid(q, p)
        q = np.expand_dims(q.flatten(), axis=-1)
        p = np.expand_dims(p.flatten(), axis=-1)
        x_train = tf.concat((q,p), axis = 1)
        x_train = tf.reshape(x_train, shape= (1,len(x_train),2))

        #Validation data
        q = np.random.uniform(low = -self.q_lim + self.q_star, high = self.q_star + self.q_lim, size = int(len(q)*(1-self.split))).astype(np.float32).flatten()
        p = np.random.uniform(low = -self.p_lim + self.p_star, high = self.p_star + self.p_lim, size = int(len(p)*(1-self.split))).astype(np.float32).flatten()
        q = np.expand_dims(q, axis = -1)
        p = np.expand_dims(p, axis = -1)
        x_test = tf.concat((q,p), axis = 1)
        x_test = tf.reshape(x_test, shape= (1,len(x_test),2))

        tf.print("Number of training samples: ", x_train.shape[0]*x_train.shape[1])
        tf.print("Number of validation samples: ", x_test.shape[0]*x_test.shape[1])
        return x_train, x_test

    def generate_data_random(self, n_samples, seed):
        #Training data
        q = np.random.uniform(low = -self.q_lim + self.q_star, high = self.q_star + self.q_lim, size = n_samples).astype(np.float32).flatten()
        p = np.random.uniform(low = -self.p_lim + self.p_star, high = self.p_star + self.p_lim, size = n_samples).astype(np.float32).flatten()
        q = np.expand_dims(q.flatten(), axis=-1)
        p = np.expand_dims(p.flatten(), axis=-1)
        x = np.concatenate([q,p], axis=1)
        x_train, x_test = np.split(x, [int(self.split*n_samples)])

        x_train = tf.data.Dataset.from_tensor_slices((x_train))
        x_train = x_train.shuffle(buffer_size=int(n_samples), seed=seed, reshuffle_each_iteration=True)
        x_train = x_train.batch(self.batch_size)

        #Validation data
        x_test = tf.reshape(x_test, shape= (1,len(x_test),2))

        tf.print("Number of training samples: ", int(self.split*n_samples))
        tf.print("Number of validation samples: ", x_test.shape[0]*x_test.shape[1])
        return x_train, x_test

        