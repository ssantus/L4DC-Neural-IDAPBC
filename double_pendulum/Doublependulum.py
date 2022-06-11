from utils.neuralnet import NN
import tensorflow as tf
import numpy as np


class Doublependulum():
    def __init__(self, parameters = [1., 1., 9.81, 1., 1.], x_star = [0., 0.], j = [[0.,0., 1.,0.],[0.,0., 0.,1.],[-1., 0., 0.,0.],[0., -1., 0.,0.]], r = [[0.,0.,0., 0.],[0.,0.,0., 0.],[0.,0.,0., 0.],[0.,0.,0., 0.]], g=[[0.,0.],[0.,0.],[1.,0.],[0.,1.]], g_perp=[[1.,0.],[0.,1.],[0.,0.],[0.,0.]]):
        # System definition
        self.dim_in = 4 # number of arguments in energy function
        self.dim_out = 1 # energy is a scalar function
        self.parameters = parameters # m1, m2 , g, l1, l2
        self.set_x_star(x_star)
        self.j = tf.constant(j, shape = (1,4,4))
        self.r = tf.constant(r, shape = (1,4,4))
        self.jd = self.j
        self.rd = self.r
        self.g = tf.constant(g, shape = (1,4,2))
        self.g_perp = tf.constant(g_perp, shape = (1,4,2))
        self.analytical = False # set to true to use analytical Ha

        # Data sample options
        self.q1_lim = 4.
        self.q2_lim = 4.
        self.p1_lim = 4.
        self.p2_lim = 4.
        self.delta = 0.5
        self.split = 0.7
        self.batch_size = 256

    def set_x_star(self, x_star):
        self.x_star = tf.constant(x_star, shape=(1,4))
        self.q1_star,self.q2_star, self.p1_star, self.p2_star = self.x_star[0]

    def set_ja(self, ja1: float, ja2: float):
        self.ja = tf.constant([[0.,0., ja1,0.],[0.,0., 0.,ja2],[-ja1, 0., 0.,0.],[0., -ja2, 0.,0.]], shape=(1,4,4))
        self.jd = self.j + self.ja

    def set_ra(self, ra1: float, ra2: float):
        self.ra = tf.constant([[0.,0.,0., 0.],[0.,0.,0., 0.],[0.,0.,ra1, 0.],[0.,0.,0., ra2]], shape=(1,4,4))
        self.rd = self.r + self.ra

    def set_nn(self, nn: NN):
        self.nn = nn

    @tf.function
    def h_fn(self, x, nn = None):
        """Energy function of simple pendulum"""
        q1, q2, p1, p2 = tf.split(x, 4, axis=1)
        m1, m2, g, l1, l2 = self.parameters

        # m11 = tf.ones(shape=(q1.shape[0], q1.shape[1])) * (m1 + m2) * l1 ** 2
        # m12 = m2 * l1 * l2 * tf.cos(q1 - q2)
        # m22 = tf.ones(shape=(q1.shape[0], q1.shape[1])) * m2 * l2 ** 2

        # M = tf.concat([m11, m12, m12, m22], 0)
        # M = tf.split(M, q1.shape[0], 0)
        # M = tf.reshape(M, [-1, 2, 2])

        # iM = tf.linalg.inv(M)
        # p = tf.concat((p1, p2), 1)
        # K = 0.5*tf.reduce_sum(tf.multiply(p, tf.linalg.matvec(iM, p)), axis = 1)
        # V = (m1+m2)*g*l1*(1-tf.cos(q1)) + m2*g*l2*(1-tf.cos(q2))

        K = p2*((p2*(m1 + m2))/(2*(- l2**2*m2**2*tf.cos(q1 - q2)**2 + l2**2*m2**2 + m1*l2**2*m2)) - (p1*tf.cos(q1 - q2))/(2*(- l1*l2*m2*tf.cos(q1 - q2)**2 + l1*l2*m1 + l1*l2*m2))) + \
            p1*(p1/(2*(l1**2*m1 + l1**2*m2 - l1**2*m2*tf.cos(q1 - q2)**2)) - (p2*tf.cos(q1 - q2))/(2*(- l1*l2*m2*tf.cos(q1 - q2)**2 + l1*l2*m1 + l1*l2*m2)))
        V = - g*l1*(tf.cos(q1) - 1)*(m1 + m2) - g*l2*m2*(tf.cos(q2) - 1)
        return K + V


    @tf.function
    def ha_fn(self, x, nn = None):
        """Auxiliary energy neural form for simple pendulum"""
        if self.analytical == True:
            q1, q2, p1, p2 = tf.split(x, 4, axis=1)
            m1, m2, g, l1, l2 = self.parameters
            V = (m1 + m2) * g * l1 * (1 - tf.cos(q1)) + m2 * g * l2 * (1 - tf.cos(q2))
            return - V + (q1-self.q1_star)**2 + (q2-self.q2_star)**2
        elif nn == None:
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
        return u[:,2], u[:,3]

    def data_gen_uniform(self):
        #Training data
        q1 = np.arange(-self.q1_lim + self.q1_star, self.q1_star + self.q1_lim, self.delta, dtype=np.float32)
        q2 = np.arange(-self.q2_lim + self.q2_star, self.q2_star + self.q2_lim, self.delta, dtype=np.float32)
        p1 = np.arange(-self.p1_lim + self.p1_star, self.p1_star + self.p1_lim, self.delta, dtype=np.float32)
        p2 = np.arange(-self.p2_lim + self.p2_star, self.p2_star + self.p2_lim, self.delta, dtype=np.float32)

        q1, q2, p1, p2 = np.meshgrid(q1, q2, p1, p2)
        q1 = np.expand_dims(q1.flatten(), axis=-1)
        q2 = np.expand_dims(q2.flatten(), axis=-1)
        p1 = np.expand_dims(p1.flatten(), axis=-1)
        p2 = np.expand_dims(p2.flatten(), axis=-1)
        x_train = tf.concat((q1, q2, p1, p2), axis = 1)
        x_train = tf.reshape(x_train, shape= (1,len(x_train),4))

        #Validation data
        q1 = np.random.uniform(low = -self.q1_lim + self.q1_star, high = self.q1_star + self.q1_lim, size = int(len(q1)*(1-self.split))).astype(np.float32).flatten()
        q2 = np.random.uniform(low = -self.q2_lim + self.q2_star, high = self.q2_star + self.q2_lim, size = int(len(q2)*(1-self.split))).astype(np.float32).flatten()
        p1 = np.random.uniform(low = -self.p1_lim + self.p1_star, high = self.p1_star + self.p1_lim, size = int(len(p1)*(1-self.split))).astype(np.float32).flatten()
        p2 = np.random.uniform(low = -self.p2_lim + self.p2_star, high = self.p2_star + self.p2_lim, size = int(len(p2)*(1-self.split))).astype(np.float32).flatten()
        q1 = np.expand_dims(q1, axis = -1)
        q2 = np.expand_dims(q2, axis = -1)
        p1 = np.expand_dims(p1, axis = -1)
        p2 = np.expand_dims(p2, axis = -1)
        x_test = tf.concat((q1, q2, p1, p2), axis = 1)
        x_test = tf.reshape(x_test, shape= (1,len(x_test),4))

        tf.print("Number of training samples: ", x_train.shape[0]*x_train.shape[1])
        tf.print("Number of validation samples: ", x_test.shape[0]*x_test.shape[1])
        return x_train, x_test

    def generate_data_random(self, n_samples, seed):
        #Training data
        q1 = np.random.uniform(low = -self.q1_lim + self.q1_star, high = self.q1_star + self.q1_lim, size = int(len(q1)*(1-self.split))).astype(np.float32).flatten()
        q2 = np.random.uniform(low = -self.q2_lim + self.q2_star, high = self.q2_star + self.q2_lim, size = int(len(q2)*(1-self.split))).astype(np.float32).flatten()
        p1 = np.random.uniform(low = -self.p1_lim + self.p1_star, high = self.p1_star + self.p1_lim, size = int(len(p1)*(1-self.split))).astype(np.float32).flatten()
        p2 = np.random.uniform(low = -self.p2_lim + self.p2_star, high = self.p2_star + self.p2_lim, size = int(len(p2)*(1-self.split))).astype(np.float32).flatten()
        q1 = np.expand_dims(q1, axis = -1)
        q2 = np.expand_dims(q2, axis = -1)
        p1 = np.expand_dims(p1, axis = -1)
        p2 = np.expand_dims(p2, axis = -1)
        x = np.concatenate([q1,q2,p1,p2], axis=1)
        x_train, x_test = np.split(x, [int(self.split*n_samples)])

        x_train = tf.data.Dataset.from_tensor_slices((x_train))
        x_train = x_train.shuffle(buffer_size=int(n_samples), seed=seed, reshuffle_each_iteration=True)
        x_train = x_train.batch(self.batch_size)

        #Validation data
        x_test = tf.reshape(x_test, shape= (1,len(x_test),4))

        tf.print("Number of training samples: ", int(self.split*n_samples))
        tf.print("Number of validation samples: ", x_test.shape[0]*x_test.shape[1])
        return x_train, x_test

        