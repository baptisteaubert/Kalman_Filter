import tensorflow as tf
import numpy as np
import pylab as pl

class KalmanFilter(object):

    def __init__(self, x = None, A = None, P = None, B = None, H = None, Q = None ):

        '''Initialize filter
        # X: State at step k-1 (apriori) [n]
        # P : State errror covariance at step k -1 (apriori) [n,n]
        # A : transition matrix [n,n]
        # Q : The process noise covariance matrix
        # B: The input effect matrix
        # U : The control Input

        Parameters

        m: int - measurement size
        n : int - state size
        I : int - control input size
        x : float32 [n,1] - initial state
        A: float 32 [n,n] - state transition matrix
        Q : float32 [n,n] - process noise covariance
        u : float32 [l, 1] - control input
        B : float32 [n, l] - control input transition matrix
        z : float32 [m, 1] - measurement
        R : float32 [m,m] - measurement noise covariance
        H : float32 [m, n] - measurment transition matrix '''

        m = self._m = H.shape[0]
        n = self._n = x.shape[0]
        I = self._I = B.shape[1]
        self._x = tf.Variable(x, dtype=tf.float32, name="x")
        self._A = tf.constant(A, dtype=tf.float32, name="A")
        self._P = tf.Variable(P, dtype=tf.float32, name="P")
        self._B = tf.constant(B, dtype=tf.float32, name="B")
        self._Q = tf.constant(Q, dtype=tf.float32, name="Q")
        self._H = tf.constant(H, dtype=tf.float32, name="H")
        self._u = tf.placeholder(dtype=tf.float32, shape=[I,1], name="u")
        self._z = tf.placeholder(dtype=tf.float32, shape=[m,1], name="z")
        self._R = tf.placeholder(dtype=tf.float32, shape=[m,m], name="R")

        #X a priori projected state at step k
        #P projected error covaraince at step k

    def predict(self):
        x = self._x
        A = self._A
        P = self._P
        B = self._B
        Q = self._Q
        u = self._u

        x_pred = x.assign(tf.matmul(A,x) + tf.matmul(B,u))
        p_pred = P.assign(tf.matmul(A, tf.matmul(P, A, transpose_b = True)) + Q)
        return x_pred, p_pred

    def correct(self):
        x = self._x
        P = self._P
        H = self._H
        z = self._z
        R = self._R

        K = tf.matmul(P, tf.matmul(tf.transpose(H), tf.matrix_inverse(tf.matmul(H, tf.matmul(P, H, transpose_b = True)) + R)))
        x_corr = x.assign(x+tf.matmul(K, z - tf.matmul(H,x)))
        P_corr = P.assign(tf.matmul((1 - tf.matmul(K, H)), P))
        return K, x_corr, P_corr

    @property
    def x(self):
        return self._x
    @property
    def A(self):
        return self._A
    @property
    def P(self):
        return self._P
    @property
    def Q(self):
        return self._Q
    @property
    def B(self):
        return self._B
    @property
    def u(self):
        return self._u
    @property
    def H(self):
        return self._H
    @property
    def R(self):
        return self._R
    @property
    def z(self):
        return self._z


rnd = np.random.RandomState(0)
n_timesteps = 200
x_axis = np.linspace(0, 5*np.pi, n_timesteps)
#observations = 20*(np.sin(x_axis) + 0.5*rnd.randn(n_timesteps))
observations = 20*(np.sin(x_axis))

n, m, I = 1, 1, 1

x = np.ones([1,1])
A = np.ones([1,1])
B = np.zeros([1,1])
P = np.ones([1,1])
Q = np.array([[0.05]])
H = np.ones([1,1])
u = np.zeros([1,1])
R = np.array([[0.5]])

predictions = []

with tf.Session() as sess:
    kf = KalmanFilter(x=x, A=A, B=B, P=P, Q=Q, H=H)
    predict = kf.predict()
    correct = kf.correct()
    tf.global_variables_initializer().run()

    for i in range(0, n_timesteps):
        x_pred,_ = sess.run(predict, feed_dict = {kf.u:u})
        predictions.append(x_pred[0,0])
        sess.run(correct, feed_dict={kf.z:np.array([[observations[i]]]), kf.R:R})

    pl.figure(figsize=(16,16))
    obs_scatter = pl.scatter(x_axis, observations, marker='x', color='b', label='observations')
    position_line =pl.plot(x_axis, np.array(predictions), linestyle='-', marker='o', color='r', label='postition est.')
    pl.legend(loc='lower right')
    pl.xlim(xmin=0, xmax=x_axis.max())
    pl.xlabel('time')
    pl.legend()
    pl.show()



x = np.ones([1,1])
x = np.ones([1,1])
