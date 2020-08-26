import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative
import sys
sys.path.append("..")
from config import cfg


def cubic_speed(x):
    x, t = timestamp(x)
    raw_f = UnivariateSpline(t,x, k=3)
    vec_f = raw_f.derivative(n=1)
    acc_f = raw_f.derivative(n=2)
    X = []
    V = []
    ACC = []
    for j in range(t.shape[0]):
        X.append(x[j])
        V.append(vec_f(t[j]))
        ACC.append(acc_f(t[j]))
    return np.array([X, V, ACC])


def timestamp(x):
    t = []
    for i in range(x.shape[0]):
        t.append(i/cfg['frame_rate'])
    t = np.array(t)
    return x, t

