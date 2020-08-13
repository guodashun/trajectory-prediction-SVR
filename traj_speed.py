import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative
import matplotlib.pyplot as plt

def cubic_speed(trajs_data):
    for i in range(np.shape(trajs_data)[1]):
        # print("traj_shape", np.shape(trajs_data))
        x = trajs_data[0][i].flatten()
        # x = trajs_data[0][i][...,0]
        # print(x.shape)
        y = trajs_data[1][i]
        # print("before cubic", x.shape, y.shape)
        raw_f = UnivariateSpline(x,y, k=3)
        # print(type(raw_f))
        vec_f = raw_f.derivative(n=1)
        # vec_f = np.polyder(raw_f,1)
        acc_f = raw_f.derivative(n=2)
        # acc_f = np.polyder(raw_f,2)
        X = []
        Y = []
        for j in range(x.shape[0]):
            # print("x,v,a", [y[j], vec_f(x[j]), acc_f(x[j])])
            X.append([y[j], vec_f(x[j])])
            Y.append(acc_f(x[j]))
        # plt.figure()
        # plt.plot(X,Y)
        # break
    return [X, Y]

def cubic_speed_single(traj_data):
    x = traj_data[:,1]
    y = traj_data[:,0]
    raw_f = UnivariateSpline(x,y, k=3)
    vec_f = raw_f.derivative(n=1)
    acc_f = raw_f.derivative(n=2)
    X = []
    Y = []
    for j in range(traj_data.shape[0]):
        X.append([y[j], vec_f(x[j])])
        Y.append(acc_f(x[j]))
    return [X, Y]
