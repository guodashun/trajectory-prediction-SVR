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
            # print("x,v", [y[j], vec_f(y[j])])
            X.append([y[j], vec_f(y[j])])
            Y.append(acc_f(y[j]))
        # plt.figure()
        # plt.plot(X,Y)
        # break
    return [X, Y]

