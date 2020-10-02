import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
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


# calculate x(t+1) v(t+1)
def intergral_x(x, v, a, DT):
    x = x + v * DT + a * DT * DT / 2
    v = v + a * DT
    return x, v


def err_cal(xTrue, xPre):
    err_rate = 0.0
    for i in range(3):
        err_rate = err_rate + abs(xPre[i] - xTrue[i]) / abs(xTrue[i])
    return err_rate / 3.0 * 100.0


def err_norm(xTrue, xPre):
    return np.linalg.norm(xTrue - xPre)


def plt_show(data, num=1, color=['red']):
    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    ax.set_zlabel('Z')  # 设置z坐标轴
    for i in range(num):
        ax.scatter(data[i][0], data[i][1], data[i][2], color=color[i], s=1)
        # ax.plot(data[i][0], data[i][1], data[i][2], color=color[i])
    plt.show()

