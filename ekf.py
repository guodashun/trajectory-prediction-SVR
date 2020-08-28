import numpy as np
import math
import os
import time
import joblib
from config import cfg
from utils.dataset import load_npz
from utils.utils import cubic_speed, intergral_x, err_cal, plt_show

F = cfg['frame_rate']
T = cfg['init_time']
B = cfg['init_start']
model_dir = './model/'
test_dir = 'test_data'
test_list = os.listdir(test_dir)
model_list = os.listdir(model_dir)
model_list.remove(".keep")
test_list.remove(".keep")
show = True
verbose = False

# ekf params
# noise_w = np.diag([0.1, 0.1, 0.1]) / 1000
noise_w = np.zeros(3)
noise_v = np.diag([0.05, 0.05, 0.05]) / 50
R = noise_v ** 2
model = [joblib.load(model_dir + i) for i in model_list]
DT = 1/F


def ekf_all(traj_name):
    pos_raw_data = [load_npz(test_dir, traj_name, i) for i in range(3)]
    test_data = np.array(pos_raw_data)
    xTrue = test_data[:, T]
    vel = init_speed(test_data[:, 0:T])
    xEst = xTrue
    PEst = np.eye(3)

    # history
    hxEst = test_data[:, 0:T]
    hxEst_plt = hxEst

    # ekf time normalize
    pre_time = pos_raw_data[0].shape[0] - T
    ekf_time = cfg['ekf_time'] if pre_time >= cfg['ekf_time'] else (pre_time-1)
    ekf_cost_t = time.time()
    for i in range(ekf_time):
        cost_t = time.time()
        frame = i + T
        xTrue = test_data[:, frame+1]
        z = observation(xTrue)
        xEst, PEst = ekf_estimation(test_data[:, i:frame], vel, xEst, PEst, z) # need *T* frames data
        # xEst, PEst = ekf_estimation(hxEst[:, i:frame], xEst, PEst, z) # need *T* frames data
        if verbose:
            print("During EKF frame err is %.2f%%."%(err_cal(xTrue, xEst)), "cost time: %.3fs"%(time.time()-cost_t))
        
        # store data history
        hxEst = np.hstack((hxEst, xEst.reshape(3,1)))
        # normalize
        xEst_n = xEst.copy()
        for i in range(xEst_n.shape[0]):
            xEst_n[i] = min(xEst_n[i], 3)
            xEst_n[i] = max(xEst_n[i], -3)
        hxEst_plt = np.hstack((hxEst_plt, xEst_n.reshape(3,1)))
        # hxEst = np.hstack((hxEst, xEst.reshape(3,1)))
    
    if verbose:
        print("EKF cost time: %.3fs"%(time.time()-ekf_cost_t))

    x_data = hxEst[:, ekf_time:ekf_time+T]
    pre_cost_t = time.time()
    for j in range(pre_time - ekf_time - 1):
        frame = j + ekf_time + T + 1
        for i in range(3):
            x_v = np.array([[x_data[i][idx], vel[i][idx]] for idx in range(T)])
            acc = model[i].predict(x_v)[T - 1]
            pre_x, pre_v = intergral_x(x_data[i][T-1], vel[i][T-1], acc, 1/F)
            x_data[i][0:T - 1] = x_data[i][1:T]
            vel[i][0:T - 1] = vel[i][1:T]
            x_data[i][T-1], vel[i][T-1] = pre_x, pre_v
            xEst[i] = pre_x

        if verbose:    
            print("After EKF frame err is %.2f%%"%(err_cal(test_data[:, frame], xEst)))
        # store data history
        # normalize
        xEst_n = xEst.copy()
        for i in range(xEst_n.shape[0]):
            xEst_n[i] = min(xEst_n[i], 3)
            xEst_n[i] = max(xEst_n[i], -3)
        hxEst_plt = np.hstack((hxEst_plt, xEst_n.reshape(3,1)))
    
    if verbose:
        print("Prediction cost time: %.3fs"%(time.time()-pre_cost_t))
    err = err_cal(test_data[:, -1], xEst)
    print("EKF time is", ekf_time, "Final err is %.2f%%"%(err))

    if show:
        show_data = [test_data, hxEst_plt]
        plt_show(show_data, 2, ['red', 'blue'])


def observation(x):

    z = observation_model(x) + (noise_w @ np.random.randn(3,1)).T
    return z.flatten()

def observation_model(x):
    H = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    z = H @ x

    return z


def ekf_estimation(x_frame, vel, xEst, PEst, z):
    xPred, vel = motion_model(x_frame, vel)
    jF = jacob_f(xEst)
    Phi_x = (np.eye(len(jF)) + jF) * DT
    Q = Phi_x @ noise_w ** 2 @ Phi_x.T * DT
    PPred = jF @ PEst @ jF.T + Q

    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ S
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst # , xPred


def jacob_f(x):
    df_dx = [0]*3
    t = time.time()
    for i in range(3):
        alpha = model[i].dual_coef_.flatten()
        s_v = model[i].support_vectors_
        gamma = model[i].get_params()['gamma']
        # df/dx
        df_dx[i] = sum([alpha[j] * np.linalg.norm(x[i] - s_v[j]) * math.exp(-gamma \
                 * np.linalg.norm(x[i] - s_v[j]) ** 2) for j in range(alpha.shape[0])]) * (-2 * gamma)

    jF = np.array([
        [df_dx[0], 0, 0],
        [0, df_dx[1], 0],
        [0, 0, df_dx[2]],
    ])
    return jF


def jacob_h():
    jH = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    return jH


def init_speed(x):
    vel = [0]*3
    for i in range(len(x)):
        speed_data = cubic_speed(x[i])
        vel[i] = speed_data[1]
    return vel


def motion_model(x, v):
    for i in range(3):
        acc = model[i].predict(np.array([[x[i][idx], v[i][idx]] for idx in range(x[i].shape[0])]))[T - 1]
        pre_x, vel = intergral_x(x[i][T - 1], v[i][T - 1], acc, DT)
        x[i][0:T - 1] = x[i][1:T]
        v[i][0:T - 1] = v[i][1:T]
        x[i][T - 1] = pre_x
        v[i][T - 1] = vel
    
    return x[:, -1], vel


if __name__ == '__main__':
    for traj in test_list:
        print("The traj now is", traj)
        ekf_all(traj)
