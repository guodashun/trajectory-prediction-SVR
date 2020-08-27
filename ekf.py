import numpy as np
import math
import os
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
    hz = hxEst
    pre_time = pos_raw_data[0].shape[0] - T - 1
    ekf_time = cfg['ekf_time'] if pre_time >= cfg['ekf_time'] else pre_time
    
    for i in range(pre_time):
        time = i + T
        xTrue = test_data[:, time+1]
        z = observation(xTrue)
        xEst, PEst = ekf_estimation(test_data[:, i:time], vel, xEst, PEst, z) # need *T* frames data
        # xEst, PEst = ekf_estimation(hxEst[:, i:time], xEst, PEst, z) # need *T* frames data
        # print("Frame err is %.2f%%"%(err_cal(xTrue, xEst)))
        
        # store data history
        # normalize
        xEst_n = xEst.copy()
        for i in range(xEst_n.shape[0]):
            xEst_n[i] = min(xEst_n[i], 3)
            xEst_n[i] = max(xEst_n[i], -3)
        hxEst = np.hstack((hxEst, xEst_n.reshape(3,1)))
        # hxEst = np.hstack((hxEst, xEst.reshape(3,1)))
        hz = np.hstack((hz, z.reshape(3,1)))
    
    err = err_cal(xTrue, xEst)
    print("Final err is %.2f%%"%(err))

    if show:
        show_data = [test_data, hxEst, hz]
        plt_show(show_data, 3, ['red', 'blue', 'green'])


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
