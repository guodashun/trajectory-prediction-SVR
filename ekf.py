import numpy as np
import math
import os
import joblib
from dataset import load_npz
from traj_speed import cubic_speed, cubic_speed_single
from config import cfg
from utils.graphic import plt_show

# ekf params
frame_rate = cfg['frame_rate']
init_time = cfg['init_time']
init_start = cfg['init_start']
model_dir = './model/'
model_list = os.listdir(model_dir)
noise_w = np.diag([0.1, 0.1, 0.1]) / 10
noise_v = np.diag([0.05, 0.05, 0.05]) * 100
R = noise_v ** 2
model = [joblib.load(model_dir + i) for i in model_list]
DT = 1/frame_rate


def ekf_all():
    pos_raw_data = [load_npz("test_data", i, frame_rate) for i in range(3)]
    test_data = np.array(pos_raw_data)[:, 1].reshape(3,-1)
    # print(pos_raw_data[0])
    # print("true acc:", cubic_speed(np.array(pos_raw_data)[0]))
    xTrue = test_data[:, init_time]
    xEst = xTrue
    PEst = np.eye(3)

    # history
    hxEst = test_data[:, 0:init_time]
    hz = hxEst


    for i in range(pos_raw_data[0][0].shape[1] - init_time - 1):
        time = i + init_time
        xTrue = test_data[:, time+1]
        # print(xTrue.shape)
        z = observation(xTrue)
        xEst, PEst = ekf_estimation(test_data[:, i:time], xEst, PEst, z) # need *init_time* frames data

        # store data history
        # print("normalize shape:", test_data.shape, hxEst.shape, xEst.shape, z.shape)
        # normalize
        xEst_n = xEst.copy()
        for i in range(xEst_n.shape[0]):
            xEst_n[i] = min(xEst_n[i], 3)
            xEst_n[i] = max(xEst_n[i], -3)
        hxEst = np.hstack((hxEst, xEst_n.reshape(3,1)))
        hz = np.hstack((hz, z.reshape(3,1)))
    
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


def ekf_estimation(x_frame, xEst, PEst, z):
    xPred = motion_model(x_frame)
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
    return xEst, PEst


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
    # print("jF:", jF)
    return jF


def jacob_h():
    jH = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    return jH


def motion_model(x):
    # x.shape (3, init_time)
    t = np.tile(np.linspace(init_start/frame_rate, (init_time-1)/frame_rate, init_time), (3,1))
    x = np.dstack((x,t))
    # print("before cubic speed", x.shape)
    for i in range(3):
        # print("add x", x[i])
        speed_data = cubic_speed_single(x[i])
        # print("speed data", speed_data)
        acc = model[i].predict(speed_data[0])[init_time - 1]
        vel = np.array(speed_data[0])[init_time - 1][1] + acc / frame_rate
        pre_x = np.array(speed_data[0])[init_time - 1][0] + np.array(speed_data[0])[init_time - 1][1] / frame_rate \
                + acc / frame_rate / frame_rate / 2 # vt + 1/2 at^2
        x[i][0:init_time - 1] = x[i][1:init_time]
        x[i][init_time - 1] = [pre_x, vel]
    return x[:, -1, 0]


if __name__ == '__main__':
    ekf_all()
