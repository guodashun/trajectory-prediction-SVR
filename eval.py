import numpy as np
import joblib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
from dataset import load_npz
from traj_speed import cubic_speed, cubic_speed_single
import time
import math
# from sklearn.svm import SVR

# npz_list = os.listdir('./npz_502/')
model_dir = './model/'
model_list = os.listdir(model_dir)
frame_rate = 120
time_start = 0
time_step = 10
verbose = False

# ekf params
noise_w = np.diag([0.1, 0.1, 0.1]) / 10
noise_v = np.diag([0.05, 0.05, 0.05]) * 100
R = noise_v ** 2
model = [joblib.load(model_dir + i) for i in model_list]
DT = 1/frame_rate


def eval_position():
    pos_raw_data = [0]*3
    pos_pre_data = [0]*3
    t_start = time.time()
    for i in range(3): # x,y,z
        pos_raw_data[i] = load_npz("test_data", i, frame_rate)
        pos_pre_data[i] = pos_raw_data[i][1][0].copy()
        # print('after load what is pre',pos_raw_data[i][1], pos_pre_data[i])
        model = joblib.load(model_dir + model_list[i])
        speed_data = cubic_speed(pos_raw_data[i])
        test_data = np.array(speed_data[0])[time_start:time_start+time_step]
        for j in range(pos_raw_data[i][0].shape[1] - time_step-time_start):
            if verbose and j == 1:
                print("raw speed", speed_data)
                print("a",test_data)
            acc = model.predict(test_data)[time_step - 1]
            if verbose and j == 1:
                print("acc", acc)
            vel = test_data[time_step - 1][1] + acc / frame_rate
            pre_x = test_data[time_step - 1][0] + test_data[time_step - 1][1] / frame_rate \
                    + acc / frame_rate / frame_rate / 2 # vt + 1/2 at^2
            if verbose and j == 1:
                print("pre_x", pre_x)
            test_data[0:time_step - 1] = test_data[1:time_step]
            test_data[time_step - 1] = [pre_x, vel]
            if verbose and j == 1:
                print("b", test_data)
            pos_pre_data[i][time_step+j+time_start] = pre_x
    # print('before plot', pos_raw_data)
    if verbose:
        print("cost time:", time.time()-t_start)
    show_data = [
        [pos_raw_data[0][1][0],pos_raw_data[1][1][0],pos_raw_data[2][1][0]],
        [pos_pre_data[0],pos_pre_data[1],pos_pre_data[2]]
    ]
    plt_show(show_data, num=2, color=['red', 'green'])


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


def ekf_all():
    pos_raw_data = [load_npz("test_data", i, frame_rate) for i in range(3)]
    test_data = np.array(pos_raw_data)[:, 1].reshape(3,-1)
    # print(pos_raw_data[0])
    # print("true acc:", cubic_speed(np.array(pos_raw_data)[0]))
    xTrue = test_data[:, time_step]
    xEst = xTrue
    PEst = np.eye(3)

    # history
    hxEst = test_data[:, 0:time_step]
    hz = hxEst


    for i in range(pos_raw_data[0][0].shape[1] - time_step - 1):
        time = i + time_step
        xTrue = test_data[:, time+1]
        # print(xTrue.shape)
        z = observation(xTrue)
        xEst, PEst = ekf_estimation(test_data[:, i:time], xEst, PEst, z) # need *time_step* frames data

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
    # 改为xv 不要每次都cubic速度
    # x.shape (3, time_step)
    # 1. 用10帧还是用1帧 10帧
    # 2. 用cubic还是用微分 我感觉区别不大？用cubic
    t = np.tile(np.linspace(time_start/frame_rate, (time_step-1)/frame_rate, time_step), (3,1))
    x = np.dstack((x,t))
    # print("before cubic speed", x.shape)
    for i in range(3):
        # print("add x", x[i])
        speed_data = cubic_speed_single(x[i])
        # print("speed data", speed_data)
        acc = model[i].predict(speed_data[0])[time_step - 1]
        vel = np.array(speed_data[0])[time_step - 1][1] + acc / frame_rate
        pre_x = np.array(speed_data[0])[time_step - 1][0] + np.array(speed_data[0])[time_step - 1][1] / frame_rate \
                + acc / frame_rate / frame_rate / 2 # vt + 1/2 at^2
        x[i][0:time_step - 1] = x[i][1:time_step]
        x[i][time_step - 1] = [pre_x, vel]
    return x[:, -1, 0]


if __name__ == '__main__':
    eval_position()
    ekf_all()
